"""Utilities for raster meteorology and dataset persistence."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import polars as pl
from eccodes import (
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
)
from pyproj import CRS, Proj, Transformer
from scipy.ndimage import map_coordinates

from config import (
    HOTSPOT_WINDOW_SIZE,
    IMG_RANGE,
    IMG_SIZE,
    MIN_HOTSPOT_NO2_FINITE_FRACTION,
    MIN_TIMESTEP_NO2_FINITE_FRACTION,
    SEQUENCE_TIMESTEPS,
)
from preprocessing.regrid import (
    AoiGrid,
    build_granule_spatial_index,
    concatenate_pixels,
    read_granule_pixels,
    regrid_aoi_raster,
    write_raster_npz,
)
from preprocessing.stratify_utils import AOI_ID_COL

NO2_RASTER_NAME = "no2"
NO2_MASK_NAME = "no2_mask"
TEMPERATURE_RASTER_NAME = "temperature_2m_k"
WIND_U_RASTER_NAME = "wind_u_80m_mps"
WIND_V_RASTER_NAME = "wind_v_80m_mps"
NO2_FINITE_FRACTION_COLUMNS = tuple(
    f"no2_finite_fraction_t{index}" for index in range(SEQUENCE_TIMESTEPS)
)
HOTSPOT_FINITE_FRACTION_COLUMNS = tuple(
    f"hotspot_no2_finite_fraction_t{index}" for index in range(SEQUENCE_TIMESTEPS)
)
MIN_NO2_FINITE_FRACTION_COL = "min_no2_finite_fraction"
MIN_HOTSPOT_FINITE_FRACTION_COL = "min_hotspot_no2_finite_fraction"
HOTSPOT_ROW_COL = "hotspot_row"
HOTSPOT_COLUMN_COL = "hotspot_column"
MEAN_RETRIEVAL_UNCERTAINTY_COL = "mean_retrieval_uncertainty"
TABULAR_FEATURE_NAMES = (
    *NO2_FINITE_FRACTION_COLUMNS,
    *HOTSPOT_FINITE_FRACTION_COLUMNS,
    MIN_NO2_FINITE_FRACTION_COL,
    MIN_HOTSPOT_FINITE_FRACTION_COL,
    "mean_weighted_cloud_fraction",
    "mean_good_quality_fraction",
    MEAN_RETRIEVAL_UNCERTAINTY_COL,
)
SOURCE_RECORD_INDEX_COL = "_source_record_index"
CANDIDATE_RASTER_PATH_COL = "_candidate_raster_path"
RASTER_BUNDLE_PATH_COL = "raster_bundle_path"
CANDIDATE_FEATURE_SCHEMA = {
    SOURCE_RECORD_INDEX_COL: pl.UInt32,
    CANDIDATE_RASTER_PATH_COL: pl.String,
    **{name: pl.Float64 for name in TABULAR_FEATURE_NAMES},
    HOTSPOT_ROW_COL: pl.UInt8,
    HOTSPOT_COLUMN_COL: pl.UInt8,
}
PROCESSING_FAILURE_SCHEMA = {"record_index": pl.Int64, "error": pl.String}
SHARD_CANDIDATES_FILE = "candidates.csv"
SHARD_FAILURES_FILE = "failures.csv"
MAX_PENDING_FACTOR = 2
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def bounded_parallel_map(
    function: Callable[[InputT], OutputT],
    tasks: Iterable[InputT],
    workers: int,
) -> Iterator[OutputT]:
    """Map a worker function over tasks while bounding pending futures.

    Args:
        function: Worker callable executed in a separate process.
        tasks: Task stream consumed lazily so large runs stay memory-safe.
        workers: Number of worker processes.

    Returns:
        Each worker result as it completes.
    """
    task_iterator = iter(tasks)
    max_pending = max(workers * MAX_PENDING_FACTOR, 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[Future[OutputT]] = set()
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(function, next(task_iterator)))
            except StopIteration:
                break

        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    pending.add(executor.submit(function, next(task_iterator)))
                except StopIteration:
                    pass


@dataclass(frozen=True)
class ShardTask:
    """One deterministic source-record range assigned to an array task."""

    task_id: int
    split: str
    shard_index: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        """Return the number of source records assigned to this shard."""
        return self.stop - self.start


def build_shard_plan(split_paths: dict[str, str], shard_size: int) -> list[ShardTask]:
    """Map split CSV rows onto consecutive Slurm array task IDs.

    Args:
        split_paths: Ordered split names and source CSV paths.
        shard_size: Maximum source records assigned to one task.

    Returns:
        Deterministic shard tasks spanning every source record.
    """
    tasks: list[ShardTask] = []
    for split, path in split_paths.items():
        row_count = int(pl.scan_csv(path).select(pl.len()).collect(engine="streaming").item())
        for shard_index, start in enumerate(range(0, row_count, shard_size)):
            tasks.append(
                ShardTask(
                    task_id=len(tasks),
                    split=split,
                    shard_index=shard_index,
                    start=start,
                    stop=min(start + shard_size, row_count),
                )
            )
    return tasks


@dataclass(frozen=True)
class DatasetShardStore:
    """Manage disposable dataset shards under one workspace."""

    root: Path

    def load(self, task: ShardTask, resolve_paths: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load and validate one shard.

        Args:
            task: Expected shard identity and source-record range.
            resolve_paths: Replace stored relative raster paths with absolute paths.

        Returns:
            Candidate features and processing failures.
        """
        shard_dir = self.root / task.split / f"{task.shard_index:06d}"
        candidates = pl.read_csv(
            shard_dir / SHARD_CANDIDATES_FILE,
            schema_overrides=CANDIDATE_FEATURE_SCHEMA,
        )
        failures = pl.read_csv(
            shard_dir / SHARD_FAILURES_FILE,
            schema_overrides=PROCESSING_FAILURE_SCHEMA,
        )
        expected_indices = list(range(task.start, task.stop))
        candidate_indices = [int(value) for value in candidates[SOURCE_RECORD_INDEX_COL].to_list()]
        failure_indices = [int(value) for value in failures["record_index"].to_list()]
        if sorted(candidate_indices + failure_indices) != expected_indices:
            raise ValueError(f"Shard {task.task_id} does not contain one outcome per source record")

        raster_directory = Path("record-rasters") / task.split
        relative_paths = [Path(str(value)) for value in candidates[CANDIDATE_RASTER_PATH_COL].to_list()]
        if any(path.is_absolute() or path.parent != raster_directory for path in relative_paths):
            raise ValueError(f"Shard {task.task_id} contains a raster path outside {raster_directory}")
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError(f"Shard {task.task_id} contains duplicate raster paths")
        if any(not (shard_dir / path).is_file() for path in relative_paths):
            raise ValueError(f"Shard {task.task_id} references a missing raster")

        if resolve_paths:
            absolute_paths = [str(shard_dir / path) for path in relative_paths]
            candidates = candidates.with_columns(pl.Series(CANDIDATE_RASTER_PATH_COL, absolute_paths, dtype=pl.String))
        return candidates, failures

    def create(self, task: ShardTask) -> Path:
        """Create the final directory for one fresh shard.

        Args:
            task: Shard that the caller will generate.

        Returns:
            Empty shard directory.
        """
        split_root = self._prepare_split_root(task.split)
        shard_dir = split_root / f"{task.shard_index:06d}"
        shard_dir.mkdir()
        return shard_dir

    def write(
        self,
        task: ShardTask,
        shard_dir: Path,
        output_rows: list[dict[str, object]],
        failure_rows: list[dict[str, object]],
    ) -> None:
        """Write candidate and failure metadata into a shard.

        Args:
            task: Shard identity and expected source-record range.
            shard_dir: Directory containing generated candidate rasters.
            output_rows: Successful record features and raster paths.
            failure_rows: Failed source-record indices and messages.
        """
        candidates = pl.DataFrame(output_rows, schema=CANDIDATE_FEATURE_SCHEMA).sort(SOURCE_RECORD_INDEX_COL)
        relative_paths = [
            str(Path(candidate_path).relative_to(shard_dir))
            for candidate_path in candidates[CANDIDATE_RASTER_PATH_COL].to_list()
        ]
        candidates = candidates.with_columns(pl.Series(CANDIDATE_RASTER_PATH_COL, relative_paths, dtype=pl.String))
        failures = pl.DataFrame(
            sorted(failure_rows, key=lambda row: int(row["record_index"])),
            schema=PROCESSING_FAILURE_SCHEMA,
        )
        write_csv_atomic(candidates, shard_dir / SHARD_CANDIDATES_FILE)
        write_csv_atomic(failures, shard_dir / SHARD_FAILURES_FILE)

    def clear(self) -> None:
        """Delete the complete disposable shard tree."""
        if self.root.exists():
            shutil.rmtree(self.root)

    def _prepare_root(self) -> Path:
        # Create the shard root before changing descendants
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _prepare_split_root(self, split: str) -> Path:
        # Create one split directory before changing shard contents
        split_root = self._prepare_root() / split
        split_root.mkdir(parents=True, exist_ok=True)
        return split_root



def _coverage_group_summary(frame: pl.DataFrame) -> dict[str, int | float]:
    # Summarize retained count and sequence coverage for one record group
    records = frame.height
    full_coverage = frame.filter(pl.col(MIN_NO2_FINITE_FRACTION_COL) >= 1.0).height
    return {
        "records": records,
        "full_coverage_records": full_coverage,
        "full_coverage_fraction": full_coverage / records if records else 0.0,
        "aoi_count": frame[AOI_ID_COL].n_unique() if records else 0,
    }


def coverage_selection_summary(frame: pl.DataFrame) -> dict[str, object]:
    """Report sequence coverage and AOI representation.

    Args:
        frame: Successfully generated records carrying sequence coverage.

    Returns:
        Coverage counts and AOI representation for the frame.
    """
    return _coverage_group_summary(frame)


@dataclass(frozen=True)
class ScanTask:
    """One unique AOI scan to regrid into the persistent cache."""

    cache_key: str
    aoi_id: int
    lon: float
    lat: float
    granule_paths: tuple[str, ...]
    cache_path: str


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one cached AOI-scan regridding operation."""

    cache_key: str
    cache_path: str
    error: str | None


@dataclass(frozen=True)
class ScanBatchTask:
    """AOI scans that can reuse the same loaded TEMPO granules."""

    granule_paths: tuple[str, ...]
    scans: tuple[ScanTask, ...]
    reuse_existing: bool = True


@dataclass(frozen=True)
class RecordTask:
    """Inputs needed to derive one temporal raster record."""

    split: str
    record_index: int
    scan_cache_paths: tuple[str, ...]
    weather_cache_paths: tuple[str, ...]
    hotspot_row: int
    hotspot_column: int
    output_path: str


@dataclass(frozen=True)
class RecordResult:
    """Tabular features or failure from one temporal record."""

    split: str
    record_index: int
    features: dict[str, int | float]
    error: str | None


@dataclass(frozen=True)
class WeatherTask:
    """One AOI-hour wind and temperature raster cache entry."""

    cache_key: str
    aoi_id: int
    lon: float
    lat: float
    wind_hrrr_path: str
    temperature_hrrr_path: str
    cache_path: str


@dataclass(frozen=True)
class WeatherResult:
    """Outcome of one aligned weather-cache operation."""

    cache_key: str
    cache_path: str
    error: str | None


@dataclass(frozen=True)
class WeatherBatchTask:
    """AOI weather rasters sharing the same HRRR source files."""

    wind_hrrr_path: str
    temperature_hrrr_path: str
    weather: tuple[WeatherTask, ...]
    reuse_existing: bool = True


@dataclass(frozen=True)
class _HrrrGrid:
    """Projection and array layout shared by HRRR fields."""

    crs: CRS
    rows: int
    columns: int
    x_origin_m: float
    y_origin_m: float
    x_spacing_m: float
    y_spacing_m: float


def parse_tempo_paths(serialized_paths: object, tempo_root: Path) -> tuple[str, ...]:
    """Parse a stratified CSV's JSON granule list into absolute paths.

    Args:
        serialized_paths: JSON string from stratification or an indexed path list.
        tempo_root: Root of the configured TEMPO Level 2 archive.

    Returns:
        Non-empty tuple of absolute granule paths.
    """
    relative_paths = json.loads(serialized_paths) if isinstance(serialized_paths, str) else serialized_paths
    return tuple(str(tempo_root / path) for path in relative_paths)


def make_scan_task(row: dict[str, object], path_column: str, tempo_root: Path, cache_dir: Path) -> ScanTask:
    """Create a stable cache task for one AOI scan.

    Args:
        row: Stratified record carrying AOI coordinates and granule paths.
        path_column: Either the current or previous TEMPO path-list column.
        tempo_root: Root of the configured TEMPO archive.
        cache_dir: Persistent directory for regridded scan bundles.

    Returns:
        Deduplicatable scan task with a content-derived cache key.
    """
    aoi_id = int(row["aoi_id"])
    lon = float(row["lon"])
    lat = float(row["lat"])
    granule_paths = parse_tempo_paths(row[path_column], tempo_root)
    scan_identity = {
        "aoi": [aoi_id, lon, lat],
        "granules": granule_paths,
    }
    identity = json.dumps(scan_identity, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(identity.encode()).hexdigest()
    return ScanTask(
        cache_key=cache_key,
        aoi_id=aoi_id,
        lon=lon,
        lat=lat,
        granule_paths=granule_paths,
        cache_path=str(cache_dir / f"{cache_key}.npz"),
    )


def cache_exists(path: str | Path) -> bool:
    """Return whether a cache entry exists.

    Args:
        path: Candidate cache path.

    Returns:
        True when the cache path exists.
    """
    return Path(path).is_file()


def _directory_file_names(directory: Path, suffix: str) -> set[str]:
    # One directory enumeration avoids a metadata lookup for every expected path
    try:
        with os.scandir(directory) as entries:
            return {
                entry.name
                for entry in entries
                if entry.name.endswith(suffix) and entry.is_file(follow_symlinks=False)
            }
    except FileNotFoundError:
        return set()


def cache_inventory(directory: str | Path) -> set[str]:
    """Return complete cache filenames from one directory enumeration.

    Args:
        directory: Persistent cache directory to scan.

    Returns:
        Names of regular ``.npz`` cache entries. Temporary files are excluded.
    """
    return _directory_file_names(Path(directory), suffix=".npz")


def scan_batches(scans: Iterable[ScanTask], reuse_existing: bool = True) -> list[ScanBatchTask]:
    """Group scan tasks so each worker reads one granule set once.

    Args:
        scans: Scan tasks awaiting regridding.
        reuse_existing: Recheck initial inventory misses inside each worker.

    Returns:
        One batch per distinct granule set.
    """
    grouped: dict[tuple[str, ...], list[ScanTask]] = {}
    for scan in scans:
        grouped.setdefault(scan.granule_paths, []).append(scan)
    return [ScanBatchTask(paths, tuple(group), reuse_existing) for paths, group in grouped.items()]


def process_scan_batch(batch: ScanBatchTask) -> list[ScanResult]:
    """Regrid several AOIs while loading each shared granule once.

    Args:
        batch: Scans sharing an identical set of TEMPO granules.

    Returns:
        One cache location or contextual failure for every scan.
    """
    reusable = {
        task.cache_key: ScanResult(task.cache_key, task.cache_path, None)
        for task in batch.scans
        if batch.reuse_existing and cache_exists(task.cache_path)
    }
    pending = [task for task in batch.scans if task.cache_key not in reusable]
    if not pending:
        return [reusable[task.cache_key] for task in batch.scans]

    try:
        granule_indices = [build_granule_spatial_index(read_granule_pixels(path)) for path in batch.granule_paths]
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"TEMPO granule read failed: {error}"
        reusable.update(
            (task.cache_key, ScanResult(task.cache_key, task.cache_path, message)) for task in pending
        )
        return [reusable[task.cache_key] for task in batch.scans]

    for task in pending:
        try:
            grid = AoiGrid.from_lon_lat(task.aoi_id, task.lon, task.lat)
            pixels = concatenate_pixels([index.select_grid(grid) for index in granule_indices])
            raster = regrid_aoi_raster(pixels, grid)
            write_raster_npz(raster, task.cache_path)
            reusable[task.cache_key] = ScanResult(task.cache_key, task.cache_path, None)
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            reusable[task.cache_key] = ScanResult(
                task.cache_key,
                task.cache_path,
                f"TEMPO regridding failed: {error}",
            )
    return [reusable[task.cache_key] for task in batch.scans]


def process_scan(task: ScanTask) -> ScanResult:
    """Regrid one AOI scan through the shared batch implementation.

    Args:
        task: Unique scan description and cache destination.

    Returns:
        Cache location or contextual failure text.
    """
    return process_scan_batch(ScanBatchTask(task.granule_paths, (task,)))[0]


def make_weather_task(
    row: dict[str, object],
    wind_path_column: str,
    temperature_path_column: str,
    hrrr_root: Path,
    cache_dir: Path,
) -> WeatherTask:
    """Create one persistent aligned-weather cache task.

    Args:
        row: Stratified record carrying its AOI and HRRR relative paths.
        wind_path_column: Column holding the wind GRIB path.
        temperature_path_column: Column holding the temperature GRIB path.
        hrrr_root: Root of the HRRR archive.
        cache_dir: Persistent aligned-weather cache directory.

    Returns:
        Deduplicatable AOI-hour weather task.
    """
    aoi_id = int(row["aoi_id"])
    lon = float(row["lon"])
    lat = float(row["lat"])
    wind_hrrr_path = str(hrrr_root / str(row[wind_path_column]))
    temperature_hrrr_path = str(hrrr_root / str(row[temperature_path_column]))
    identity = json.dumps(
        {
            "aoi": [aoi_id, lon, lat],
            "fields": [WIND_U_RASTER_NAME, WIND_V_RASTER_NAME, TEMPERATURE_RASTER_NAME],
            "temperature_hrrr": temperature_hrrr_path,
            "wind_hrrr": wind_hrrr_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(identity.encode()).hexdigest()
    return WeatherTask(
        cache_key,
        aoi_id,
        lon,
        lat,
        wind_hrrr_path,
        temperature_hrrr_path,
        str(cache_dir / f"{cache_key}.npz"),
    )


def _longitude_180(longitude: float) -> float:
    # Normalize GRIB longitudes for PROJ
    return (longitude + 180.0) % 360.0 - 180.0


def _hrrr_grid(message: int) -> _HrrrGrid:
    # Build the spherical Lambert grid declared by the GRIB message
    central_longitude = _longitude_180(float(codes_get(message, "LoVInDegrees")))
    latitude_origin = float(codes_get(message, "LaDInDegrees"))
    standard_parallel_1 = float(codes_get(message, "Latin1InDegrees"))
    standard_parallel_2 = float(codes_get(message, "Latin2InDegrees"))
    radius = float(codes_get(message, "radiusInMetres"))
    crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={standard_parallel_1} +lat_2={standard_parallel_2} "
        f"+lat_0={latitude_origin} +lon_0={central_longitude} +R={radius} +units=m +no_defs"
    )
    first_longitude = _longitude_180(float(codes_get(message, "longitudeOfFirstGridPointInDegrees")))
    first_latitude = float(codes_get(message, "latitudeOfFirstGridPointInDegrees"))
    x_origin, y_origin = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        first_longitude,
        first_latitude,
    )
    return _HrrrGrid(
        crs=crs,
        rows=int(codes_get(message, "Ny")),
        columns=int(codes_get(message, "Nx")),
        x_origin_m=float(x_origin),
        y_origin_m=float(y_origin),
        x_spacing_m=float(codes_get(message, "DxInMetres")),
        y_spacing_m=float(codes_get(message, "DyInMetres")),
    )


def _read_hrrr_fields(path: str) -> tuple[_HrrrGrid, dict[str, np.ndarray]]:
    # Read each required full-grid field once
    fields: dict[str, np.ndarray] = {}
    grid: _HrrrGrid | None = None
    with Path(path).open("rb") as source:
        while (message := codes_grib_new_from_file(source)) is not None:
            try:
                short_name = str(codes_get(message, "shortName"))
                if short_name not in {"2t", "u", "v"}:
                    continue
                if grid is None:
                    grid = _hrrr_grid(message)
                fields[short_name] = np.asarray(codes_get_array(message, "values"), dtype=np.float32).reshape(
                    grid.rows,
                    grid.columns,
                )
            finally:
                codes_release(message)
    return grid, fields


def _hrrr_coordinates(grid: _HrrrGrid, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Express EPSG:5070 coordinates as fractional HRRR row and column positions
    x_hrrr, y_hrrr = Transformer.from_crs("EPSG:5070", grid.crs, always_xy=True).transform(x_m, y_m)
    rows = (y_hrrr - grid.y_origin_m) / grid.y_spacing_m
    columns = (x_hrrr - grid.x_origin_m) / grid.x_spacing_m
    return rows, columns


def _interpolate_hrrr(field: np.ndarray, coordinates: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    # Bilinearly sample a native HRRR field at target cell centres
    return map_coordinates(field, coordinates, order=1, mode="nearest")


def _align_weather(
    wind_grid: _HrrrGrid,
    wind_fields: dict[str, np.ndarray],
    temperature_grid: _HrrrGrid,
    temperature_fields: dict[str, np.ndarray],
    task: WeatherTask,
) -> dict[str, np.ndarray]:
    # Interpolate weather and rotate grid-relative wind to geographic coordinates
    target_grid = AoiGrid.from_lon_lat(task.aoi_id, task.lon, task.lat)
    x_m, y_m = target_grid.cell_centres()
    wind_coordinates = _hrrr_coordinates(wind_grid, x_m, y_m)
    temperature_coordinates = _hrrr_coordinates(temperature_grid, x_m, y_m)
    grid_u = _interpolate_hrrr(wind_fields["u"], wind_coordinates)
    grid_v = _interpolate_hrrr(wind_fields["v"], wind_coordinates)
    temperature = _interpolate_hrrr(temperature_fields["2t"], temperature_coordinates)
    longitudes, latitudes = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True).transform(x_m, y_m)
    convergence = np.deg2rad(Proj(wind_grid.crs).get_factors(longitudes, latitudes).meridian_convergence)
    eastward = grid_u * np.cos(convergence) + grid_v * np.sin(convergence)
    northward = -grid_u * np.sin(convergence) + grid_v * np.cos(convergence)
    return {
        WIND_U_RASTER_NAME: eastward.astype(np.float32),
        WIND_V_RASTER_NAME: northward.astype(np.float32),
        TEMPERATURE_RASTER_NAME: temperature.astype(np.float32),
    }


def weather_batches(
    weather: Iterable[WeatherTask],
    reuse_existing: bool = True,
) -> list[WeatherBatchTask]:
    """Group weather tasks so each HRRR field is read once.

    Args:
        weather: Weather tasks awaiting alignment.
        reuse_existing: Recheck initial inventory misses inside each worker.

    Returns:
        One batch per distinct pair of HRRR files.
    """
    grouped: dict[tuple[str, str], list[WeatherTask]] = {}
    for task in weather:
        key = (task.wind_hrrr_path, task.temperature_hrrr_path)
        grouped.setdefault(key, []).append(task)
    return [
        WeatherBatchTask(wind_path, temperature_path, tuple(group), reuse_existing)
        for (wind_path, temperature_path), group in grouped.items()
    ]


def process_weather_batch(batch: WeatherBatchTask) -> list[WeatherResult]:
    """Align all AOIs sharing the same HRRR source files.

    Args:
        batch: Weather tasks sharing the same HRRR source files.

    Returns:
        One cache result for each requested AOI-hour.
    """
    reusable = {
        task.cache_key: WeatherResult(task.cache_key, task.cache_path, None)
        for task in batch.weather
        if batch.reuse_existing and cache_exists(task.cache_path)
    }
    pending = [task for task in batch.weather if task.cache_key not in reusable]
    if not pending:
        return [reusable[task.cache_key] for task in batch.weather]

    try:
        wind_grid, wind_fields = _read_hrrr_fields(batch.wind_hrrr_path)
        if batch.temperature_hrrr_path == batch.wind_hrrr_path:
            temperature_grid, temperature_fields = wind_grid, wind_fields
        else:
            temperature_grid, temperature_fields = _read_hrrr_fields(batch.temperature_hrrr_path)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"HRRR read failed: {error}"
        reusable.update(
            (task.cache_key, WeatherResult(task.cache_key, task.cache_path, message)) for task in pending
        )
        return [reusable[task.cache_key] for task in batch.weather]

    for task in pending:
        try:
            arrays = _align_weather(
                wind_grid,
                wind_fields,
                temperature_grid,
                temperature_fields,
                task,
            )
            _write_npz_atomic(task.cache_path, **arrays)
            reusable[task.cache_key] = WeatherResult(task.cache_key, task.cache_path, None)
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            reusable[task.cache_key] = WeatherResult(
                task.cache_key,
                task.cache_path,
                f"HRRR weather alignment failed: {error}",
            )
    return [reusable[task.cache_key] for task in batch.weather]


def extract_weather_cache(path: str) -> dict[str, np.ndarray]:
    """Read aligned wind and temperature rasters from one cache entry.

    Args:
        path: Persistent AOI-hour weather cache path.

    Returns:
        Eastward wind, northward wind, and temperature rasters.
    """
    with np.load(path, allow_pickle=False) as cache:
        return {
            name: np.asarray(cache[name], dtype=np.float32)
            for name in (WIND_U_RASTER_NAME, WIND_V_RASTER_NAME, TEMPERATURE_RASTER_NAME)
        }


def _sequence_mean(values_by_timestep: list[np.ndarray], masks: list[np.ndarray]) -> float:
    # Average one diagnostic over each timestep's independent NO2 support
    values = np.concatenate([values[valid] for values, valid in zip(values_by_timestep, masks, strict=True)])
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


class _InsufficientRasterCoverageError(ValueError):
    """Signal that a derived TEMPO raster misses a fixed coverage gate."""


def _require_coverage(valid: np.ndarray, threshold: float, raster_name: str) -> float:
    # Return coverage after enforcing one inclusive raster eligibility gate
    fraction = float(np.mean(valid))
    return _require_fraction(fraction, threshold, raster_name)


def _require_fraction(fraction: float, threshold: float, raster_name: str) -> float:
    # Enforce an inclusive eligibility gate on a precomputed coverage fraction
    if fraction < threshold:
        raise _InsufficientRasterCoverageError(
            f"{raster_name} coverage must be at least {threshold:.0%}; got {fraction:.2%}"
        )
    return fraction


def select_hotspot_cell(
    source_east_km: tuple[float, ...],
    source_north_km: tuple[float, ...],
    source_unit_counts: tuple[int, ...],
) -> tuple[int, int]:
    """Select the highest-unit source cell with a centroid-distance tie-break.

    Args:
        source_east_km: Facility offsets east of the AOI centre.
        source_north_km: Facility offsets north of the AOI centre.
        source_unit_counts: Modeled unit counts aligned with the offsets.

    Returns:
        Zero-indexed hotspot row and column in the model raster.
    """
    source_count = len(source_east_km)
    if source_count == 0 or len(source_north_km) != source_count or len(source_unit_counts) != source_count:
        raise ValueError("Source coordinates and unit counts must be non-empty and aligned")
    east = np.asarray(source_east_km, dtype=np.float64)
    north = np.asarray(source_north_km, dtype=np.float64)
    counts = np.asarray(source_unit_counts, dtype=np.int64)

    cell_size_km = IMG_RANGE / IMG_SIZE
    half_extent_km = IMG_RANGE / 2
    columns = np.floor((east + half_extent_km) / cell_size_km).astype(np.int64)
    rows = np.floor((half_extent_km - north) / cell_size_km).astype(np.int64)

    clusters: dict[tuple[int, int], tuple[int, float, float]] = {}
    for row, column, source_east, source_north, unit_count in zip(
        rows,
        columns,
        east,
        north,
        counts,
        strict=True,
    ):
        key = (int(row), int(column))
        total, weighted_east, weighted_north = clusters.get(key, (0, 0.0, 0.0))
        clusters[key] = (
            total + int(unit_count),
            weighted_east + float(source_east * unit_count),
            weighted_north + float(source_north * unit_count),
        )

    ranked = []
    for (row, column), (unit_count, weighted_east, weighted_north) in clusters.items():
        centroid_distance_squared = (weighted_east / unit_count) ** 2 + (weighted_north / unit_count) ** 2
        ranked.append((-unit_count, centroid_distance_squared, row, column))
    _, _, row, column = min(ranked)
    return row, column


def hotspot_finite_fraction(valid: np.ndarray, hotspot_row: int, hotspot_column: int) -> float:
    """Calculate finite coverage in the configured hotspot window.

    Args:
        valid: Two-dimensional NO2 validity mask.
        hotspot_row: Selected source-cluster row.
        hotspot_column: Selected source-cluster column.

    Returns:
        Fraction of valid cells in the complete hotspot window.
    """
    radius = HOTSPOT_WINDOW_SIZE // 2
    if not radius <= hotspot_row < IMG_SIZE - radius or not radius <= hotspot_column < IMG_SIZE - radius:
        raise ValueError("Hotspot is too close to the AOI boundary for a complete window")
    window = valid[
        hotspot_row - radius : hotspot_row + radius + 1,
        hotspot_column - radius : hotspot_column + radius + 1,
    ]
    return float(np.mean(window))


def derive_raster_features(
    scan_paths: tuple[str, ...],
    weather_paths: tuple[str, ...],
    hotspot_row: int,
    hotspot_column: int,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    """Build time-major model rasters and scan-quality scalar features.

    Args:
        scan_paths: Oldest-to-newest cached TEMPO scan bundles.
        weather_paths: Matching oldest-to-newest weather cache bundles.
        hotspot_row: Row containing the selected largest source cluster.
        hotspot_column: Column containing the selected largest source cluster.

    Returns:
        Model raster arrays and their retrieval-quality diagnostics.
    """
    no2_values: list[np.ndarray] = []
    no2_masks: list[np.ndarray] = []
    cloud_values: list[np.ndarray] = []
    quality_values: list[np.ndarray] = []
    uncertainty_values: list[np.ndarray] = []
    finite_fractions: list[float] = []
    hotspot_fractions: list[float] = []
    for index, path in enumerate(scan_paths):
        with np.load(path, allow_pickle=False) as scan:
            no2 = np.asarray(scan["no2"], dtype=np.float32)
            valid = np.isfinite(no2)
            finite_fractions.append(
                _require_coverage(
                    valid,
                    MIN_TIMESTEP_NO2_FINITE_FRACTION,
                    f"Timestep {index} NO2",
                )
            )
            hotspot_fraction = hotspot_finite_fraction(valid, hotspot_row, hotspot_column)
            hotspot_fractions.append(
                _require_fraction(
                    hotspot_fraction,
                    MIN_HOTSPOT_NO2_FINITE_FRACTION,
                    f"Timestep {index} hotspot NO2",
                )
            )
            no2_values.append(no2)
            no2_masks.append(valid)
            cloud_values.append(np.asarray(scan["weighted_cloud_fraction"], dtype=np.float32))
            quality_values.append(np.asarray(scan["good_quality_fraction"], dtype=np.float32))
            uncertainty_values.append(np.asarray(scan["retrieval_uncertainty"], dtype=np.float32))

    weather = [extract_weather_cache(path) for path in weather_paths]
    features = {
        **dict(zip(NO2_FINITE_FRACTION_COLUMNS, finite_fractions, strict=True)),
        **dict(zip(HOTSPOT_FINITE_FRACTION_COLUMNS, hotspot_fractions, strict=True)),
        MIN_NO2_FINITE_FRACTION_COL: min(finite_fractions),
        MIN_HOTSPOT_FINITE_FRACTION_COL: min(hotspot_fractions),
        HOTSPOT_ROW_COL: hotspot_row,
        HOTSPOT_COLUMN_COL: hotspot_column,
        "mean_weighted_cloud_fraction": _sequence_mean(cloud_values, no2_masks),
        "mean_good_quality_fraction": _sequence_mean(quality_values, no2_masks),
        MEAN_RETRIEVAL_UNCERTAINTY_COL: _sequence_mean(uncertainty_values, no2_masks),
    }
    rasters = {
        NO2_RASTER_NAME: np.stack(no2_values),
        NO2_MASK_NAME: np.stack(no2_masks).astype(np.uint8),
        TEMPERATURE_RASTER_NAME: np.stack([item[TEMPERATURE_RASTER_NAME] for item in weather]),
        WIND_U_RASTER_NAME: np.stack([item[WIND_U_RASTER_NAME] for item in weather]),
        WIND_V_RASTER_NAME: np.stack([item[WIND_V_RASTER_NAME] for item in weather]),
    }
    return rasters, features


def _write_npz_atomic(destination: str, **arrays: np.ndarray | float) -> None:
    # Keep interrupted workers from leaving apparently complete samples
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def _build_model_bundle(task: RecordTask) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    # Build and validate one complete raster and scalar feature bundle
    rasters, features = derive_raster_features(
        task.scan_cache_paths,
        task.weather_cache_paths,
        task.hotspot_row,
        task.hotspot_column,
    )
    return rasters, features


def process_record(task: RecordTask) -> RecordResult:
    """Create one persistent model raster bundle and its tabular features.

    Args:
        task: Cached TEMPO, HRRR, and output locations for one record.

    Returns:
        Derived scalar features or contextual failure text.
    """
    try:
        rasters, features = _build_model_bundle(task)
        _write_npz_atomic(task.output_path, **rasters)
        return RecordResult(task.split, task.record_index, features, None)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return RecordResult(task.split, task.record_index, {}, f"Record processing failed: {error}")


def write_json_atomic(values: dict[str, object], destination: Path) -> None:
    """Write a JSON object through an atomic replacement.

    Args:
        values: JSON-safe object to persist.
        destination: Final JSON path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(values, temporary, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def write_csv_atomic(frame: pl.DataFrame, destination: Path) -> None:
    """Write a CSV through an atomic replacement.

    Args:
        frame: Output rows to persist.
        destination: Final CSV path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.write_csv(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
