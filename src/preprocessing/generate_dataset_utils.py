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
    LABEL_COL,
    MIN_CURRENT_NO2_FINITE_FRACTION,
    MIN_DELTA_NO2_FINITE_FRACTION,
    MODEL_IMAGE_KEYS,
    MODEL_MASK_KEYS,
)
from preprocessing.flux_model import estimate_aggregate_flux
from preprocessing.regrid import (
    AoiGrid,
    build_granule_spatial_index,
    concatenate_pixels,
    read_granule_pixels,
    regrid_aoi_raster,
    write_raster_npz,
)
from preprocessing.smoothing import smooth_no2
from preprocessing.stratify_utils import AOI_ID_COL

CURRENT_RASTER_NAME, DELTA_RASTER_NAME, WIND_U_RASTER_NAME, WIND_V_RASTER_NAME = MODEL_IMAGE_KEYS
CURRENT_MASK_NAME, DELTA_MASK_NAME = MODEL_MASK_KEYS
CURRENT_FINITE_FRACTION_COL = "current_finite_fraction"
PAIRED_FINITE_FRACTION_COL = "paired_finite_fraction"
MEAN_RETRIEVAL_UNCERTAINTY_COL = "mean_retrieval_uncertainty"
FLUX_NOX_COL = "flux_nox"
FLUX_CONFIDENCE_COL = "flux_confidence"
FLUX_LOG_RATIO_PREV_QTR_COL = "flux_log_ratio_prev_qtr"
SELECTION_HELPER_COLUMNS = (
    "_selection_year",
    "_selection_quarter",
    "_selection_hour_bin",
    "_stratum_rank",
    "_aoi_round",
)
HRRR_FIELDS = {
    "2t": "temperature_2m_k",
    "blh": "boundary_layer_height_m",
}
TABULAR_FEATURE_NAMES = (
    "plume_score",
    CURRENT_FINITE_FRACTION_COL,
    PAIRED_FINITE_FRACTION_COL,
    "mean_weighted_cloud_fraction",
    "mean_good_quality_fraction",
    MEAN_RETRIEVAL_UNCERTAINTY_COL,
    FLUX_NOX_COL,
    FLUX_CONFIDENCE_COL,
    *HRRR_FIELDS.values(),
)
SOURCE_RECORD_INDEX_COL = "_source_record_index"
CANDIDATE_RASTER_PATH_COL = "_candidate_raster_path"
DELTA_NO2_PATH_COL = "delta_no2_path"
CANDIDATE_FEATURE_SCHEMA = {
    SOURCE_RECORD_INDEX_COL: pl.UInt32,
    CANDIDATE_RASTER_PATH_COL: pl.String,
    **{name: pl.Float64 for name in TABULAR_FEATURE_NAMES},
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
    # Summarize retained count and paired coverage for one record group
    records = frame.height
    full_coverage = frame.filter(pl.col(PAIRED_FINITE_FRACTION_COL) >= 1.0).height
    return {
        "records": records,
        "full_coverage_records": full_coverage,
        "full_coverage_fraction": full_coverage / records if records else 0.0,
        "aoi_count": frame[AOI_ID_COL].n_unique() if records else 0,
    }


def coverage_selection_summary(frame: pl.DataFrame) -> dict[str, object]:
    """Report paired coverage and AOI representation overall and by class.

    Args:
        frame: Records carrying paired coverage and class labels.

    Returns:
        Coverage counts and AOI representation for the frame and each class.
    """
    return {
        **_coverage_group_summary(frame),
        "by_class": {str(label): _coverage_group_summary(frame.filter(pl.col(LABEL_COL) == label)) for label in (0, 1)},
    }


def select_final_records(frame: pl.DataFrame, size: int) -> pl.DataFrame:
    """Select the largest requested balanced coverage-ranked AOI subset.

    Args:
        frame: Successfully generated candidate records with finite paired coverage.
        size: Maximum number of records to select.

    Returns:
        Selected records without temporary ranking columns.
    """
    eligible_by_class = {label: frame.filter(pl.col(LABEL_COL) == label).height for label in (0, 1)}
    class_size = min(size // 2, *eligible_by_class.values())
    selected_classes = []
    for label in (0, 1):
        class_records = frame.filter(pl.col(LABEL_COL) == label)
        selected_classes.append(_rank_final_records(class_records).head(class_size))
    return pl.concat(selected_classes, how="vertical").sort(AOI_ID_COL, "date", "hour").drop(*SELECTION_HELPER_COLUMNS)


def _rank_final_records(frame: pl.DataFrame) -> pl.DataFrame:
    # Rank by coverage while retaining temporal and AOI round-robin ordering

    strata = [AOI_ID_COL, "_selection_year", "_selection_quarter", "_selection_hour_bin"]
    return (
        frame.with_columns(
            pl.col("date").dt.year().alias("_selection_year"),
            pl.col("date").dt.quarter().alias("_selection_quarter"),
            (pl.col("hour") // 4).alias("_selection_hour_bin"),
        )
        .sort(
            [*strata, PAIRED_FINITE_FRACTION_COL, "date", "hour"],
            descending=[False, False, False, False, True, False, False],
        )
        .with_columns(pl.col(AOI_ID_COL).cum_count().over(strata).alias("_stratum_rank"))
        .sort(
            [AOI_ID_COL, "_stratum_rank", PAIRED_FINITE_FRACTION_COL, "date", "hour"],
            descending=[False, False, True, False, False],
        )
        .with_columns(pl.col(AOI_ID_COL).cum_count().over(AOI_ID_COL).alias("_aoi_round"))
        .sort(
            ["_aoi_round", PAIRED_FINITE_FRACTION_COL, AOI_ID_COL, "date", "hour"],
            descending=[False, True, False, False, False],
        )
    )


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
    """Inputs needed to derive one paired record."""

    split: str
    record_index: int
    current_cache_path: str
    previous_cache_path: str
    current_wind_cache_path: str
    previous_wind_cache_path: str
    output_path: str
    source_east_km: tuple[float, ...] = (0.0,)
    source_north_km: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class RecordResult:
    """Tabular features or failure from one paired record."""

    split: str
    record_index: int
    features: dict[str, float]
    error: str | None


@dataclass(frozen=True)
class WindTask:
    """One AOI-hour wind raster and scalar meteorology cache entry."""

    cache_key: str
    aoi_id: int
    lon: float
    lat: float
    hrrr_path: str
    cache_path: str


@dataclass(frozen=True)
class WindResult:
    """Outcome of one aligned wind-cache operation."""

    cache_key: str
    cache_path: str
    error: str | None


@dataclass(frozen=True)
class WindBatchTask:
    """AOI wind rasters sharing one HRRR source file."""

    hrrr_path: str
    winds: tuple[WindTask, ...]
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


def make_wind_task(row: dict[str, object], hrrr_root: Path, cache_dir: Path) -> WindTask:
    """Create one persistent aligned-wind cache task.

    Args:
        row: Stratified record carrying its AOI and HRRR relative path.
        hrrr_root: Root of the HRRR archive.
        cache_dir: Persistent aligned-wind cache directory.

    Returns:
        Deduplicatable AOI-hour wind task.
    """
    aoi_id = int(row["aoi_id"])
    lon = float(row["lon"])
    lat = float(row["lat"])
    hrrr_path = str(hrrr_root / str(row["hrrr"]))
    identity = json.dumps(
        {"aoi": [aoi_id, lon, lat], "hrrr": hrrr_path},
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(identity.encode()).hexdigest()
    return WindTask(cache_key, aoi_id, lon, lat, hrrr_path, str(cache_dir / f"{cache_key}.npz"))


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
                if short_name not in {"2t", "u", "v", "blh"}:
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


def _align_wind(grid: _HrrrGrid, fields: dict[str, np.ndarray], task: WindTask) -> dict[str, np.ndarray]:
    # Interpolate grid-relative wind then rotate it to geographic east and north
    target_grid = AoiGrid.from_lon_lat(task.aoi_id, task.lon, task.lat)
    x_m, y_m = target_grid.cell_centres()
    coordinates = _hrrr_coordinates(grid, x_m, y_m)
    grid_u = _interpolate_hrrr(fields["u"], coordinates)
    grid_v = _interpolate_hrrr(fields["v"], coordinates)
    longitudes, latitudes = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True).transform(x_m, y_m)
    convergence = np.deg2rad(Proj(grid.crs).get_factors(longitudes, latitudes).meridian_convergence)
    eastward = grid_u * np.cos(convergence) + grid_v * np.sin(convergence)
    northward = -grid_u * np.sin(convergence) + grid_v * np.cos(convergence)
    return {
        WIND_U_RASTER_NAME: eastward.astype(np.float32),
        WIND_V_RASTER_NAME: northward.astype(np.float32),
    }


def _centre_hrrr_features(grid: _HrrrGrid, fields: dict[str, np.ndarray], task: WindTask) -> dict[str, float]:
    # Interpolate scalar weather fields at the AOI centre
    target = AoiGrid.from_lon_lat(task.aoi_id, task.lon, task.lat)
    coordinates = _hrrr_coordinates(
        grid,
        np.asarray([[target.x_m]]),
        np.asarray([[target.y_m]]),
    )
    return {
        output_name: float(_interpolate_hrrr(fields[short_name], coordinates).item())
        for short_name, output_name in HRRR_FIELDS.items()
    }


def wind_batches(winds: Iterable[WindTask], reuse_existing: bool = True) -> list[WindBatchTask]:
    """Group wind tasks so each HRRR field is read once.

    Args:
        winds: Wind tasks awaiting alignment.
        reuse_existing: Recheck initial inventory misses inside each worker.

    Returns:
        One batch per distinct HRRR file.
    """
    grouped: dict[str, list[WindTask]] = {}
    for wind in winds:
        grouped.setdefault(wind.hrrr_path, []).append(wind)
    return [WindBatchTask(path, tuple(group), reuse_existing) for path, group in grouped.items()]


def process_wind_batch(batch: WindBatchTask) -> list[WindResult]:
    """Align all AOIs sharing one HRRR source file.

    Args:
        batch: Wind tasks sharing one HRRR analysis file.

    Returns:
        One cache result for each requested AOI-hour.
    """
    reusable = {
        task.cache_key: WindResult(task.cache_key, task.cache_path, None)
        for task in batch.winds
        if batch.reuse_existing and cache_exists(task.cache_path)
    }
    pending = [task for task in batch.winds if task.cache_key not in reusable]
    if not pending:
        return [reusable[task.cache_key] for task in batch.winds]

    try:
        grid, fields = _read_hrrr_fields(batch.hrrr_path)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"HRRR read failed: {error}"
        reusable.update(
            (task.cache_key, WindResult(task.cache_key, task.cache_path, message)) for task in pending
        )
        return [reusable[task.cache_key] for task in batch.winds]

    for task in pending:
        try:
            arrays = _align_wind(grid, fields, task)
            arrays.update(_centre_hrrr_features(grid, fields, task))
            _write_npz_atomic(task.cache_path, **arrays)
            reusable[task.cache_key] = WindResult(task.cache_key, task.cache_path, None)
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            reusable[task.cache_key] = WindResult(
                task.cache_key,
                task.cache_path,
                f"HRRR alignment failed: {error}",
            )
    return [reusable[task.cache_key] for task in batch.winds]


def extract_wind_cache(path: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read aligned wind rasters and scalar weather from one cache entry.

    Args:
        path: Persistent AOI-hour wind cache path.

    Returns:
        Eastward/northward rasters and scalar weather features.
    """
    with np.load(path, allow_pickle=False) as cache:
        rasters = {
            WIND_U_RASTER_NAME: np.asarray(cache[WIND_U_RASTER_NAME], dtype=np.float32),
            WIND_V_RASTER_NAME: np.asarray(cache[WIND_V_RASTER_NAME], dtype=np.float32),
        }
        features = {name: float(cache[name]) for name in HRRR_FIELDS.values()}
    return rasters, features


def _paired_mean(current: np.ndarray, previous: np.ndarray, valid: np.ndarray) -> float:
    # Average both diagnostics over original paired NO2 support
    values = np.concatenate([current[valid], previous[valid]])
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


class _InsufficientRasterCoverageError(ValueError):
    """Signal that a derived TEMPO raster misses a fixed coverage gate."""


def _require_coverage(valid: np.ndarray, threshold: float, raster_name: str) -> float:
    # Return coverage after enforcing one strict raster eligibility gate
    fraction = float(np.mean(valid))
    if fraction <= threshold:
        raise _InsufficientRasterCoverageError(
            f"{raster_name} coverage must exceed {threshold:.0%}; got {fraction:.2%}"
        )
    return fraction


def derive_raster_features(
    current_path: str,
    previous_path: str,
    current_wind_path: str,
    previous_wind_path: str,
    source_east_km: tuple[float, ...] = (0.0,),
    source_north_km: tuple[float, ...] = (0.0,),
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Derive paired model rasters and scan-quality scalar features.

    Args:
        current_path: Cached scan bundle for the current observation.
        previous_path: Cached scan bundle for the prior observation.
        current_wind_path: Aligned wind cache for the current observation.
        previous_wind_path: Aligned wind cache for the prior observation.
        source_east_km: Facility offsets east of the AOI centre.
        source_north_km: Facility offsets north of the AOI centre.

    Returns:
        Model raster arrays and their plume, cloud, quality, and uncertainty
        summaries.
    """
    current_wind, weather_features = extract_wind_cache(current_wind_path)
    previous_wind, _ = extract_wind_cache(previous_wind_path)
    with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as previous:
        current_no2 = np.asarray(current["no2"], dtype=np.float64)
        previous_no2 = np.asarray(previous["no2"], dtype=np.float64)
        current_valid = np.isfinite(current_no2)
        previous_valid = np.isfinite(previous_no2)
        current_fraction = _require_coverage(
            current_valid,
            MIN_CURRENT_NO2_FINITE_FRACTION,
            "Current NO2",
        )

        paired_valid = current_valid & previous_valid
        paired_fraction = _require_coverage(
            paired_valid,
            MIN_DELTA_NO2_FINITE_FRACTION,
            "One-hour delta",
        )
        current_smoothed = smooth_no2(
            current_no2,
            np.asarray(current["retrieval_uncertainty"], dtype=np.float64),
            current_wind[WIND_U_RASTER_NAME],
            current_wind[WIND_V_RASTER_NAME],
        )
        previous_smoothed = smooth_no2(
            previous_no2,
            np.asarray(previous["retrieval_uncertainty"], dtype=np.float64),
            previous_wind[WIND_U_RASTER_NAME],
            previous_wind[WIND_V_RASTER_NAME],
        )
        flux = estimate_aggregate_flux(
            current_smoothed,
            np.asarray(current["retrieval_uncertainty"], dtype=np.float64),
            current_wind[WIND_U_RASTER_NAME],
            current_wind[WIND_V_RASTER_NAME],
            source_east_km,
            source_north_km,
        )
        delta_no2 = np.full_like(current_no2, np.nan)
        np.subtract(current_smoothed, previous_smoothed, out=delta_no2, where=paired_valid)

        p10, p50, p99 = np.percentile(delta_no2[paired_valid], [10, 50, 99])
        denominator = p50 - p10
        epsilon = np.finfo(np.float64).eps * max(abs(p10), abs(p50), 1.0)
        features = {
            "plume_score": float((p99 - p50) / max(denominator, epsilon)),
            CURRENT_FINITE_FRACTION_COL: current_fraction,
            PAIRED_FINITE_FRACTION_COL: paired_fraction,
            "mean_weighted_cloud_fraction": _paired_mean(
                current["weighted_cloud_fraction"], previous["weighted_cloud_fraction"], paired_valid
            ),
            "mean_good_quality_fraction": _paired_mean(
                current["good_quality_fraction"], previous["good_quality_fraction"], paired_valid
            ),
            MEAN_RETRIEVAL_UNCERTAINTY_COL: _paired_mean(
                current["retrieval_uncertainty"], previous["retrieval_uncertainty"], paired_valid
            ),
            FLUX_NOX_COL: flux.flux_nox,
            FLUX_CONFIDENCE_COL: flux.confidence,
            **weather_features,
        }
    rasters = {
        CURRENT_RASTER_NAME: current_smoothed.astype(np.float32),
        DELTA_RASTER_NAME: delta_no2.astype(np.float32),
        **current_wind,
        CURRENT_MASK_NAME: current_valid.astype(np.uint8),
        DELTA_MASK_NAME: paired_valid.astype(np.uint8),
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


def _build_model_bundle(task: RecordTask) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    # Build and validate one complete raster and scalar feature bundle
    rasters, features = derive_raster_features(
        task.current_cache_path,
        task.previous_cache_path,
        task.current_wind_cache_path,
        task.previous_wind_cache_path,
        task.source_east_km,
        task.source_north_km,
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
