"""Utilities for raster meteorology and dataset persistence."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
    EMA_HALF_LIFE_DAYS,
    EMA_MIN_PIXEL_OBSERVATIONS,
    EMA_MIN_SCANS,
    IMG_SIZE,
    LABEL_COL,
    MIN_CURRENT_NO2_FINITE_FRACTION,
    MIN_DELTA_NO2_FINITE_FRACTION,
    MIN_EMA_DELTA_NO2_FINITE_FRACTION,
    MODEL_IMAGE_KEYS,
    MODEL_MASK_KEYS,
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

CURRENT_RASTER_NAME, DELTA_RASTER_NAME, EMA_DELTA_RASTER_NAME, WIND_U_RASTER_NAME, WIND_V_RASTER_NAME = MODEL_IMAGE_KEYS
CURRENT_MASK_NAME, DELTA_MASK_NAME, EMA_DELTA_MASK_NAME = MODEL_MASK_KEYS
CURRENT_FINITE_FRACTION_COL = "current_finite_fraction"
PAIRED_FINITE_FRACTION_COL = "paired_finite_fraction"
EMA_PAIRED_FINITE_FRACTION_COL = "ema_paired_finite_fraction"
MEAN_RETRIEVAL_UNCERTAINTY_COL = "mean_retrieval_uncertainty"
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
    EMA_PAIRED_FINITE_FRACTION_COL,
    "mean_weighted_cloud_fraction",
    "mean_good_quality_fraction",
    MEAN_RETRIEVAL_UNCERTAINTY_COL,
    *HRRR_FIELDS.values(),
)
SOURCE_RECORD_INDEX_COL = "_source_record_index"
CANDIDATE_RASTER_PATH_COL = "_candidate_raster_path"
CANDIDATE_FEATURE_SCHEMA = {
    SOURCE_RECORD_INDEX_COL: pl.UInt32,
    CANDIDATE_RASTER_PATH_COL: pl.String,
    **{name: pl.Float64 for name in TABULAR_FEATURE_NAMES},
}
PROCESSING_FAILURE_SCHEMA = {"record_index": pl.Int64, "error": pl.String}
SHARD_CANDIDATES_FILE = "candidates.csv"
SHARD_FAILURES_FILE = "failures.csv"
SHARD_COMPLETION_FILE = "complete.json"


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
    if shard_size <= 0:
        raise ValueError("Shard size must be greater than zero")
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
    """Manage resumable dataset shards under one validated workspace."""

    root: Path

    def load(self, task: ShardTask, resolve_paths: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load and validate one completed shard.

        Args:
            task: Expected shard identity and source-record range.
            resolve_paths: Replace stored relative raster paths with absolute paths.

        Returns:
            Candidate features and processing failures.
        """
        shard_dir = self.root / task.split / f"{task.shard_index:06d}"
        with (shard_dir / SHARD_COMPLETION_FILE).open() as source:
            completion = json.load(source)
        candidates = pl.read_csv(
            shard_dir / SHARD_CANDIDATES_FILE,
            schema_overrides=CANDIDATE_FEATURE_SCHEMA,
        )
        failures = pl.read_csv(
            shard_dir / SHARD_FAILURES_FILE,
            schema_overrides=PROCESSING_FAILURE_SCHEMA,
        )
        expected_completion = self._completion_values(task, candidates.height, failures.height)
        if completion != expected_completion:
            raise ValueError(f"Shard {task.task_id} has an inconsistent completion marker")
        if candidates.schema != CANDIDATE_FEATURE_SCHEMA:
            raise ValueError(f"Shard {task.task_id} has an unexpected candidate schema")
        expected_indices = list(range(task.start, task.stop))
        candidate_indices = [int(value) for value in candidates[SOURCE_RECORD_INDEX_COL].to_list()]
        failure_indices = [int(value) for value in failures["record_index"].to_list()]
        if sorted(candidate_indices + failure_indices) != expected_indices:
            raise ValueError(f"Shard {task.task_id} does not contain one outcome per source record")

        absolute_paths = []
        for record_index, serialized_path in zip(
            candidate_indices,
            candidates[CANDIDATE_RASTER_PATH_COL].to_list(),
            strict=True,
        ):
            relative_path = Path(str(serialized_path))
            expected_path = Path("record-rasters") / task.split / f"{record_index:06d}.npz"
            if relative_path != expected_path:
                raise ValueError(f"Shard {task.task_id} has an unexpected raster path for record {record_index}")
            absolute_path = shard_dir / relative_path
            if not absolute_path.is_file():
                raise ValueError(f"Shard {task.task_id} is missing raster {relative_path}")
            absolute_paths.append(str(absolute_path))
        if resolve_paths:
            candidates = candidates.with_columns(pl.Series(CANDIDATE_RASTER_PATH_COL, absolute_paths, dtype=pl.String))
        return candidates, failures

    def is_complete(self, task: ShardTask) -> bool:
        """Return whether one shard has a valid terminal outcome for each source record.

        Args:
            task: Expected shard identity and source-record range.

        Returns:
            True when the shard can be reused or finalized.
        """
        try:
            self.load(task)
        except (json.JSONDecodeError, OSError, TypeError, ValueError, pl.exceptions.PolarsError):
            return False
        return True

    def create_staging(self, task: ShardTask) -> Path:
        """Clear incomplete attempts and create a private staging directory.

        Args:
            task: Shard that the caller will generate.

        Returns:
            Empty directory on the shard filesystem.
        """
        split_root = self._prepare_split_root(task.split)
        self._remove_directory(split_root / f"{task.shard_index:06d}", split_root)
        temporary_prefix = f".{task.shard_index:06d}-"
        for path in split_root.iterdir():
            if path.name.startswith(temporary_prefix):
                self._remove_directory(path, split_root)
        return Path(tempfile.mkdtemp(prefix=temporary_prefix, dir=split_root))

    def complete(
        self,
        task: ShardTask,
        staging: Path,
        output_rows: list[dict[str, object]],
        failure_rows: list[dict[str, object]],
    ) -> None:
        """Write terminal shard artifacts and publish the directory atomically.

        Args:
            task: Shard identity and expected source-record range.
            staging: Directory containing generated candidate rasters.
            output_rows: Successful record features and raster paths.
            failure_rows: Failed source-record indices and messages.
        """
        candidates = pl.DataFrame(output_rows, schema=CANDIDATE_FEATURE_SCHEMA).sort(SOURCE_RECORD_INDEX_COL)
        relative_paths = [
            str(Path(candidate_path).relative_to(staging))
            for candidate_path in candidates[CANDIDATE_RASTER_PATH_COL].to_list()
        ]
        candidates = candidates.with_columns(pl.Series(CANDIDATE_RASTER_PATH_COL, relative_paths, dtype=pl.String))
        failures = pl.DataFrame(
            sorted(failure_rows, key=lambda row: int(row["record_index"])),
            schema=PROCESSING_FAILURE_SCHEMA,
        )
        outcome_indices = [int(value) for value in candidates[SOURCE_RECORD_INDEX_COL].to_list()]
        outcome_indices.extend(int(value) for value in failures["record_index"].to_list())
        if sorted(outcome_indices) != list(range(task.start, task.stop)):
            raise ValueError(f"Shard {task.task_id} did not produce one outcome per source record")

        write_csv_atomic(candidates, staging / SHARD_CANDIDATES_FILE)
        write_csv_atomic(failures, staging / SHARD_FAILURES_FILE)
        write_json_atomic(
            self._completion_values(task, candidates.height, failures.height),
            staging / SHARD_COMPLETION_FILE,
        )
        os.replace(staging, self.root / task.split / f"{task.shard_index:06d}")

    def clear_splits(self, splits: Iterable[str]) -> None:
        """Delete shard workspaces for the supplied split names.

        Args:
            splits: Iterable of split names under this store.
        """
        shard_root = self._prepare_root()
        for split in splits:
            split_path = shard_root / str(split)
            if split_path.exists():
                self._remove_directory(split_path, shard_root)
        if not any(shard_root.iterdir()):
            shard_root.rmdir()

    def _prepare_root(self) -> Path:
        # Validate the shard root before changing descendants
        dataset_root = self.root.parent.resolve()
        if self.root.name != "shards" or self.root.is_symlink():
            raise ValueError(f"Refusing to use shard workspace outside {dataset_root}: {self.root}")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"Shard workspace is not a directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _prepare_split_root(self, split: str) -> Path:
        # Validate one split directory before changing shard contents
        shard_root = self._prepare_root()
        split_root = shard_root / split
        if split_root.parent.resolve() != shard_root.resolve() or split_root.is_symlink():
            raise ValueError(f"Refusing to use split shard workspace outside {shard_root}: {split_root}")
        if split_root.exists() and not split_root.is_dir():
            raise ValueError(f"Split shard workspace is not a directory: {split_root}")
        split_root.mkdir(parents=True, exist_ok=True)
        return split_root

    @staticmethod
    def _remove_directory(path: Path, parent: Path) -> None:
        # Restrict recursive deletion to the expected parent directory
        if path.parent.resolve() != parent.resolve():
            raise ValueError(f"Refusing to clear shard path outside {parent}: {path}")
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Shard path is not a directory: {path}")
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _completion_values(task: ShardTask, successful_count: int, failure_count: int) -> dict[str, object]:
        # Describe the shard boundary and its terminal outcomes
        return {
            "split": task.split,
            "shard_index": task.shard_index,
            "start": task.start,
            "stop": task.stop,
            "source_count": task.size,
            "successful_count": successful_count,
            "failure_count": failure_count,
        }


def select_final_records(frame: pl.DataFrame, size: int) -> pl.DataFrame:
    """Select the largest requested balanced coverage-ranked AOI subset.

    Args:
        frame: Successfully generated candidate records with finite paired coverage.
        size: Maximum number of records to select.

    Returns:
        Selected records without temporary ranking columns.
    """
    if PAIRED_FINITE_FRACTION_COL not in frame.columns:
        raise ValueError(f"Generated records are missing {PAIRED_FINITE_FRACTION_COL}")
    if not frame[PAIRED_FINITE_FRACTION_COL].is_finite().all():
        raise ValueError("Generated records contain non-finite paired raster coverage")

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


@dataclass(frozen=True)
class RecordTask:
    """Inputs needed to derive one paired record."""

    split: str
    record_index: int
    current_cache_path: str
    previous_cache_path: str
    ema_scan_paths: tuple[str, ...]
    ema_scan_age_days: tuple[float, ...]
    wind_cache_path: str
    output_path: str


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


def process_scan_batch(batch: ScanBatchTask) -> list[ScanResult]:
    """Regrid several AOIs while loading each shared granule once.

    Args:
        batch: Scans sharing an identical set of TEMPO granules.

    Returns:
        One cache location or contextual failure for every scan.
    """
    try:
        granule_indices = [build_granule_spatial_index(read_granule_pixels(path)) for path in batch.granule_paths]
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"TEMPO granule read failed: {error}"
        return [ScanResult(task.cache_key, task.cache_path, message) for task in batch.scans]

    results: list[ScanResult] = []
    for task in batch.scans:
        try:
            grid = AoiGrid.from_lon_lat(task.aoi_id, task.lon, task.lat)
            pixels = concatenate_pixels([index.select_grid(grid) for index in granule_indices])
            raster = regrid_aoi_raster(pixels, grid)
            write_raster_npz(raster, task.cache_path)
            results.append(ScanResult(task.cache_key, task.cache_path, None))
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            results.append(ScanResult(task.cache_key, task.cache_path, f"TEMPO regridding failed: {error}"))
    return results


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
                if short_name not in {"2t", "10u", "10v", "blh"}:
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
    grid_u = _interpolate_hrrr(fields["10u"], coordinates)
    grid_v = _interpolate_hrrr(fields["10v"], coordinates)
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


def process_wind_batch(batch: WindBatchTask) -> list[WindResult]:
    """Align all AOIs sharing one HRRR source file.

    Args:
        batch: Wind tasks sharing one HRRR analysis file.

    Returns:
        One cache result for each requested AOI-hour.
    """
    try:
        grid, fields = _read_hrrr_fields(batch.hrrr_path)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"HRRR read failed: {error}"
        return [WindResult(task.cache_key, task.cache_path, message) for task in batch.winds]

    results = []
    for task in batch.winds:
        try:
            arrays = _align_wind(grid, fields, task)
            arrays.update(_centre_hrrr_features(grid, fields, task))
            _write_npz_atomic(task.cache_path, **arrays)
            results.append(WindResult(task.cache_key, task.cache_path, None))
        except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            results.append(WindResult(task.cache_key, task.cache_path, f"HRRR alignment failed: {error}"))
    return results


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


def _ema_from_scans(
    ema_scan_paths: tuple[str, ...],
    ema_scan_age_days: tuple[float, ...],
) -> np.ndarray:
    # Build one causal per-pixel EMA from the available historical dates
    if len(ema_scan_paths) != len(ema_scan_age_days):
        raise ValueError("EMA scan paths and ages must have equal length")
    if len(ema_scan_paths) < EMA_MIN_SCANS:
        raise ValueError(f"Only {len(ema_scan_paths)} EMA dates are available; {EMA_MIN_SCANS} are required")

    ema_scans: list[np.ndarray] = []
    for path in ema_scan_paths:
        with np.load(path, allow_pickle=False) as cache:
            ema_scans.append(np.asarray(cache["no2"], dtype=np.float64))

    return _masked_ema(np.stack(ema_scans), np.asarray(ema_scan_age_days, dtype=np.float64))


def _masked_ema(stack: np.ndarray, age_days: np.ndarray) -> np.ndarray:
    # Compute support-aware temporal weights across the stacked scan axis
    finite = np.isfinite(stack)
    support = np.count_nonzero(finite, axis=0)
    weights = np.exp2(-age_days / EMA_HALF_LIFE_DAYS)[:, None, None]
    # Broadcasting per-date weights against the finite mask renormalizes each cell over its own dates
    valid_weights = finite * weights
    weight_sum = np.sum(valid_weights, axis=0)
    weighted_sum = np.sum(np.where(finite, stack, 0.0) * weights, axis=0)
    ema = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    valid = support >= EMA_MIN_PIXEL_OBSERVATIONS
    # The where clause leaves the NaN prefill at cells below the support floor
    np.divide(weighted_sum, weight_sum, out=ema, where=valid)
    return ema


def derive_raster_features(
    current_path: str,
    previous_path: str,
    ema_scan_paths: tuple[str, ...],
    ema_scan_age_days: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Derive paired model rasters and scan-quality scalar features.

    Args:
        current_path: Cached five-raster bundle for the current scan.
        previous_path: Cached five-raster bundle for the prior scan.
        ema_scan_paths: Cached same-time scans from the preceding two weeks.
        ema_scan_age_days: Exact age of each EMA scan before current.

    Returns:
        Model raster arrays and their plume, cloud, quality, and uncertainty
        summaries.
    """
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
        delta_no2 = np.full_like(current_no2, np.nan)
        np.subtract(current_no2, previous_no2, out=delta_no2, where=paired_valid)

        ema_no2 = _ema_from_scans(ema_scan_paths, ema_scan_age_days)
        ema_paired_valid = current_valid & np.isfinite(ema_no2)
        ema_paired_fraction = _require_coverage(
            ema_paired_valid,
            MIN_EMA_DELTA_NO2_FINITE_FRACTION,
            "EMA delta",
        )
        ema_delta_no2 = np.full_like(current_no2, np.nan)
        np.subtract(current_no2, ema_no2, out=ema_delta_no2, where=ema_paired_valid)
        p10, p50, p99 = np.percentile(delta_no2[paired_valid], [10, 50, 99])
        denominator = p50 - p10
        epsilon = np.finfo(np.float64).eps * max(abs(p10), abs(p50), 1.0)
        features = {
            "plume_score": float((p99 - p50) / max(denominator, epsilon)),
            CURRENT_FINITE_FRACTION_COL: current_fraction,
            PAIRED_FINITE_FRACTION_COL: paired_fraction,
            EMA_PAIRED_FINITE_FRACTION_COL: ema_paired_fraction,
            "mean_weighted_cloud_fraction": _paired_mean(
                current["weighted_cloud_fraction"], previous["weighted_cloud_fraction"], paired_valid
            ),
            "mean_good_quality_fraction": _paired_mean(
                current["good_quality_fraction"], previous["good_quality_fraction"], paired_valid
            ),
            MEAN_RETRIEVAL_UNCERTAINTY_COL: _paired_mean(
                current["retrieval_uncertainty"], previous["retrieval_uncertainty"], paired_valid
            ),
        }
    rasters = {
        CURRENT_RASTER_NAME: current_no2.astype(np.float32),
        DELTA_RASTER_NAME: delta_no2.astype(np.float32),
        EMA_DELTA_RASTER_NAME: ema_delta_no2.astype(np.float32),
        CURRENT_MASK_NAME: current_valid.astype(np.uint8),
        DELTA_MASK_NAME: paired_valid.astype(np.uint8),
        EMA_DELTA_MASK_NAME: ema_paired_valid.astype(np.uint8),
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


def _validate_model_rasters(rasters: dict[str, np.ndarray]) -> None:
    # Validate the complete model bundle before its atomic write
    expected_names = set((*MODEL_IMAGE_KEYS, *MODEL_MASK_KEYS))
    if set(rasters) != expected_names:
        raise ValueError(f"Model raster bundle keys differ from {sorted(expected_names)}")
    for name in MODEL_IMAGE_KEYS:
        raster = rasters[name]
        if raster.shape != (IMG_SIZE, IMG_SIZE) or raster.dtype != np.float32:
            raise ValueError(f"{name} must be a float32 {(IMG_SIZE, IMG_SIZE)} raster")
    for channel, name in enumerate(MODEL_MASK_KEYS):
        mask = rasters[name]
        if mask.shape != (IMG_SIZE, IMG_SIZE) or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{name} must be a binary uint8 {(IMG_SIZE, IMG_SIZE)} mask")
        if not np.array_equal(mask.astype(bool), np.isfinite(rasters[MODEL_IMAGE_KEYS[channel]])):
            raise ValueError(f"{name} disagrees with finite {MODEL_IMAGE_KEYS[channel]} values")
    for name in (WIND_U_RASTER_NAME, WIND_V_RASTER_NAME):
        if not np.isfinite(rasters[name]).all():
            raise ValueError(f"{name} must be finite across the full grid")


def _build_model_bundle(task: RecordTask) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    # Build and validate one complete raster and scalar feature bundle
    rasters, features = derive_raster_features(
        task.current_cache_path,
        task.previous_cache_path,
        task.ema_scan_paths,
        task.ema_scan_age_days,
    )
    wind_rasters, weather_features = extract_wind_cache(task.wind_cache_path)
    rasters.update(wind_rasters)
    features.update(weather_features)
    _validate_model_rasters(rasters)
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


def audit_record(task: RecordTask) -> RecordResult:
    """Evaluate one record entirely from persistent caches without writing a raster.

    Args:
        task: Cached TEMPO and HRRR locations for one record.

    Returns:
        Derived scalar features or contextual failure text.
    """
    try:
        _, features = _build_model_bundle(task)
        return RecordResult(task.split, task.record_index, features, None)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return RecordResult(task.split, task.record_index, {}, f"Record audit failed: {error}")


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
