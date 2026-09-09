"""Utilities for raster meteorology and dataset persistence."""

import hashlib
import json
import os
import tempfile
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
    CENTRAL_COVERAGE_WINDOW_SIZE,
    EMA_HALF_LIFE_DAYS,
    EMA_MIN_PIXEL_OBSERVATIONS,
    EMA_MIN_SCANS,
    IMG_SIZE,
    LABEL_COL,
    MIN_PAIRED_FINITE_FRACTION,
    MODEL_IMAGE_KEYS,
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
NO_PAIRED_FINITE_NO2_ERROR = "Paired TEMPO scans have no cells with finite NO2 in both rasters"
PAIRED_FINITE_FRACTION_COL = "paired_finite_fraction"
CENTRAL_FINITE_FRACTION_COL = "central_finite_fraction"
MEAN_RETRIEVAL_UNCERTAINTY_COL = "mean_retrieval_uncertainty"
RASTER_QUALITY_SCORE_COL = "raster_quality_score"
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
    PAIRED_FINITE_FRACTION_COL,
    CENTRAL_FINITE_FRACTION_COL,
    "mean_weighted_cloud_fraction",
    "mean_good_quality_fraction",
    MEAN_RETRIEVAL_UNCERTAINTY_COL,
    *HRRR_FIELDS.values(),
)


def eligible_generated_records(frame: pl.DataFrame) -> pl.DataFrame:
    """Filter records by raster quality and add a bounded score.

    Args:
        frame: Generated candidate records with raster-quality summaries.

    Returns:
        Eligible records carrying a bounded raster-quality score.
    """
    paired = pl.col(PAIRED_FINITE_FRACTION_COL)
    return frame.filter(paired.is_finite() & (paired >= MIN_PAIRED_FINITE_FRACTION)).with_columns(
        paired.alias(RASTER_QUALITY_SCORE_COL)
    )


def select_final_records(frame: pl.DataFrame, size: int) -> pl.DataFrame:
    """Select an exactly balanced quality-ranked AOI subset.

    Args:
        frame: Generated candidate records eligible for final selection.
        size: Exact number of records to select.

    Returns:
        Selected records without temporary ranking columns.
    """
    eligible = eligible_generated_records(frame)
    class_size = size // 2
    selected_classes = []
    for label in (0, 1):
        class_records = eligible.filter(pl.col(LABEL_COL) == label)
        if class_records.height < class_size:
            raise ValueError(
                f"Only {class_records.height:,} class {label} records pass raster-quality gates; "
                f"cannot produce the requested {class_size:,}"
            )
        selected_classes.append(_rank_final_records(class_records).head(class_size))
    return pl.concat(selected_classes, how="vertical").sort(AOI_ID_COL, "date", "hour").drop(*SELECTION_HELPER_COLUMNS)


def _rank_final_records(eligible: pl.DataFrame) -> pl.DataFrame:
    # Interleave temporal strata within each AOI before global AOI rounds

    strata = [AOI_ID_COL, "_selection_year", "_selection_quarter", "_selection_hour_bin"]
    return (
        eligible.with_columns(
            pl.col("date").dt.year().alias("_selection_year"),
            pl.col("date").dt.quarter().alias("_selection_quarter"),
            (pl.col("hour") // 4).alias("_selection_hour_bin"),
        )
        .sort(
            [*strata, RASTER_QUALITY_SCORE_COL, "date", "hour"],
            descending=[False, False, False, False, True, False, False],
        )
        .with_columns(pl.col(AOI_ID_COL).cum_count().over(strata).alias("_stratum_rank"))
        .sort(
            [AOI_ID_COL, "_stratum_rank", RASTER_QUALITY_SCORE_COL, "date", "hour"],
            descending=[False, False, True, False, False],
        )
        .with_columns(pl.col(AOI_ID_COL).cum_count().over(AOI_ID_COL).alias("_aoi_round"))
        .sort(
            ["_aoi_round", RASTER_QUALITY_SCORE_COL, AOI_ID_COL, "date", "hour"],
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
    historical_cache_paths: tuple[str, ...]
    historical_age_days: tuple[float, ...]
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
    # Average both scans over the cells that contribute to the NO2 delta
    values = np.concatenate([current[valid], previous[valid]])
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _historical_ema(
    historical_paths: tuple[str, ...],
    age_days: tuple[float, ...],
) -> np.ndarray:
    # Build one causal EMA from persistent regridded scan-cache entries
    if len(historical_paths) != len(age_days):
        raise ValueError("Historical cache paths and ages must have equal length")
    if len(historical_paths) < EMA_MIN_SCANS:
        raise ValueError(f"Only {len(historical_paths)} historical scans are available")
    historical = []
    for path in historical_paths:
        with np.load(path, allow_pickle=False) as cache:
            historical.append(np.asarray(cache["no2"], dtype=np.float64))
    stack = np.stack(historical)
    weights = np.exp2(-np.asarray(age_days, dtype=np.float64) / EMA_HALF_LIFE_DAYS)[:, None, None]
    finite = np.isfinite(stack)
    support = finite.sum(axis=0)
    weighted_sum = np.nansum(stack * weights, axis=0)
    weight_sum = np.sum(finite * weights, axis=0)
    ema = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    usable = support >= EMA_MIN_PIXEL_OBSERVATIONS
    ema[usable] = weighted_sum[usable] / weight_sum[usable]
    return ema


def derive_raster_features(
    current_path: str,
    previous_path: str,
    historical_paths: tuple[str, ...],
    historical_age_days: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Derive paired model rasters and scan-quality scalar features.

    Args:
        current_path: Cached five-raster bundle for the current scan.
        previous_path: Cached five-raster bundle for the prior scan.
        historical_paths: Cached same-time scans from the preceding two weeks.
        historical_age_days: Exact age of each historical scan before current.

    Returns:
        Model raster arrays and their plume, cloud, quality, and uncertainty
        summaries.
    """
    with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as previous:
        current_no2 = np.asarray(current["no2"], dtype=np.float64)
        previous_no2 = np.asarray(previous["no2"], dtype=np.float64)
        valid = np.isfinite(current_no2) & np.isfinite(previous_no2)
        if not valid.any():
            raise ValueError(NO_PAIRED_FINITE_NO2_ERROR)

        centre_start = (IMG_SIZE - CENTRAL_COVERAGE_WINDOW_SIZE) // 2
        centre_stop = centre_start + CENTRAL_COVERAGE_WINDOW_SIZE
        central_valid = valid[centre_start:centre_stop, centre_start:centre_stop]

        paired_current_no2 = np.full(current_no2.shape, np.nan, dtype=np.float32)
        paired_current_no2[valid] = current_no2[valid].astype(np.float32)
        delta_no2 = np.full(current_no2.shape, np.nan, dtype=np.float32)
        delta_values = current_no2[valid].astype(np.float64) - previous_no2[valid].astype(np.float64)
        delta_no2[valid] = delta_values.astype(np.float32)
        historical_ema = _historical_ema(historical_paths, historical_age_days)
        ema_valid = np.isfinite(current_no2) & np.isfinite(historical_ema)
        ema_delta_no2 = np.full(current_no2.shape, np.nan, dtype=np.float32)
        ema_delta_no2[ema_valid] = (current_no2[ema_valid] - historical_ema[ema_valid]).astype(np.float32)
        p10, p50, p99 = np.percentile(delta_values, [10, 50, 99])
        denominator = p50 - p10
        epsilon = np.finfo(np.float64).eps * max(abs(p10), abs(p50), 1.0)
        features = {
            "plume_score": float((p99 - p50) / max(denominator, epsilon)),
            PAIRED_FINITE_FRACTION_COL: float(np.mean(valid)),
            CENTRAL_FINITE_FRACTION_COL: float(np.mean(central_valid)),
            "mean_weighted_cloud_fraction": _paired_mean(
                current["weighted_cloud_fraction"], previous["weighted_cloud_fraction"], valid
            ),
            "mean_good_quality_fraction": _paired_mean(
                current["good_quality_fraction"], previous["good_quality_fraction"], valid
            ),
            MEAN_RETRIEVAL_UNCERTAINTY_COL: _paired_mean(
                current["retrieval_uncertainty"], previous["retrieval_uncertainty"], valid
            ),
        }
    rasters = {
        CURRENT_RASTER_NAME: paired_current_no2,
        DELTA_RASTER_NAME: delta_no2,
        EMA_DELTA_RASTER_NAME: ema_delta_no2,
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


def process_record(task: RecordTask) -> RecordResult:
    """Create one persistent model raster bundle and its tabular features.

    Args:
        task: Cached TEMPO, HRRR, and output locations for one record.

    Returns:
        Derived scalar features or contextual failure text.
    """
    try:
        rasters, features = derive_raster_features(
            task.current_cache_path,
            task.previous_cache_path,
            task.historical_cache_paths,
            task.historical_age_days,
        )
        wind_rasters, weather_features = extract_wind_cache(task.wind_cache_path)
        rasters.update(wind_rasters)
        features.update(weather_features)
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
