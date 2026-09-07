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
    codes_get_double_element,
    codes_grib_find_nearest,
    codes_grib_new_from_file,
    codes_release,
)

from config import (
    CENTRAL_COVERAGE_WINDOW_SIZE,
    IMG_SIZE,
    LABEL_COL,
    MIN_CENTRAL_FINITE_FRACTION,
    MIN_PAIRED_FINITE_FRACTION,
    MODEL_IMAGE_KEYS,
    MODEL_VALID_MASK_KEY,
    RASTER_UNCERTAINTY_WEIGHT,
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

CURRENT_RASTER_NAME, DELTA_RASTER_NAME = MODEL_IMAGE_KEYS
VALID_MASK_NAME = MODEL_VALID_MASK_KEY
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
    "10u": "wind_u_10m_mps",
    "10v": "wind_v_10m_mps",
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
    central = pl.col(CENTRAL_FINITE_FRACTION_COL)
    keep = (
        paired.is_finite()
        & central.is_finite()
        & (paired >= MIN_PAIRED_FINITE_FRACTION)
        & (central >= MIN_CENTRAL_FINITE_FRACTION)
    )
    if RASTER_UNCERTAINTY_WEIGHT > 0:
        keep &= pl.col(MEAN_RETRIEVAL_UNCERTAINTY_COL).is_finite()
    eligible = frame.filter(keep)
    coverage_quality = 2 * paired * central / (paired + central)
    if RASTER_UNCERTAINTY_WEIGHT == 0:
        uncertainty_quality = pl.lit(0.0)
    elif eligible.height == 1:
        uncertainty_quality = pl.lit(1.0)
    else:
        uncertainty_quality = 1 - (
            (pl.col(MEAN_RETRIEVAL_UNCERTAINTY_COL).rank(method="average") - 1)
            / (eligible.height - 1)
        )
    return eligible.with_columns(
        (
            (1 - RASTER_UNCERTAINTY_WEIGHT) * coverage_quality
            + RASTER_UNCERTAINTY_WEIGHT * uncertainty_quality
        ).alias(RASTER_QUALITY_SCORE_COL)
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
    hrrr_path: str
    hrrr_grid_index: int
    output_path: str


@dataclass(frozen=True)
class RecordResult:
    """Tabular features or failure from one paired record."""

    split: str
    record_index: int
    features: dict[str, float]
    error: str | None


def parse_tempo_paths(serialized_paths: object, tempo_root: Path) -> tuple[str, ...]:
    """Parse a stratified CSV's JSON granule list into absolute paths.

    Args:
        serialized_paths: JSON string emitted by the stratification stage.
        tempo_root: Root of the configured TEMPO Level 2 archive.

    Returns:
        Non-empty tuple of absolute granule paths.
    """
    relative_paths = json.loads(str(serialized_paths))
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


def scan_cache_exists(path: str | Path) -> bool:
    """Return whether a cached scan exists.

    Args:
        path: Candidate persistent scan bundle.

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
            results.append(
                ScanResult(task.cache_key, task.cache_path, f"TEMPO regridding failed: {error}")
            )
    return results


def process_scan(task: ScanTask) -> ScanResult:
    """Regrid one AOI scan through the shared batch implementation.

    Args:
        task: Unique scan description and cache destination.

    Returns:
        Cache location or contextual failure text.
    """
    return process_scan_batch(ScanBatchTask(task.granule_paths, (task,)))[0]


def build_hrrr_grid_indices(
    reference_path: str,
    locations: dict[int, tuple[float, float]],
) -> dict[int, int]:
    """Find the nearest native HRRR grid element for each AOI centroid.

    Args:
        reference_path: Any available HRRR surface-analysis subset.
        locations: AOI IDs mapped to latitude and longitude in degrees.

    Returns:
        AOI IDs mapped to flat native-grid element indices.
    """
    with Path(reference_path).open("rb") as source:
        message = codes_grib_new_from_file(source)
        if message is None:
            raise ValueError(f"HRRR file contains no GRIB messages: {reference_path}")
        try:
            return {
                aoi_id: int(codes_grib_find_nearest(message, lat, lon, is_lsm=False, npoints=1)[0]["index"])
                for aoi_id, (lat, lon) in locations.items()
            }
        finally:
            codes_release(message)


def extract_hrrr_features(path: str, grid_index: int) -> dict[str, float]:
    """Read four meteorological values at one native HRRR grid element.

    Args:
        path: HRRR GRIB2 subset containing the configured four fields.
        grid_index: Flat grid element nearest to the record's AOI centroid.

    Returns:
        Temperature, U/V wind, and boundary-layer-height features with units.
    """
    features: dict[str, float] = {}
    with Path(path).open("rb") as source:
        while (message := codes_grib_new_from_file(source)) is not None:
            try:
                short_name = str(codes_get(message, "shortName"))
                output_name = HRRR_FIELDS.get(short_name)
                if output_name is not None:
                    features[output_name] = float(codes_get_double_element(message, "values", grid_index))
            finally:
                codes_release(message)
    missing = set(HRRR_FIELDS.values()).difference(features)
    if missing:
        raise ValueError(f"HRRR file is missing fields: {', '.join(sorted(missing))}")
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("HRRR features contain non-finite values")
    return features


def _paired_mean(current: np.ndarray, previous: np.ndarray, valid: np.ndarray) -> float:
    # Average both scans over the cells that contribute to the NO2 delta
    values = np.concatenate([current[valid], previous[valid]])
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def derive_raster_features(
    current_path: str,
    previous_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Derive paired model rasters and scan-quality scalar features.

    Args:
        current_path: Cached five-raster bundle for the current scan.
        previous_path: Cached five-raster bundle for the prior scan.

    Returns:
        Model raster arrays and their plume, cloud, quality, and uncertainty
        summaries.
    """
    with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as previous:
        current_no2 = current["no2"]
        previous_no2 = previous["no2"]
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
        VALID_MASK_NAME: valid.astype(np.float32),
    }
    return rasters, features


def _write_npz_atomic(destination: str, **arrays: np.ndarray) -> None:
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
        rasters, features = derive_raster_features(task.current_cache_path, task.previous_cache_path)
        features.update(extract_hrrr_features(task.hrrr_path, task.hrrr_grid_index))
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
