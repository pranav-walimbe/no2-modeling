"""Raster, meteorology, and persistence utilities for dataset generation."""

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

from config import IMG_SIZE
from preprocessing.regrid import AoiGrid, regrid_aoi_scan, write_raster_npz

DELTA_RASTER_NAME = "delta_no2"
HRRR_GRID_SPACING_M = 3_000.0
HRRR_FIELDS = {
    "2t": "temperature_2m_k",
    "10u": "wind_u_10m_mps",
    "10v": "wind_v_10m_mps",
    "blh": "boundary_layer_height_m",
}
TABULAR_FEATURE_NAMES = (
    "plume_score",
    "mean_weighted_cloud_fraction",
    "mean_good_quality_fraction",
    *HRRR_FIELDS.values(),
)


@dataclass(frozen=True)
class ScanTask:
    """One unique AOI scan to regrid into the run cache."""

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
    if not isinstance(serialized_paths, str):
        raise TypeError("TEMPO path list must be a JSON string")
    try:
        relative_paths = json.loads(serialized_paths)
    except json.JSONDecodeError as error:
        raise ValueError("TEMPO path list is not valid JSON") from error
    if not isinstance(relative_paths, list) or not relative_paths:
        raise ValueError("TEMPO path list must contain at least one granule")
    if any(not isinstance(path, str) or not path for path in relative_paths):
        raise ValueError("TEMPO path list contains an invalid granule path")
    return tuple(str(tempo_root / path) for path in relative_paths)


def make_scan_task(row: dict[str, object], path_column: str, tempo_root: Path, cache_dir: Path) -> ScanTask:
    """Create a stable cache task for one AOI scan.

    Args:
        row: Stratified record carrying AOI coordinates and granule paths.
        path_column: Either the current or previous TEMPO path-list column.
        tempo_root: Root of the configured TEMPO archive.
        cache_dir: Run-scoped directory for regridded scan bundles.

    Returns:
        Deduplicatable scan task with a content-derived cache key.
    """
    aoi_id = int(row["aoi_id"])
    lon = float(row["lon"])
    lat = float(row["lat"])
    granule_paths = parse_tempo_paths(row[path_column], tempo_root)
    identity = json.dumps([aoi_id, lon, lat, granule_paths], separators=(",", ":"))
    cache_key = hashlib.sha256(identity.encode()).hexdigest()
    return ScanTask(
        cache_key=cache_key,
        aoi_id=aoi_id,
        lon=lon,
        lat=lat,
        granule_paths=granule_paths,
        cache_path=str(cache_dir / f"{cache_key}.npz"),
    )


def process_scan(task: ScanTask) -> ScanResult:
    """Regrid one unique AOI scan and persist it in the run cache.

    Args:
        task: Unique scan description and cache destination.

    Returns:
        Cache location or contextual failure text.
    """
    try:
        grid = AoiGrid.from_lon_lat(task.aoi_id, task.lon, task.lat)
        raster = regrid_aoi_scan(list(task.granule_paths), grid)
        write_raster_npz(raster, task.cache_path)
        return ScanResult(task.cache_key, task.cache_path, None)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return ScanResult(task.cache_key, task.cache_path, f"TEMPO regridding failed: {error}")


def build_hrrr_grid_indices(
    reference_path: str,
    locations: dict[int, tuple[float, float]],
) -> dict[int, int]:
    """Find the nearest native HRRR grid element for each AOI centroid.

    HRRR's CONUS surface product uses a 3 km Lambert grid. The grid is fixed
    across the analysis hours in this dataset, so each AOI index can be reused.

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
            spacing_x = float(codes_get(message, "DxInMetres"))
            spacing_y = float(codes_get(message, "DyInMetres"))
            if not np.isclose(spacing_x, HRRR_GRID_SPACING_M) or not np.isclose(spacing_y, HRRR_GRID_SPACING_M):
                raise ValueError(f"Expected a 3 km HRRR grid, found {spacing_x:g} by {spacing_y:g} m")
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


def derive_delta_features(
    current_path: str,
    previous_path: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Derive a paired NO2 delta and scan-quality scalar features.

    Args:
        current_path: Cached five-raster bundle for the current scan.
        previous_path: Cached five-raster bundle for the prior scan.

    Returns:
        Delta NO2 raster and its plume, cloud, and quality summaries.
    """
    with np.load(current_path, allow_pickle=False) as current, np.load(previous_path, allow_pickle=False) as previous:
        current_no2 = current["no2"]
        previous_no2 = previous["no2"]
        if current_no2.shape != (IMG_SIZE, IMG_SIZE) or previous_no2.shape != current_no2.shape:
            raise ValueError("Paired TEMPO rasters do not share the configured grid shape")
        valid = np.isfinite(current_no2) & np.isfinite(previous_no2)
        if not valid.any():
            raise ValueError("Paired TEMPO scans have no cells with finite NO2 in both rasters")

        delta_no2 = np.full(current_no2.shape, np.nan, dtype=np.float32)
        delta_values = current_no2[valid].astype(np.float64) - previous_no2[valid].astype(np.float64)
        delta_no2[valid] = delta_values.astype(np.float32)
        p10, p50, p99 = np.percentile(delta_values, [10, 50, 99])
        denominator = p50 - p10
        epsilon = np.finfo(np.float64).eps * max(abs(p10), abs(p50), 1.0)
        features = {
            "plume_score": float((p99 - p50) / max(denominator, epsilon)),
            "mean_weighted_cloud_fraction": _paired_mean(
                current["weighted_cloud_fraction"], previous["weighted_cloud_fraction"], valid
            ),
            "mean_good_quality_fraction": _paired_mean(
                current["good_quality_fraction"], previous["good_quality_fraction"], valid
            ),
        }
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Derived TEMPO features contain non-finite values")
    return delta_no2, features


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
    """Create one persistent delta raster and its tabular features.

    Args:
        task: Cached TEMPO, HRRR, and output locations for one record.

    Returns:
        Derived scalar features or contextual failure text.
    """
    try:
        delta_no2, features = derive_delta_features(task.current_cache_path, task.previous_cache_path)
        features.update(extract_hrrr_features(task.hrrr_path, task.hrrr_grid_index))
        _write_npz_atomic(task.output_path, **{DELTA_RASTER_NAME: delta_no2})
        return RecordResult(task.split, task.record_index, features, None)
    except (IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return RecordResult(task.split, task.record_index, {}, f"Record processing failed: {error}")


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
