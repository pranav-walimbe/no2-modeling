"""Shared utilities for masked-pretraining dataset generation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from preprocessing.generate_dataset_utils import (
    NO2_MASK_NAME,
    NO2_RASTER_NAME,
    TEMPERATURE_RASTER_NAME,
    WIND_U_RASTER_NAME,
    WIND_V_RASTER_NAME,
    ScanTask,
    WeatherTask,
    bounded_parallel_map,
    extract_weather_cache,
    make_scan_task,
    make_weather_task,
    process_scan_batch,
    process_weather_batch,
    scan_batches,
    weather_batches,
)

from config import IMG_SIZE

VALID_STATUS = "valid"
INVALID_STATUS = "invalid"
RETRYABLE_STATUS = "retryable"
MASKED_NO2_RASTER_NAME = "masked_no2"
ARTIFICIAL_MASK_NAME = "artificial_mask"
MIN_MASK_FRACTION = 0.01
MAX_MASK_FRACTION = 0.10
MASK_EDGE_STRENGTH = 2.0
MASK_EDGE_DECAY_PIXELS = 2.0
MASK_CLUMP_STRENGTH = 0.35
MASK_CLUMP_SIGMA_PIXELS = 1.0
MASK_MAX_CLUMP_MULTIPLIER = 2.0

_MASK_ROWS, _MASK_COLUMNS = np.indices((IMG_SIZE, IMG_SIZE))
_MASK_EDGE_DISTANCE = np.minimum.reduce(
    (_MASK_ROWS, _MASK_COLUMNS, IMG_SIZE - 1 - _MASK_ROWS, IMG_SIZE - 1 - _MASK_COLUMNS)
)
_MASK_BASE_WEIGHTS = 1.0 + MASK_EDGE_STRENGTH * np.exp(-_MASK_EDGE_DISTANCE / MASK_EDGE_DECAY_PIXELS)
_MASK_COORDINATES = np.column_stack((_MASK_ROWS.ravel(), _MASK_COLUMNS.ravel()))
_MASK_PAIRWISE_SQUARED_DISTANCE = ((_MASK_COORDINATES[:, None] - _MASK_COORDINATES[None, :]) ** 2).sum(axis=2)
_MASK_CLUMP_KERNELS = np.exp(-_MASK_PAIRWISE_SQUARED_DISTANCE / (2.0 * MASK_CLUMP_SIGMA_PIXELS**2))

CANDIDATE_SCHEMA = {
    "candidate_index": pl.UInt64,
    "aoi_id": pl.Int64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "scan_date": pl.Date,
    "scan_num": pl.Int32,
    "tempo_time": pl.Datetime(time_zone="UTC"),
    "granule_paths": pl.List(pl.String),
    "weather_path": pl.String,
}

VALIDITY_INDEX_SCHEMA = {
    "cache_key": pl.String,
    "status": pl.String,
    "tempo_cache_path": pl.String,
    "weather_cache_path": pl.String,
    "reason": pl.String,
}

FINAL_RECORD_SCHEMA = {
    "candidate_index": pl.UInt64,
    "aoi_id": pl.Int64,
    "scan_date": pl.Date,
    "scan_num": pl.Int32,
    "tempo_time": pl.Datetime(time_zone="UTC"),
    "cache_key": pl.String,
    "raster_bundle_path": pl.String,
}


@dataclass(frozen=True)
class CandidateOutcome:
    """Cache paths or failure state for one candidate."""

    status: str
    row: dict[str, object]
    cache_key: str
    tempo_cache_path: str | None = None
    weather_cache_path: str | None = None
    reason: str | None = None

    def validity_row(self) -> dict[str, object]:
        """Return this outcome in persistent validity-index form."""
        return {
            "cache_key": self.cache_key,
            "status": self.status,
            "tempo_cache_path": self.tempo_cache_path,
            "weather_cache_path": self.weather_cache_path,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MaskedRecordTask:
    """One masked raster bundle to assemble from source caches."""

    row: dict[str, object]
    cache_key: str
    tempo_cache_path: str
    weather_cache_path: str
    output_path: str
    mask_seed: int


def write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    """Publish a Parquet file atomically.

    Args:
        frame: Data to write.
        destination: Final file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(frame: pl.DataFrame, destination: Path) -> None:
    """Publish a CSV file atomically.

    Args:
        frame: Data to write.
        destination: Final file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        frame.write_csv(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(values: dict[str, object], destination: Path) -> None:
    """Publish JSON metadata atomically.

    Args:
        values: JSON-compatible metadata.
        destination: Final file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as output:
            json.dump(values, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_atomic(destination: Path, **arrays: np.ndarray) -> None:
    """Publish a compressed raster bundle atomically.

    Args:
        destination: Final file path.
        **arrays: Named arrays stored in the bundle.
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
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def empty_validity_index() -> pl.DataFrame:
    """Return an empty validity-index frame."""
    return pl.DataFrame(schema=VALIDITY_INDEX_SCHEMA)


def empty_final_records() -> pl.DataFrame:
    """Return an empty final-record frame."""
    return pl.DataFrame(schema=FINAL_RECORD_SCHEMA)


def merge_validity_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Merge validity-index frames with later rows taking precedence.

    Args:
        frames: Index snapshots and ordered update frames.

    Returns:
        One row per cache key.
    """
    populated = [frame.cast(VALIDITY_INDEX_SCHEMA) for frame in frames if frame.height]
    if not populated:
        return empty_validity_index()
    return (
        pl.concat(populated, how="vertical")
        .unique(subset="cache_key", keep="last", maintain_order=True)
        .sort("cache_key")
    )


def validity_lookup(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    """Create a cache-key lookup from a validity-index frame.

    Args:
        frame: Persistent validity records.

    Returns:
        Mapping from cache key to validity row.
    """
    return {str(row["cache_key"]): row for row in frame.iter_rows(named=True)}


def indexed_candidate_outcome(
    row: dict[str, object],
    cache_key: str,
    indexed: dict[str, object],
) -> CandidateOutcome:
    """Build a candidate outcome from one validity-index row.

    Args:
        row: Candidate metadata.
        cache_key: Stable candidate cache key.
        indexed: Stored validity-index values.

    Returns:
        Validated candidate outcome.
    """
    status = str(indexed["status"])
    if status not in {VALID_STATUS, INVALID_STATUS}:
        raise ValueError(f"Unsupported validity-index status for {cache_key}: {status}")
    tempo_path = indexed.get("tempo_cache_path")
    weather_path = indexed.get("weather_cache_path")
    if status == VALID_STATUS and (not tempo_path or not weather_path):
        raise ValueError(f"Valid index entry lacks source cache paths: {cache_key}")
    return CandidateOutcome(
        status,
        row,
        cache_key,
        tempo_cache_path=str(tempo_path) if tempo_path else None,
        weather_cache_path=str(weather_path) if weather_path else None,
        reason=str(indexed["reason"]) if indexed.get("reason") else None,
    )


def candidate_cache_tasks(
    row: dict[str, object],
    *,
    tempo_root: Path,
    tempo_cache_dir: Path,
    hrrr_root: Path,
    weather_cache_dir: Path,
) -> tuple[ScanTask, WeatherTask]:
    """Build the TEMPO and weather cache tasks for one candidate.

    Args:
        row: Candidate metadata.
        tempo_root: Raw TEMPO archive root.
        tempo_cache_dir: Regridded TEMPO cache.
        hrrr_root: Raw HRRR archive root.
        weather_cache_dir: Aligned weather cache.

    Returns:
        TEMPO and weather tasks with stable cache paths.
    """
    scan = make_scan_task(row, "granule_paths", tempo_root, tempo_cache_dir)
    weather = make_weather_task(row, "weather_path", hrrr_root, weather_cache_dir)
    return scan, weather


def discover_candidate_batch(
    rows: list[dict[str, object]],
    *,
    tempo_root: Path,
    tempo_cache_dir: Path,
    hrrr_root: Path,
    weather_cache_dir: Path,
    workers: int,
) -> list[CandidateOutcome]:
    """Resolve cache misses and classify one candidate batch.

    Args:
        rows: Candidate rows absent from the validity index.
        tempo_root: Raw TEMPO archive root.
        tempo_cache_dir: Regridded TEMPO cache.
        hrrr_root: Raw HRRR archive root.
        weather_cache_dir: Aligned weather cache.
        workers: Maximum worker processes for source processing.

    Returns:
        Outcomes in input order.
    """
    outcomes: dict[str, CandidateOutcome] = {}
    scans: dict[str, ScanTask] = {}
    weather: dict[str, WeatherTask] = {}
    rows_by_key: dict[str, dict[str, object]] = {}
    ordered_keys: list[str] = []
    for row in rows:
        scan, weather_task = candidate_cache_tasks(
            row,
            tempo_root=tempo_root,
            tempo_cache_dir=tempo_cache_dir,
            hrrr_root=hrrr_root,
            weather_cache_dir=weather_cache_dir,
        )
        scans[scan.cache_key] = scan
        weather[scan.cache_key] = weather_task
        rows_by_key[scan.cache_key] = row
        ordered_keys.append(scan.cache_key)

    scan_errors: dict[str, str | None] = {}
    for batch_results in bounded_parallel_map(process_scan_batch, scan_batches(scans.values()), workers):
        scan_errors.update({result.cache_key: result.error for result in batch_results})

    complete_keys: list[str] = []
    for key, scan in scans.items():
        row = rows_by_key[key]
        error = scan_errors.get(key)
        if error is not None:
            outcomes[key] = CandidateOutcome(RETRYABLE_STATUS, row, key, reason=error)
            continue
        try:
            with np.load(scan.cache_path, allow_pickle=False) as bundle:
                no2 = np.asarray(bundle[NO2_RASTER_NAME], dtype=np.float32)
        except (KeyError, OSError, TypeError, ValueError) as error:
            outcomes[key] = CandidateOutcome(
                RETRYABLE_STATUS,
                row,
                key,
                reason=f"TEMPO cache read failed: {error}",
            )
            continue
        if no2.shape != (IMG_SIZE, IMG_SIZE) or not np.isfinite(no2).all():
            outcomes[key] = CandidateOutcome(
                INVALID_STATUS,
                row,
                key,
                tempo_cache_path=scan.cache_path,
                reason="NO2 coverage is below 100%",
            )
            continue
        complete_keys.append(key)

    weather_errors: dict[str, str | None] = {}
    weather_tasks = [weather[key] for key in complete_keys]
    for batch_results in bounded_parallel_map(process_weather_batch, weather_batches(weather_tasks), workers):
        weather_errors.update({result.cache_key: result.error for result in batch_results})

    for key in complete_keys:
        row = rows_by_key[key]
        scan = scans[key]
        weather_task = weather[key]
        error = weather_errors.get(weather_task.cache_key)
        if error is not None:
            outcomes[key] = CandidateOutcome(RETRYABLE_STATUS, row, key, reason=error)
            continue
        outcomes[key] = CandidateOutcome(
            VALID_STATUS,
            row,
            key,
            tempo_cache_path=scan.cache_path,
            weather_cache_path=weather_task.cache_path,
        )
    return [outcomes[key] for key in ordered_keys]


def load_clean_raster_bundle(tempo_cache_path: str, weather_cache_path: str) -> dict[str, np.ndarray]:
    """Load one clean model bundle from the shared source caches.

    Args:
        tempo_cache_path: Regridded TEMPO NPZ path.
        weather_cache_path: Aligned weather NPZ path.

    Returns:
        Complete model raster arrays.
    """
    with np.load(tempo_cache_path, allow_pickle=False) as bundle:
        no2 = np.asarray(bundle[NO2_RASTER_NAME], dtype=np.float32)
    weather = extract_weather_cache(weather_cache_path)
    arrays = {
        NO2_RASTER_NAME: no2,
        NO2_MASK_NAME: np.ones_like(no2, dtype=np.uint8),
        TEMPERATURE_RASTER_NAME: np.asarray(weather[TEMPERATURE_RASTER_NAME], dtype=np.float32),
        WIND_U_RASTER_NAME: np.asarray(weather[WIND_U_RASTER_NAME], dtype=np.float32),
        WIND_V_RASTER_NAME: np.asarray(weather[WIND_V_RASTER_NAME], dtype=np.float32),
    }
    if any(array.shape != (IMG_SIZE, IMG_SIZE) or not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Clean raster bundle contains an invalid shape or non-finite values")
    return arrays


def mask_no2_raster(
    no2: np.ndarray,
    mask_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create masked NO2 and its observed-pixel mask.

    Args:
        no2: Complete NO2 raster.
        mask_fraction: Fraction of raster pixels to mask.
        rng: Random generator for mask sampling.

    Returns:
        Masked NO2 and a binary mask where one marks observed pixels.
    """
    masked_pixels = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    clump_influence = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.float64)
    masked_pixel_count = round(mask_fraction * masked_pixels.size)

    for _ in range(masked_pixel_count):
        clump_multiplier = np.minimum(
            1.0 + MASK_CLUMP_STRENGTH * clump_influence,
            MASK_MAX_CLUMP_MULTIPLIER,
        )
        weights = _MASK_BASE_WEIGHTS.ravel() * clump_multiplier
        weights[masked_pixels.ravel()] = 0.0
        selected_pixel = int(rng.choice(masked_pixels.size, p=weights / weights.sum()))
        masked_pixels.ravel()[selected_pixel] = True
        clump_influence += _MASK_CLUMP_KERNELS[selected_pixel]

    masked_no2 = no2.copy()
    masked_no2[masked_pixels] = 0.0
    observed_mask = (~masked_pixels).astype(np.uint8)
    return masked_no2, observed_mask


def write_masked_record(task: MaskedRecordTask) -> dict[str, object]:
    """Write one masked output bundle from indexed source caches.

    Args:
        task: Source cache paths, output metadata, and mask seed.

    Returns:
        Published manifest row.
    """
    arrays = load_clean_raster_bundle(task.tempo_cache_path, task.weather_cache_path)
    no2 = np.asarray(arrays[NO2_RASTER_NAME], dtype=np.float32)
    rng = np.random.default_rng(task.mask_seed)
    mask_fraction = rng.uniform(MIN_MASK_FRACTION, MAX_MASK_FRACTION)
    masked_no2, artificial_mask = mask_no2_raster(no2, mask_fraction, rng)
    write_npz_atomic(
        Path(task.output_path),
        **arrays,
        **{
            MASKED_NO2_RASTER_NAME: np.asarray(masked_no2, dtype=np.float32),
            ARTIFICIAL_MASK_NAME: np.asarray(artificial_mask, dtype=np.uint8),
        },
    )
    row = task.row
    return {
        "candidate_index": int(row["candidate_index"]),
        "aoi_id": int(row["aoi_id"]),
        "scan_date": row["scan_date"],
        "scan_num": int(row["scan_num"]),
        "tempo_time": row["tempo_time"],
        "cache_key": task.cache_key,
        "raster_bundle_path": task.output_path,
    }
