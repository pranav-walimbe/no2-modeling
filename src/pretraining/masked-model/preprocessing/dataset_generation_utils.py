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
    "tempo_cached": pl.Boolean,
    "weather_cached": pl.Boolean,
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
class PretrainingRecordTask:
    """One pretraining raster bundle to assemble from source caches."""

    row: dict[str, object]
    cache_key: str
    tempo_cache_path: str
    weather_cache_path: str
    output_path: str


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
    tempo_cache_additions: set[str],
    weather_cache_additions: set[str],
) -> list[CandidateOutcome]:
    """Resolve cache misses and classify one candidate batch.

    Args:
        rows: Candidate rows absent from the validity index.
        tempo_root: Raw TEMPO archive root.
        tempo_cache_dir: Regridded TEMPO cache.
        hrrr_root: Raw HRRR archive root.
        weather_cache_dir: Aligned weather cache.
        workers: Maximum worker processes for source processing.
        tempo_cache_additions: TEMPO files created since preparation.
        weather_cache_additions: Weather files created since preparation.

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

    cached_scan_keys = {
        key
        for key, scan in scans.items()
        if bool(rows_by_key[key]["tempo_cached"]) or Path(scan.cache_path).name in tempo_cache_additions
    }
    scan_errors: dict[str, str | None] = {key: None for key in cached_scan_keys}
    missing_scans = [scan for key, scan in scans.items() if key not in cached_scan_keys]
    for batch_results in bounded_parallel_map(process_scan_batch, scan_batches(missing_scans), workers):
        scan_errors.update({result.cache_key: result.error for result in batch_results})
        tempo_cache_additions.update(Path(result.cache_path).name for result in batch_results if result.error is None)

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

    weather_by_key: dict[str, WeatherTask] = {}
    cached_weather_keys: set[str] = set()
    for key in complete_keys:
        task = weather[key]
        weather_by_key[task.cache_key] = task
        if bool(rows_by_key[key]["weather_cached"]) or Path(task.cache_path).name in weather_cache_additions:
            cached_weather_keys.add(task.cache_key)
    missing_weather = {
        key: task for key, task in weather_by_key.items() if key not in cached_weather_keys
    }
    weather_errors: dict[str, str | None] = {key: None for key in cached_weather_keys}
    for batch_results in bounded_parallel_map(process_weather_batch, weather_batches(missing_weather.values()), workers):
        weather_errors.update({result.cache_key: result.error for result in batch_results})
        weather_cache_additions.update(
            Path(result.cache_path).name for result in batch_results if result.error is None
        )

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


def write_pretraining_record(task: PretrainingRecordTask) -> dict[str, object]:
    """Write one pretraining bundle from indexed source caches.

    Args:
        task: Source cache paths and output metadata.

    Returns:
        Published manifest row.
    """
    arrays = load_clean_raster_bundle(task.tempo_cache_path, task.weather_cache_path)
    write_npz_atomic(Path(task.output_path), **arrays)
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
