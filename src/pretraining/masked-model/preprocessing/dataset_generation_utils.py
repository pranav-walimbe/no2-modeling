"""Shared utilities for masked-pretraining dataset generation."""

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
SHARD_RECORDS_FILE = "records.csv"

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
    "shard_key": pl.UInt64,
    "selection_key": pl.UInt64,
}

VALID_RECORD_SCHEMA = {
    "candidate_index": pl.UInt64,
    "aoi_id": pl.Int64,
    "scan_date": pl.Date,
    "scan_num": pl.Int32,
    "tempo_time": pl.Datetime(time_zone="UTC"),
    "cache_key": pl.String,
    "validity_cache_path": pl.String,
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
class PretrainingShardTask:
    """One deterministic output-record range assigned to an array task."""

    task_id: int
    split: str
    shard_index: int
    start: int
    stop: int
    split_target: int

    @property
    def size(self) -> int:
        """Return the maximum number of output records in this shard."""
        return self.stop - self.start


@dataclass(frozen=True)
class CandidateResult:
    """Terminal or retryable outcome for one AOI-scene candidate."""

    status: str
    row: dict[str, object]
    cache_key: str
    raster_path: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MaskedRecordTask:
    """One selected validity-cache entry to materialize in a fresh shard."""

    row: dict[str, object]
    output_path: str
    mask_seed: int


@dataclass(frozen=True)
class MaskedDatasetShardStore:
    """Manage disposable masked-pretraining dataset shards."""

    root: Path

    def create(self, task: PretrainingShardTask) -> Path:
        """Create one new shard directory after the run-wide reset."""
        split_root = self.root / task.split
        split_root.mkdir(parents=True, exist_ok=True)
        shard_dir = split_root / f"{task.shard_index:06d}"
        shard_dir.mkdir()
        return shard_dir

    def write(self, task: PretrainingShardTask, shard_dir: Path, rows: list[dict[str, object]]) -> None:
        """Publish one shard manifest with paths relative to its directory."""
        frame = pl.DataFrame(rows, schema=FINAL_RECORD_SCHEMA).sort("candidate_index")
        relative_paths = [str(Path(path).relative_to(shard_dir)) for path in frame["raster_bundle_path"].to_list()]
        frame = frame.with_columns(pl.Series("raster_bundle_path", relative_paths, dtype=pl.String))
        write_csv_atomic(frame, shard_dir / SHARD_RECORDS_FILE)
        print(f"[{task.split} shard {task.shard_index}] wrote {frame.height:,} masked records")

    def load(self, task: PretrainingShardTask, *, resolve_paths: bool = False) -> pl.DataFrame:
        """Load and validate one completed shard manifest and its bundles."""
        shard_dir = self.root / task.split / f"{task.shard_index:06d}"
        frame = pl.read_csv(shard_dir / SHARD_RECORDS_FILE, schema_overrides=FINAL_RECORD_SCHEMA)
        if frame.height > task.size:
            raise ValueError(f"Shard {task.task_id} contains more than {task.size:,} records")
        raster_directory = Path("record-rasters") / task.split
        relative_paths = [Path(str(path)) for path in frame["raster_bundle_path"].to_list()]
        if any(path.is_absolute() or path.parent != raster_directory for path in relative_paths):
            raise ValueError(f"Shard {task.task_id} contains a raster path outside {raster_directory}")
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError(f"Shard {task.task_id} contains duplicate raster paths")
        if any(not (shard_dir / path).is_file() for path in relative_paths):
            raise ValueError(f"Shard {task.task_id} references a missing raster bundle")
        if resolve_paths:
            frame = frame.with_columns(
                pl.Series(
                    "raster_bundle_path",
                    [str(shard_dir / path) for path in relative_paths],
                    dtype=pl.String,
                )
            )
        return frame


def write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    """Atomically publish a Parquet file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv_atomic(frame: pl.DataFrame, destination: Path) -> None:
    """Atomically publish a CSV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        frame.write_csv(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(values: dict[str, object], destination: Path) -> None:
    """Atomically publish JSON metadata."""
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
    """Atomically publish a compressed raster bundle."""
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


def build_shard_tasks(
    targets: dict[str, int],
    shard_size: int,
) -> list[PretrainingShardTask]:
    """Partition each requested split into fixed-size output shards."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    tasks: list[PretrainingShardTask] = []
    for split, target_count in targets.items():
        for shard_index, start in enumerate(range(0, target_count, shard_size)):
            tasks.append(
                PretrainingShardTask(
                    task_id=len(tasks),
                    split=split,
                    shard_index=shard_index,
                    start=start,
                    stop=min(start + shard_size, target_count),
                    split_target=target_count,
                )
            )
    return tasks


def load_shard_candidates(path: Path, task: PretrainingShardTask, split_shard_count: int) -> pl.DataFrame:
    """Load the candidates owned by one modulo-partitioned shard."""
    return (
        pl.scan_parquet(path)
        .filter((pl.col("shard_key") % split_shard_count) == task.shard_index)
        .sort("selection_key", "scan_date", "scan_num", "aoi_id")
        .collect(engine="streaming")
    )


def _valid_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "valid" / cache_key[:2] / f"{cache_key}.npz"


def _invalid_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "invalid" / cache_key[:2] / f"{cache_key}.json"


def cached_candidate_result(
    row: dict[str, object],
    scan_task: ScanTask,
    cache_dir: Path,
) -> CandidateResult | None:
    """Return a terminal cached result, if one exists."""
    valid_path = _valid_path(cache_dir, scan_task.cache_key)
    if valid_path.is_file():
        return CandidateResult(VALID_STATUS, row, scan_task.cache_key, str(valid_path))
    invalid_path = _invalid_path(cache_dir, scan_task.cache_key)
    if invalid_path.is_file():
        try:
            reason = str(json.loads(invalid_path.read_text()).get("reason", "cached invalid raster"))
        except (OSError, TypeError, ValueError):
            reason = "cached invalid raster"
        return CandidateResult(INVALID_STATUS, row, scan_task.cache_key, reason=reason)
    return None


def _mark_invalid(
    row: dict[str, object],
    scan_task: ScanTask,
    cache_dir: Path,
    reason: str,
    valid_pixel_count: int,
) -> CandidateResult:
    invalid_path = _invalid_path(cache_dir, scan_task.cache_key)
    write_json_atomic(
        {
            "cache_key": scan_task.cache_key,
            "aoi_id": int(row["aoi_id"]),
            "tempo_time": str(row["tempo_time"]),
            "valid": False,
            "valid_pixel_count": valid_pixel_count,
            "pixel_count": IMG_SIZE * IMG_SIZE,
            "reason": reason,
        },
        invalid_path,
    )
    return CandidateResult(INVALID_STATUS, row, scan_task.cache_key, reason=reason)


def _make_tasks(
    rows: list[dict[str, object]],
    tempo_root: Path,
    tempo_cache_dir: Path,
    hrrr_root: Path,
    weather_cache_dir: Path,
) -> tuple[dict[str, ScanTask], dict[str, WeatherTask]]:
    scans: dict[str, ScanTask] = {}
    weather: dict[str, WeatherTask] = {}
    for row in rows:
        scan = make_scan_task(row, "granule_paths", tempo_root, tempo_cache_dir)
        item = make_weather_task(row, "weather_path", hrrr_root, weather_cache_dir)
        scans[scan.cache_key] = scan
        weather[scan.cache_key] = item
    return scans, weather


def _run_scan_tasks(scans: list[ScanTask], workers: int) -> dict[str, str | None]:
    outcomes: dict[str, str | None] = {}
    batches = scan_batches(scans)
    for results in bounded_parallel_map(process_scan_batch, batches, workers):
        outcomes.update({result.cache_key: result.error for result in results})
    return outcomes


def _run_weather_tasks(weather: list[WeatherTask], workers: int) -> dict[str, str | None]:
    outcomes: dict[str, str | None] = {}
    batches = weather_batches(weather)
    for results in bounded_parallel_map(process_weather_batch, batches, workers):
        outcomes.update({result.cache_key: result.error for result in results})
    return outcomes


def process_candidate_batch(
    rows: list[dict[str, object]],
    *,
    validity_cache_dir: Path,
    tempo_root: Path,
    tempo_cache_dir: Path,
    hrrr_root: Path,
    weather_cache_dir: Path,
    workers: int,
) -> list[CandidateResult]:
    """Resolve cached outcomes, then generate uncached complete raster bundles."""
    scans, weather = _make_tasks(rows, tempo_root, tempo_cache_dir, hrrr_root, weather_cache_dir)
    results: dict[str, CandidateResult] = {}
    pending_scans: list[ScanTask] = []
    rows_by_key: dict[str, dict[str, object]] = {}
    for row in rows:
        scan = make_scan_task(row, "granule_paths", tempo_root, tempo_cache_dir)
        rows_by_key[scan.cache_key] = row
        cached = cached_candidate_result(row, scan, validity_cache_dir)
        if cached is None:
            pending_scans.append(scan)
        else:
            results[scan.cache_key] = cached

    scan_errors = _run_scan_tasks(pending_scans, workers) if pending_scans else {}
    full_coverage_keys: list[str] = []
    for scan in pending_scans:
        row = rows_by_key[scan.cache_key]
        error = scan_errors.get(scan.cache_key)
        if error is not None:
            results[scan.cache_key] = CandidateResult(
                RETRYABLE_STATUS,
                row,
                scan.cache_key,
                reason=error,
            )
            continue
        try:
            with np.load(scan.cache_path, allow_pickle=False) as bundle:
                no2 = np.asarray(bundle[NO2_RASTER_NAME], dtype=np.float32)
            valid_pixel_count = int(np.isfinite(no2).sum())
        except (KeyError, OSError, TypeError, ValueError) as error:
            results[scan.cache_key] = CandidateResult(
                RETRYABLE_STATUS,
                row,
                scan.cache_key,
                reason=f"TEMPO cache read failed: {error}",
            )
            continue
        if no2.shape != (IMG_SIZE, IMG_SIZE) or valid_pixel_count != IMG_SIZE * IMG_SIZE:
            results[scan.cache_key] = _mark_invalid(
                row,
                scan,
                validity_cache_dir,
                "NO2 coverage is below 100%",
                valid_pixel_count,
            )
            continue
        full_coverage_keys.append(scan.cache_key)

    pending_weather = [weather[key] for key in full_coverage_keys]
    weather_errors = _run_weather_tasks(pending_weather, workers) if pending_weather else {}
    for key in full_coverage_keys:
        row = rows_by_key[key]
        scan = scans[key]
        item = weather[key]
        error = weather_errors.get(item.cache_key)
        if error is not None:
            results[key] = CandidateResult(RETRYABLE_STATUS, row, key, reason=error)
            continue
        try:
            with np.load(scan.cache_path, allow_pickle=False) as bundle:
                no2 = np.asarray(bundle[NO2_RASTER_NAME], dtype=np.float32)
            weather_arrays = extract_weather_cache(item.cache_path)
            arrays = {
                NO2_RASTER_NAME: no2,
                NO2_MASK_NAME: np.ones_like(no2, dtype=np.uint8),
                TEMPERATURE_RASTER_NAME: weather_arrays[TEMPERATURE_RASTER_NAME],
                WIND_U_RASTER_NAME: weather_arrays[WIND_U_RASTER_NAME],
                WIND_V_RASTER_NAME: weather_arrays[WIND_V_RASTER_NAME],
            }
            if any(array.shape != (IMG_SIZE, IMG_SIZE) or not np.isfinite(array).all() for array in arrays.values()):
                results[key] = CandidateResult(
                    RETRYABLE_STATUS,
                    row,
                    key,
                    reason="Weather or NO2 bundle is not completely finite",
                )
                continue
            destination = _valid_path(validity_cache_dir, key)
            write_npz_atomic(destination, **arrays)
            results[key] = CandidateResult(VALID_STATUS, row, key, str(destination))
        except (KeyError, OSError, TypeError, ValueError) as error:
            results[key] = CandidateResult(
                RETRYABLE_STATUS,
                row,
                key,
                reason=f"Raster bundle assembly failed: {error}",
            )
    return [results[make_scan_task(row, "granule_paths", tempo_root, tempo_cache_dir).cache_key] for row in rows]


def valid_record_frame(results: list[CandidateResult]) -> pl.DataFrame:
    """Convert valid outcomes to the stable shard schema."""
    return pl.DataFrame(
        [
            {
                "candidate_index": int(result.row["candidate_index"]),
                "aoi_id": int(result.row["aoi_id"]),
                "scan_date": result.row["scan_date"],
                "scan_num": int(result.row["scan_num"]),
                "tempo_time": result.row["tempo_time"],
                "cache_key": result.cache_key,
                "validity_cache_path": str(result.raster_path),
            }
            for result in results
            if result.status == VALID_STATUS
        ],
        schema=VALID_RECORD_SCHEMA,
    )


def mask_no2_raster(no2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    """Return masked NO2 and its artificial mask once EDA defines the algorithm."""
    del no2, rng
    # Intentionally unimplemented until missingness EDA fixes the masking contract.
    return None


def materialize_masked_record(task: MaskedRecordTask) -> dict[str, object]:
    """Combine one persistent clean bundle with a newly sampled mask."""
    row = task.row
    with np.load(str(row["validity_cache_path"]), allow_pickle=False) as cached:
        arrays = {
            name: np.asarray(cached[name])
            for name in (
                NO2_RASTER_NAME,
                NO2_MASK_NAME,
                TEMPERATURE_RASTER_NAME,
                WIND_U_RASTER_NAME,
                WIND_V_RASTER_NAME,
            )
        }
    no2 = np.asarray(arrays[NO2_RASTER_NAME], dtype=np.float32)
    masked = mask_no2_raster(no2, np.random.default_rng(task.mask_seed))
    if masked is None:
        raise NotImplementedError(
            "mask_no2_raster is intentionally empty until missingness EDA defines the masking algorithm"
        )
    masked_no2, artificial_mask = masked
    if masked_no2.shape != no2.shape or artificial_mask.shape != no2.shape:
        raise ValueError("Masked NO2 and artificial mask must match the original NO2 shape")
    write_npz_atomic(
        Path(task.output_path),
        **arrays,
        **{
            MASKED_NO2_RASTER_NAME: np.asarray(masked_no2, dtype=np.float32),
            ARTIFICIAL_MASK_NAME: np.asarray(artificial_mask, dtype=np.uint8),
        },
    )
    return {
        **{name: row[name] for name in FINAL_RECORD_SCHEMA if name != "raster_bundle_path"},
        "raster_bundle_path": task.output_path,
    }
