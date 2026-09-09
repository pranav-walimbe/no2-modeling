"""Generate paired TEMPO rasters and tabular features for every data split."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import polars as pl

from config import (
    DATASET_DF,
    DATASET_DIR,
    DATASET_RASTER_DIR,
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WIND_CACHE_DIR,
    DELTA_THRESHOLD,
    EMA_HISTORY_DAYS,
    EMA_MIN_SCANS,
    EMA_SAME_TIME_TOLERANCE_MINUTES,
    HRRR_DIR,
    NUM_CORES,
    TEMPO_AOI_MAPPING,
    TEMPO_DIR,
    TEST_RECORDS_CSV,
    TEST_SIZE,
    TRAIN_RECORDS_CSV,
    TRAIN_SIZE,
    VAL_RECORDS_CSV,
    VAL_SIZE,
)
from preprocessing.generate_dataset_utils import (
    TABULAR_FEATURE_NAMES,
    RecordTask,
    ScanBatchTask,
    ScanTask,
    WindBatchTask,
    WindTask,
    cache_exists,
    make_scan_task,
    make_wind_task,
    process_record,
    process_scan_batch,
    process_wind_batch,
    select_final_records,
    write_csv_atomic,
    write_json_atomic,
)
from preprocessing.stratify_utils import classification_summary

SPLIT_PATHS = {
    "train": TRAIN_RECORDS_CSV,
    "val": VAL_RECORDS_CSV,
    "test": TEST_RECORDS_CSV,
}
ARRAY_SPLITS = tuple(SPLIT_PATHS)
MAX_PENDING_FACTOR = 2
PROGRESS_INTERVAL = 1_000
SHARD_DIR_NAME = "shards"
SHARD_CANDIDATES_FILE = "candidates.csv"
SHARD_FAILURES_FILE = "failures.csv"
SHARD_COMPLETION_FILE = "complete.json"
LEGACY_DATASET_JOB_NAME = "generate-dataset"
SHARD_WORKER_JOB_NAME = "generate-dataset-shard"
SHARD_FINALIZER_JOB_NAME = "generate-dataset-finalize"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_BATCH_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "generate_dataset.sh"
SOURCE_RECORD_INDEX_COL = "_source_record_index"
DELTA_NO2_PATH_COL = "delta_no2_path"
CANDIDATE_RASTER_PATH_COL = "_candidate_raster_path"
FINAL_SPLIT_SIZES = {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE}
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class PreparedRecord:
    """One source row resolved to its cache and output paths."""

    split: str
    record_index: int
    current_scan_key: str
    previous_scan_key: str
    ema_scan_keys: tuple[str, ...]
    ema_scan_age_days: tuple[float, ...]
    wind_cache_key: str
    delta_no2_path: str


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


def _positive_int(value: str) -> int:
    # Parse a strictly positive command-line integer
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _source_row_count(path: str) -> int:
    # Count records without materializing the source split
    return int(pl.scan_csv(path).select(pl.len()).collect(engine="streaming").item())


def _shard_plan(split_paths: dict[str, str], shard_size: int) -> list[ShardTask]:
    # Assign consecutive array IDs across deterministic split ranges
    tasks: list[ShardTask] = []
    for split, path in split_paths.items():
        row_count = _source_row_count(path)
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


def _slurm_array_spec(task_ids: list[int]) -> str:
    # Compress consecutive task IDs into Slurm array ranges
    if not task_ids:
        return "none"
    ranges: list[str] = []
    range_start = task_ids[0]
    previous = task_ids[0]
    for task_id in task_ids[1:]:
        if task_id == previous + 1:
            previous = task_id
            continue
        ranges.append(str(range_start) if range_start == previous else f"{range_start}-{previous}")
        range_start = task_id
        previous = task_id
    ranges.append(str(range_start) if range_start == previous else f"{range_start}-{previous}")
    return ",".join(ranges)


def _shard_root() -> Path:
    # Keep resumable work under the configured dataset root
    return Path(DATASET_DIR) / SHARD_DIR_NAME


def _prepare_shard_root() -> Path:
    # Validate the shard root before creating or removing descendants
    dataset_root = Path(DATASET_DIR).resolve()
    shard_root = _shard_root()
    if shard_root.parent.resolve() != dataset_root or shard_root.is_symlink():
        raise ValueError(f"Refusing to use shard workspace outside {dataset_root}: {shard_root}")
    if shard_root.exists() and not shard_root.is_dir():
        raise ValueError(f"Shard workspace is not a directory: {shard_root}")
    shard_root.mkdir(parents=True, exist_ok=True)
    return shard_root


def _prepare_split_shard_root(split: str) -> Path:
    # Validate one split directory before using its shard contents
    shard_root = _prepare_shard_root()
    split_root = shard_root / split
    if split_root.parent.resolve() != shard_root.resolve() or split_root.is_symlink():
        raise ValueError(f"Refusing to use split shard workspace outside {shard_root}: {split_root}")
    if split_root.exists() and not split_root.is_dir():
        raise ValueError(f"Split shard workspace is not a directory: {split_root}")
    split_root.mkdir(parents=True, exist_ok=True)
    return split_root


def _shard_path(task: ShardTask) -> Path:
    # Name shards by stable split-local indices
    return _shard_root() / task.split / f"{task.shard_index:06d}"


def _feature_schema() -> dict[str, pl.DataType]:
    # Define the intermediate feature contract once
    return {
        SOURCE_RECORD_INDEX_COL: pl.UInt32,
        CANDIDATE_RASTER_PATH_COL: pl.String,
        **{name: pl.Float64 for name in TABULAR_FEATURE_NAMES},
    }


def _failure_schema() -> dict[str, pl.DataType]:
    # Define the intermediate failure contract once
    return {"record_index": pl.Int64, "error": pl.String}


def _bounded_parallel_map(
    function: Callable[[InputT], OutputT],
    tasks: Iterable[InputT],
    workers: int,
) -> Iterator[OutputT]:
    # Bound pending futures to keep large production runs memory-safe
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


def _load_splits(split_paths: dict[str, str]) -> dict[str, pl.DataFrame]:
    # Load inputs before starting expensive worker processes
    splits: dict[str, pl.DataFrame] = {}
    for split, path in split_paths.items():
        splits[split] = (
            pl.scan_csv(path, try_parse_dates=True).with_row_index(SOURCE_RECORD_INDEX_COL).collect(engine="streaming")
        )
    return splits


def _load_shard(task: ShardTask) -> dict[str, pl.DataFrame]:
    # Materialize only the source range assigned to this worker
    frame = (
        pl.scan_csv(SPLIT_PATHS[task.split], try_parse_dates=True)
        .with_row_index(SOURCE_RECORD_INDEX_COL)
        .slice(task.start, task.size)
        .collect(engine="streaming")
    )
    if frame.height != task.size:
        raise ValueError(f"Shard {task.task_id} expected {task.size:,} source records but loaded {frame.height:,}")
    return {task.split: frame}


ObservationIndex = dict[int, dict[date, list[dict[str, object]]]]


def _load_observations(aoi_ids: list[int]) -> ObservationIndex:
    # Index the shared TEMPO observations once by AOI and calendar day
    paths = sorted(Path(TEMPO_AOI_MAPPING).rglob("date=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No AOI observation shards found under {TEMPO_AOI_MAPPING}")
    observations = (
        pl.scan_parquet(paths)
        .select("aoi_id", "scan_date", "tempo_time", "granule_paths", "sampled_pixel_count")
        .filter(pl.col("aoi_id").is_in(aoi_ids))
        .collect()
    )
    index: ObservationIndex = {}
    for observation in observations.iter_rows(named=True):
        aoi_id = int(observation["aoi_id"])
        scan_date = observation["scan_date"]
        index.setdefault(aoi_id, {}).setdefault(scan_date, []).append(observation)
    return index


def _same_time_ema_observations(
    observations_by_day: dict[date, list[dict[str, object]]], target_time: datetime
) -> list[dict[str, object]]:
    # Keep the closest scan per prior day inside the same-time tolerance
    target_seconds = target_time.hour * 3600 + target_time.minute * 60 + target_time.second
    selected = []
    for age_days in range(1, EMA_HISTORY_DAYS + 1):
        scan_date = target_time.date() - timedelta(days=age_days)
        candidates = []
        for observation in observations_by_day.get(scan_date, []):
            observation_time = observation["tempo_time"]
            observation_seconds = observation_time.hour * 3600 + observation_time.minute * 60 + observation_time.second
            difference = abs((observation_seconds - target_seconds + 43_200) % 86_400 - 43_200)
            if difference <= EMA_SAME_TIME_TOLERANCE_MINUTES * 60:
                candidates.append((difference, -int(observation["sampled_pixel_count"]), observation))
        if candidates:
            selected.append(min(candidates, key=lambda priority: priority[:2])[2])
    return sorted(selected, key=lambda row: row["tempo_time"])


def _prepare_records(
    splits: dict[str, pl.DataFrame],
    tempo_cache_dir: Path,
    wind_cache_dir: Path,
    run_dir: Path,
) -> tuple[list[PreparedRecord], dict[str, ScanTask], dict[str, WindTask], dict[str, list[dict[str, object]]]]:
    # Build one global scan plan across train, validation, and test
    records: list[PreparedRecord] = []
    scans: dict[str, ScanTask] = {}
    winds: dict[str, WindTask] = {}
    failures: dict[str, list[dict[str, object]]] = {split: [] for split in splits}
    aoi_ids = sorted({int(aoi_id) for frame in splits.values() for aoi_id in frame["aoi_id"].unique()})
    observations_by_aoi = _load_observations(aoi_ids)
    for split, frame in splits.items():
        output_dir = run_dir / "record-rasters" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in frame.iter_rows(named=True):
            record_index = int(row[SOURCE_RECORD_INDEX_COL])
            try:
                current = make_scan_task(row, "tempo", Path(TEMPO_DIR), tempo_cache_dir)
                previous = make_scan_task(row, "prev_tempo", Path(TEMPO_DIR), tempo_cache_dir)
                wind = make_wind_task(row, Path(HRRR_DIR), wind_cache_dir)
                ema_observations = _same_time_ema_observations(
                    observations_by_aoi[int(row["aoi_id"])], row["tempo_time"]
                )
                if len(ema_observations) < EMA_MIN_SCANS:
                    raise ValueError(f"Only {len(ema_observations)} same-time EMA scans are available")
                ema_scans = []
                ema_scan_age_days = []
                for observation in ema_observations:
                    ema_row = {
                        "aoi_id": row["aoi_id"],
                        "lon": row["lon"],
                        "lat": row["lat"],
                        "tempo": observation["granule_paths"],
                    }
                    ema_scan = make_scan_task(ema_row, "tempo", Path(TEMPO_DIR), tempo_cache_dir)
                    ema_scans.append(ema_scan)
                    age = (row["tempo_time"] - observation["tempo_time"]).total_seconds() / 86_400
                    ema_scan_age_days.append(float(age))
                delta_no2_path = output_dir / f"{record_index:06d}.npz"
                records.append(
                    PreparedRecord(
                        split=split,
                        record_index=record_index,
                        current_scan_key=current.cache_key,
                        previous_scan_key=previous.cache_key,
                        ema_scan_keys=tuple(scan.cache_key for scan in ema_scans),
                        ema_scan_age_days=tuple(ema_scan_age_days),
                        wind_cache_key=wind.cache_key,
                        delta_no2_path=str(delta_no2_path),
                    )
                )
                scans.setdefault(current.cache_key, current)
                scans.setdefault(previous.cache_key, previous)
                for ema_scan in ema_scans:
                    scans.setdefault(ema_scan.cache_key, ema_scan)
                winds.setdefault(wind.cache_key, wind)
            except (KeyError, TypeError, ValueError) as error:
                failures[split].append({"record_index": record_index, "error": str(error)})
    return records, scans, winds, failures


def _scan_batches(scans: Iterable[ScanTask]) -> list[ScanBatchTask]:
    # Group AOIs by source files so each worker reads a granule set once
    grouped: dict[tuple[str, ...], list[ScanTask]] = {}
    for scan in scans:
        grouped.setdefault(scan.granule_paths, []).append(scan)
    return [ScanBatchTask(paths, tuple(group)) for paths, group in grouped.items()]


def _run_tempo_regridding(
    scans: dict[str, ScanTask],
    workers: int,
    refresh_tempo: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    # Reuse existing entries and batch only cache misses
    tempo_cache_paths = {
        key: task.cache_path for key, task in scans.items() if not refresh_tempo and cache_exists(task.cache_path)
    }
    failures: dict[str, str] = {}
    missing = [task for key, task in scans.items() if key not in tempo_cache_paths]
    print(f"TEMPO cache: {len(tempo_cache_paths):,} hits; {len(missing):,} scans to generate")
    if not missing:
        return tempo_cache_paths, failures
    batches = _scan_batches(missing)
    granule_reads = sum(len(batch.granule_paths) for batch in batches)
    print(f"Grouped cache misses into {len(batches):,} batches requiring {granule_reads:,} granule reads")
    completed = 0
    for batch_results in _bounded_parallel_map(process_scan_batch, batches, workers):
        for result in batch_results:
            completed += 1
            if result.error is None:
                tempo_cache_paths[result.cache_key] = result.cache_path
            else:
                failures[result.cache_key] = result.error
        if completed % PROGRESS_INTERVAL < len(batch_results) or completed == len(missing):
            print(f"Regridded {completed:,}/{len(missing):,} cache-missing AOI scans")
    return tempo_cache_paths, failures


def _wind_batches(winds: Iterable[WindTask]) -> list[WindBatchTask]:
    # Group AOIs by HRRR hour so each full field is read once
    grouped: dict[str, list[WindTask]] = {}
    for wind in winds:
        grouped.setdefault(wind.hrrr_path, []).append(wind)
    return [WindBatchTask(path, tuple(group)) for path, group in grouped.items()]


def _run_wind_alignment(
    winds: dict[str, WindTask],
    workers: int,
    refresh_wind: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    # Reuse cached AOI-hour wind rasters unless refresh is explicit
    cache_paths = {
        key: task.cache_path for key, task in winds.items() if not refresh_wind and cache_exists(task.cache_path)
    }
    failures: dict[str, str] = {}
    missing = [task for key, task in winds.items() if key not in cache_paths]
    print(f"Wind cache: {len(cache_paths):,} hits; {len(missing):,} AOI-hours to align")
    if not missing:
        return cache_paths, failures
    completed = 0
    for batch_results in _bounded_parallel_map(process_wind_batch, _wind_batches(missing), workers):
        for result in batch_results:
            completed += 1
            if result.error is None:
                cache_paths[result.cache_key] = result.cache_path
            else:
                failures[result.cache_key] = result.error
        if completed % PROGRESS_INTERVAL < len(batch_results) or completed == len(missing):
            print(f"Aligned {completed:,}/{len(missing):,} cache-missing AOI-hour winds")
    return cache_paths, failures


def _record_tasks(
    records: list[PreparedRecord],
    tempo_cache_paths: dict[str, str],
    tempo_failures: dict[str, str],
    wind_cache_paths: dict[str, str],
    wind_failures: dict[str, str],
    failures: dict[str, list[dict[str, object]]],
) -> tuple[list[RecordTask], dict[tuple[str, int], PreparedRecord]]:
    # Exclude records whose current or previous scan failed to regrid
    tasks: list[RecordTask] = []
    records_by_id: dict[tuple[str, int], PreparedRecord] = {}
    for record in records:
        missing_keys = [
            key for key in (record.current_scan_key, record.previous_scan_key) if key not in tempo_cache_paths
        ]
        if missing_keys:
            reasons = [tempo_failures.get(key, "TEMPO cache unavailable") for key in missing_keys]
            failures[record.split].append({"record_index": record.record_index, "error": "; ".join(reasons)})
            continue
        available_ema_scans = [
            (tempo_cache_paths[key], age)
            for key, age in zip(record.ema_scan_keys, record.ema_scan_age_days, strict=True)
            if key in tempo_cache_paths
        ]
        if len(available_ema_scans) < EMA_MIN_SCANS:
            missing_count = len(record.ema_scan_keys) - len(available_ema_scans)
            failures[record.split].append(
                {
                    "record_index": record.record_index,
                    "error": (
                        f"Only {len(available_ema_scans)} EMA scans remain after {missing_count} TEMPO cache failures"
                    ),
                }
            )
            continue
        if record.wind_cache_key not in wind_cache_paths:
            reason = wind_failures.get(record.wind_cache_key, "wind cache unavailable")
            failures[record.split].append({"record_index": record.record_index, "error": reason})
            continue
        record_id = (record.split, record.record_index)
        records_by_id[record_id] = record
        tasks.append(
            RecordTask(
                split=record.split,
                record_index=record.record_index,
                current_cache_path=tempo_cache_paths[record.current_scan_key],
                previous_cache_path=tempo_cache_paths[record.previous_scan_key],
                ema_scan_paths=tuple(path for path, _ in available_ema_scans),
                ema_scan_age_days=tuple(age for _, age in available_ema_scans),
                wind_cache_path=wind_cache_paths[record.wind_cache_key],
                output_path=record.delta_no2_path,
            )
        )
    return tasks, records_by_id


def _run_record_processing(
    tasks: list[RecordTask],
    records_by_id: dict[tuple[str, int], PreparedRecord],
    failures: dict[str, list[dict[str, object]]],
    workers: int,
) -> dict[str, list[dict[str, object]]]:
    # Derive delta rasters and scalar features in parallel
    output_rows: dict[str, list[dict[str, object]]] = {split: [] for split in failures}
    total = len(tasks)
    for completed, result in enumerate(_bounded_parallel_map(process_record, tasks, workers), start=1):
        record_id = (result.split, result.record_index)
        if result.error is not None:
            failures[result.split].append({"record_index": result.record_index, "error": result.error})
        else:
            record = records_by_id[record_id]
            output_row: dict[str, object] = {
                SOURCE_RECORD_INDEX_COL: result.record_index,
                CANDIDATE_RASTER_PATH_COL: record.delta_no2_path,
            }
            output_row.update(result.features)
            output_rows[result.split].append(output_row)
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            print(f"Processed {completed:,}/{total:,} paired records")
    return output_rows


def _write_outputs(
    output_rows: dict[str, list[dict[str, object]]],
    failures: dict[str, list[dict[str, object]]],
    source_splits: dict[str, pl.DataFrame],
) -> None:
    # Sort asynchronous results back into source-record order
    prepared_outputs: dict[str, tuple[pl.DataFrame, dict[str, object], pl.DataFrame]] = {}
    for split, source_frame in source_splits.items():
        rows = output_rows[split]
        failure_rows = sorted(failures[split], key=lambda row: int(row["record_index"]))
        features = pl.DataFrame(rows, schema=_feature_schema())
        candidates = source_frame.join(features, on=SOURCE_RECORD_INDEX_COL, how="inner", maintain_order="left").sort(
            SOURCE_RECORD_INDEX_COL
        )
        output_frame = select_final_records(candidates, FINAL_SPLIT_SIZES[split])
        print(f"[{split}] {candidates.height:,} generated; {output_frame.height:,} selected")
        classification_report = {
            "split": split,
            "raw_delta_nox_threshold": DELTA_THRESHOLD,
            "final_balance": classification_summary(candidates, output_frame),
        }
        prepared_outputs[split] = (
            output_frame,
            classification_report,
            pl.DataFrame(failure_rows, schema=_failure_schema()),
        )

    for split, (output_frame, classification_report, failure_frame) in prepared_outputs.items():
        output_frame = _install_selected_rasters(split, output_frame)
        write_csv_atomic(
            output_frame.drop(SOURCE_RECORD_INDEX_COL, CANDIDATE_RASTER_PATH_COL),
            Path(DATASET_DF) / f"{split}_df.csv",
        )
        write_json_atomic(
            classification_report,
            Path(DATASET_DF) / f"{split}_classification_summary.json",
        )
        write_csv_atomic(
            failure_frame,
            Path(DATASET_DF) / f"{split}_failures.csv",
        )
        print(f"[{split}] wrote {output_frame.height:,} records; {failure_frame.height:,} processing failures")


def _install_selected_rasters(split: str, frame: pl.DataFrame) -> pl.DataFrame:
    # Atomically replace one split without consuming resumable shard rasters
    raster_root = Path(DATASET_RASTER_DIR)
    raster_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{split}-staging-", dir=raster_root))
    final_dir = raster_root / split
    backup = Path(tempfile.mkdtemp(prefix=f".{split}-backup-", dir=raster_root))
    backup.rmdir()
    had_previous = final_dir.exists()
    relative_paths = []
    try:
        for output_index, candidate_path in enumerate(frame[CANDIDATE_RASTER_PATH_COL].to_list()):
            filename = f"{output_index:06d}.npz"
            destination = staging / filename
            try:
                os.link(candidate_path, destination)
            except OSError:
                shutil.copy2(candidate_path, destination)
            relative_paths.append(str(Path("rasters") / split / filename))
        if had_previous:
            os.replace(final_dir, backup)
        os.replace(staging, final_dir)
    except Exception:
        if final_dir.exists() and had_previous and backup.exists():
            shutil.rmtree(final_dir)
            os.replace(backup, final_dir)
        elif had_previous and backup.exists():
            os.replace(backup, final_dir)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not had_previous:
            shutil.rmtree(backup)
    return frame.with_columns(pl.Series(DELTA_NO2_PATH_COL, relative_paths, dtype=pl.String))


def _completion_values(task: ShardTask, successful_count: int, failure_count: int) -> dict[str, object]:
    # Describe the deterministic shard boundary and its terminal outcomes
    return {
        "split": task.split,
        "shard_index": task.shard_index,
        "start": task.start,
        "stop": task.stop,
        "source_count": task.size,
        "successful_count": successful_count,
        "failure_count": failure_count,
    }


def _load_completed_shard(task: ShardTask, resolve_paths: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Validate the completion marker and every source-record outcome
    shard_dir = _shard_path(task)
    with (shard_dir / SHARD_COMPLETION_FILE).open() as source:
        completion = json.load(source)
    candidates = pl.read_csv(
        shard_dir / SHARD_CANDIDATES_FILE,
        schema_overrides=_feature_schema(),
    )
    failures = pl.read_csv(
        shard_dir / SHARD_FAILURES_FILE,
        schema_overrides={"record_index": pl.Int64, "error": pl.String},
    )
    expected_completion = _completion_values(task, candidates.height, failures.height)
    if completion != expected_completion:
        raise ValueError(f"Shard {task.task_id} has an inconsistent completion marker")
    if candidates.schema != _feature_schema():
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


def _is_shard_complete(task: ShardTask) -> bool:
    # Treat any missing or inconsistent artifact as incomplete
    try:
        _load_completed_shard(task)
    except (json.JSONDecodeError, OSError, TypeError, ValueError, pl.exceptions.PolarsError):
        return False
    return True


def _remove_shard_directory(path: Path, split_root: Path) -> None:
    # Restrict recursive deletion to one split's shard directory
    if path.parent.resolve() != split_root.resolve():
        raise ValueError(f"Refusing to clear shard path outside {split_root}: {path}")
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError(f"Shard path is not a directory: {path}")
    if path.exists():
        shutil.rmtree(path)


def _clear_task_workspace(task: ShardTask) -> None:
    # Remove one invalid shard and temporary directories from interrupted attempts
    split_root = _prepare_split_shard_root(task.split)
    _remove_shard_directory(_shard_path(task), split_root)
    temporary_prefix = f".{task.shard_index:06d}-"
    for path in split_root.iterdir():
        if path.name.startswith(temporary_prefix):
            _remove_shard_directory(path, split_root)


def _clear_shard_splits(split_paths: dict[str, str]) -> None:
    # Remove only selected split directories under the configured shard root
    shard_root = _prepare_shard_root()
    for split in split_paths:
        split_path = shard_root / split
        if split_path.exists():
            _remove_shard_directory(split_path, shard_root)
    if shard_root.exists() and not any(shard_root.iterdir()):
        shard_root.rmdir()


def _write_shard_outputs(
    task: ShardTask,
    staging: Path,
    output_rows: dict[str, list[dict[str, object]]],
    failures: dict[str, list[dict[str, object]]],
) -> None:
    # Persist relative raster paths before publishing the completed shard
    candidates = pl.DataFrame(output_rows[task.split], schema=_feature_schema()).sort(SOURCE_RECORD_INDEX_COL)
    relative_paths = []
    for candidate_path in candidates[CANDIDATE_RASTER_PATH_COL].to_list():
        relative_paths.append(str(Path(candidate_path).relative_to(staging)))
    candidates = candidates.with_columns(pl.Series(CANDIDATE_RASTER_PATH_COL, relative_paths, dtype=pl.String))
    failure_rows = sorted(failures[task.split], key=lambda row: int(row["record_index"]))
    failure_frame = pl.DataFrame(failure_rows, schema=_failure_schema())
    outcome_indices = [int(value) for value in candidates[SOURCE_RECORD_INDEX_COL].to_list()]
    outcome_indices.extend(int(value) for value in failure_frame["record_index"].to_list())
    if sorted(outcome_indices) != list(range(task.start, task.stop)):
        raise ValueError(f"Shard {task.task_id} did not produce one outcome per source record")

    write_csv_atomic(candidates, staging / SHARD_CANDIDATES_FILE)
    write_csv_atomic(failure_frame, staging / SHARD_FAILURES_FILE)
    write_json_atomic(
        _completion_values(task, candidates.height, failure_frame.height),
        staging / SHARD_COMPLETION_FILE,
    )


def _run_shard(task: ShardTask) -> None:
    # Generate one resumable source-record range
    if _is_shard_complete(task):
        print(f"Shard {task.task_id} is already complete")
        return
    _clear_task_workspace(task)
    split_root = _prepare_split_shard_root(task.split)
    staging = Path(tempfile.mkdtemp(prefix=f".{task.shard_index:06d}-", dir=split_root))
    tempo_cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    wind_cache_dir = Path(DATASET_WIND_CACHE_DIR)
    tempo_cache_dir.mkdir(parents=True, exist_ok=True)
    wind_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        splits = _load_shard(task)
        records, scans, winds, failures = _prepare_records(
            splits,
            tempo_cache_dir,
            wind_cache_dir,
            staging,
        )
        print(
            f"Planned shard {task.task_id} with {len(records):,} records, "
            f"{len(scans):,} TEMPO scans, and {len(winds):,} wind rasters"
        )
        tempo_cache_paths, tempo_failures = _run_tempo_regridding(scans, NUM_CORES, False)
        wind_cache_paths, wind_failures = _run_wind_alignment(winds, NUM_CORES, False)
        record_tasks, records_by_id = _record_tasks(
            records,
            tempo_cache_paths,
            tempo_failures,
            wind_cache_paths,
            wind_failures,
            failures,
        )
        output_rows = _run_record_processing(record_tasks, records_by_id, failures, NUM_CORES)
        _write_shard_outputs(task, staging, output_rows, failures)
        os.replace(staging, _shard_path(task))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Completed shard {task.task_id}: {task.split} records {task.start:,}:{task.stop:,}")


def _finalize_shards(tasks: list[ShardTask], split_paths: dict[str, str]) -> None:
    # Combine validated shard outputs before applying global split selection
    output_rows: dict[str, list[dict[str, object]]] = {split: [] for split in split_paths}
    failures: dict[str, list[dict[str, object]]] = {split: [] for split in split_paths}
    for task in tasks:
        try:
            candidates, failure_frame = _load_completed_shard(task, resolve_paths=True)
        except (json.JSONDecodeError, OSError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(f"Cannot finalize incomplete shard {task.task_id}: {error}") from error
        output_rows[task.split].extend(candidates.to_dicts())
        failures[task.split].extend(failure_frame.to_dicts())

    source_splits = _load_splits(split_paths)
    _write_outputs(output_rows, failures, source_splits)
    _clear_shard_splits(split_paths)
    print(f"Finalized {len(tasks):,} shards and removed their staging workspace")


def parse_args() -> argparse.Namespace:
    """Parse dataset-generation command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", *SPLIT_PATHS), default=_default_split())
    parser.add_argument(
        "--shard-size",
        type=_positive_int,
        help="source records processed by each Slurm array task",
    )
    parser.add_argument(
        "--refresh-shards",
        action="store_true",
        help="discard completed and partial shards before a sharded run",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="empty both image caches before rebuilding entries for the selected split",
    )
    parser.add_argument(
        "--refresh-tempo",
        action="store_true",
        help="empty the TEMPO image cache before rebuilding entries for the selected split",
    )
    parser.add_argument(
        "--refresh-wind",
        action="store_true",
        help="empty the wind image cache before rebuilding entries for the selected split",
    )
    return parser.parse_args()


def _default_split() -> str:
    # Preserve the legacy split-per-task array when no internal stage is set
    if os.getenv("DATASET_GENERATION_STAGE") is not None:
        return "all"
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id is None or os.getenv("SLURM_JOB_ID") is None:
        return "all"
    return ARRAY_SPLITS[int(task_id)]


def _selected_split_paths(split: str) -> dict[str, str]:
    # Preserve the optional single-split local workflow
    return SPLIT_PATHS if split == "all" else {split: SPLIT_PATHS[split]}


def _reset_cache_directory(cache_dir: Path) -> None:
    # Restrict recursive deletion to a configured direct child of DATASET_DIR
    dataset_root = Path(DATASET_DIR).resolve()
    resolved_cache = cache_dir.resolve()
    if resolved_cache.parent != dataset_root:
        raise ValueError(f"Refusing to clear cache outside {dataset_root}: {resolved_cache}")
    if resolved_cache.exists() and not resolved_cache.is_dir():
        raise ValueError(f"Cache path is not a directory: {resolved_cache}")
    if resolved_cache.exists():
        print(f"Clearing persistent image cache: {resolved_cache}")
        shutil.rmtree(resolved_cache)
    resolved_cache.mkdir(parents=True)


def _reset_requested_caches(args: argparse.Namespace, tempo_cache_dir: Path, wind_cache_dir: Path) -> None:
    # Clear shared caches only from a single non-array process
    refresh_tempo = args.refresh_cache or args.refresh_tempo
    refresh_wind = args.refresh_cache or args.refresh_wind
    if not refresh_tempo and not refresh_wind:
        return
    if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
        raise ValueError("Cache refresh cannot run inside a Slurm array because its tasks share cache directories")
    if refresh_tempo:
        _reset_cache_directory(tempo_cache_dir)
    if refresh_wind:
        _reset_cache_directory(wind_cache_dir)


def _initialize_output_directories() -> tuple[Path, Path]:
    # Create persistent output and cache directories
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATASET_DF).mkdir(parents=True, exist_ok=True)
    Path(DATASET_RASTER_DIR).mkdir(parents=True, exist_ok=True)
    tempo_cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    wind_cache_dir = Path(DATASET_WIND_CACHE_DIR)
    tempo_cache_dir.mkdir(parents=True, exist_ok=True)
    wind_cache_dir.mkdir(parents=True, exist_ok=True)
    return tempo_cache_dir, wind_cache_dir


def _run_monolithic(args: argparse.Namespace) -> None:
    # Preserve direct single-process generation outside the shard launcher
    tempo_cache_dir, wind_cache_dir = _initialize_output_directories()
    _reset_requested_caches(args, tempo_cache_dir, wind_cache_dir)
    splits = _load_splits(_selected_split_paths(args.split))
    refresh_tempo = args.refresh_cache or args.refresh_tempo
    refresh_wind = args.refresh_cache or args.refresh_wind
    with tempfile.TemporaryDirectory(prefix=".dataset-run-", dir=DATASET_DIR) as temporary_dir:
        records, scans, winds, failures = _prepare_records(
            splits,
            tempo_cache_dir,
            wind_cache_dir,
            Path(temporary_dir),
        )
        print(f"Planned {len(records):,} records using {len(scans):,} TEMPO scans and {len(winds):,} wind rasters")
        tempo_cache_paths, tempo_failures = _run_tempo_regridding(scans, NUM_CORES, refresh_tempo)
        wind_cache_paths, wind_failures = _run_wind_alignment(winds, NUM_CORES, refresh_wind)
        tasks, records_by_id = _record_tasks(
            records,
            tempo_cache_paths,
            tempo_failures,
            wind_cache_paths,
            wind_failures,
            failures,
        )
        output_rows = _run_record_processing(tasks, records_by_id, failures, NUM_CORES)
        _write_outputs(output_rows, failures, splits)


def _required_shard_size(args: argparse.Namespace) -> int:
    # Require explicit sizing for every internal sharded stage
    if args.shard_size is None:
        raise ValueError("Sharded dataset generation requires --shard-size")
    return int(args.shard_size)


def _active_dataset_job_ids() -> list[str]:
    # Find overlapping worker or finalizer jobs owned by the current user
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--user",
            str(os.environ["USER"]),
            f"--name={LEGACY_DATASET_JOB_NAME},{SHARD_WORKER_JOB_NAME},{SHARD_FINALIZER_JOB_NAME}",
            "--format=%i",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _submit_job(arguments: list[str]) -> str:
    # Submit one Slurm job and return its cluster-local numeric ID
    result = subprocess.run(
        ["sbatch", "--parsable", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    job_id = result.stdout.strip().partition(";")[0]
    if not job_id:
        raise RuntimeError("sbatch returned an empty job ID")
    return job_id


def _shard_cli_arguments(args: argparse.Namespace, shard_size: int) -> list[str]:
    # Forward only immutable plan inputs to workers and the finalizer
    arguments = ["--shard-size", str(shard_size)]
    if args.split != "all":
        arguments.extend(("--split", args.split))
    return arguments


def _launch_sharded_run(args: argparse.Namespace) -> None:
    # Prepare resumable state before submitting array workers and a finalizer
    if os.getenv("SLURM_JOB_ID") is not None:
        raise ValueError("Launch sharded dataset generation from a login node")
    if not DATASET_BATCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Dataset batch script is missing: {DATASET_BATCH_SCRIPT}")
    (REPOSITORY_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    active_job_ids = _active_dataset_job_ids()
    if active_job_ids:
        joined_ids = ", ".join(active_job_ids)
        raise RuntimeError(f"Dataset-generation jobs are already active: {joined_ids}")

    shard_size = _required_shard_size(args)
    split_paths = _selected_split_paths(args.split)
    tempo_cache_dir, wind_cache_dir = _initialize_output_directories()
    _reset_requested_caches(args, tempo_cache_dir, wind_cache_dir)
    if args.refresh_shards or args.refresh_cache or args.refresh_tempo or args.refresh_wind:
        _clear_shard_splits(split_paths)
    tasks = _shard_plan(split_paths, shard_size)
    pending_task_ids = [task.task_id for task in tasks if not _is_shard_complete(task)]
    array_spec = _slurm_array_spec(pending_task_ids)
    shard_arguments = _shard_cli_arguments(args, shard_size)
    worker_job_id: str | None = None
    if pending_task_ids:
        worker_job_id = _submit_job(
            [
                f"--array={array_spec}",
                f"--job-name={SHARD_WORKER_JOB_NAME}",
                "--export=ALL,DATASET_GENERATION_STAGE=worker",
                str(DATASET_BATCH_SCRIPT),
                *shard_arguments,
            ]
        )
        print(f"Dataset shard array: {worker_job_id}")
    else:
        print("All dataset shards are already complete")

    finalizer_options = [
        "--array=0",
        "--cpus-per-task=1",
        "--time=01:00:00",
        f"--job-name={SHARD_FINALIZER_JOB_NAME}",
        "--export=ALL,DATASET_GENERATION_STAGE=finalize",
    ]
    if worker_job_id is not None:
        finalizer_options.append(f"--dependency=afterany:{worker_job_id}")
    finalizer_job_id = _submit_job([*finalizer_options, str(DATASET_BATCH_SCRIPT), *shard_arguments])
    print(f"Planned {len(tasks):,} shards with {len(pending_task_ids):,} pending")
    print(f"Dataset finalizer: {finalizer_job_id}")


def _run_array_shard(args: argparse.Namespace) -> None:
    # Resolve this array index through the deterministic source-row plan
    task_id_text = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise ValueError("Shard workers require SLURM_ARRAY_TASK_ID")
    tasks = _shard_plan(_selected_split_paths(args.split), _required_shard_size(args))
    task_id = int(task_id_text)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Array task {task_id} is outside the {len(tasks):,}-shard plan")
    _run_shard(tasks[task_id])


def _run_shard_finalizer(args: argparse.Namespace) -> None:
    # Validate and combine the complete deterministic shard plan
    split_paths = _selected_split_paths(args.split)
    tasks = _shard_plan(split_paths, _required_shard_size(args))
    _initialize_output_directories()
    _finalize_shards(tasks, split_paths)


def main() -> None:
    """Generate paired raster NPZ files and metadata CSVs for all splits."""
    args = parse_args()
    stage = os.getenv("DATASET_GENERATION_STAGE", "generate")
    if stage == "worker":
        _run_array_shard(args)
    elif stage == "finalize":
        _run_shard_finalizer(args)
    elif stage == "generate":
        if args.shard_size is not None:
            _launch_sharded_run(args)
        elif args.refresh_shards:
            raise ValueError("--refresh-shards requires --shard-size")
        else:
            _run_monolithic(args)
    else:
        raise ValueError(f"Unsupported internal dataset-generation stage: {stage}")


if __name__ == "__main__":
    main()
