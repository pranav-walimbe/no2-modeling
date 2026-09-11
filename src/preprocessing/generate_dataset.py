"""Generate paired TEMPO rasters and tabular features for every data split."""

import argparse
import os
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from config import (
    DATASET_DF,
    DATASET_DIR,
    DATASET_RASTER_DIR,
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WIND_CACHE_DIR,
    DELTA_THRESHOLD,
    HRRR_DIR,
    LABEL_COL,
    NUM_CORES,
    TEMPO_DIR,
    TEST_RECORDS_CSV,
    TEST_SIZE,
    TRAIN_RECORDS_CSV,
    TRAIN_SIZE,
    VAL_RECORDS_CSV,
    VAL_SIZE,
)
from preprocessing.generate_dataset_utils import (
    CANDIDATE_FEATURE_SCHEMA,
    CANDIDATE_RASTER_PATH_COL,
    DELTA_NO2_PATH_COL,
    PROCESSING_FAILURE_SCHEMA,
    SOURCE_RECORD_INDEX_COL,
    DatasetShardStore,
    RecordTask,
    ScanTask,
    ShardTask,
    WindTask,
    bounded_parallel_map,
    build_shard_plan,
    cache_inventory,
    coverage_selection_summary,
    make_scan_task,
    make_wind_task,
    process_record,
    process_scan_batch,
    process_wind_batch,
    scan_batches,
    select_final_records,
    wind_batches,
    write_csv_atomic,
    write_json_atomic,
)
from preprocessing.stratify_utils import HRRR_FIELD_SLUG, HRRR_PRODUCT, classification_summary

SPLIT_PATHS = {
    "train": TRAIN_RECORDS_CSV,
    "val": VAL_RECORDS_CSV,
    "test": TEST_RECORDS_CSV,
}
ARRAY_SPLITS = tuple(SPLIT_PATHS)
PROGRESS_INTERVAL = 1_000
LEGACY_DATASET_JOB_NAME = "generate-dataset"
SHARD_WORKER_JOB_NAME = "generate-dataset-shard"
SHARD_FINALIZER_JOB_NAME = "generate-dataset-finalize"
TRAINING_JOB_NAME = "train-no2"
RUN_STARTED_ENV = "DATASET_RUN_STARTED_AT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_BATCH_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "generate_dataset.sh"
FINAL_SPLIT_SIZES = {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE}


@dataclass(frozen=True)
class PreparedRecord:
    """One source row resolved to its cache and output paths."""

    split: str
    record_index: int
    current_scan_key: str
    previous_scan_key: str
    current_wind_cache_key: str
    previous_wind_cache_key: str
    delta_no2_path: str


def _positive_int(value: str) -> int:
    # Parse a strictly positive command-line integer
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


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


def _observation_wind_task(
    row: dict[str, object],
    time_column: str,
    hrrr_root: Path,
    wind_cache_dir: Path,
) -> WindTask:
    # Resolve wind to the HRRR analysis nearest the TEMPO observation
    observation_time = row[time_column]
    if not isinstance(observation_time, datetime):
        raise TypeError(f"{time_column} must be a datetime")
    matched_time = (observation_time + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    relative_path = (
        f"raw/{matched_time:%Y/%m/%d}/hrrr_{matched_time:%Y%m%d_%H}z_"
        f"{HRRR_PRODUCT}_{HRRR_FIELD_SLUG}.grib2"
    )
    return make_wind_task({**row, "hrrr": relative_path}, hrrr_root, wind_cache_dir)


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
    source_count = sum(frame.height for frame in splits.values())
    prepared_count = 0
    for split, frame in splits.items():
        output_dir = run_dir / "record-rasters" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in frame.iter_rows(named=True):
            prepared_count += 1
            record_index = int(row[SOURCE_RECORD_INDEX_COL])
            try:
                current = make_scan_task(row, "tempo", Path(TEMPO_DIR), tempo_cache_dir)
                previous = make_scan_task(row, "prev_tempo", Path(TEMPO_DIR), tempo_cache_dir)
                current_wind = _observation_wind_task(
                    row,
                    "tempo_time",
                    Path(HRRR_DIR),
                    wind_cache_dir,
                )
                previous_wind = _observation_wind_task(
                    row,
                    "prev_tempo_time",
                    Path(HRRR_DIR),
                    wind_cache_dir,
                )
                delta_no2_path = output_dir / f"{record_index:06d}.npz"
                records.append(
                    PreparedRecord(
                        split=split,
                        record_index=record_index,
                        current_scan_key=current.cache_key,
                        previous_scan_key=previous.cache_key,
                        current_wind_cache_key=current_wind.cache_key,
                        previous_wind_cache_key=previous_wind.cache_key,
                        delta_no2_path=str(delta_no2_path),
                    )
                )
                scans.setdefault(current.cache_key, current)
                scans.setdefault(previous.cache_key, previous)
                winds.setdefault(current_wind.cache_key, current_wind)
                winds.setdefault(previous_wind.cache_key, previous_wind)
            except (KeyError, TypeError, ValueError) as error:
                failures[split].append({"record_index": record_index, "error": str(error)})
            if prepared_count % PROGRESS_INTERVAL == 0 or prepared_count == source_count:
                print(f"Prepared cache plan for {prepared_count:,}/{source_count:,} source records")
    return records, scans, winds, failures


def _cached_task_paths(tasks: dict[str, ScanTask] | dict[str, WindTask]) -> dict[str, str]:
    # Scan each cache directory once instead of issuing one metadata lookup per task
    directories = {Path(task.cache_path).parent for task in tasks.values()}
    inventories = {directory: cache_inventory(directory) for directory in directories}
    return {
        key: task.cache_path
        for key, task in tasks.items()
        if Path(task.cache_path).name in inventories[Path(task.cache_path).parent]
    }


def _run_tempo_regridding(
    scans: dict[str, ScanTask],
    workers: int,
    refresh_tempo: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    # Reuse existing entries and batch only cache misses
    tempo_cache_paths = {} if refresh_tempo else _cached_task_paths(scans)
    failures: dict[str, str] = {}
    missing = [task for key, task in scans.items() if key not in tempo_cache_paths]
    print(f"TEMPO cache: {len(tempo_cache_paths):,} hits; {len(missing):,} scans to generate")
    if not missing:
        return tempo_cache_paths, failures
    batches = scan_batches(missing, reuse_existing=not refresh_tempo)
    granule_reads = sum(len(batch.granule_paths) for batch in batches)
    print(f"Grouped cache misses into {len(batches):,} batches requiring {granule_reads:,} granule reads")
    completed = 0
    for batch_results in bounded_parallel_map(process_scan_batch, batches, workers):
        for result in batch_results:
            completed += 1
            if result.error is None:
                tempo_cache_paths[result.cache_key] = result.cache_path
            else:
                failures[result.cache_key] = result.error
        if completed % PROGRESS_INTERVAL < len(batch_results) or completed == len(missing):
            print(f"Regridded {completed:,}/{len(missing):,} cache-missing AOI scans")
    return tempo_cache_paths, failures


def _run_wind_alignment(
    winds: dict[str, WindTask],
    workers: int,
    refresh_wind: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    # Reuse cached AOI-hour wind rasters unless refresh is explicit
    cache_paths = {} if refresh_wind else _cached_task_paths(winds)
    failures: dict[str, str] = {}
    missing = [task for key, task in winds.items() if key not in cache_paths]
    print(f"Wind cache: {len(cache_paths):,} hits; {len(missing):,} AOI-hours to align")
    if not missing:
        return cache_paths, failures
    completed = 0
    batches = wind_batches(missing, reuse_existing=not refresh_wind)
    for batch_results in bounded_parallel_map(process_wind_batch, batches, workers):
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
        missing_wind_keys = [
            key
            for key in (record.current_wind_cache_key, record.previous_wind_cache_key)
            if key not in wind_cache_paths
        ]
        if missing_wind_keys:
            reasons = [wind_failures.get(key, "wind cache unavailable") for key in missing_wind_keys]
            failures[record.split].append({"record_index": record.record_index, "error": "; ".join(reasons)})
            continue
        record_id = (record.split, record.record_index)
        records_by_id[record_id] = record
        tasks.append(
            RecordTask(
                split=record.split,
                record_index=record.record_index,
                current_cache_path=tempo_cache_paths[record.current_scan_key],
                previous_cache_path=tempo_cache_paths[record.previous_scan_key],
                current_wind_cache_path=wind_cache_paths[record.current_wind_cache_key],
                previous_wind_cache_path=wind_cache_paths[record.previous_wind_cache_key],
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
    for completed, result in enumerate(bounded_parallel_map(process_record, tasks, workers), start=1):
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
        features = pl.DataFrame(rows, schema=CANDIDATE_FEATURE_SCHEMA)
        candidates = source_frame.join(features, on=SOURCE_RECORD_INDEX_COL, how="inner", maintain_order="left").sort(
            SOURCE_RECORD_INDEX_COL
        )
        output_frame = select_final_records(candidates, FINAL_SPLIT_SIZES[split])
        selected_coverage = coverage_selection_summary(output_frame)
        eligible_by_class = {str(label): candidates.filter(pl.col(LABEL_COL) == label).height for label in (0, 1)}
        selection_size = {
            "requested_size": FINAL_SPLIT_SIZES[split],
            "actual_size": output_frame.height,
            "shortfall": FINAL_SPLIT_SIZES[split] - output_frame.height,
            "eligible_by_class": eligible_by_class,
        }
        print(f"[{split}] {candidates.height:,} generated; {output_frame.height:,} selected")
        if selection_size["shortfall"]:
            print(
                f"[{split}] requested {selection_size['requested_size']:,}; "
                f"using largest balanced subset with {selection_size['shortfall']:,} fewer records"
            )
        print(
            f"[{split}] full paired coverage: {selected_coverage['full_coverage_records']:,}/"
            f"{selected_coverage['records']:,} selected across {selected_coverage['aoi_count']:,} AOIs"
        )
        classification_report = {
            "split": split,
            "raw_delta_nox_threshold": DELTA_THRESHOLD,
            "selection_size": selection_size,
            "final_balance": classification_summary(candidates, output_frame),
            "coverage_selection": {
                "generated": coverage_selection_summary(candidates),
                "selected": selected_coverage,
            },
        }
        prepared_outputs[split] = (
            output_frame,
            classification_report,
            pl.DataFrame(failure_rows, schema=PROCESSING_FAILURE_SCHEMA),
        )

    for split, (output_frame, classification_report, failure_frame) in prepared_outputs.items():
        relative_paths = [
            str(Path(candidate_path).relative_to(DATASET_DIR))
            for candidate_path in output_frame[CANDIDATE_RASTER_PATH_COL].to_list()
        ]
        output_frame = output_frame.with_columns(pl.Series(DELTA_NO2_PATH_COL, relative_paths, dtype=pl.String))
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


def _run_shard(task: ShardTask, store: DatasetShardStore) -> None:
    # Generate one source-record range directly in its final shard directory
    started_at = time.perf_counter()
    shard_dir = store.create(task)
    tempo_cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    wind_cache_dir = Path(DATASET_WIND_CACHE_DIR)
    tempo_cache_dir.mkdir(parents=True, exist_ok=True)
    wind_cache_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_shard(task)
    records, scans, winds, failures = _prepare_records(
        splits,
        tempo_cache_dir,
        wind_cache_dir,
        shard_dir,
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
    store.write(task, shard_dir, output_rows[task.split], failures[task.split])
    elapsed = time.perf_counter() - started_at
    peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"Completed shard {task.task_id}: {task.split} records {task.start:,}:{task.stop:,}; "
        f"worker_time_seconds={elapsed:.1f}; peak_memory_mib={peak_memory_mib:.1f}"
    )


def _finalize_shards(
    tasks: list[ShardTask],
    split_paths: dict[str, str],
    store: DatasetShardStore,
) -> None:
    # Combine validated shard outputs before applying global split selection
    started_at = time.perf_counter()
    output_rows: dict[str, list[dict[str, object]]] = {split: [] for split in split_paths}
    failures: dict[str, list[dict[str, object]]] = {split: [] for split in split_paths}
    for task in tasks:
        try:
            candidates, failure_frame = store.load(task, resolve_paths=True)
        except (OSError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(f"Cannot finalize incomplete shard {task.task_id}: {error}") from error
        output_rows[task.split].extend(candidates.to_dicts())
        failures[task.split].extend(failure_frame.to_dicts())

    source_splits = _load_splits(split_paths)
    _write_outputs(output_rows, failures, source_splits)
    elapsed = time.perf_counter() - started_at
    peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    benchmark = f"finalizer_time_seconds={elapsed:.1f}; peak_memory_mib={peak_memory_mib:.1f}"
    run_started_at = os.getenv(RUN_STARTED_ENV)
    if run_started_at is not None:
        benchmark += f"; total_wall_seconds={time.time() - float(run_started_at):.1f}"
    print(f"Finalized {len(tasks):,} shards; {benchmark}")


def parse_args() -> argparse.Namespace:
    """Parse dataset-generation command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "A sharded launch replaces the current shards and published metadata. "
            "Do not launch while dataset generation or model training is active."
        ),
    )
    parser.add_argument("--split", choices=("all", *SPLIT_PATHS), default=_default_split())
    parser.add_argument(
        "--shard-size",
        type=_positive_int,
        help="source records per Slurm task in a fresh disposable-shard run",
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


def _reset_generated_outputs() -> None:
    # Remove disposable rasters and published metadata while preserving caches
    DatasetShardStore(Path(DATASET_DIR) / "shards").clear()
    for output_dir in (Path(DATASET_DF), Path(DATASET_RASTER_DIR)):
        if output_dir.exists():
            shutil.rmtree(output_dir)


def _run_monolithic(args: argparse.Namespace, split_paths: dict[str, str]) -> None:
    # Preserve direct single-process generation with persistent raster paths
    _refuse_active_dataset_jobs()
    _reset_generated_outputs()
    tempo_cache_dir, wind_cache_dir = _initialize_output_directories()
    _reset_requested_caches(args, tempo_cache_dir, wind_cache_dir)
    splits = _load_splits(split_paths)
    refresh_tempo = args.refresh_cache or args.refresh_tempo
    refresh_wind = args.refresh_cache or args.refresh_wind
    records, scans, winds, failures = _prepare_records(
        splits,
        tempo_cache_dir,
        wind_cache_dir,
        Path(DATASET_RASTER_DIR),
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


def _active_dataset_job_ids() -> list[str]:
    # Find dataset generation or training jobs owned by the current user
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--user",
            str(os.environ["USER"]),
            f"--name={LEGACY_DATASET_JOB_NAME},{SHARD_WORKER_JOB_NAME},{SHARD_FINALIZER_JOB_NAME},{TRAINING_JOB_NAME}",
            "--format=%i",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _refuse_active_dataset_jobs() -> None:
    # Protect dataset replacement from generation and training readers
    active_job_ids = _active_dataset_job_ids()
    if active_job_ids:
        joined_ids = ", ".join(active_job_ids)
        raise RuntimeError(f"Dataset-generation or training jobs are already active: {joined_ids}")


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


def _launch_sharded_run(args: argparse.Namespace, split_paths: dict[str, str], shard_size: int) -> None:
    # Replace disposable outputs before submitting workers and a finalizer
    if os.getenv("SLURM_JOB_ID") is not None:
        raise ValueError("Launch sharded dataset generation from a login node")
    if not DATASET_BATCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Dataset batch script is missing: {DATASET_BATCH_SCRIPT}")
    (REPOSITORY_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    _refuse_active_dataset_jobs()

    _reset_generated_outputs()
    tempo_cache_dir, wind_cache_dir = _initialize_output_directories()
    _reset_requested_caches(args, tempo_cache_dir, wind_cache_dir)
    tasks = build_shard_plan(split_paths, shard_size)
    task_ids = [task.task_id for task in tasks]
    array_spec = _slurm_array_spec(task_ids)
    run_started_at = str(time.time())
    shard_arguments = ["--shard-size", str(shard_size)]
    if args.split != "all":
        shard_arguments.extend(("--split", args.split))
    worker_job_id: str | None = None
    if task_ids:
        worker_job_id = _submit_job(
            [
                f"--array={array_spec}",
                f"--job-name={SHARD_WORKER_JOB_NAME}",
                f"--export=ALL,DATASET_GENERATION_STAGE=worker,{RUN_STARTED_ENV}={run_started_at}",
                str(DATASET_BATCH_SCRIPT),
                *shard_arguments,
            ]
        )
        print(f"Dataset shard array: {worker_job_id}")
    finalizer_options = [
        "--array=0",
        "--cpus-per-task=1",
        "--time=01:00:00",
        f"--job-name={SHARD_FINALIZER_JOB_NAME}",
        f"--export=ALL,DATASET_GENERATION_STAGE=finalize,{RUN_STARTED_ENV}={run_started_at}",
    ]
    if worker_job_id is not None:
        finalizer_options.append(f"--dependency=afterok:{worker_job_id}")
    finalizer_job_id = _submit_job([*finalizer_options, str(DATASET_BATCH_SCRIPT), *shard_arguments])
    print(f"Planned {len(tasks):,} fresh shards")
    print(f"Dataset finalizer: {finalizer_job_id}")


def _run_array_shard(split_paths: dict[str, str], shard_size: int) -> None:
    # Resolve this array index through the deterministic source-row plan
    task_id_text = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise ValueError("Shard workers require SLURM_ARRAY_TASK_ID")
    tasks = build_shard_plan(split_paths, shard_size)
    task_id = int(task_id_text)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Array task {task_id} is outside the {len(tasks):,}-shard plan")
    _run_shard(tasks[task_id], DatasetShardStore(Path(DATASET_DIR) / "shards"))


def main() -> None:
    """Generate paired raster NPZ files and metadata CSVs for all splits."""
    args = parse_args()
    stage = os.getenv("DATASET_GENERATION_STAGE", "generate")
    split_paths = SPLIT_PATHS if args.split == "all" else {args.split: SPLIT_PATHS[args.split]}
    if stage in {"worker", "finalize"} and args.shard_size is None:
        raise ValueError("Sharded dataset generation requires --shard-size")
    if stage == "worker":
        _run_array_shard(split_paths, int(args.shard_size))
    elif stage == "finalize":
        _initialize_output_directories()
        tasks = build_shard_plan(split_paths, int(args.shard_size))
        _finalize_shards(tasks, split_paths, DatasetShardStore(Path(DATASET_DIR) / "shards"))
    elif stage == "generate":
        if args.shard_size is not None:
            _launch_sharded_run(args, split_paths, args.shard_size)
        elif os.getenv("SLURM_JOB_ID") is not None:
            raise ValueError("Launch sharded dataset generation from a login node with --shard-size")
        else:
            _run_monolithic(args, split_paths)
    else:
        raise ValueError(f"Unsupported internal dataset-generation stage: {stage}")


if __name__ == "__main__":
    main()
