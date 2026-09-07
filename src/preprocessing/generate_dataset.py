"""Generate paired TEMPO rasters and tabular features for every data split."""

import argparse
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import polars as pl

from config import (
    DATASET_DF,
    DATASET_DIR,
    DATASET_RASTER_DIR,
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WIND_CACHE_DIR,
    DEADBAND_THRESHOLD_COL,
    HRRR_DIR,
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
    NO_PAIRED_FINITE_NO2_ERROR,
    TABULAR_FEATURE_NAMES,
    RecordTask,
    ScanBatchTask,
    ScanTask,
    WindBatchTask,
    WindTask,
    cache_exists,
    eligible_generated_records,
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
SOURCE_RECORD_INDEX_COL = "_source_record_index"
DELTA_NO2_PATH_COL = "delta_no2_path"
CANDIDATE_RASTER_PATH_COL = "_candidate_raster_path"
FINAL_SPLIT_SIZES = {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE}
NO_PAIRED_FINITE_NO2_FAILURE = f"Record processing failed: {NO_PAIRED_FINITE_NO2_ERROR}"

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class PreparedRecord:
    """One source row resolved to its cache and output paths."""

    split: str
    record_index: int
    current_scan_key: str
    previous_scan_key: str
    wind_cache_key: str
    delta_no2_path: str


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
    for split, frame in splits.items():
        output_dir = run_dir / "record-rasters" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in frame.iter_rows(named=True):
            record_index = int(row[SOURCE_RECORD_INDEX_COL])
            try:
                current = make_scan_task(row, "tempo", Path(TEMPO_DIR), tempo_cache_dir)
                previous = make_scan_task(row, "prev_tempo", Path(TEMPO_DIR), tempo_cache_dir)
                wind = make_wind_task(row, Path(HRRR_DIR), wind_cache_dir)
                delta_no2_path = output_dir / f"{record_index:06d}.npz"
                records.append(
                    PreparedRecord(
                        split=split,
                        record_index=record_index,
                        current_scan_key=current.cache_key,
                        previous_scan_key=previous.cache_key,
                        wind_cache_key=wind.cache_key,
                        delta_no2_path=str(delta_no2_path),
                    )
                )
                scans.setdefault(current.cache_key, current)
                scans.setdefault(previous.cache_key, previous)
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
    feature_schema = {
        SOURCE_RECORD_INDEX_COL: pl.UInt32,
        CANDIDATE_RASTER_PATH_COL: pl.String,
        **{name: pl.Float64 for name in TABULAR_FEATURE_NAMES},
    }
    for split, source_frame in source_splits.items():
        rows = output_rows[split]
        failure_rows = sorted(failures[split], key=lambda row: int(row["record_index"]))
        features = pl.DataFrame(rows, schema=feature_schema)
        candidates = source_frame.join(features, on=SOURCE_RECORD_INDEX_COL, how="inner", maintain_order="left").sort(
            SOURCE_RECORD_INDEX_COL
        )
        eligible = eligible_generated_records(candidates)
        output_frame = select_final_records(candidates, FINAL_SPLIT_SIZES[split])
        print(
            f"[{split}] {candidates.height:,} regridded; "
            f"{eligible.height:,} passed raster QC; {output_frame.height:,} selected"
        )
        thresholds = candidates[DEADBAND_THRESHOLD_COL].unique()
        classification_report = {
            "split": split,
            "raw_delta_nox_threshold": float(thresholds.item()),
            "raster_qc": classification_summary(candidates, eligible),
            "final_balance": classification_summary(eligible, output_frame),
        }
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
            pl.DataFrame(failure_rows, schema={"record_index": pl.Int64, "error": pl.String}),
            Path(DATASET_DF) / f"{split}_failures.csv",
        )
        no_paired_coverage, processing_failures = _count_failure_outcomes(failure_rows)
        print(
            f"[{split}] wrote {output_frame.height:,} records; "
            f"{no_paired_coverage:,} rejected with no paired finite NO2; "
            f"{processing_failures:,} processing failures"
        )


def _count_failure_outcomes(failure_rows: list[dict[str, object]]) -> tuple[int, int]:
    # Separate expected coverage rejection from operational failures
    no_paired_coverage = sum(row.get("error") == NO_PAIRED_FINITE_NO2_FAILURE for row in failure_rows)
    return no_paired_coverage, len(failure_rows) - no_paired_coverage


def _install_selected_rasters(split: str, frame: pl.DataFrame) -> pl.DataFrame:
    # Atomically replace one split's raster directory with selected files
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
            os.replace(candidate_path, staging / filename)
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


def parse_args() -> argparse.Namespace:
    """Parse dataset-generation command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=NUM_CORES)
    parser.add_argument("--split", choices=("all", *SPLIT_PATHS), default=_default_split())
    parser.add_argument(
        "--refresh-tempo",
        action="store_true",
        help="rebuild cached TEMPO rasters for the selected split",
    )
    parser.add_argument(
        "--refresh-wind",
        action="store_true",
        help="rebuild cached wind rasters for the selected split",
    )
    return parser.parse_args()


def _default_split() -> str:
    # A three-task Slurm array maps directly to train, validation, and test
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return "all"
    return ARRAY_SPLITS[int(task_id)]


def _selected_split_paths(split: str) -> dict[str, str]:
    # A split-per-array-task layout preserves every useful cache hit
    return SPLIT_PATHS if split == "all" else {split: SPLIT_PATHS[split]}


def _worker_count(requested_workers: int) -> int:
    # NUM_CORES reflects SLURM_CPUS_PER_TASK inside a Savio allocation
    workers = min(requested_workers, NUM_CORES)
    if workers < requested_workers:
        print(f"Capping workers at the allocated core count: {workers}")
    return workers


def main() -> None:
    """Generate paired raster NPZ files and metadata CSVs for all splits."""
    args = parse_args()
    workers = _worker_count(args.workers)

    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATASET_DF).mkdir(parents=True, exist_ok=True)
    Path(DATASET_RASTER_DIR).mkdir(parents=True, exist_ok=True)
    tempo_cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    tempo_cache_dir.mkdir(parents=True, exist_ok=True)
    wind_cache_dir = Path(DATASET_WIND_CACHE_DIR)
    wind_cache_dir.mkdir(parents=True, exist_ok=True)
    splits = _load_splits(_selected_split_paths(args.split))
    with tempfile.TemporaryDirectory(prefix=".dataset-run-", dir=DATASET_DIR) as temporary_dir:
        records, scans, winds, failures = _prepare_records(
            splits,
            tempo_cache_dir,
            wind_cache_dir,
            Path(temporary_dir),
        )
        print(f"Planned {len(records):,} records using {len(scans):,} TEMPO scans and {len(winds):,} wind rasters")
        tempo_cache_paths, tempo_failures = _run_tempo_regridding(scans, workers, args.refresh_tempo)
        wind_cache_paths, wind_failures = _run_wind_alignment(winds, workers, args.refresh_wind)
        tasks, records_by_id = _record_tasks(
            records,
            tempo_cache_paths,
            tempo_failures,
            wind_cache_paths,
            wind_failures,
            failures,
        )
        output_rows = _run_record_processing(tasks, records_by_id, failures, workers)
        _write_outputs(output_rows, failures, splits)


if __name__ == "__main__":
    main()
