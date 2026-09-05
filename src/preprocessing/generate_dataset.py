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
    TABULAR_FEATURE_NAMES,
    RecordTask,
    ScanTask,
    build_hrrr_grid_indices,
    eligible_generated_records,
    make_scan_task,
    process_record,
    process_scan,
    select_final_records,
    validate_coverage_config,
    write_csv_atomic,
)

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

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class PreparedRecord:
    """One source row resolved to its cached scan and output paths."""

    split: str
    record_index: int
    aoi_id: int
    lat: float
    lon: float
    current_scan_key: str
    previous_scan_key: str
    hrrr_path: str
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
    # Validate inputs before starting expensive worker processes
    splits: dict[str, pl.DataFrame] = {}
    for split, path in split_paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {split} records: {path}")
        splits[split] = (
            pl.scan_csv(path, try_parse_dates=True).with_row_index(SOURCE_RECORD_INDEX_COL).collect(engine="streaming")
        )
    return splits


def _prepare_records(
    splits: dict[str, pl.DataFrame],
    cache_dir: Path,
) -> tuple[list[PreparedRecord], dict[str, ScanTask], dict[str, list[dict[str, object]]]]:
    # Build one global scan plan across train, validation, and test
    records: list[PreparedRecord] = []
    scans: dict[str, ScanTask] = {}
    failures: dict[str, list[dict[str, object]]] = {split: [] for split in splits}
    for split, frame in splits.items():
        output_dir = cache_dir / "record-rasters" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in frame.iter_rows(named=True):
            record_index = int(row[SOURCE_RECORD_INDEX_COL])
            try:
                current = make_scan_task(row, "tempo", Path(TEMPO_DIR), cache_dir)
                previous = make_scan_task(row, "prev_tempo", Path(TEMPO_DIR), cache_dir)
                hrrr_path = Path(HRRR_DIR) / str(row["hrrr"])
                delta_no2_path = output_dir / f"{record_index:06d}.npz"
                records.append(
                    PreparedRecord(
                        split=split,
                        record_index=record_index,
                        aoi_id=int(row["aoi_id"]),
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        current_scan_key=current.cache_key,
                        previous_scan_key=previous.cache_key,
                        hrrr_path=str(hrrr_path),
                        delta_no2_path=str(delta_no2_path),
                    )
                )
                scans.setdefault(current.cache_key, current)
                scans.setdefault(previous.cache_key, previous)
            except (KeyError, TypeError, ValueError) as error:
                failures[split].append({"record_index": record_index, "error": str(error)})
    return records, scans, failures


def _run_scan_regridding(scans: dict[str, ScanTask], workers: int) -> tuple[dict[str, str], dict[str, str]]:
    # Cache every unique AOI scan once for this process run
    cache_paths: dict[str, str] = {}
    failures: dict[str, str] = {}
    total = len(scans)
    for completed, result in enumerate(_bounded_parallel_map(process_scan, scans.values(), workers), start=1):
        if result.error is None:
            cache_paths[result.cache_key] = result.cache_path
        else:
            failures[result.cache_key] = result.error
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            print(f"Regridded {completed:,}/{total:,} unique AOI scans")
    return cache_paths, failures


def _hrrr_grid_indices(records: list[PreparedRecord]) -> dict[int, int]:
    # HRRR uses one fixed CONUS grid across the archived analysis hours
    locations: dict[int, tuple[float, float]] = {}
    reference_path: str | None = None
    for record in records:
        locations.setdefault(record.aoi_id, (record.lat, record.lon))
        if reference_path is None and Path(record.hrrr_path).is_file():
            reference_path = record.hrrr_path
    if reference_path is None:
        raise FileNotFoundError("No HRRR file from the stratified records exists")
    return build_hrrr_grid_indices(reference_path, locations)


def _record_tasks(
    records: list[PreparedRecord],
    cache_paths: dict[str, str],
    scan_failures: dict[str, str],
    hrrr_indices: dict[int, int],
    failures: dict[str, list[dict[str, object]]],
) -> tuple[list[RecordTask], dict[tuple[str, int], PreparedRecord]]:
    # Exclude records whose current or previous scan failed to regrid
    tasks: list[RecordTask] = []
    records_by_id: dict[tuple[str, int], PreparedRecord] = {}
    for record in records:
        missing_keys = [key for key in (record.current_scan_key, record.previous_scan_key) if key not in cache_paths]
        if missing_keys:
            reasons = [scan_failures.get(key, "scan cache unavailable") for key in missing_keys]
            failures[record.split].append({"record_index": record.record_index, "error": "; ".join(reasons)})
            continue
        record_id = (record.split, record.record_index)
        records_by_id[record_id] = record
        tasks.append(
            RecordTask(
                split=record.split,
                record_index=record.record_index,
                current_cache_path=cache_paths[record.current_scan_key],
                previous_cache_path=cache_paths[record.previous_scan_key],
                hrrr_path=record.hrrr_path,
                hrrr_grid_index=hrrr_indices[record.aoi_id],
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
        candidates = (
            source_frame.join(features, on=SOURCE_RECORD_INDEX_COL, how="inner", maintain_order="left")
            .sort(SOURCE_RECORD_INDEX_COL)
        )
        eligible_count = eligible_generated_records(candidates).height
        output_frame = select_final_records(candidates, FINAL_SPLIT_SIZES[split])
        print(
            f"[{split}] {candidates.height:,} regridded; "
            f"{eligible_count:,} passed coverage QC; {output_frame.height:,} selected"
        )
        output_frame = _install_selected_rasters(split, output_frame)
        write_csv_atomic(
            output_frame.drop(SOURCE_RECORD_INDEX_COL, CANDIDATE_RASTER_PATH_COL),
            Path(DATASET_DF) / f"{split}_df.csv",
        )
        write_csv_atomic(
            pl.DataFrame(failure_rows, schema={"record_index": pl.Int64, "error": pl.String}),
            Path(DATASET_DF) / f"{split}_failures.csv",
        )
        print(f"[{split}] wrote {output_frame.height:,} records; {len(failure_rows):,} processing failures")


def _install_selected_rasters(split: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Atomically replace one split's raster directory with selected files."""
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
    """Parse dataset-generation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=NUM_CORES)
    parser.add_argument("--split", choices=("all", *SPLIT_PATHS), default=_default_split())
    return parser.parse_args()


def _default_split() -> str:
    # A three-task Slurm array maps directly to train, validation, and test
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return "all"
    try:
        return ARRAY_SPLITS[int(task_id)]
    except (IndexError, ValueError) as error:
        raise ValueError("SLURM_ARRAY_TASK_ID must be 0, 1, or 2 for dataset generation") from error


def _selected_split_paths(split: str) -> dict[str, str]:
    # A split-per-array-task layout preserves every useful cache hit
    return SPLIT_PATHS if split == "all" else {split: SPLIT_PATHS[split]}


def _worker_count(requested_workers: int) -> int:
    # NUM_CORES reflects SLURM_CPUS_PER_TASK inside a Savio allocation
    if requested_workers < 1:
        raise ValueError("workers must be at least one")
    workers = min(requested_workers, NUM_CORES)
    if workers < requested_workers:
        print(f"Capping workers at the allocated core count: {workers}")
    return workers


def main() -> None:
    """Generate paired raster NPZ files and metadata CSVs for all splits."""
    args = parse_args()
    workers = _worker_count(args.workers)
    validate_coverage_config()

    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATASET_DF).mkdir(parents=True, exist_ok=True)
    Path(DATASET_RASTER_DIR).mkdir(parents=True, exist_ok=True)
    splits = _load_splits(_selected_split_paths(args.split))
    with tempfile.TemporaryDirectory(prefix=".regrid-cache-", dir=DATASET_DIR) as temporary_dir:
        records, scans, failures = _prepare_records(splits, Path(temporary_dir))
        print(f"Planned {len(records):,} records using {len(scans):,} unique AOI scans")
        cache_paths, scan_failures = _run_scan_regridding(scans, workers)
        hrrr_indices = _hrrr_grid_indices(records)
        tasks, records_by_id = _record_tasks(records, cache_paths, scan_failures, hrrr_indices, failures)
        output_rows = _run_record_processing(tasks, records_by_id, failures, workers)
        _write_outputs(output_rows, failures, splits)


if __name__ == "__main__":
    main()
