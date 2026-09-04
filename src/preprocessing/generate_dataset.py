"""Generate paired TEMPO rasters and tabular features for every data split."""

import argparse
import os
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
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
)
from preprocessing.generate_dataset_utils import (
    TABULAR_FEATURE_NAMES,
    RecordTask,
    ScanTask,
    build_hrrr_grid_indices,
    make_scan_task,
    process_record,
    process_scan,
    write_csv_atomic,
)

SPLIT_PATHS = {
    "train": TRAIN_RECORDS_CSV,
    "val": VAL_RECORDS_CSV,
    "test": TEST_RECORDS_CSV,
}
MAX_PENDING_FACTOR = 2
PROGRESS_INTERVAL = 1_000

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class PreparedRecord:
    """One source row resolved to its cached scan and output paths."""

    split: str
    record_index: int
    row: dict[str, object]
    current_scan_key: str
    previous_scan_key: str
    hrrr_path: str
    raster_path: str


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


def _load_splits() -> dict[str, pl.DataFrame]:
    # Validate inputs before starting expensive worker processes
    splits: dict[str, pl.DataFrame] = {}
    for split, path in SPLIT_PATHS.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {split} records: {path}")
        splits[split] = pl.read_csv(path, try_parse_dates=True)
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
        output_dir = Path(DATASET_RASTER_DIR) / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for record_index, row in enumerate(frame.iter_rows(named=True)):
            try:
                current = make_scan_task(row, "tempo", Path(TEMPO_DIR), cache_dir)
                previous = make_scan_task(row, "prev_tempo", Path(TEMPO_DIR), cache_dir)
                hrrr_path = Path(HRRR_DIR) / str(row["hrrr"])
                raster_path = output_dir / f"{record_index:06d}.npz"
                records.append(
                    PreparedRecord(
                        split=split,
                        record_index=record_index,
                        row=row,
                        current_scan_key=current.cache_key,
                        previous_scan_key=previous.cache_key,
                        hrrr_path=str(hrrr_path),
                        raster_path=str(raster_path),
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
        aoi_id = int(record.row["aoi_id"])
        locations.setdefault(aoi_id, (float(record.row["lat"]), float(record.row["lon"])))
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
                hrrr_grid_index=hrrr_indices[int(record.row["aoi_id"])],
                output_path=record.raster_path,
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
    output_rows: dict[str, list[dict[str, object]]] = {split: [] for split in SPLIT_PATHS}
    total = len(tasks)
    for completed, result in enumerate(_bounded_parallel_map(process_record, tasks, workers), start=1):
        record_id = (result.split, result.record_index)
        if result.error is not None:
            failures[result.split].append({"record_index": result.record_index, "error": result.error})
        else:
            record = records_by_id[record_id]
            output_row = dict(record.row)
            output_row.update(result.features)
            output_row["raster_path"] = os.path.relpath(record.raster_path, DATASET_DIR)
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
    for split in SPLIT_PATHS:
        rows = sorted(output_rows[split], key=lambda row: str(row["raster_path"]))
        failure_rows = sorted(failures[split], key=lambda row: int(row["record_index"]))
        if rows:
            output_frame = pl.DataFrame(rows)
        else:
            output_frame = (
                source_splits[split]
                .head(0)
                .with_columns(
                    *(pl.lit(None, dtype=pl.Float64).alias(name) for name in TABULAR_FEATURE_NAMES),
                    pl.lit(None, dtype=pl.String).alias("raster_path"),
                )
            )
        write_csv_atomic(output_frame, Path(DATASET_DF) / f"{split}_df.csv")
        write_csv_atomic(
            pl.DataFrame(failure_rows, schema={"record_index": pl.Int64, "error": pl.String}),
            Path(DATASET_DF) / f"{split}_failures.csv",
        )
        print(f"[{split}] wrote {len(rows):,} records; {len(failure_rows):,} failed")


def parse_args() -> argparse.Namespace:
    """Parse dataset-generation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=NUM_CORES)
    return parser.parse_args()


def main() -> None:
    """Generate paired raster NPZ files and metadata CSVs for all splits."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least one")

    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATASET_DF).mkdir(parents=True, exist_ok=True)
    Path(DATASET_RASTER_DIR).mkdir(parents=True, exist_ok=True)
    splits = _load_splits()
    with tempfile.TemporaryDirectory(prefix=".regrid-cache-", dir=DATASET_DIR) as temporary_dir:
        records, scans, failures = _prepare_records(splits, Path(temporary_dir))
        print(f"Planned {len(records):,} records using {len(scans):,} unique AOI scans")
        cache_paths, scan_failures = _run_scan_regridding(scans, args.workers)
        hrrr_indices = _hrrr_grid_indices(records)
        tasks, records_by_id = _record_tasks(records, cache_paths, scan_failures, hrrr_indices, failures)
        output_rows = _run_record_processing(tasks, records_by_id, failures, args.workers)
        _write_outputs(output_rows, failures, splits)


if __name__ == "__main__":
    main()
