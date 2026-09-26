"""Generate temporal TEMPO and HRRR raster bundles for every data split."""

import argparse
import os
import resource
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
from preprocessing.generate_dataset_utils import (
    CANDIDATE_FEATURE_SCHEMA,
    CANDIDATE_RASTER_PATH_COL,
    PROCESSING_FAILURE_SCHEMA,
    RASTER_BUNDLE_PATH_COL,
    SOURCE_RECORD_INDEX_COL,
    BackgroundFileWriter,
    DatasetShardStore,
    RecordTask,
    ScanTask,
    ShardTask,
    WeatherTask,
    bounded_parallel_map,
    build_shard_plan,
    coverage_selection_summary,
    make_scan_task,
    make_weather_task,
    process_record,
    process_scan_batch,
    process_weather_batch,
    scan_batches,
    stage_files,
    weather_batches,
    write_csv_atomic,
    write_json_atomic,
)

from config import (
    DATASET_DF,
    DATASET_DIR,
    DATASET_MAX_PARALLEL_SHARDS,
    DATASET_RASTER_DIR,
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WEATHER_CACHE_DIR,
    DATASET_WORKERS_PER_SHARD,
    HRRR_DIR,
    MIN_TIMESTEP_NO2_FINITE_FRACTION,
    NUM_CORES,
    SEQUENCE_TIMESTEPS,
    TEMPO_DIR,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
)

SPLIT_PATHS = {
    "train": TRAIN_RECORDS_CSV,
    "val": VAL_RECORDS_CSV,
    "test": TEST_RECORDS_CSV,
}
REQUIRED_SOURCE_COLUMNS = frozenset(
    {
        "aoi_id",
        "lat",
        "lon",
        *(f"no2_paths_t{index}" for index in range(SEQUENCE_TIMESTEPS)),
        *(f"weather_path_t{index}" for index in range(SEQUENCE_TIMESTEPS)),
    }
)
ARRAY_SPLITS = tuple(SPLIT_PATHS)
PROGRESS_INTERVAL = 1_000
LEGACY_DATASET_JOB_NAME = "generate-dataset"
SHARD_WORKER_JOB_NAME = "generate-dataset-shard"
SHARD_FINALIZER_JOB_NAME = "generate-dataset-finalize"
TRAINING_JOB_NAME = "train-no2"
RUN_STARTED_ENV = "DATASET_RUN_STARTED_AT"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_BATCH_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "generate_dataset.sh"
SHARD_INPUT_DIRECTORY = Path(DATASET_DIR) / "shard-inputs"


@dataclass(frozen=True)
class PreparedRecord:
    """One source row resolved to its cache and output paths."""

    split: str
    record_index: int
    scan_keys: tuple[str, ...]
    weather_cache_keys: tuple[str, ...]
    raster_bundle_path: str


@dataclass(frozen=True)
class StagedBatch:
    """Batch tasks and cache hits resolved to node-local paths."""

    scan_misses: dict[str, ScanTask]
    weather_misses: dict[str, WeatherTask]
    tempo_cache_paths: dict[str, str]
    weather_cache_paths: dict[str, str]


def _positive_int(value: str) -> int:
    # Parse a strictly positive command-line integer
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _scan_split(path: str) -> pl.LazyFrame:
    # Load split rows lazily for bounded orchestration memory
    frame = pl.scan_csv(path, try_parse_dates=True)
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS.difference(frame.collect_schema().names()))
    if missing_columns:
        raise ValueError(f"Stratified split {path} is missing dataset-generation columns: {', '.join(missing_columns)}")
    return frame


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
        splits[split] = _scan_split(path).with_row_index(SOURCE_RECORD_INDEX_COL).collect(engine="streaming")
    return splits


def _load_shard(task: ShardTask) -> dict[str, pl.DataFrame]:
    # Load the locality-ordered manifest prepared by the launcher
    frame = pl.read_parquet(_shard_input_path(task))
    if frame.height != task.size:
        raise ValueError(f"Shard {task.task_id} expected {task.size:,} source records but loaded {frame.height:,}")
    return {task.split: frame}


def _locality_order(frame: pl.DataFrame) -> pl.DataFrame:
    # Group nearby AOIs within each target hour while preserving source identity
    return (
        frame.with_columns(
            pl.col("lat").floor().alias("_latitude_tile"),
            pl.col("lon").floor().alias("_longitude_tile"),
        )
        .sort(
            "emissions_hour_utc",
            "_latitude_tile",
            "_longitude_tile",
            "aoi_id",
            SOURCE_RECORD_INDEX_COL,
        )
        .drop("_latitude_tile", "_longitude_tile")
    )


def _shard_input_path(task: ShardTask) -> Path:
    # Keep ordered source rows separate from generated shard outputs
    return SHARD_INPUT_DIRECTORY / task.split / f"{task.shard_index:06d}.parquet"


def _write_shard_inputs(split_paths: dict[str, str], tasks: list[ShardTask]) -> None:
    # Sort each split once and persist the exact rows assigned to each worker
    tasks_by_split: dict[str, list[ShardTask]] = {split: [] for split in split_paths}
    for task in tasks:
        tasks_by_split[task.split].append(task)
    for split, path in split_paths.items():
        frame = _locality_order(_scan_split(path).with_row_index(SOURCE_RECORD_INDEX_COL).collect(engine="streaming"))
        split_dir = SHARD_INPUT_DIRECTORY / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks_by_split[split]:
            frame.slice(task.start, task.size).write_parquet(_shard_input_path(task))


def _prepare_records(
    splits: dict[str, pl.DataFrame],
    tempo_cache_dir: Path,
    weather_cache_dir: Path,
    run_dir: Path,
) -> tuple[list[PreparedRecord], dict[str, ScanTask], dict[str, WeatherTask], dict[str, list[dict[str, object]]]]:
    # Build one global scan plan across train, validation, and test
    records: list[PreparedRecord] = []
    scans: dict[str, ScanTask] = {}
    weather: dict[str, WeatherTask] = {}
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
                record_scans = tuple(
                    make_scan_task(row, f"no2_paths_t{index}", Path(TEMPO_DIR), tempo_cache_dir)
                    for index in range(SEQUENCE_TIMESTEPS)
                )
                record_weather = tuple(
                    make_weather_task(
                        row,
                        f"weather_path_t{index}",
                        Path(HRRR_DIR),
                        weather_cache_dir,
                    )
                    for index in range(SEQUENCE_TIMESTEPS)
                )
                raster_bundle_path = output_dir / f"{record_index:06d}.npz"
                records.append(
                    PreparedRecord(
                        split=split,
                        record_index=record_index,
                        scan_keys=tuple(scan.cache_key for scan in record_scans),
                        weather_cache_keys=tuple(item.cache_key for item in record_weather),
                        raster_bundle_path=str(raster_bundle_path),
                    )
                )
                for scan in record_scans:
                    scans.setdefault(scan.cache_key, scan)
                for item in record_weather:
                    weather.setdefault(item.cache_key, item)
            except (KeyError, TypeError, ValueError) as error:
                failures[split].append({"record_index": record_index, "error": str(error)})
            if prepared_count % PROGRESS_INTERVAL == 0 or prepared_count == source_count:
                print(f"Prepared cache plan for {prepared_count:,}/{source_count:,} source records")
    return records, scans, weather, failures


def _cached_task_paths(tasks: dict[str, ScanTask] | dict[str, WeatherTask]) -> dict[str, str]:
    # Check only cache entries referenced by the current batch
    return {key: task.cache_path for key, task in tasks.items() if Path(task.cache_path).is_file()}


def _localize_batch(
    scans: dict[str, ScanTask],
    weather: dict[str, WeatherTask],
    batch_dir: Path,
) -> StagedBatch:
    # Resolve hits directly and stage every input needed by the remaining tasks
    tempo_hits = _cached_task_paths(scans)
    weather_hits = _cached_task_paths(weather)
    missing_scans = {key: task for key, task in scans.items() if key not in tempo_hits}
    missing_weather = {key: task for key, task in weather.items() if key not in weather_hits}
    source_paths = {path for task in missing_scans.values() for path in task.granule_paths} | {
        path for task in missing_weather.values() for path in (task.wind_hrrr_path, task.temperature_hrrr_path)
    }
    staged_sources = stage_files(source_paths, batch_dir / "sources")
    staged_tempo_hits = stage_files(set(tempo_hits.values()), batch_dir / "tempo-cache")
    staged_weather_hits = stage_files(set(weather_hits.values()), batch_dir / "weather-cache")
    local_scans = {
        key: replace(
            task,
            granule_paths=tuple(staged_sources[path] for path in task.granule_paths),
            cache_path=str(batch_dir / "generated-tempo-cache" / Path(task.cache_path).name),
        )
        for key, task in missing_scans.items()
    }
    local_weather = {
        key: replace(
            task,
            wind_hrrr_path=staged_sources[task.wind_hrrr_path],
            temperature_hrrr_path=staged_sources[task.temperature_hrrr_path],
            cache_path=str(batch_dir / "generated-weather-cache" / Path(task.cache_path).name),
        )
        for key, task in missing_weather.items()
    }
    return StagedBatch(
        scan_misses=local_scans,
        weather_misses=local_weather,
        tempo_cache_paths={key: staged_tempo_hits[path] for key, path in tempo_hits.items()},
        weather_cache_paths={key: staged_weather_hits[path] for key, path in weather_hits.items()},
    )


def _publish_generated_caches(
    local_paths: dict[str, str],
    persistent_tasks: dict[str, ScanTask] | dict[str, WeatherTask],
    writer: BackgroundFileWriter,
) -> None:
    # Publish successful cache misses through the batch writer
    for key, local_path in local_paths.items():
        writer.publish(local_path, persistent_tasks[key].cache_path)


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


def _run_weather_alignment(
    weather: dict[str, WeatherTask],
    workers: int,
    refresh_weather: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    # Reuse cached AOI-hour weather rasters unless refresh is explicit
    cache_paths = {} if refresh_weather else _cached_task_paths(weather)
    failures: dict[str, str] = {}
    missing = [task for key, task in weather.items() if key not in cache_paths]
    print(f"Weather cache: {len(cache_paths):,} hits; {len(missing):,} AOI-hours to align")
    if not missing:
        return cache_paths, failures
    completed = 0
    batches = weather_batches(missing, reuse_existing=not refresh_weather)
    for batch_results in bounded_parallel_map(process_weather_batch, batches, workers):
        for result in batch_results:
            completed += 1
            if result.error is None:
                cache_paths[result.cache_key] = result.cache_path
            else:
                failures[result.cache_key] = result.error
        if completed % PROGRESS_INTERVAL < len(batch_results) or completed == len(missing):
            print(f"Aligned {completed:,}/{len(missing):,} cache-missing AOI-hour weather rasters")
    return cache_paths, failures


def _record_tasks(
    records: list[PreparedRecord],
    tempo_cache_paths: dict[str, str],
    tempo_failures: dict[str, str],
    weather_cache_paths: dict[str, str],
    weather_failures: dict[str, str],
    failures: dict[str, list[dict[str, object]]],
    output_dir: Path | None = None,
) -> tuple[list[RecordTask], dict[tuple[str, int], PreparedRecord]]:
    # Exclude records when any timestep cache is unavailable
    tasks: list[RecordTask] = []
    records_by_id: dict[tuple[str, int], PreparedRecord] = {}
    for record in records:
        missing_keys = [key for key in record.scan_keys if key not in tempo_cache_paths]
        if missing_keys:
            reasons = [tempo_failures.get(key, "TEMPO cache unavailable") for key in missing_keys]
            failures[record.split].append({"record_index": record.record_index, "error": "; ".join(reasons)})
            continue
        missing_weather_keys = [key for key in record.weather_cache_keys if key not in weather_cache_paths]
        if missing_weather_keys:
            reasons = [weather_failures.get(key, "weather cache unavailable") for key in missing_weather_keys]
            failures[record.split].append({"record_index": record.record_index, "error": "; ".join(reasons)})
            continue
        record_id = (record.split, record.record_index)
        records_by_id[record_id] = record
        output_path = record.raster_bundle_path
        if output_dir is not None:
            output_path = str(output_dir / record.split / f"{record.record_index:06d}.npz")
        tasks.append(
            RecordTask(
                split=record.split,
                record_index=record.record_index,
                scan_cache_paths=tuple(tempo_cache_paths[key] for key in record.scan_keys),
                weather_cache_paths=tuple(weather_cache_paths[key] for key in record.weather_cache_keys),
                output_path=output_path,
            )
        )
    return tasks, records_by_id


def _run_record_processing(
    tasks: list[RecordTask],
    records_by_id: dict[tuple[str, int], PreparedRecord],
    failures: dict[str, list[dict[str, object]]],
    workers: int,
    on_success: Callable[[RecordTask, PreparedRecord], None] | None = None,
) -> dict[str, list[dict[str, object]]]:
    # Build temporal raster bundles and scalar diagnostics in parallel
    output_rows: dict[str, list[dict[str, object]]] = {split: [] for split in failures}
    tasks_by_id = {(task.split, task.record_index): task for task in tasks}
    total = len(tasks)
    for completed, result in enumerate(bounded_parallel_map(process_record, tasks, workers), start=1):
        record_id = (result.split, result.record_index)
        if result.error is not None:
            failures[result.split].append({"record_index": result.record_index, "error": result.error})
        else:
            record = records_by_id[record_id]
            if on_success is not None:
                on_success(tasks_by_id[record_id], record)
            output_row: dict[str, object] = {
                SOURCE_RECORD_INDEX_COL: result.record_index,
                CANDIDATE_RASTER_PATH_COL: record.raster_bundle_path,
            }
            output_row.update(result.features)
            output_rows[result.split].append(output_row)
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            print(f"Processed {completed:,}/{total:,} temporal records")
    return output_rows


def _process_shard_batch(
    task: ShardTask,
    frame: pl.DataFrame,
    shard_dir: Path,
    tempo_cache_dir: Path,
    weather_cache_dir: Path,
    workers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    # Stage and publish one locality batch before releasing its local files
    splits = {task.split: frame}
    records, scans, weather, failures = _prepare_records(
        splits,
        tempo_cache_dir,
        weather_cache_dir,
        shard_dir,
    )
    with tempfile.TemporaryDirectory(prefix=f"delta-dataset-{task.task_id}-", dir="/tmp") as temporary:
        batch_dir = Path(temporary)
        local = _localize_batch(scans, weather, batch_dir)
        with BackgroundFileWriter() as writer:
            generated_tempo, tempo_failures = _run_tempo_regridding(local.scan_misses, workers, True)
            _publish_generated_caches(generated_tempo, scans, writer)
            tempo_paths = {**local.tempo_cache_paths, **generated_tempo}

            generated_weather, weather_failures = _run_weather_alignment(local.weather_misses, workers, True)
            _publish_generated_caches(generated_weather, weather, writer)
            weather_paths = {**local.weather_cache_paths, **generated_weather}

            record_tasks, records_by_id = _record_tasks(
                records,
                tempo_paths,
                tempo_failures,
                weather_paths,
                weather_failures,
                failures,
                batch_dir / "record-rasters",
            )

            def publish_record(local_task: RecordTask, record: PreparedRecord) -> None:
                writer.publish(local_task.output_path, record.raster_bundle_path)

            output_rows = _run_record_processing(
                record_tasks,
                records_by_id,
                failures,
                workers,
                publish_record,
            )
    return output_rows[task.split], failures[task.split]


def _write_outputs(
    output_rows: dict[str, list[dict[str, object]]],
    failures: dict[str, list[dict[str, object]]],
    source_splits: dict[str, pl.DataFrame],
) -> None:
    # Sort asynchronous results back into source-record order
    candidates_by_split: dict[str, pl.DataFrame] = {}
    failure_frames: dict[str, pl.DataFrame] = {}
    for split, source_frame in source_splits.items():
        features = pl.DataFrame(output_rows[split], schema=CANDIDATE_FEATURE_SCHEMA)
        candidates = source_frame.join(
            features,
            on=SOURCE_RECORD_INDEX_COL,
            how="inner",
            maintain_order="left",
        ).sort(SOURCE_RECORD_INDEX_COL)
        candidates_by_split[split] = candidates
        failure_frames[split] = pl.DataFrame(
            sorted(failures[split], key=lambda row: int(row["record_index"])),
            schema=PROCESSING_FAILURE_SCHEMA,
        )

    prepared_outputs: dict[str, tuple[pl.DataFrame, dict[str, object], pl.DataFrame]] = {}
    for split, candidates in candidates_by_split.items():
        coverage_summary = coverage_selection_summary(candidates)
        print(f"[{split}] {candidates.height:,} records passed raster generation")
        print(
            f"[{split}] full sequence coverage: {coverage_summary['full_coverage_records']:,}/"
            f"{coverage_summary['records']:,} records across {coverage_summary['aoi_count']:,} AOIs"
        )
        generation_report = {
            "split": split,
            "raster_contract": {
                "sequence_timesteps": SEQUENCE_TIMESTEPS,
                "minimum_no2_finite_fraction_per_timestep": MIN_TIMESTEP_NO2_FINITE_FRACTION,
            },
            "source_records": source_splits[split].height,
            "generated_records": candidates.height,
            "processing_failures": failure_frames[split].height,
            "coverage": coverage_summary,
        }
        prepared_outputs[split] = (
            candidates,
            generation_report,
            failure_frames[split],
        )

    for split, (output_frame, generation_report, failure_frame) in prepared_outputs.items():
        relative_paths = [
            str(Path(candidate_path).relative_to(DATASET_DIR))
            for candidate_path in output_frame[CANDIDATE_RASTER_PATH_COL].to_list()
        ]
        output_frame = output_frame.drop(
            "_source_east_km",
            "_source_north_km",
            "_source_unit_count",
            strict=False,
        ).with_columns(pl.Series(RASTER_BUNDLE_PATH_COL, relative_paths, dtype=pl.String))
        write_csv_atomic(
            output_frame.drop(SOURCE_RECORD_INDEX_COL, CANDIDATE_RASTER_PATH_COL),
            Path(DATASET_DF) / f"{split}_df.csv",
        )
        write_json_atomic(
            generation_report,
            Path(DATASET_DF) / f"{split}_generation_summary.json",
        )
        write_csv_atomic(
            failure_frame,
            Path(DATASET_DF) / f"{split}_failures.csv",
        )
        print(f"[{split}] wrote {output_frame.height:,} records; {failure_frame.height:,} processing failures")


def _run_shard(task: ShardTask, store: DatasetShardStore, batch_size: int, workers: int) -> None:
    # Generate locality batches into the shard's existing output layout
    started_at = time.perf_counter()
    shard_dir = store.create(task)
    tempo_cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    weather_cache_dir = Path(DATASET_WEATHER_CACHE_DIR)
    tempo_cache_dir.mkdir(parents=True, exist_ok=True)
    weather_cache_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_shard(task)[task.split]
    output_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for offset in range(0, frame.height, batch_size):
        batch_rows, batch_failures = _process_shard_batch(
            task,
            frame.slice(offset, batch_size),
            shard_dir,
            tempo_cache_dir,
            weather_cache_dir,
            workers,
        )
        output_rows.extend(batch_rows)
        failures.extend(batch_failures)
    store.write(task, shard_dir, output_rows, failures)
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
    # Combine validated shard outputs into the published split datasets
    started_at = time.perf_counter()
    output_rows: dict[str, list[dict[str, object]]] = {split: [] for split in split_paths}
    failures: dict[str, list[dict[str, object]]] = {split: [] for split in split_paths}
    for task in tasks:
        expected_indices = pl.read_parquet(
            _shard_input_path(task),
            columns=[SOURCE_RECORD_INDEX_COL],
        )[SOURCE_RECORD_INDEX_COL].to_list()
        try:
            candidates, failure_frame = store.load(
                task,
                resolve_paths=True,
                expected_indices=expected_indices,
            )
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
        "--max-parallel-shards",
        type=_positive_int,
        default=DATASET_MAX_PARALLEL_SHARDS,
        help="maximum Slurm shard tasks allowed to run concurrently",
    )
    parser.add_argument(
        "--workers-per-shard",
        type=_positive_int,
        default=DATASET_WORKERS_PER_SHARD,
        help="worker processes and CPUs allocated to each Slurm shard task",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="locality-ordered records staged together on each shard worker",
    )
    parser.add_argument(
        "--afterok-job-id",
        type=_positive_int,
        help="hold the shard array until this Slurm job completes successfully",
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
        "--refresh-weather",
        "--refresh-wind",
        dest="refresh_weather",
        action="store_true",
        help="empty the weather raster cache before rebuilding entries for the selected split",
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


def _reset_requested_caches(args: argparse.Namespace, tempo_cache_dir: Path, weather_cache_dir: Path) -> None:
    # Clear shared caches only from a single non-array process
    refresh_tempo = args.refresh_cache or args.refresh_tempo
    refresh_weather = args.refresh_cache or args.refresh_weather
    if not refresh_tempo and not refresh_weather:
        return
    if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
        raise ValueError("Cache refresh cannot run inside a Slurm array because its tasks share cache directories")
    if refresh_tempo:
        _reset_cache_directory(tempo_cache_dir)
    if refresh_weather:
        _reset_cache_directory(weather_cache_dir)


def _initialize_output_directories() -> tuple[Path, Path]:
    # Create persistent output and cache directories
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATASET_DF).mkdir(parents=True, exist_ok=True)
    Path(DATASET_RASTER_DIR).mkdir(parents=True, exist_ok=True)
    tempo_cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    weather_cache_dir = Path(DATASET_WEATHER_CACHE_DIR)
    tempo_cache_dir.mkdir(parents=True, exist_ok=True)
    weather_cache_dir.mkdir(parents=True, exist_ok=True)
    return tempo_cache_dir, weather_cache_dir


def _reset_generated_outputs() -> None:
    # Remove disposable rasters and published metadata while preserving caches
    DatasetShardStore(Path(DATASET_DIR) / "shards").clear()
    for output_dir in (Path(DATASET_DF), Path(DATASET_RASTER_DIR), SHARD_INPUT_DIRECTORY):
        if output_dir.exists():
            shutil.rmtree(output_dir)


def _run_monolithic(args: argparse.Namespace, split_paths: dict[str, str]) -> None:
    # Preserve direct single-process generation with persistent raster paths
    _refuse_active_dataset_jobs()
    _reset_generated_outputs()
    tempo_cache_dir, weather_cache_dir = _initialize_output_directories()
    _reset_requested_caches(args, tempo_cache_dir, weather_cache_dir)
    splits = _load_splits(split_paths)
    refresh_tempo = args.refresh_cache or args.refresh_tempo
    refresh_weather = args.refresh_cache or args.refresh_weather
    records, scans, weather, failures = _prepare_records(
        splits,
        tempo_cache_dir,
        weather_cache_dir,
        Path(DATASET_RASTER_DIR),
    )
    print(f"Planned {len(records):,} records using {len(scans):,} TEMPO scans and {len(weather):,} weather rasters")
    tempo_cache_paths, tempo_failures = _run_tempo_regridding(scans, NUM_CORES, refresh_tempo)
    weather_cache_paths, weather_failures = _run_weather_alignment(
        weather,
        NUM_CORES,
        refresh_weather,
    )
    tasks, records_by_id = _record_tasks(
        records,
        tempo_cache_paths,
        tempo_failures,
        weather_cache_paths,
        weather_failures,
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
    tempo_cache_dir, weather_cache_dir = _initialize_output_directories()
    _reset_requested_caches(args, tempo_cache_dir, weather_cache_dir)
    tasks = build_shard_plan(split_paths, shard_size)
    _write_shard_inputs(split_paths, tasks)
    task_ids = [task.task_id for task in tasks]
    array_spec = _slurm_array_spec(task_ids)
    run_started_at = str(time.time())
    shard_arguments = ["--shard-size", str(shard_size), "--batch-size", str(args.batch_size)]
    if args.split != "all":
        shard_arguments.extend(("--split", args.split))
    external_dependency = [f"--dependency=afterok:{args.afterok_job_id}"] if args.afterok_job_id is not None else []
    worker_job_id: str | None = None
    if task_ids:
        worker_job_id = _submit_job(
            [
                *external_dependency,
                f"--array={array_spec}%{args.max_parallel_shards}",
                f"--cpus-per-task={args.workers_per_shard}",
                f"--job-name={SHARD_WORKER_JOB_NAME}",
                f"--export=ALL,DATASET_GENERATION_STAGE=worker,{RUN_STARTED_ENV}={run_started_at}",
                str(DATASET_BATCH_SCRIPT),
                *shard_arguments,
            ]
        )
        print(
            f"Dataset shard array: {worker_job_id}; at most {args.max_parallel_shards} concurrent shards "
            f"with {args.workers_per_shard} workers each"
        )
    finalizer_options = [
        "--array=0",
        "--cpus-per-task=1",
        "--time=01:00:00",
        f"--job-name={SHARD_FINALIZER_JOB_NAME}",
        f"--export=ALL,DATASET_GENERATION_STAGE=finalize,{RUN_STARTED_ENV}={run_started_at}",
    ]
    if worker_job_id is not None:
        finalizer_options.append(f"--dependency=afterok:{worker_job_id}")
    else:
        finalizer_options.extend(external_dependency)
    finalizer_job_id = _submit_job([*finalizer_options, str(DATASET_BATCH_SCRIPT), *shard_arguments])
    print(f"Planned {len(tasks):,} fresh shards")
    print(f"Dataset finalizer: {finalizer_job_id}")


def _run_array_shard(split_paths: dict[str, str], shard_size: int, batch_size: int) -> None:
    # Resolve this array index through the deterministic source-row plan
    task_id_text = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise ValueError("Shard workers require SLURM_ARRAY_TASK_ID")
    tasks = build_shard_plan(split_paths, shard_size)
    task_id = int(task_id_text)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Array task {task_id} is outside the {len(tasks):,}-shard plan")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", DATASET_WORKERS_PER_SHARD))
    _run_shard(tasks[task_id], DatasetShardStore(Path(DATASET_DIR) / "shards"), batch_size, workers)


def main() -> None:
    """Generate temporal raster NPZ files and metadata CSVs for all splits."""
    args = parse_args()
    stage = os.getenv("DATASET_GENERATION_STAGE", "generate")
    split_paths = SPLIT_PATHS if args.split == "all" else {args.split: SPLIT_PATHS[args.split]}
    if stage in {"worker", "finalize"} and args.shard_size is None:
        raise ValueError("Sharded dataset generation requires --shard-size")
    if stage == "worker":
        _run_array_shard(split_paths, int(args.shard_size), args.batch_size)
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
