"""Build the AOI-disjoint masked-pretraining raster dataset on Savio."""

import argparse
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from dataset_generation_utils import (
    CANDIDATE_SCHEMA,
    FINAL_RECORD_SCHEMA,
    VALID_RECORD_SCHEMA,
    VALID_STATUS,
    MaskedDatasetShardStore,
    MaskedRecordTask,
    PretrainingShardTask,
    bounded_parallel_map,
    build_shard_tasks,
    materialize_masked_record,
    process_candidate_batch,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    add_projected_coordinates,
    add_sequence_weather_paths,
    build_aoi_spatial_frame,
    cluster_aois,
)
from preprocessing.tempo_mapping import read_aoi_mapping

from config import (
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WEATHER_CACHE_DIR,
    HRRR_DIR,
    MASKED_PRETRAINING_BASE_DIR,
    MASKED_PRETRAINING_DF_DIR,
    MASKED_PRETRAINING_MAX_PARALLEL_SHARDS,
    MASKED_PRETRAINING_SHARD_DIR,
    MASKED_PRETRAINING_SHARD_SIZE,
    MASKED_PRETRAINING_SPLIT_SEED,
    MASKED_PRETRAINING_TEST_RECORDS,
    MASKED_PRETRAINING_TRAIN_RECORDS,
    MASKED_PRETRAINING_VAL_RECORDS,
    MASKED_PRETRAINING_VALIDITY_CACHE_DIR,
    MASKED_PRETRAINING_WORK_DIR,
    MASKED_PRETRAINING_WORKERS_PER_SHARD,
    TEMPO_AOI_MAPPING,
    TEMPO_DIR,
    TRAIN_RECORDS_CSV,
)

SPLIT_TARGETS = {
    "train": MASKED_PRETRAINING_TRAIN_RECORDS,
    "val": MASKED_PRETRAINING_VAL_RECORDS,
    "test": MASKED_PRETRAINING_TEST_RECORDS,
}
STAGE_ENV = "MASKED_PRETRAINING_STAGE"
DEFAULT_BATCH_SIZE = 64
PROGRESS_INTERVAL = 1_000

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BATCH_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "generate_masked_pretraining_dataset.sh"


def parse_args() -> argparse.Namespace:
    """Parse launcher and worker options.

    Returns:
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-cache",
        "--refresh-cache",
        dest="clear_cache",
        action="store_true",
        help="clear persistent positive and negative validity-cache entries before this run",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=MASKED_PRETRAINING_SHARD_SIZE,
        help="maximum selected records stored in each disposable dataset shard",
    )
    parser.add_argument("--max-parallel-shards", type=int, default=MASKED_PRETRAINING_MAX_PARALLEL_SHARDS)
    parser.add_argument("--workers-per-shard", type=int, default=MASKED_PRETRAINING_WORKERS_PER_SHARD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def _safe_reset(path: Path, base: Path) -> None:
    resolved = path.resolve()
    resolved_base = base.resolve()
    if resolved == resolved_base or resolved_base not in resolved.parents:
        raise ValueError(f"Refusing to clear unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _load_aois() -> pl.DataFrame:
    # Keep pretraining within downstream-training geography
    # This avoids transductive access to validation and test geography
    aois = (
        pl.scan_csv(TRAIN_RECORDS_CSV)
        .select(AOI_ID_COL, "lat", "lon")
        .drop_nulls()
        .unique(subset=AOI_ID_COL, keep="first")
        .collect(engine="streaming")
    )
    return add_projected_coordinates(aois)


def _load_candidate_pool(aois: pl.DataFrame) -> pl.DataFrame:
    observations = read_aoi_mapping(
        TEMPO_AOI_MAPPING,
        columns=[AOI_ID_COL, "scan_date", "scan_num", "tempo_time", "granule_paths"],
    )
    candidates = (
        observations.join(aois.select(AOI_ID_COL, "lat", "lon"), on=AOI_ID_COL, how="inner")
        .unique(subset=[AOI_ID_COL, "scan_date", "scan_num"], keep="first")
        .with_columns(pl.col("tempo_time").alias("timestep_time_t0"))
    )
    return (
        add_sequence_weather_paths(candidates, timesteps=1)
        .rename({"weather_path_t0": "weather_path"})
        .drop("timestep_time_t0")
    )


def _assign_clusters(frame: pl.DataFrame) -> pl.DataFrame:
    target_total = sum(SPLIT_TARGETS.values())
    fractions = {split: count / target_total for split, count in SPLIT_TARGETS.items()}
    cluster_counts = (
        frame.group_by("cluster")
        .agg(pl.len().alias("records"))
        .with_columns(pl.col("cluster").hash(seed=MASKED_PRETRAINING_SPLIT_SEED).alias("tie_breaker"))
        .sort("records", "tie_breaker", descending=[True, False])
    )
    if cluster_counts.height < len(SPLIT_TARGETS):
        raise ValueError("At least three non-overlapping AOI groups are required")
    total_records = float(cluster_counts["records"].sum())
    targets = {split: total_records * fraction for split, fraction in fractions.items()}
    assigned = {split: 0.0 for split in SPLIT_TARGETS}
    assigned_groups = {split: 0 for split in SPLIT_TARGETS}
    rows: list[dict[str, object]] = []
    for index, group in enumerate(cluster_counts.iter_rows(named=True)):
        empty = [split for split, count in assigned_groups.items() if count == 0]
        remaining = cluster_counts.height - index
        choices = empty if remaining == len(empty) else list(SPLIT_TARGETS)
        destination = min(
            choices,
            key=lambda choice: sum(
                (assigned[split] + (float(group["records"]) if split == choice else 0.0) - targets[split]) ** 2
                for split in SPLIT_TARGETS
            ),
        )
        rows.append({"cluster": group["cluster"], "split": destination})
        assigned[destination] += float(group["records"])
        assigned_groups[destination] += 1
    assignments = pl.DataFrame(rows, schema={"cluster": frame.schema["cluster"], "split": pl.String})
    return frame.join(assignments, on="cluster", how="inner")


def _write_candidate_manifests() -> None:
    aois = _load_aois()
    candidate_pool = _load_candidate_pool(aois)
    candidate_aois = aois.join(candidate_pool.select(AOI_ID_COL).unique(), on=AOI_ID_COL, how="semi")
    clustered = candidate_pool.join(
        cluster_aois(candidate_aois, build_aoi_spatial_frame(candidate_aois)),
        on=AOI_ID_COL,
        how="inner",
    )
    assigned = _assign_clusters(clustered)
    for split in SPLIT_TARGETS:
        frame = (
            assigned.filter(pl.col("split") == split)
            .with_columns(
                pl.struct("scan_date", "scan_num").hash(seed=MASKED_PRETRAINING_SPLIT_SEED).alias("shard_key"),
                pl.struct("scan_date", "scan_num").hash(seed=MASKED_PRETRAINING_SPLIT_SEED + 1).alias("selection_key"),
            )
            .sort("selection_key", AOI_ID_COL, "tempo_time")
            .with_row_index("candidate_index")
            .select(list(CANDIDATE_SCHEMA))
        )
        manifest = Path(MASKED_PRETRAINING_WORK_DIR) / "candidates" / f"{split}.parquet"
        write_parquet_atomic(frame.cast(CANDIDATE_SCHEMA), manifest)
        print(f"[{split}] candidate pool: {frame.height:,} scenes across {frame[AOI_ID_COL].n_unique():,} AOIs")


def _submit(options: list[str], script_arguments: list[str]) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable", *options, str(BATCH_SCRIPT), *script_arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    job_id = result.stdout.strip().partition(";")[0]
    if not job_id:
        raise RuntimeError("sbatch returned an empty job ID")
    return job_id


def _launch(args: argparse.Namespace) -> None:
    if os.getenv("SLURM_JOB_ID") is not None:
        raise ValueError("Launch masked-pretraining generation from a login node")
    if not BATCH_SCRIPT.is_file():
        raise FileNotFoundError(BATCH_SCRIPT)
    active_jobs = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--user",
            str(os.environ["USER"]),
            "--name=masked-data-prepare,masked-data-discover,masked-data-shard,masked-data-finalize",
            "--format=%i",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.split()
    if active_jobs:
        raise RuntimeError(f"Masked-pretraining generation jobs are already active: {', '.join(active_jobs)}")
    (REPOSITORY_ROOT / "logs").mkdir(exist_ok=True)
    tasks = build_shard_tasks(SPLIT_TARGETS, args.shard_size)
    array_spec = f"0-{len(tasks) - 1}"
    script_arguments = [
        "--shard-size",
        str(args.shard_size),
        "--workers-per-shard",
        str(args.workers_per_shard),
        "--batch-size",
        str(args.batch_size),
    ]
    prepare_arguments = [*script_arguments]
    if args.clear_cache:
        prepare_arguments.append("--clear-cache")
    prepare_job = _submit(
        [
            f"--cpus-per-task={args.workers_per_shard}",
            "--time=04:00:00",
            "--job-name=masked-data-prepare",
            f"--export=ALL,{STAGE_ENV}=prepare",
        ],
        prepare_arguments,
    )
    discovery_job = _submit(
        [
            f"--array={array_spec}%{args.max_parallel_shards}",
            f"--cpus-per-task={args.workers_per_shard}",
            "--job-name=masked-data-discover",
            f"--dependency=afterok:{prepare_job}",
            f"--export=ALL,{STAGE_ENV}=discover",
        ],
        script_arguments,
    )
    materialize_job = _submit(
        [
            f"--array={array_spec}%{args.max_parallel_shards}",
            f"--cpus-per-task={args.workers_per_shard}",
            "--job-name=masked-data-shard",
            f"--dependency=afterok:{discovery_job}",
            f"--export=ALL,{STAGE_ENV}=materialize",
        ],
        script_arguments,
    )
    finalizer_job = _submit(
        [
            "--array=0",
            "--cpus-per-task=1",
            "--time=04:00:00",
            "--job-name=masked-data-finalize",
            f"--dependency=afterok:{materialize_job}",
            f"--export=ALL,{STAGE_ENV}=finalize",
        ],
        script_arguments,
    )
    print(f"Masked-pretraining preparation: {prepare_job}")
    print(f"Masked-pretraining validity discovery: {discovery_job}")
    print(f"Masked-pretraining shard materialization: {materialize_job}")
    print(f"Masked-pretraining finalizer: {finalizer_job}")


def _run_prepare(args: argparse.Namespace) -> None:
    base = Path(MASKED_PRETRAINING_BASE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    _safe_reset(Path(MASKED_PRETRAINING_WORK_DIR), base)
    _safe_reset(Path(MASKED_PRETRAINING_SHARD_DIR), base)
    _safe_reset(Path(MASKED_PRETRAINING_DF_DIR), base)
    if args.clear_cache:
        _safe_reset(Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR), base)
    else:
        Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    _write_candidate_manifests()


def _candidate_batches(candidates: pl.DataFrame, batch_size: int) -> Iterator[pl.DataFrame]:
    # Batch complete scans to reuse granule reads
    pending: list[pl.DataFrame] = []
    pending_rows = 0
    for group in candidates.partition_by(["scan_date", "scan_num"], maintain_order=True):
        if pending and pending_rows + group.height > batch_size:
            yield pl.concat(pending, how="vertical")
            pending = []
            pending_rows = 0
        pending.append(group)
        pending_rows += group.height
    if pending:
        yield pl.concat(pending, how="vertical")


def _resolve_array_task(shard_size: int) -> tuple[PretrainingShardTask, list[PretrainingShardTask]]:
    tasks = build_shard_tasks(SPLIT_TARGETS, shard_size)
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    return tasks[task_id], tasks


def _run_discovery(args: argparse.Namespace) -> None:
    task, tasks = _resolve_array_task(args.shard_size)
    split_shard_count = sum(candidate.split == task.split for candidate in tasks)
    manifest = Path(MASKED_PRETRAINING_WORK_DIR) / "candidates" / f"{task.split}.parquet"
    candidates = (
        pl.scan_parquet(manifest)
        .filter((pl.col("shard_key") % split_shard_count) == task.shard_index)
        .sort("selection_key", "scan_date", "scan_num", "aoi_id")
        .collect(engine="streaming")
    )
    result_path = (
        Path(MASKED_PRETRAINING_WORK_DIR) / "validity-results" / task.split / f"{task.shard_index:06d}.parquet"
    )
    valid_rows: list[dict[str, object]] = []
    write_parquet_atomic(pl.DataFrame(schema=VALID_RECORD_SCHEMA), result_path)
    invalid_count = 0
    retryable_count = 0
    completed = 0
    for batch_frame in _candidate_batches(candidates, args.batch_size):
        result_paths = (Path(MASKED_PRETRAINING_WORK_DIR) / "validity-results" / task.split).glob("*.parquet")
        valid_count = sum(
            pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item() for path in result_paths
        )
        if valid_count >= task.split_target:
            break
        results = process_candidate_batch(
            list(batch_frame.iter_rows(named=True)),
            validity_cache_dir=Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR),
            tempo_root=Path(TEMPO_DIR),
            tempo_cache_dir=Path(DATASET_TEMPO_CACHE_DIR),
            hrrr_root=Path(HRRR_DIR),
            weather_cache_dir=Path(DATASET_WEATHER_CACHE_DIR),
            workers=args.workers_per_shard,
        )
        valid_rows.extend(
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
        )
        invalid_count += sum(result.status != VALID_STATUS and result.status != "retryable" for result in results)
        retryable_count += sum(result.status == "retryable" for result in results)
        write_parquet_atomic(pl.DataFrame(valid_rows, schema=VALID_RECORD_SCHEMA), result_path)
        completed += batch_frame.height
        if completed % PROGRESS_INTERVAL == 0 or completed == candidates.height:
            print(
                f"[{task.split} shard {task.shard_index}] {completed:,}/{candidates.height:,} candidates; "
                f"{len(valid_rows):,} valid, {invalid_count:,} invalid, {retryable_count:,} retryable"
            )
    print(f"[{task.split} discovery {task.shard_index}] finished with {len(valid_rows):,} valid scenes")


def _selected_valid_records(split: str, target_count: int) -> pl.DataFrame:
    # Select deterministic cache-backed records from this run
    paths = sorted((Path(MASKED_PRETRAINING_WORK_DIR) / "validity-results" / split).glob("*.parquet"))
    frames = [pl.read_parquet(path).cast(VALID_RECORD_SCHEMA) for path in paths]
    return (
        pl.concat(frames, how="vertical")
        .unique(subset="cache_key", keep="first")
        .sort("candidate_index")
        .head(target_count)
        if frames
        else pl.DataFrame(schema=VALID_RECORD_SCHEMA)
    )


def _run_materialize(args: argparse.Namespace) -> None:
    task, _ = _resolve_array_task(args.shard_size)
    selected = _selected_valid_records(task.split, task.split_target).slice(task.start, task.size)
    store = MaskedDatasetShardStore(Path(MASKED_PRETRAINING_SHARD_DIR))
    shard_dir = store.create(task)
    raster_dir = shard_dir / "record-rasters" / task.split
    raster_dir.mkdir(parents=True)
    record_tasks = [
        MaskedRecordTask(
            row=row,
            output_path=str(raster_dir / f"{row['cache_key']}.npz"),
            mask_seed=MASKED_PRETRAINING_SPLIT_SEED + task.task_id * args.shard_size + offset,
        )
        for offset, row in enumerate(selected.iter_rows(named=True))
    ]
    output_rows = list(bounded_parallel_map(materialize_masked_record, record_tasks, args.workers_per_shard))
    store.write(task, shard_dir, output_rows)


def _run_finalizer(args: argparse.Namespace) -> None:
    tasks = build_shard_tasks(SPLIT_TARGETS, args.shard_size)
    store = MaskedDatasetShardStore(Path(MASKED_PRETRAINING_SHARD_DIR))
    output_by_split: dict[str, list[pl.DataFrame]] = {split: [] for split in SPLIT_TARGETS}
    for task in tasks:
        try:
            output_by_split[task.split].append(store.load(task, resolve_paths=True))
        except (OSError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(f"Cannot finalize incomplete shard {task.task_id}: {error}") from error

    summaries: list[dict[str, object]] = []
    base = Path(MASKED_PRETRAINING_BASE_DIR)
    for split, target_count in SPLIT_TARGETS.items():
        expected = _selected_valid_records(split, target_count)
        frames = output_by_split[split]
        output = (
            pl.concat(frames, how="vertical").sort("candidate_index")
            if frames
            else pl.DataFrame(schema=FINAL_RECORD_SCHEMA)
        )
        if output["cache_key"].to_list() != expected["cache_key"].to_list():
            raise ValueError(f"[{split}] shard records do not match the selected validity-cache records")
        relative_paths = [str(Path(path).relative_to(base)) for path in output["raster_bundle_path"].to_list()]
        output = output.with_columns(pl.Series("raster_bundle_path", relative_paths, dtype=pl.String))
        write_csv_atomic(output, Path(MASKED_PRETRAINING_DF_DIR) / f"{split}_df.csv")
        summaries.append(
            {
                "split": split,
                "target_records": target_count,
                "available_valid_records": expected.height,
                "published_records": output.height,
                "target_met": output.height == target_count,
            }
        )

    write_json_atomic(
        {"shard_size": args.shard_size, "splits": summaries},
        Path(MASKED_PRETRAINING_DF_DIR) / "generation_summary.json",
    )
    for summary in summaries:
        print(
            f"[{summary['split']}] published {summary['published_records']:,}/"
            f"{summary['target_records']:,} requested records"
        )


def main() -> None:
    """Launch, execute, or finalize masked-pretraining generation."""
    args = parse_args()
    stage = os.getenv(STAGE_ENV, "launch")
    if stage == "launch":
        _launch(args)
    elif stage == "prepare":
        _run_prepare(args)
    elif stage == "discover":
        _run_discovery(args)
    elif stage == "materialize":
        _run_materialize(args)
    elif stage == "finalize":
        _run_finalizer(args)
    else:
        raise ValueError(f"Unsupported {STAGE_ENV}: {stage}")


if __name__ == "__main__":
    main()
