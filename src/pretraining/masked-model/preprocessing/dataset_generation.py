"""Build the AOI-disjoint masked-pretraining raster dataset on Savio."""

import argparse
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
from dataset_generation_utils import (
    CANDIDATE_SCHEMA,
    FINAL_RECORD_SCHEMA,
    VALID_RECORD_SCHEMA,
    VALID_STATUS,
    PretrainingShardTask,
    build_shard_tasks,
    load_shard_candidates,
    mask_no2_raster,
    process_candidate_batch,
    valid_record_frame,
    write_json_atomic,
    write_npz_atomic,
    write_parquet_atomic,
)
from preprocessing.generate_dataset_utils import NO2_RASTER_NAME
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
    MASKED_PRETRAINING_MASKED_RASTER_DIR,
    MASKED_PRETRAINING_MAX_PARALLEL_SHARDS,
    MASKED_PRETRAINING_SHARDS_PER_SPLIT,
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
SHARD_COUNT_ENV = "MASKED_PRETRAINING_SHARD_COUNT"
BATCH_SIZE_ENV = "MASKED_PRETRAINING_BATCH_SIZE"
DEFAULT_BATCH_SIZE = 64
PROGRESS_INTERVAL = 1_000

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BATCH_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "generate_masked_pretraining_dataset.sh"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse public launcher and internal worker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--shards-per-split", type=_positive_int, default=MASKED_PRETRAINING_SHARDS_PER_SPLIT)
    parser.add_argument("--max-parallel-shards", type=_positive_int, default=MASKED_PRETRAINING_MAX_PARALLEL_SHARDS)
    parser.add_argument("--workers-per-shard", type=_positive_int, default=MASKED_PRETRAINING_WORKERS_PER_SHARD)
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def _candidate_manifest(split: str) -> Path:
    return Path(MASKED_PRETRAINING_WORK_DIR) / "candidates" / f"{split}.parquet"


def _shard_result_path(task: PretrainingShardTask) -> Path:
    return Path(MASKED_PRETRAINING_WORK_DIR) / "shards" / task.split / f"{task.shard_index:06d}.parquet"


def _safe_reset(path: Path, base: Path) -> None:
    resolved = path.resolve()
    resolved_base = base.resolve()
    if resolved == resolved_base or resolved_base not in resolved.parents:
        raise ValueError(f"Refusing to clear unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _load_aois() -> pl.DataFrame:
    # Restrict all three pretraining splits to downstream-training geography.
    # This prevents unlabeled raster pretraining from making downstream
    # validation/test geography transductive.
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
    """Assign overlapping AOI groups toward the requested record ratios."""
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
                (
                    assigned[split]
                    + (float(group["records"]) if split == choice else 0.0)
                    - targets[split]
                )
                ** 2
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
                pl.struct("scan_date", "scan_num")
                .hash(seed=MASKED_PRETRAINING_SPLIT_SEED)
                .alias("shard_key"),
                pl.struct("scan_date", "scan_num")
                .hash(seed=MASKED_PRETRAINING_SPLIT_SEED + 1)
                .alias("selection_key")
            )
            .sort("selection_key", AOI_ID_COL, "tempo_time")
            .with_row_index("candidate_index")
            .select(list(CANDIDATE_SCHEMA))
        )
        write_parquet_atomic(frame.cast(CANDIDATE_SCHEMA), _candidate_manifest(split))
        print(f"[{split}] candidate pool: {frame.height:,} scenes across {frame[AOI_ID_COL].n_unique():,} AOIs")


def _array_spec(tasks: list[PretrainingShardTask]) -> str:
    return f"0-{len(tasks) - 1}"


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
    (REPOSITORY_ROOT / "logs").mkdir(exist_ok=True)
    tasks = build_shard_tasks(SPLIT_TARGETS, args.shards_per_split)
    script_arguments = [
        "--shards-per-split",
        str(args.shards_per_split),
        "--workers-per-shard",
        str(args.workers_per_shard),
        "--batch-size",
        str(args.batch_size),
    ]
    prepare_arguments = [*script_arguments]
    if args.refresh_cache:
        prepare_arguments.append("--refresh-cache")
    prepare_job = _submit(
        [
            f"--cpus-per-task={args.workers_per_shard}",
            "--time=04:00:00",
            "--job-name=masked-data-prepare",
            f"--export=ALL,{STAGE_ENV}=prepare",
        ],
        prepare_arguments,
    )
    exported = (
        f"ALL,{STAGE_ENV}=worker,{SHARD_COUNT_ENV}={args.shards_per_split},"
        f"{BATCH_SIZE_ENV}={args.batch_size}"
    )
    worker_job = _submit(
        [
            f"--array={_array_spec(tasks)}%{args.max_parallel_shards}",
            f"--cpus-per-task={args.workers_per_shard}",
            "--job-name=masked-data-shard",
            f"--dependency=afterok:{prepare_job}",
            f"--export={exported}",
        ],
        script_arguments,
    )
    finalizer_job = _submit(
        [
            "--array=0",
            "--cpus-per-task=1",
            "--time=04:00:00",
            "--job-name=masked-data-finalize",
            f"--dependency=afterok:{worker_job}",
            f"--export=ALL,{STAGE_ENV}=finalize,{SHARD_COUNT_ENV}={args.shards_per_split}",
        ],
        script_arguments,
    )
    print(f"Masked-pretraining preparation: {prepare_job}")
    print(f"Masked-pretraining shard array: {worker_job}")
    print(f"Masked-pretraining finalizer: {finalizer_job}")


def _run_prepare(args: argparse.Namespace) -> None:
    """Reset disposable outputs and write immutable split manifests."""
    base = Path(MASKED_PRETRAINING_BASE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    _safe_reset(Path(MASKED_PRETRAINING_WORK_DIR), base)
    _safe_reset(Path(MASKED_PRETRAINING_MASKED_RASTER_DIR), base)
    if args.refresh_cache:
        _safe_reset(Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR), base)
    else:
        Path(MASKED_PRETRAINING_VALIDITY_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    _write_candidate_manifests()


def _global_valid_count(split: str) -> int:
    paths = sorted((Path(MASKED_PRETRAINING_WORK_DIR) / "shards" / split).glob("*.parquet"))
    if not paths:
        return 0
    return sum(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item() for path in paths)


def _candidate_batches(candidates: pl.DataFrame, batch_size: int) -> Iterator[pl.DataFrame]:
    """Batch complete TEMPO scans together so granule reads are reused."""
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


def _run_worker(args: argparse.Namespace) -> None:
    task_id_text = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise ValueError("Worker stage requires SLURM_ARRAY_TASK_ID")
    shard_count = int(os.getenv(SHARD_COUNT_ENV, str(args.shards_per_split)))
    tasks = build_shard_tasks(SPLIT_TARGETS, shard_count)
    task_id = int(task_id_text)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Array task {task_id} is outside the {len(tasks)}-task plan")
    task = tasks[task_id]
    candidates = load_shard_candidates(_candidate_manifest(task.split), task)
    result_path = _shard_result_path(task)
    valid_rows: list[dict[str, object]] = []
    write_parquet_atomic(pl.DataFrame(schema=VALID_RECORD_SCHEMA), result_path)
    invalid_count = 0
    retryable_count = 0
    completed = 0
    for batch_frame in _candidate_batches(candidates, args.batch_size):
        if _global_valid_count(task.split) >= task.target_count:
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
        valid_rows.extend(valid_record_frame(results).iter_rows(named=True))
        invalid_count += sum(result.status != VALID_STATUS and result.status != "retryable" for result in results)
        retryable_count += sum(result.status == "retryable" for result in results)
        write_parquet_atomic(pl.DataFrame(valid_rows, schema=VALID_RECORD_SCHEMA), result_path)
        completed += batch_frame.height
        if completed % PROGRESS_INTERVAL == 0 or completed == candidates.height:
            print(
                f"[{task.split} shard {task.shard_index}] {completed:,}/{candidates.height:,} candidates; "
                f"{len(valid_rows):,} valid, {invalid_count:,} invalid, {retryable_count:,} retryable"
            )
    print(f"[{task.split} shard {task.shard_index}] finished with {len(valid_rows):,} valid scenes")


def _write_csv_atomic(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        frame.write_csv(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_split(split: str, target_count: int) -> dict[str, object]:
    paths = sorted((Path(MASKED_PRETRAINING_WORK_DIR) / "shards" / split).glob("*.parquet"))
    frames = [pl.read_parquet(path).cast(VALID_RECORD_SCHEMA) for path in paths]
    valid = (
        pl.concat(frames, how="vertical")
        .unique(subset="cache_key", keep="first")
        .sort("candidate_index")
        .head(target_count)
        if frames
        else pl.DataFrame(schema=VALID_RECORD_SCHEMA)
    )
    final_rows: list[dict[str, object]] = []
    masked_root = Path(MASKED_PRETRAINING_MASKED_RASTER_DIR) / split
    for row in valid.iter_rows(named=True):
        with np.load(str(row["original_raster_path"]), allow_pickle=False) as bundle:
            no2 = np.asarray(bundle[NO2_RASTER_NAME], dtype=np.float32)
        rng = np.random.default_rng(MASKED_PRETRAINING_SPLIT_SEED + int(row["candidate_index"]))
        masked = mask_no2_raster(no2, rng)
        if masked is None:
            raise NotImplementedError(
                "mask_no2_raster is intentionally empty until missingness EDA defines the masking algorithm"
            )
        masked_no2, artificial_mask = masked
        masked_path = masked_root / f"{row['cache_key']}.npz"
        write_npz_atomic(masked_path, masked_no2=masked_no2, artificial_mask=artificial_mask)
        final_rows.append({**row, "masked_raster_path": str(masked_path)})
    output = pl.DataFrame(final_rows, schema=FINAL_RECORD_SCHEMA)
    _write_csv_atomic(output, Path(MASKED_PRETRAINING_DF_DIR) / f"{split}_df.csv")
    return {
        "split": split,
        "target_records": target_count,
        "available_valid_records": valid.height,
        "published_records": output.height,
        "target_met": output.height == target_count,
    }


def _run_finalizer() -> None:
    summaries = [_finalize_split(split, target) for split, target in SPLIT_TARGETS.items()]
    write_json_atomic({"splits": summaries}, Path(MASKED_PRETRAINING_DF_DIR) / "generation_summary.json")
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
    elif stage == "worker":
        _run_worker(args)
    elif stage == "finalize":
        _run_finalizer()
    else:
        raise ValueError(f"Unsupported {STAGE_ENV}: {stage}")


if __name__ == "__main__":
    main()
