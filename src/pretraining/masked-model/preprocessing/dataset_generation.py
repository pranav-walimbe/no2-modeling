"""Build masked-pretraining rasters with locality-aware discovery shards."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from dataset_generation_utils import (
    CANDIDATE_SCHEMA,
    FINAL_RECORD_SCHEMA,
    RETRYABLE_STATUS,
    VALID_STATUS,
    VALIDITY_INDEX_SCHEMA,
    CandidateOutcome,
    MaskedRecordTask,
    candidate_cache_tasks,
    discover_candidate_batch,
    empty_final_records,
    empty_validity_index,
    indexed_candidate_outcome,
    merge_validity_frames,
    validity_lookup,
    write_csv_atomic,
    write_json_atomic,
    write_masked_record,
    write_parquet_atomic,
)
from preprocessing.generate_dataset_utils import cache_inventory
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    add_sequence_weather_paths,
    build_aoi_spatial_frame,
    build_aois,
    cluster_aois,
)
from preprocessing.tempo_mapping import read_aoi_mapping

from config import (
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WEATHER_CACHE_DIR,
    FULL_DATA_PARQUET,
    HRRR_DIR,
    MASKED_PRETRAINING_BASE_DIR,
    MASKED_PRETRAINING_DF_DIR,
    MASKED_PRETRAINING_NUM_SHARDS,
    MASKED_PRETRAINING_SHARD_DIR,
    MASKED_PRETRAINING_SPLIT_SEED,
    MASKED_PRETRAINING_TEST_RECORDS,
    MASKED_PRETRAINING_TRAIN_RECORDS,
    MASKED_PRETRAINING_VAL_RECORDS,
    MASKED_PRETRAINING_VALIDITY_INDEX,
    MASKED_PRETRAINING_VALIDITY_UPDATES_DIR,
    MASKED_PRETRAINING_WORK_DIR,
    MASKED_PRETRAINING_WORKERS_PER_SHARD,
    TEMPO_AOI_MAPPING,
    TEMPO_DIR,
)

SPLIT_TARGETS = {
    "train": MASKED_PRETRAINING_TRAIN_RECORDS,
    "val": MASKED_PRETRAINING_VAL_RECORDS,
    "test": MASKED_PRETRAINING_TEST_RECORDS,
}
STAGE_ENV = "MASKED_PRETRAINING_STAGE"
DEFAULT_BATCH_SIZE = 64
PROGRESS_INTERVAL = 1_000
DISCOVERY_TIME_LIMIT = "24:00:00"

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BATCH_SCRIPT = REPOSITORY_ROOT / "scripts" / "slurm" / "generate_masked_pretraining_dataset.sh"


def parse_args() -> argparse.Namespace:
    """Parse launcher and worker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="clear the persistent validity index before this run",
    )
    parser.add_argument("--num-shards", type=int, default=MASKED_PRETRAINING_NUM_SHARDS)
    parser.add_argument("--workers-per-shard", type=int, default=MASKED_PRETRAINING_WORKERS_PER_SHARD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.num_shards < 1 or args.workers_per_shard < 1 or args.batch_size < 1:
        parser.error("shard, worker, and batch counts must be positive")
    if args.run_id is not None and (Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}):
        parser.error("--run-id must be one path-safe name")
    return args


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_work_dir(run_id: str) -> Path:
    return Path(MASKED_PRETRAINING_WORK_DIR) / run_id


def _run_shard_dir(run_id: str) -> Path:
    return Path(MASKED_PRETRAINING_SHARD_DIR) / run_id


def _candidate_path(run_id: str, split: str, shard_id: int) -> Path:
    return _run_work_dir(run_id) / "candidates" / split / f"{shard_id:06d}.parquet"


def _snapshot_path(run_id: str) -> Path:
    return _run_work_dir(run_id) / "validity-index.parquet"


def _update_path(run_id: str, split: str, shard_id: int, candidate_offset: int) -> Path:
    filename = f"{shard_id:06d}-{candidate_offset:012d}.parquet"
    return Path(MASKED_PRETRAINING_VALIDITY_UPDATES_DIR) / run_id / split / filename


def _shard_dir(run_id: str, split: str, shard_id: int) -> Path:
    return _run_shard_dir(run_id) / split / f"{shard_id:06d}"


def _record_path(run_id: str, split: str, shard_id: int) -> Path:
    return _shard_dir(run_id, split, shard_id) / "records.parquet"


def _split_quota(target: int, shard_count: int, shard_id: int) -> int:
    base, remainder = divmod(target, shard_count)
    return base + int(shard_id < remainder)


def _safe_reset(path: Path, base: Path) -> None:
    resolved = path.resolve()
    resolved_base = base.resolve()
    if resolved == resolved_base or resolved_base not in resolved.parents:
        raise ValueError(f"Refusing to clear unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _load_global_aois() -> pl.DataFrame:
    facilities = (
        pl.scan_parquet(FULL_DATA_PARQUET)
        .select("facilityId", "lat", "lon")
        .drop_nulls()
        .unique(subset="facilityId", keep="first")
        .collect(engine="streaming")
    )
    return build_aois(facilities)


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


def _write_candidate_manifests(
    run_id: str,
    shard_count: int,
    tempo_inventory: set[str],
    weather_inventory: set[str],
) -> None:
    aois = _load_global_aois()
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
                pl.col("tempo_time").dt.truncate("1h").alias("_processing_hour"),
            )
            .sort("_processing_hour", "cluster", AOI_ID_COL, "tempo_time", "scan_num")
            .drop("_processing_hour")
            .with_row_index("candidate_index")
        )
        inventory_started = time.monotonic()
        tempo_cached: list[bool] = []
        weather_cached: list[bool] = []
        for row in frame.iter_rows(named=True):
            scan, weather = candidate_cache_tasks(
                row,
                tempo_root=Path(TEMPO_DIR),
                tempo_cache_dir=Path(DATASET_TEMPO_CACHE_DIR),
                hrrr_root=Path(HRRR_DIR),
                weather_cache_dir=Path(DATASET_WEATHER_CACHE_DIR),
            )
            tempo_cached.append(Path(scan.cache_path).name in tempo_inventory)
            weather_cached.append(Path(weather.cache_path).name in weather_inventory)
        frame = (
            frame.with_columns(
                pl.Series("tempo_cached", tempo_cached, dtype=pl.Boolean),
                pl.Series("weather_cached", weather_cached, dtype=pl.Boolean),
            )
            .select(list(CANDIDATE_SCHEMA))
            .cast(CANDIDATE_SCHEMA)
        )
        for shard_id in range(shard_count):
            start = frame.height * shard_id // shard_count
            stop = frame.height * (shard_id + 1) // shard_count
            write_parquet_atomic(frame.slice(start, stop - start), _candidate_path(run_id, split, shard_id))
        print(
            f"[{_timestamp()}] [{split}] wrote {frame.height:,} locality-ordered candidates across "
            f"{shard_count} shards in {time.monotonic() - inventory_started:.1f}s"
        )


def _load_persistent_validity_index() -> pl.DataFrame:
    index_path = Path(MASKED_PRETRAINING_VALIDITY_INDEX)
    frames = []
    if index_path.is_file():
        frames.append(pl.read_parquet(index_path).cast(VALIDITY_INDEX_SCHEMA))
    update_root = Path(MASKED_PRETRAINING_VALIDITY_UPDATES_DIR)
    frames.extend(pl.read_parquet(path) for path in sorted(update_root.glob("*/*/*.parquet")))
    return merge_validity_frames(frames)


def _build_cache_inventory(cache_name: str, cache_dir: Path) -> set[str]:
    started = time.monotonic()
    print(f"[{_timestamp()}] [{cache_name}] cache inventory started: {cache_dir}")
    filenames = cache_inventory(cache_dir)
    elapsed = time.monotonic() - started
    print(f"[{_timestamp()}] [{cache_name}] cache inventory finished: {len(filenames):,} entries in {elapsed:.1f}s")
    return filenames


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
            "--name=masked-data-prepare,masked-data-discover,masked-data-finalize",
            "--format=%i",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.split()
    if active_jobs:
        raise RuntimeError(f"Masked-pretraining generation jobs are already active: {', '.join(active_jobs)}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("clean-%Y%m%d-%H%M%S")
    (REPOSITORY_ROOT / "logs").mkdir(exist_ok=True)
    script_arguments = [
        "--run-id",
        run_id,
        "--num-shards",
        str(args.num_shards),
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
            f"--array=0-{args.num_shards - 1}",
            f"--cpus-per-task={args.workers_per_shard}",
            f"--time={DISCOVERY_TIME_LIMIT}",
            "--job-name=masked-data-discover",
            "--kill-on-invalid-dep=yes",
            f"--dependency=afterok:{prepare_job}",
            f"--export=ALL,{STAGE_ENV}=discover",
        ],
        script_arguments,
    )
    finalizer_job = _submit(
        [
            "--array=0",
            "--cpus-per-task=1",
            "--time=02:00:00",
            "--job-name=masked-data-finalize",
            "--kill-on-invalid-dep=yes",
            f"--dependency=afterok:{discovery_job}",
            f"--export=ALL,{STAGE_ENV}=finalize",
        ],
        script_arguments,
    )
    print(f"Masked-pretraining run: {run_id}")
    print(f"Preparation: {prepare_job}")
    print(f"Masked-raster discovery: {discovery_job}")
    print(f"Manifest finalizer: {finalizer_job}")


def _run_prepare(args: argparse.Namespace) -> None:
    if args.run_id is None:
        raise ValueError("Prepare requires --run-id")
    started = time.monotonic()
    base = Path(MASKED_PRETRAINING_BASE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    if args.clear_cache:
        Path(MASKED_PRETRAINING_VALIDITY_INDEX).unlink(missing_ok=True)
        _safe_reset(Path(MASKED_PRETRAINING_VALIDITY_UPDATES_DIR), base)
        index = empty_validity_index()
    else:
        index = _load_persistent_validity_index()
    _safe_reset(Path(MASKED_PRETRAINING_WORK_DIR), base)
    _safe_reset(Path(MASKED_PRETRAINING_SHARD_DIR), base)
    _safe_reset(Path(MASKED_PRETRAINING_DF_DIR), base)
    work_dir = _run_work_dir(args.run_id)
    shard_dir = _run_shard_dir(args.run_id)
    work_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    write_parquet_atomic(index, _snapshot_path(args.run_id))
    print(f"[{_timestamp()}] validity index snapshot: {index.height:,} entries")
    tempo_inventory = _build_cache_inventory("tempo-cache", Path(DATASET_TEMPO_CACHE_DIR))
    weather_inventory = _build_cache_inventory("weather-cache", Path(DATASET_WEATHER_CACHE_DIR))
    _write_candidate_manifests(args.run_id, args.num_shards, tempo_inventory, weather_inventory)
    write_json_atomic(
        {
            "run_id": args.run_id,
            "num_shards": args.num_shards,
            "tempo_cache_entries": len(tempo_inventory),
            "weather_cache_entries": len(weather_inventory),
            "prepared_at": _timestamp(),
        },
        work_dir / "run.json",
    )
    print(f"[{_timestamp()}] preparation finished in {time.monotonic() - started:.1f}s")


def _run_discovery_split(
    args: argparse.Namespace,
    split: str,
    shard_id: int,
    index: dict[str, dict[str, object]],
    tempo_cache_additions: set[str],
    weather_cache_additions: set[str],
) -> None:
    if args.run_id is None:
        raise ValueError("Discovery requires --run-id")
    quota = _split_quota(SPLIT_TARGETS[split], args.num_shards, shard_id)
    candidates = pl.read_parquet(_candidate_path(args.run_id, split, shard_id)).cast(CANDIDATE_SCHEMA)
    records: list[dict[str, object]] = []
    pending_updates: list[dict[str, object]] = []
    offset = 0
    raster_dir = _shard_dir(args.run_id, split, shard_id) / "record-rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_published_offset = offset
    index_hits = 0
    cache_misses = 0
    record_writer = ProcessPoolExecutor(max_workers=args.workers_per_shard)

    try:
        while offset < candidates.height and len(records) < quota:
            batch = candidates.slice(offset, args.batch_size).to_dicts()
            keyed_rows: list[tuple[str, dict[str, object]]] = []
            resolved: dict[str, CandidateOutcome] = {}
            unknown: list[dict[str, object]] = []
            for row in batch:
                scan, _ = candidate_cache_tasks(
                    row,
                    tempo_root=Path(TEMPO_DIR),
                    tempo_cache_dir=Path(DATASET_TEMPO_CACHE_DIR),
                    hrrr_root=Path(HRRR_DIR),
                    weather_cache_dir=Path(DATASET_WEATHER_CACHE_DIR),
                )
                keyed_rows.append((scan.cache_key, row))
                indexed = index.get(scan.cache_key)
                if indexed is None:
                    unknown.append(row)
                    cache_misses += 1
                else:
                    resolved[scan.cache_key] = indexed_candidate_outcome(row, scan.cache_key, indexed)
                    index_hits += 1

            if unknown:
                discovered = discover_candidate_batch(
                    unknown,
                    tempo_root=Path(TEMPO_DIR),
                    tempo_cache_dir=Path(DATASET_TEMPO_CACHE_DIR),
                    hrrr_root=Path(HRRR_DIR),
                    weather_cache_dir=Path(DATASET_WEATHER_CACHE_DIR),
                    workers=args.workers_per_shard,
                    tempo_cache_additions=tempo_cache_additions,
                    weather_cache_additions=weather_cache_additions,
                )
                for outcome in discovered:
                    resolved[outcome.cache_key] = outcome
                    if outcome.status != RETRYABLE_STATUS:
                        validity_row = outcome.validity_row()
                        pending_updates.append(validity_row)
                        index[outcome.cache_key] = validity_row

            remaining = quota - len(records)
            selected = [resolved[key] for key, _ in keyed_rows if resolved[key].status == VALID_STATUS][:remaining]
            tasks = [
                MaskedRecordTask(
                    row=outcome.row,
                    cache_key=outcome.cache_key,
                    tempo_cache_path=str(outcome.tempo_cache_path),
                    weather_cache_path=str(outcome.weather_cache_path),
                    output_path=str(raster_dir / f"{outcome.cache_key}.npz"),
                    mask_seed=MASKED_PRETRAINING_SPLIT_SEED + int(outcome.row["candidate_index"]),
                )
                for outcome in selected
            ]
            records.extend(record_writer.map(write_masked_record, tasks))
            offset += len(batch)

            should_publish = (
                offset - last_published_offset >= PROGRESS_INTERVAL
                or len(records) == quota
                or offset == candidates.height
            )
            if should_publish:
                if pending_updates:
                    updates = pl.DataFrame(pending_updates, schema=VALIDITY_INDEX_SCHEMA)
                    write_parquet_atomic(updates, _update_path(args.run_id, split, shard_id, offset))
                    pending_updates.clear()
                write_parquet_atomic(
                    pl.DataFrame(records, schema=FINAL_RECORD_SCHEMA),
                    _record_path(args.run_id, split, shard_id),
                )
                last_published_offset = offset
                print(
                    f"[{_timestamp()}] [{split} shard {shard_id}] {len(records):,}/{quota:,} masked records; "
                    f"{offset:,}/{candidates.height:,} candidates; {index_hits:,} index hits; "
                    f"{cache_misses:,} cache misses"
                )
    finally:
        record_writer.shutdown(cancel_futures=True)

    if len(records) != quota:
        raise ValueError(
            f"[{split} shard {shard_id}] candidate segment exhausted with {len(records):,}/{quota:,} records"
        )
    print(f"[{_timestamp()}] [{split} shard {shard_id}] finished in {time.monotonic() - started:.1f}s")


def _run_discovery(args: argparse.Namespace) -> None:
    if args.run_id is None:
        raise ValueError("Discovery requires --run-id")
    shard_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if shard_id >= args.num_shards:
        raise ValueError(f"Shard {shard_id} is outside configured count {args.num_shards}")
    index = validity_lookup(pl.read_parquet(_snapshot_path(args.run_id)).cast(VALIDITY_INDEX_SCHEMA))
    tempo_cache_additions: set[str] = set()
    weather_cache_additions: set[str] = set()
    for split in SPLIT_TARGETS:
        _run_discovery_split(
            args,
            split,
            shard_id,
            index,
            tempo_cache_additions,
            weather_cache_additions,
        )


def _run_finalizer(args: argparse.Namespace) -> None:
    if args.run_id is None:
        raise ValueError("Finalizer requires --run-id")
    base = Path(MASKED_PRETRAINING_BASE_DIR)
    summaries: list[dict[str, object]] = []
    for split, target in SPLIT_TARGETS.items():
        frames = [
            pl.read_parquet(_record_path(args.run_id, split, shard_id)).cast(FINAL_RECORD_SCHEMA)
            for shard_id in range(args.num_shards)
        ]
        records = pl.concat(frames, how="vertical").sort("candidate_index") if frames else empty_final_records()
        if records.height != target:
            raise ValueError(f"[{split}] discovered {records.height:,}/{target:,} requested records")
        relative_paths = [str(Path(path).relative_to(base)) for path in records["raster_bundle_path"].to_list()]
        output = records.with_columns(pl.Series("raster_bundle_path", relative_paths, dtype=pl.String))
        write_csv_atomic(output, Path(MASKED_PRETRAINING_DF_DIR) / f"{split}_df.csv")
        summaries.append({"split": split, "target_records": target, "published_records": output.height})
        print(f"[{_timestamp()}] [{split}] published {output.height:,} records")

    index_frames = [pl.read_parquet(_snapshot_path(args.run_id)).cast(VALIDITY_INDEX_SCHEMA)]
    update_paths = sorted(Path(MASKED_PRETRAINING_VALIDITY_UPDATES_DIR).glob("*/*/*.parquet"))
    index_frames.extend(pl.read_parquet(path).cast(VALIDITY_INDEX_SCHEMA) for path in update_paths)
    merged_index = merge_validity_frames(index_frames)
    write_parquet_atomic(merged_index, Path(MASKED_PRETRAINING_VALIDITY_INDEX))
    write_json_atomic(
        {
            "run_id": args.run_id,
            "num_shards": args.num_shards,
            "validity_index_entries": merged_index.height,
            "splits": summaries,
        },
        Path(MASKED_PRETRAINING_DF_DIR) / "generation_summary.json",
    )
    print(f"[{_timestamp()}] compacted {merged_index.height:,} validity-index entries")
    _safe_reset(Path(MASKED_PRETRAINING_VALIDITY_UPDATES_DIR), Path(MASKED_PRETRAINING_BASE_DIR))


def main() -> None:
    """Launch or execute one masked-raster generation stage."""
    args = parse_args()
    stage = os.getenv(STAGE_ENV, "launch")
    if stage == "launch":
        _launch(args)
    elif stage == "prepare":
        _run_prepare(args)
    elif stage == "discover":
        _run_discovery(args)
    elif stage == "finalize":
        _run_finalizer(args)
    else:
        raise ValueError(f"Unsupported {STAGE_ENV}: {stage}")


if __name__ == "__main__":
    main()
