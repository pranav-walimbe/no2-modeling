"""Build and publish the cache-backed AOI plume-quality ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import matplotlib
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from preprocessing.generate_dataset_utils import (
    WIND_U_RASTER_NAME,
    WIND_V_RASTER_NAME,
    ScanBatchTask,
    ScanTask,
    WeatherBatchTask,
    WeatherTask,
    bounded_parallel_map,
    cache_inventory,
    extract_weather_cache,
    make_scan_task,
    make_weather_task,
    process_scan_batch,
    process_weather_batch,
    scan_batches,
    select_hotspot_cell,
    weather_batches,
)
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    add_ema_targets,
    add_projected_coordinates,
    add_sequence_weather_paths,
    add_timestep_nox,
    build_aoi_membership,
    build_aois,
    filter_usable_nox_measurements,
    usable_nox_measurement_expr,
)
from scipy.ndimage import gaussian_filter, label

from config import (
    AOI_SCORE_JSON,
    DATASET_TEMPO_CACHE_DIR,
    DATASET_WEATHER_CACHE_DIR,
    FULL_DATA_PARQUET,
    HRRR_DIR,
    STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR,
    STRATIFICATION_INNOVATION_RELATIVE_FLOOR,
    TEMPO_AOI_MAPPING,
    TEMPO_DIR,
    TEMPO_MAX_DELTA_MINUTES,
    TEMPO_MIN_DELTA_MINUTES,
    VIS_DIR,
)

REPOSITORY = Path("/global/home/users/pranavwalimbe/no2-modeling")
RUN_ROOT = Path("/global/scratch/projects/fc_nitrates/ddp/nox/aoi-heuristic")
LOG_DIR = REPOSITORY / "logs"
EMAIL = "pranav.walimbe@berkeley.edu"
MODULE = "preprocessing.aoi_heuristic"
ACCOUNT = "fc_nitrates"
PARTITION = "savio3_htc"
QOS = "savio_normal"
PYTHON_MODULE = "python/3.11.6-gcc-11.4.0"

LABEL_TIMESTEPS = 4
CLASS_COL = "current_delta_category"
DELTA_COL = "current_effective_delta_nox"
RASTER_PATH_COL = "raster_bundle_path"
CLASS_NAMES = ("decrease", "steady", "increase")
DEFAULT_SEED = 20260923
DEFAULT_NODES = 8
DEFAULT_WORKERS_PER_NODE = 8
DEFAULT_CANDIDATES_PER_CLASS = 64
DEFAULT_MINIMUM_PER_CLASS = 8
ROW_BATCH_SIZE = 100_000
PROGRESS_FILES = 100
PROGRESS_RECORDS = 1_000

NORMALIZATION_CENTER = 1868138303979520.0
NORMALIZATION_SCALE = 1199722224989936.2
NORMALIZATION_CLIP = 8.0
ABSOLUTE_CONTRAST_SCALE = 0.2
DIRECTION_SCORE_WEIGHT = 0.2
MIN_VALID_TIMESTEPS = 2
MIN_WIND_SPEED_MPS = 1.0
MIN_REGION_COVERAGE = 0.60
CORRIDOR_LENGTH_PIXELS = 10.0
SOURCE_EXCLUSION_RADIUS_PIXELS = 3.0
PLUME_CROSSWIND_SIGMA_PIXELS = 1.5
FLANK_CENTER_PIXELS = 4.5
FLANK_SIGMA_PIXELS = 1.0
BACKGROUND_BLUR_SIGMA_PIXELS = 4.0
WIND_SEARCH_OFFSETS_DEGREES = (-45, -30, -15, 0, 15, 30, 45)
BACKGROUND_MAD_MULTIPLIER = 1.4826
NOISE_FLOOR_STANDARDIZED = 0.10
LOCALIZATION_REFERENCE_RATIO = 3.0
BROAD_SIGNAL_DECAY = 3.0
MIN_PLUME_DELTA_SCALE = 0.1
TaskType = TypeVar("TaskType")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse AOI heuristic orchestration and worker options."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="submit prepare, cache-array, and final jobs")
    submit.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    submit.add_argument("--workers-per-node", type=int, default=DEFAULT_WORKERS_PER_NODE)
    submit.add_argument("--seed", type=int, default=DEFAULT_SEED)

    prepare = subparsers.add_parser("prepare", help="prepare candidates and cache-task manifests")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--nodes", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)

    cache = subparsers.add_parser("cache-worker", help="generate one cache-task shard")
    cache.add_argument("--run-dir", type=Path, required=True)
    cache.add_argument("--workers", type=int, required=True)
    cache.add_argument("--shard-index", type=int, default=None)

    final = subparsers.add_parser("finalize", help="score candidates and publish the ranking")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--workers", type=int, required=True)
    final.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _run_id() -> str:
    # Use UTC so run paths sort in execution order
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _job_command(command: str) -> str:
    # Build the environment shared by each scheduled phase
    body = (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(REPOSITORY))}; "
        f"module load {shlex.quote(PYTHON_MODULE)}; "
        "source .venv/bin/activate; "
        f"export PYTHONPATH={shlex.quote(str(REPOSITORY / 'src'))}:"
        f"{shlex.quote(str(REPOSITORY / 'src/delta-model'))}; "
        'export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"; '
        f"srun {command}"
    )
    return f"bash -lc {shlex.quote(body)}"


def _submit_job(
    name: str,
    command: str,
    cpus: int,
    time_limit: str,
    dependency: str | None = None,
    array: str | None = None,
) -> str:
    # Submit one noninteractive Savio phase and return its job ID
    suffix = "%A_%a" if array else "%j"
    arguments = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        f"--qos={QOS}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--time={time_limit}",
        "--mail-type=BEGIN,END,FAIL",
        f"--mail-user={EMAIL}",
        f"--output={LOG_DIR}/%x-{suffix}.log",
        f"--error={LOG_DIR}/%x-{suffix}.err",
        f"--chdir={REPOSITORY}",
    ]
    if dependency:
        arguments.append(f"--dependency=afterok:{dependency}")
    if array:
        arguments.extend((f"--array={array}", "--exclusive"))
    arguments.extend(("--wrap", _job_command(command)))
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return completed.stdout.strip().split(";", maxsplit=1)[0]


def submit_pipeline(nodes: int, workers_per_node: int, seed: int) -> None:
    """Submit the dependent prepare, cache-array, and final scoring jobs.

    Args:
        nodes: Number of concurrent cache-array tasks.
        workers_per_node: Process workers assigned to every cache task.
        seed: Deterministic candidate and visualization seed.
    """
    if nodes <= 0 or workers_per_node <= 0:
        raise ValueError("nodes and workers-per-node must be positive")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True)
    output_dir = Path(VIS_DIR) / f"aoi-heuristic-{run_id}"
    output_dir.mkdir(parents=True)
    configuration = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "score_map": str(AOI_SCORE_JSON),
        "nodes": nodes,
        "workers_per_node": workers_per_node,
        "seed": seed,
    }
    _write_json_atomic(configuration, run_dir / "configuration.json")

    prepare_command = (
        f"python -u -m {MODULE} prepare --run-dir {shlex.quote(str(run_dir))} --nodes {nodes} --seed {seed}"
    )
    prepare_job = _submit_job("aoi-heuristic-prep", prepare_command, 16, "06:00:00")
    cache_command = (
        f"python -u -m {MODULE} cache-worker --run-dir {shlex.quote(str(run_dir))} --workers {workers_per_node}"
    )
    cache_job = _submit_job(
        "aoi-heuristic-cache",
        cache_command,
        workers_per_node,
        "12:00:00",
        dependency=prepare_job,
        array=f"0-{nodes - 1}%{nodes}",
    )
    final_command = (
        f"python -u -m {MODULE} finalize --run-dir {shlex.quote(str(run_dir))} "
        f"--workers {workers_per_node} --seed {seed}"
    )
    final_job = _submit_job(
        "aoi-heuristic-final",
        final_command,
        workers_per_node,
        "12:00:00",
        dependency=cache_job,
    )
    submission = {**configuration, "prepare_job": prepare_job, "cache_job": cache_job, "final_job": final_job}
    _write_json_atomic(submission, run_dir / "submission.json")
    print(json.dumps(submission, indent=2), flush=True)


def _write_json_atomic(value: object, path: Path) -> None:
    # Replace JSON only after the complete payload reaches disk
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _inventory_cache(name: str, directory: Path) -> set[str]:
    # Inventory each flat cache once so candidate resolution avoids repeated metadata calls
    started = time.monotonic()
    print(f"{datetime.now(timezone.utc).isoformat()} {name} cache inventory started", flush=True)
    inventory = cache_inventory(directory)
    elapsed = time.monotonic() - started
    print(
        f"{datetime.now(timezone.utc).isoformat()} {name} cache inventory finished: "
        f"{len(inventory):,} entries in {elapsed:.1f}s",
        flush=True,
    )
    return inventory


def _sample_order(aoi_id: int, label_time: datetime, seed: int) -> int:
    # Use a stable digest across Python processes
    identity = f"{seed}:{aoi_id}:{label_time.isoformat()}".encode()
    return int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _sequence_row(history: deque[tuple[datetime, list[str]]], aoi_id: int) -> dict[str, object] | None:
    # Require the four observations consumed by the score
    if len(history) != LABEL_TIMESTEPS:
        return None
    intervals = [(history[index][0] - history[index - 1][0]).total_seconds() / 60 for index in range(1, len(history))]
    if not all(TEMPO_MIN_DELTA_MINUTES <= value <= TEMPO_MAX_DELTA_MINUTES for value in intervals):
        return None
    row: dict[str, object] = {AOI_ID_COL: aoi_id}
    for index, (timestamp, paths) in enumerate(history):
        row[f"timestep_time_t{index}"] = timestamp
        row[f"no2_paths_t{index}"] = paths
    return row


def sample_mapping_sequences(
    mapping_dir: Path,
    aoi_ids: set[int],
    seed: int,
    output_dir: Path,
) -> tuple[list[Path], int]:
    """Enumerate four-observation histories into bounded Parquet batches.

    Args:
        mapping_dir: Partitioned AOI-observation Parquet root.
        aoi_ids: AOIs represented by the emissions table.
        seed: Deterministic candidate-order seed.
        output_dir: Run-local directory for compact sequence batches.

    Returns:
        Written batch paths and the valid sequence count.
    """
    paths = sorted(mapping_dir.rglob("date=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No TEMPO mappings found under {mapping_dir}")
    histories: dict[int, deque[tuple[datetime, list[str]]]] = defaultdict(lambda: deque(maxlen=LABEL_TIMESTEPS))
    rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True)
    batch_paths: list[Path] = []
    count = 0

    def flush_rows() -> None:
        if not rows:
            return
        output = output_dir / f"part-{len(batch_paths):05d}.parquet"
        pl.DataFrame(rows).write_parquet(output)
        batch_paths.append(output)
        rows.clear()

    for file_index, path in enumerate(paths, start=1):
        daily = (
            pl.read_parquet(path, columns=[AOI_ID_COL, "tempo_time", "granule_paths"])
            .filter(pl.col(AOI_ID_COL).is_in(list(aoi_ids)))
            .sort(AOI_ID_COL, "tempo_time")
        )
        for aoi_id, timestamp, granule_paths in daily.iter_rows():
            key = int(aoi_id)
            histories[key].append((timestamp, granule_paths))
            row = _sequence_row(histories[key], key)
            if row is None:
                continue
            identity = f"{key}:{timestamp.isoformat()}"
            rows.append(
                {
                    **row,
                    "_sample_order": _sample_order(key, timestamp, seed),
                    RASTER_PATH_COL: f"cache://{identity}",
                }
            )
            count += 1
            if len(rows) >= ROW_BATCH_SIZE:
                flush_rows()
        if file_index % PROGRESS_FILES == 0 or file_index == len(paths):
            print(f"TEMPO mapping: {file_index:,}/{len(paths):,} files, {count:,} histories", flush=True)
    flush_rows()
    if not batch_paths:
        raise ValueError("No valid four-observation TEMPO histories were found")
    return batch_paths, count


def _calculate_hourly_aoi_nox(raw_records: pl.LazyFrame, membership: pl.DataFrame) -> pl.DataFrame:
    # Aggregate hours only when every contributing measurement is usable
    invalid_hours = (
        raw_records.filter(~usable_nox_measurement_expr() | ~pl.col("noxMass").is_finite())
        .join(membership.lazy(), on="facilityId", how="inner")
        .select(AOI_ID_COL, "emissions_hour_utc")
        .unique()
        .with_columns(pl.lit(True).alias("_has_invalid_nox"))
        .collect(engine="streaming")
    )
    return (
        raw_records.pipe(filter_usable_nox_measurements)
        .filter(pl.col("noxMass").is_finite())
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(pl.col("noxMass").sum().alias("nox_mass"))
        .collect(engine="streaming")
        .join(invalid_hours, on=[AOI_ID_COL, "emissions_hour_utc"], how="left")
        .filter(pl.col("_has_invalid_nox").is_null())
        .drop("_has_invalid_nox")
    )


def _interpolate_targets(sequences: pl.DataFrame, hourly_nox: pl.DataFrame) -> pl.DataFrame:
    # Point-interpolate emissions then calculate the four-step EMA innovation
    return add_ema_targets(add_timestep_nox(sequences, hourly_nox, LABEL_TIMESTEPS), LABEL_TIMESTEPS)


def _active_nox_values(targets: pl.DataFrame) -> pl.DataFrame:
    # Retain positive finite timestep values for the AOI-relative threshold
    columns = [f"t{index}_nox" for index in range(LABEL_TIMESTEPS)]
    return (
        targets.select(AOI_ID_COL, pl.concat_list(columns).alias("_nox"))
        .explode("_nox", empty_as_null=True)
        .filter(pl.col("_nox").is_finite() & (pl.col("_nox") > 0))
    )


def _apply_labels(targets: pl.DataFrame, scales: pl.DataFrame) -> pl.DataFrame:
    # Classify the attenuation-corrected innovation at t3
    interval_hours = (pl.col("timestep_time_t3") - pl.col("timestep_time_t2")).dt.total_seconds() / 3600
    alpha = 1 - (-interval_hours / 2.0).exp()
    threshold = (STRATIFICATION_INNOVATION_RELATIVE_FLOOR * pl.col("aoi_active_median_nox")).clip(
        lower_bound=STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR
    )
    return (
        targets.join(scales, on=AOI_ID_COL, how="inner")
        .with_columns(alpha.alias("ema_update_alpha"), threshold.alias("hybrid_innovation_threshold"))
        .filter(pl.col("ema_update_alpha").is_finite() & (pl.col("ema_update_alpha") > 0))
        .with_columns((pl.col("effective_delta_nox") / pl.col("ema_update_alpha")).alias(DELTA_COL))
        .filter(pl.col(DELTA_COL).is_finite())
        .with_columns(
            pl.when(pl.col(DELTA_COL) <= -pl.col("hybrid_innovation_threshold"))
            .then(pl.lit("decrease"))
            .when(pl.col(DELTA_COL) >= pl.col("hybrid_innovation_threshold"))
            .then(pl.lit("increase"))
            .otherwise(pl.lit("steady"))
            .alias(CLASS_COL)
        )
    )


def _sample_class_candidates(records: pl.DataFrame, limit: int) -> pl.DataFrame:
    # Keep deterministic lowest hashes within every AOI and class
    return (
        records.sort(AOI_ID_COL, CLASS_COL, "_sample_order")
        .group_by(AOI_ID_COL, CLASS_COL, maintain_order=True)
        .head(limit)
    )


def _class_counts(records: pl.DataFrame) -> pl.DataFrame:
    # Count each label class by AOI
    return records.group_by(AOI_ID_COL).agg(
        *[(pl.col(CLASS_COL) == name).sum().alias(f"{name}_histories") for name in CLASS_NAMES]
    )


def select_labeled_candidates(
    sequence_paths: list[Path],
    hourly_nox: pl.DataFrame,
    active_value_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Label batches and retain bounded candidates per AOI and class.

    Args:
        sequence_paths: Compact four-observation history batches.
        hourly_nox: Valid AOI-hour emissions totals.
        active_value_dir: Run-local storage for median inputs.

    Returns:
        Candidate histories, complete class audit, and labeled-history count.
    """
    active_value_dir.mkdir(parents=True)
    active_paths = []
    for index, path in enumerate(sequence_paths):
        active_path = active_value_dir / f"part-{index:05d}.parquet"
        _active_nox_values(_interpolate_targets(pl.read_parquet(path), hourly_nox)).write_parquet(active_path)
        active_paths.append(active_path)
    scales = (
        pl.scan_parquet(active_paths)
        .group_by(AOI_ID_COL)
        .agg(pl.col("_nox").median().alias("aoi_active_median_nox"))
        .collect(engine="streaming")
    )
    count_batches: list[pl.DataFrame] = []
    retained: pl.DataFrame | None = None
    labeled_count = 0
    for index, path in enumerate(sequence_paths, start=1):
        batch = pl.read_parquet(path)
        labeled = _apply_labels(_interpolate_targets(batch, hourly_nox), scales)
        labeled_count += labeled.height
        count_batches.append(_class_counts(labeled))
        candidates = _sample_class_candidates(labeled, DEFAULT_CANDIDATES_PER_CLASS)
        retained = (
            candidates
            if retained is None
            else _sample_class_candidates(
                pl.concat([retained, candidates], how="vertical", rechunk=False),
                DEFAULT_CANDIDATES_PER_CLASS,
            )
        )
        print(f"Labeling batch {index:,}: {labeled_count:,} histories", flush=True)
    if retained is None:
        raise ValueError("No finite emissions labels were produced")
    counts = (
        pl.concat(count_batches, how="vertical", rechunk=False)
        .group_by(AOI_ID_COL)
        .agg(*(pl.col(f"{name}_histories").sum() for name in CLASS_NAMES))
        .with_columns(
            pl.min_horizontal(*(f"{name}_histories" for name in CLASS_NAMES)).alias("minimum_class_histories")
        )
        .with_columns((pl.col("minimum_class_histories") >= DEFAULT_MINIMUM_PER_CLASS).alias("meets_three_class_floor"))
        .sort(AOI_ID_COL)
    )
    viable = counts.filter(pl.col("meets_three_class_floor"))[AOI_ID_COL].implode()
    return retained.filter(pl.col(AOI_ID_COL).is_in(viable)), counts, labeled_count


def _hotspot_table(raw_records: pl.LazyFrame, membership: pl.DataFrame, aois: pl.DataFrame) -> pl.DataFrame:
    # Resolve the dominant unit cluster within each AOI
    unit_counts = raw_records.group_by("facilityId").agg(pl.col("unitId").n_unique().alias("_unit_count"))
    facilities = (
        raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId", keep="first").collect()
    )
    projected = add_projected_coordinates(facilities).lazy()
    sources = (
        projected.join(unit_counts, on="facilityId", how="inner")
        .join(membership.lazy(), on="facilityId", how="inner")
        .join(
            aois.select(AOI_ID_COL, "x_m", "y_m").lazy().rename({"x_m": "_aoi_x", "y_m": "_aoi_y"}),
            on=AOI_ID_COL,
            how="inner",
        )
        .with_columns(
            ((pl.col("x_m") - pl.col("_aoi_x")) / 1_000).alias("_east_km"),
            ((pl.col("y_m") - pl.col("_aoi_y")) / 1_000).alias("_north_km"),
        )
        .sort(AOI_ID_COL, "facilityId")
        .group_by(AOI_ID_COL, maintain_order=True)
        .agg("_east_km", "_north_km", "_unit_count")
        .collect()
    )
    rows = []
    for row in sources.iter_rows(named=True):
        hotspot_row, hotspot_column = select_hotspot_cell(
            tuple(row["_east_km"]), tuple(row["_north_km"]), tuple(row["_unit_count"])
        )
        rows.append({AOI_ID_COL: row[AOI_ID_COL], "hotspot_row": hotspot_row, "hotspot_column": hotspot_column})
    return pl.DataFrame(rows)


def _unique_scan_tasks(records: pl.DataFrame) -> dict[str, ScanTask]:
    # Resolve each scan cache key once
    tasks: dict[str, ScanTask] = {}
    cache_dir = Path(DATASET_TEMPO_CACHE_DIR)
    for row in records.iter_rows(named=True):
        for step in range(LABEL_TIMESTEPS):
            task = make_scan_task(row, f"no2_paths_t{step}", Path(TEMPO_DIR), cache_dir)
            tasks.setdefault(task.cache_key, task)
    return tasks


def _unique_weather_tasks(records: pl.DataFrame) -> dict[str, WeatherTask]:
    # Resolve each weather cache key once
    tasks: dict[str, WeatherTask] = {}
    cache_dir = Path(DATASET_WEATHER_CACHE_DIR)
    for row in records.iter_rows(named=True):
        for step in range(LABEL_TIMESTEPS):
            task = make_weather_task(row, f"weather_path_t{step}", Path(HRRR_DIR), cache_dir)
            tasks.setdefault(task.cache_key, task)
    return tasks


def _balanced_shards(tasks: list[TaskType], sizes: list[int], shard_count: int) -> list[list[TaskType]]:
    # Use longest-processing-time placement to balance estimated work
    shards: list[list[TaskType]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for task, size in sorted(zip(tasks, sizes, strict=True), key=lambda item: item[1], reverse=True):
        target = int(np.argmin(loads))
        shards[target].append(task)
        loads[target] += size
    return shards


def prepare_run(run_dir: Path, nodes: int, seed: int) -> None:
    """Prepare score candidates and targeted missing-cache manifests.

    Args:
        run_dir: Shared state directory for the pipeline run.
        nodes: Number of cache-array shards.
        seed: Deterministic candidate seed.
    """
    raw_records = pl.scan_parquet(FULL_DATA_PARQUET)
    facilities = raw_records.select("facilityId", "lat", "lon").unique(subset="facilityId").collect()
    aois = build_aois(facilities)
    membership = build_aoi_membership(aois, facilities)
    sequence_paths, sequence_count = sample_mapping_sequences(
        Path(TEMPO_AOI_MAPPING),
        set(aois[AOI_ID_COL].to_list()),
        seed,
        run_dir / "sequences",
    )
    hourly_nox = _calculate_hourly_aoi_nox(raw_records, membership)
    candidates, audit, labeled_count = select_labeled_candidates(
        sequence_paths,
        hourly_nox,
        run_dir / "active-nox-values",
    )
    candidates = candidates.join(aois.select(AOI_ID_COL, "lat", "lon"), on=AOI_ID_COL, how="inner")
    candidates = add_sequence_weather_paths(candidates, timesteps=LABEL_TIMESTEPS).join(
        _hotspot_table(raw_records, membership, aois), on=AOI_ID_COL, how="inner"
    )
    candidates.write_parquet(run_dir / "candidates.parquet")
    audit.write_csv(run_dir / "candidate_class_history_audit.csv")

    scan_tasks = _unique_scan_tasks(candidates)
    weather_tasks = _unique_weather_tasks(candidates)
    scan_inventory = _inventory_cache("TEMPO", Path(DATASET_TEMPO_CACHE_DIR))
    weather_inventory = _inventory_cache("weather", Path(DATASET_WEATHER_CACHE_DIR))
    missing_scans = [task for task in scan_tasks.values() if Path(task.cache_path).name not in scan_inventory]
    missing_weather = [task for task in weather_tasks.values() if Path(task.cache_path).name not in weather_inventory]
    scan_work = scan_batches(missing_scans)
    weather_work = weather_batches(missing_weather)
    scan_shards = _balanced_shards(scan_work, [len(task.scans) for task in scan_work], nodes)
    weather_shards = _balanced_shards(weather_work, [len(task.weather) for task in weather_work], nodes)
    for index in range(nodes):
        with (run_dir / f"cache-tasks-{index:03d}.pkl").open("wb") as handle:
            pickle.dump({"scan": scan_shards[index], "weather": weather_shards[index]}, handle)
    summary = {
        "aois": aois.height,
        "sequence_candidates": sequence_count,
        "finite_label_candidates": labeled_count,
        "viable_aois": audit.filter(pl.col("meets_three_class_floor")).height,
        "retained_candidates": candidates.height,
        "unique_scan_tasks": len(scan_tasks),
        "cached_scan_tasks": len(scan_tasks) - len(missing_scans),
        "missing_scan_tasks": len(missing_scans),
        "unique_weather_tasks": len(weather_tasks),
        "cached_weather_tasks": len(weather_tasks) - len(missing_weather),
        "missing_weather_tasks": len(missing_weather),
        "nodes": nodes,
    }
    _write_json_atomic(summary, run_dir / "prepare-summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def _failure_rows(cache_type: str, results: list[object]) -> list[dict[str, str]]:
    # Convert failed cache results into a stable table shape
    return [
        {
            "cache_type": cache_type,
            "cache_key": str(result.cache_key),
            "cache_path": str(result.cache_path),
            "error": str(result.error),
        }
        for result in results
        if result.error is not None
    ]


def run_cache_worker(run_dir: Path, workers: int, shard_index: int | None) -> None:
    """Generate one prepared shard of missing cache entries.

    Args:
        run_dir: Shared state directory for the pipeline run.
        workers: Process workers on this array task.
        shard_index: Explicit shard or the Slurm array index.
    """
    index = int(os.environ["SLURM_ARRAY_TASK_ID"]) if shard_index is None else shard_index
    with (run_dir / f"cache-tasks-{index:03d}.pkl").open("rb") as handle:
        tasks: dict[str, list[ScanBatchTask] | list[WeatherBatchTask]] = pickle.load(handle)
    failures: list[dict[str, str]] = []
    scan_work = tasks["scan"]
    for position, results in enumerate(bounded_parallel_map(process_scan_batch, scan_work, workers), start=1):
        failures.extend(_failure_rows("tempo", results))
        if position % PROGRESS_FILES == 0 or position == len(scan_work):
            print(f"TEMPO batches: {position:,}/{len(scan_work):,}", flush=True)
    weather_work = tasks["weather"]
    for position, results in enumerate(bounded_parallel_map(process_weather_batch, weather_work, workers), start=1):
        failures.extend(_failure_rows("weather", results))
        if position % PROGRESS_FILES == 0 or position == len(weather_work):
            print(f"Weather batches: {position:,}/{len(weather_work):,}", flush=True)
    failure_frame = pl.DataFrame(
        failures,
        schema={"cache_type": pl.String, "cache_key": pl.String, "cache_path": pl.String, "error": pl.String},
    )
    failure_frame.write_parquet(run_dir / f"cache-failures-{index:03d}.parquet")
    summary = {
        "shard_index": index,
        "scan_batches": len(scan_work),
        "weather_batches": len(weather_work),
        "failures": failure_frame.height,
    }
    _write_json_atomic(summary, run_dir / f"cache-summary-{index:03d}.json")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"Cache shard {index} recorded {len(failures)} failures")


def _weighted_mean(values: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> float | None:
    # Require adequate observed kernel support
    total_weight = float(weights.sum())
    valid_weight = float(weights[valid].sum())
    if total_weight <= 0 or valid_weight / total_weight < MIN_REGION_COVERAGE:
        return None
    return float(np.sum(values[valid] * weights[valid]) / valid_weight)


def _local_wind_angle(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    hotspot_row: int,
    hotspot_column: int,
) -> float | None:
    # Estimate source-local wind direction in radians from east
    rows, columns = np.indices(wind_u.shape)
    east = columns - hotspot_column
    north = hotspot_row - rows
    neighborhood = np.hypot(east, north) <= SOURCE_EXCLUSION_RADIUS_PIXELS
    valid = neighborhood & np.isfinite(wind_u) & np.isfinite(wind_v)
    if not valid.any():
        return np.nan
    local_u = float(np.median(wind_u[valid]))
    local_v = float(np.median(wind_v[valid]))
    if np.hypot(local_u, local_v) < MIN_WIND_SPEED_MPS:
        return None
    return float(np.arctan2(local_v, local_u))


def _masked_high_pass(standardized: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Remove broad background with mask-normalized Gaussian smoothing
    weights = gaussian_filter(valid.astype(np.float64), BACKGROUND_BLUR_SIGMA_PIXELS, mode="nearest")
    numerator = gaussian_filter(np.where(valid, standardized, 0.0), BACKGROUND_BLUR_SIGMA_PIXELS, mode="nearest")
    background_valid = valid & (weights >= MIN_REGION_COVERAGE)
    background = np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 0)
    return np.where(background_valid, standardized - background, np.nan), background_valid


def _direction_score(
    residual: np.ndarray,
    valid: np.ndarray,
    angle: float,
    hotspot_row: int,
    hotspot_column: int,
) -> tuple[float, float]:
    # Compare the downwind core with crosswind flanks
    rows, columns = np.indices(residual.shape)
    east = columns - hotspot_column
    north = hotspot_row - rows
    along_wind = east * np.cos(angle) + north * np.sin(angle)
    crosswind = -east * np.sin(angle) + north * np.cos(angle)
    downwind = (along_wind >= 0) & (along_wind <= CORRIDOR_LENGTH_PIXELS)
    along_weight = np.exp(-along_wind.clip(min=0) / CORRIDOR_LENGTH_PIXELS)
    core_weights = downwind * along_weight * np.exp(-0.5 * np.square(crosswind / PLUME_CROSSWIND_SIGMA_PIXELS))
    flank_weights = (
        downwind
        * along_weight
        * np.exp(-0.5 * np.square((np.abs(crosswind) - FLANK_CENTER_PIXELS) / FLANK_SIGMA_PIXELS))
    )
    core_response = _weighted_mean(residual, valid, core_weights)
    flank_response = _weighted_mean(residual, valid, flank_weights)
    if core_response is None or flank_response is None:
        return np.nan, np.nan
    distance = np.hypot(rows - hotspot_row, columns - hotspot_column)
    source_neighborhood = distance <= SOURCE_EXCLUSION_RADIUS_PIXELS
    background = residual[valid & ~source_neighborhood]
    if background.size == 0:
        return np.nan, np.nan
    background_median = float(np.median(background))
    background_mad = float(np.median(np.abs(background - background_median)))
    noise = max(BACKGROUND_MAD_MULTIPLIER * background_mad, NOISE_FLOOR_STANDARDIZED)
    signed_amplitude_snr = (core_response - flank_response) / noise
    raw_snr = max(signed_amplitude_snr, 0.0)
    if raw_snr == 0:
        return 0.0, signed_amplitude_snr
    positive = np.maximum(residual, 0.0)
    core_positive = _weighted_mean(positive, valid, core_weights)
    scene_positive = float(np.mean(positive[valid]))
    if core_positive is None or scene_positive <= 0:
        return 0.0, signed_amplitude_snr
    localization_ratio = core_positive / scene_positive
    localization = np.clip((localization_ratio - 1.0) / (LOCALIZATION_REFERENCE_RATIO - 1.0), 0.0, 1.0)
    threshold = max(0.5 * noise, NOISE_FLOOR_STANDARDIZED)
    detected, _ = label((residual > threshold) & valid, structure=np.ones((3, 3), dtype=np.int8))
    source_labels = np.unique(detected[source_neighborhood & valid])
    source_labels = source_labels[source_labels > 0]
    anchored = np.isin(detected, source_labels) if source_labels.size else np.zeros_like(valid)
    core_total = float(np.sum(positive[valid] * core_weights[valid]))
    anchored_total = float(np.sum(positive[valid & anchored] * core_weights[valid & anchored]))
    anchored_fraction = anchored_total / core_total if core_total > 0 else 0.0
    broad_fraction = float(np.mean(residual[valid] > threshold))
    morphology = np.sqrt(localization * anchored_fraction) * np.exp(-BROAD_SIGNAL_DECAY * broad_fraction)
    return raw_snr * morphology, signed_amplitude_snr * morphology


def _candidate_wind_angles(current: float | None, previous: float | None) -> tuple[float, ...]:
    # Search the accepted offsets around current and preceding winds
    bases = [angle for angle in (current, previous) if angle is not None]
    candidates = {
        round(float((base + np.deg2rad(offset)) % (2 * np.pi)), 8)
        for base in bases
        for offset in WIND_SEARCH_OFFSETS_DEGREES
    }
    return tuple(candidates)


def _timestep_score(
    no2: np.ndarray,
    wind_angles: tuple[float, ...],
    hotspot_row: int,
    hotspot_column: int,
) -> tuple[float, float, float]:
    # Return the strongest wind-constrained response
    if not wind_angles:
        return np.nan, np.nan, np.nan
    standardized = (no2 - NORMALIZATION_CENTER) / NORMALIZATION_SCALE
    valid = np.isfinite(standardized)
    residual, residual_valid = _masked_high_pass(standardized, valid)
    metrics = [_direction_score(residual, residual_valid, angle, hotspot_row, hotspot_column) for angle in wind_angles]
    scores = np.asarray([value[0] for value in metrics])
    if not np.isfinite(scores).any():
        return np.nan, np.nan, np.nan
    best = int(np.nanargmax(scores))
    return float(scores[best]), float(wind_angles[best]), float(metrics[best][1])


def _score_rasters(
    no2: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    hotspot_row: int,
    hotspot_column: int,
) -> dict[str, float | int]:
    # Score four aligned NO2 and wind rasters
    wind_angles = [
        _local_wind_angle(wind_u[index], wind_v[index], hotspot_row, hotspot_column) for index in range(LABEL_TIMESTEPS)
    ]
    metrics = [
        _timestep_score(
            no2[index],
            _candidate_wind_angles(wind_angles[index], wind_angles[index - 1] if index else None),
            hotspot_row,
            hotspot_column,
        )
        for index in range(LABEL_TIMESTEPS)
    ]
    timestep_scores = np.asarray([value[0] for value in metrics])
    finite = timestep_scores[np.isfinite(timestep_scores)]
    record_snr = float(np.median(finite)) if finite.size >= MIN_VALID_TIMESTEPS else np.nan
    return {
        "record_plume_snr": record_snr,
        "snr_valid_timesteps": int(finite.size),
        **{f"plume_snr_t{index}": float(metrics[index][0]) for index in range(LABEL_TIMESTEPS)},
        **{f"matched_angle_t{index}": float(metrics[index][1]) for index in range(LABEL_TIMESTEPS)},
        **{f"plume_amplitude_t{index}": float(metrics[index][2]) for index in range(LABEL_TIMESTEPS)},
    }


def _score_cache_task(task: tuple[str, tuple[str, ...], tuple[str, ...], int, int]) -> dict[str, object]:
    # Load one history and calculate plume metrics
    record_id, scan_paths, weather_paths, hotspot_row, hotspot_column = task
    try:
        no2_values = []
        for path in scan_paths:
            with np.load(path, allow_pickle=False) as scan:
                no2_values.append(np.asarray(scan["no2"], dtype=np.float64))
        weather = [extract_weather_cache(path) for path in weather_paths]
        no2 = np.stack(no2_values)
        blankness: dict[str, float] = {}
        for index, timestep in enumerate(no2):
            standardized = (timestep - NORMALIZATION_CENTER) / NORMALIZATION_SCALE
            valid = np.isfinite(standardized)
            residual, residual_valid = _masked_high_pass(standardized, valid)
            finite_residual = residual[residual_valid]
            blankness[f"valid_fraction_t{index}"] = float(np.mean(valid))
            blankness[f"residual_mad_t{index}"] = (
                float(np.median(np.abs(finite_residual - np.median(finite_residual))))
                if finite_residual.size
                else np.nan
            )
        metrics = _score_rasters(
            no2,
            np.stack([item[WIND_U_RASTER_NAME] for item in weather]),
            np.stack([item[WIND_V_RASTER_NAME] for item in weather]),
            hotspot_row,
            hotspot_column,
        )
        return {RASTER_PATH_COL: record_id, "cache_error": None, **metrics, **blankness}
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {RASTER_PATH_COL: record_id, "cache_error": str(error)}


def _resolve_cached_candidates(records: pl.DataFrame) -> pl.DataFrame:
    # Attach exact cache paths and reject incomplete histories
    scan_inventory = _inventory_cache("TEMPO", Path(DATASET_TEMPO_CACHE_DIR))
    weather_inventory = _inventory_cache("weather", Path(DATASET_WEATHER_CACHE_DIR))
    rows = []
    for index, row in enumerate(records.iter_rows(named=True), start=1):
        scans = tuple(
            make_scan_task(row, f"no2_paths_t{step}", Path(TEMPO_DIR), Path(DATASET_TEMPO_CACHE_DIR)).cache_path
            for step in range(LABEL_TIMESTEPS)
        )
        weather = tuple(
            make_weather_task(row, f"weather_path_t{step}", Path(HRRR_DIR), Path(DATASET_WEATHER_CACHE_DIR)).cache_path
            for step in range(LABEL_TIMESTEPS)
        )
        scans_complete = all(Path(path).name in scan_inventory for path in scans)
        weather_complete = all(Path(path).name in weather_inventory for path in weather)
        if not scans_complete or not weather_complete:
            raise FileNotFoundError(f"Prepared history lacks a cache entry: {row[RASTER_PATH_COL]}")
        rows.append({**row, "_scan_cache_paths": scans, "_weather_cache_paths": weather})
        if index % 10_000 == 0 or index == records.height:
            print(f"Cache resolution: {index:,}/{records.height:,}", flush=True)
    return pl.DataFrame(rows)


def _score_candidates(records: pl.DataFrame, workers: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Score cache-complete histories with bounded process concurrency
    tasks = [
        (
            str(row[RASTER_PATH_COL]),
            tuple(row["_scan_cache_paths"]),
            tuple(row["_weather_cache_paths"]),
            int(row["hotspot_row"]),
            int(row["hotspot_column"]),
        )
        for row in records.iter_rows(named=True)
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_score_cache_task, tasks, chunksize=32), start=1):
            results.append(result)
            if index % PROGRESS_RECORDS == 0 or index == len(tasks):
                print(f"Plume scoring: {index:,}/{len(tasks):,}", flush=True)
    frame = pl.DataFrame(results)
    failures = frame.filter(pl.col("cache_error").is_not_null()).select(RASTER_PATH_COL, "cache_error")
    metrics = frame.filter(pl.col("cache_error").is_null()).drop("cache_error")
    return metrics, failures


def _add_plume_response(records: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    # Apply label EMA timing to the four signed plume amplitudes
    plume_ema = pl.col("plume_amplitude_t0")
    previous_ema = plume_ema
    for index in range(1, LABEL_TIMESTEPS):
        interval_hours = (
            pl.col(f"timestep_time_t{index}") - pl.col(f"timestep_time_t{index - 1}")
        ).dt.total_seconds() / 3600
        retention = (-interval_hours / 2.0).exp()
        previous_ema = plume_ema
        plume_ema = retention * plume_ema + (1 - retention) * pl.col(f"plume_amplitude_t{index}")
    with_delta = records.with_columns((plume_ema - previous_ema).alias("plume_effective_delta")).filter(
        pl.col("plume_effective_delta").is_finite()
        & pl.col("record_plume_snr").is_finite()
        & pl.col(CLASS_COL).is_in(CLASS_NAMES)
    )
    lower = with_delta["plume_effective_delta"].quantile(0.25)
    upper = with_delta["plume_effective_delta"].quantile(0.75)
    if lower is None or upper is None:
        raise ValueError("Cannot derive a robust plume-delta scale")
    plume_delta_scale = max(float(upper - lower) / 1.349, MIN_PLUME_DELTA_SCALE)
    return with_delta.with_columns(
        (pl.col("plume_effective_delta") / plume_delta_scale).alias("plume_effective_delta_z")
    ), plume_delta_scale


def _calculate_scores(records: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Combine class-balanced detectability with directional response separation
    contrast_columns = []
    for index in range(LABEL_TIMESTEPS):
        noise = (BACKGROUND_MAD_MULTIPLIER * pl.col(f"residual_mad_t{index}")).clip(
            lower_bound=NOISE_FLOOR_STANDARDIZED
        )
        contrast_columns.append(
            (pl.col(f"plume_amplitude_t{index}").abs() * noise).alias(f"_absolute_contrast_t{index}")
        )
    scored = (
        records.with_columns(*contrast_columns)
        .with_columns(
            pl.concat_list([f"_absolute_contrast_t{index}" for index in range(LABEL_TIMESTEPS)])
            .list.median()
            .alias("record_absolute_contrast")
        )
        .with_columns(
            (
                pl.col("record_plume_snr").tanh()
                * (pl.col("record_absolute_contrast") / ABSOLUTE_CONTRAST_SCALE).tanh()
            ).alias("record_detectability")
        )
    )
    by_class = (
        scored.filter(pl.col("record_detectability").is_finite() & pl.col("plume_effective_delta_z").is_finite())
        .group_by(AOI_ID_COL, CLASS_COL)
        .agg(
            pl.col("record_detectability").mean().alias("class_detectability"),
            pl.col("plume_effective_delta_z").mean().alias("class_mean_plume_response"),
            pl.len().alias("finite_class_records"),
        )
    )
    eligible = (
        by_class.group_by(AOI_ID_COL)
        .agg(
            pl.col(CLASS_COL).n_unique().alias("finite_classes"),
            pl.col("finite_class_records").min().alias("minimum_finite_class_records"),
        )
        .filter(
            (pl.col("finite_classes") == len(CLASS_NAMES))
            & (pl.col("minimum_finite_class_records") >= DEFAULT_MINIMUM_PER_CLASS)
        )
        .select(AOI_ID_COL)
    )
    eligible_classes = by_class.join(eligible, on=AOI_ID_COL, how="inner")
    responses = eligible_classes.pivot(on=CLASS_COL, index=AOI_ID_COL, values="class_mean_plume_response").rename(
        {name: f"{name}_mean_plume_response" for name in CLASS_NAMES}
    )
    scores = (
        eligible_classes.group_by(AOI_ID_COL)
        .agg(
            pl.col("class_detectability").mean().alias("class_balanced_detectability"),
            pl.col("finite_class_records").sum().alias("finite_record_count"),
            pl.col("finite_class_records").min().alias("minimum_finite_class_records"),
        )
        .join(responses, on=AOI_ID_COL, how="inner")
        .with_columns(
            (pl.col("increase_mean_plume_response") - pl.col("decrease_mean_plume_response")).alias(
                "directional_response_separation"
            )
        )
        .with_columns(
            ((pl.col("class_balanced_detectability").rank("average") - 0.5) / pl.len()).alias(
                "detectability_percentile"
            ),
            ((pl.col("directional_response_separation").rank("average") - 0.5) / pl.len()).alias(
                "directional_separation_percentile"
            ),
        )
        .with_columns(
            (
                (1 - DIRECTION_SCORE_WEIGHT) * pl.col("detectability_percentile")
                + DIRECTION_SCORE_WEIGHT * pl.col("directional_separation_percentile")
            ).alias("final_aoi_score")
        )
        .sort("final_aoi_score", AOI_ID_COL, descending=[True, False])
    )
    return scored, scores


def _select_montage_records(records: pl.DataFrame, scores: pl.DataFrame, seed: int) -> pl.DataFrame:
    # Choose one deterministic history from 10 AOIs in each score quartile
    ranked = scores.with_columns(
        ((pl.col("final_aoi_score").rank("average") - 0.5) / pl.len()).alias("score_percentile")
    )
    joined = records.join(ranked.select(AOI_ID_COL, "final_aoi_score", "score_percentile"), on=AOI_ID_COL)
    selected = []
    for group, expression in (
        ("bottom quartile", pl.col("score_percentile") <= 0.25),
        ("top quartile", pl.col("score_percentile") >= 0.75),
    ):
        choices = (
            joined.filter(expression)
            .with_columns(pl.col(RASTER_PATH_COL).hash(seed=seed).alias("_montage_order"))
            .sort("_montage_order")
            .group_by(AOI_ID_COL, maintain_order=True)
            .head(1)
            .head(10)
            .with_columns(pl.lit(group).alias("score_group"))
        )
        if choices.height != 10:
            raise ValueError(f"Expected 10 montage histories in {group}, found {choices.height}")
        selected.append(choices)
    return pl.concat(selected, how="vertical")


def _load_no2_history(row: dict[str, object]) -> np.ndarray:
    # Load the four cache rasters selected for one montage history
    arrays = []
    for step in range(LABEL_TIMESTEPS):
        path = make_scan_task(row, f"no2_paths_t{step}", Path(TEMPO_DIR), Path(DATASET_TEMPO_CACHE_DIR)).cache_path
        with np.load(path, allow_pickle=False) as cache:
            arrays.append(np.asarray(cache["no2"], dtype=np.float64))
    return np.stack(arrays)


def _draw_sample_row(
    axes: np.ndarray,
    row: dict[str, object],
    column_offset: int,
) -> None:
    # Draw one four-raster history in its half of the montage row
    no2 = np.clip(
        ((_load_no2_history(row) - NORMALIZATION_CENTER) / NORMALIZATION_SCALE), -NORMALIZATION_CLIP, NORMALIZATION_CLIP
    )
    for step in range(LABEL_TIMESTEPS):
        axis: Axes = axes[column_offset + step]
        axis.imshow(no2[step], cmap="RdBu_r", vmin=-2, vmax=2, origin="upper")
        axis.scatter(row["hotspot_column"], row["hotspot_row"], marker="+", s=28, c="black", linewidths=1)
        axis.set_xticks([])
        axis.set_yticks([])
        if step == 0:
            axis.set_ylabel(
                f"AOI {row[AOI_ID_COL]}\n{row[CLASS_COL]}\nscore {row['final_aoi_score']:.3f}",
                fontsize=7,
            )


def write_montage(records: pl.DataFrame, scores: pl.DataFrame, output: Path, seed: int) -> Path:
    """Write paired low- and high-score four-timestep histories.

    Args:
        records: Scored histories with cache lookup fields.
        scores: Final AOI score table.
        output: Destination PNG path.
        seed: Deterministic history-selection seed.

    Returns:
        Written PNG path.
    """
    manifest = _select_montage_records(records, scores, seed)
    low = manifest.filter(pl.col("score_group") == "bottom quartile").sort("final_aoi_score")
    high = manifest.filter(pl.col("score_group") == "top quartile").sort("final_aoi_score", descending=True)
    figure, axes = plt.subplots(10, 8, figsize=(16, 22), constrained_layout=True)
    for index, (low_row, high_row) in enumerate(
        zip(low.iter_rows(named=True), high.iter_rows(named=True), strict=True)
    ):
        _draw_sample_row(axes[index], low_row, 0)
        _draw_sample_row(axes[index], high_row, 4)
    for column in range(4):
        axes[0, column].set_title(f"Bottom quartile t{column}", fontsize=9)
        axes[0, column + 4].set_title(f"Top quartile t{column}", fontsize=9)
    colorbar = figure.colorbar(axes[0, 0].images[0], ax=axes, orientation="horizontal", fraction=0.012, pad=0.015)
    colorbar.set_label("Robust standardized NO₂")
    figure.suptitle("AOI heuristic examples by score quartile", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    manifest.select("score_group", AOI_ID_COL, CLASS_COL, "final_aoi_score", RASTER_PATH_COL).write_csv(
        output.with_suffix(".csv")
    )
    return output


def _email_montage(path: Path, run_id: str) -> None:
    # Send the requested artifact and inspect Postfix only when its log is readable
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Montage does not exist: {path}")
    subject = f"AOI heuristic samples {run_id}"
    log_path = Path("/var/log/maillog")
    can_read_log = log_path.is_file() and os.access(log_path, os.R_OK)
    offset = log_path.stat().st_size if can_read_log else 0
    message = "The full AOI heuristic scoring montage is attached.\n"
    subprocess.run(
        ["mailx", "-s", subject, "-a", str(path), EMAIL],
        input=message,
        text=True,
        check=True,
    )
    if not can_read_log:
        print(f"mailx accepted the montage for {EMAIL}; Postfix log is not readable", flush=True)
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if log_path.is_file():
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                recent = handle.read()
            if EMAIL in recent and "status=sent" in recent:
                return
        time.sleep(2)
    raise RuntimeError(f"Postfix did not confirm delivery to {EMAIL}")


def _resume_completed_publication(output_dir: Path, run_id: str) -> bool:
    # Recover publication after a post-scoring notification failure
    mapping_path = output_dir / "aoi_scores.json"
    summary_path = output_dir / "summary.json"
    montage_path = output_dir / "score-quartile-samples.png"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (mapping_path, summary_path, montage_path)):
        return False
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _email_montage(montage_path, run_id)
    _write_json_atomic(mapping, Path(AOI_SCORE_JSON))
    print(json.dumps({**summary, "resumed_publication": True}, indent=2), flush=True)
    return True


def finalize_run(run_dir: Path, workers: int, seed: int) -> None:
    """Score all prepared candidates and publish the final mapping.

    Args:
        run_dir: Shared state directory for the pipeline run.
        workers: Cache-scoring process count.
        seed: Deterministic montage seed.
    """
    configuration = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    output_dir = Path(configuration["output_dir"])
    if _resume_completed_publication(output_dir, str(configuration["run_id"])):
        return
    candidates = pl.read_parquet(run_dir / "candidates.parquet")
    resolved = _resolve_cached_candidates(candidates)
    metrics, failures = _score_candidates(resolved, workers)
    if failures.height:
        failures.write_csv(output_dir / "score-failures.csv")
        raise RuntimeError(f"Plume scoring failed for {failures.height} histories")
    scored_records = resolved.drop("_scan_cache_paths", "_weather_cache_paths").join(
        metrics, on=RASTER_PATH_COL, how="inner"
    )
    scored_records, plume_delta_scale = _add_plume_response(scored_records)
    record_scores, aoi_scores = _calculate_scores(scored_records)
    if aoi_scores.is_empty():
        raise ValueError("Full scoring produced no eligible AOIs")
    output_dir.mkdir(parents=True, exist_ok=True)
    record_scores.write_parquet(output_dir / "record_scores.parquet")
    aoi_scores.write_csv(output_dir / "aoi_scores.csv")
    mapping = {
        str(aoi_id): float(score) for aoi_id, score in aoi_scores.select(AOI_ID_COL, "final_aoi_score").iter_rows()
    }
    _write_json_atomic(mapping, output_dir / "aoi_scores.json")
    montage = write_montage(record_scores, aoi_scores, output_dir / "score-quartile-samples.png", seed)
    summary = {
        "run_id": configuration["run_id"],
        "candidate_histories": candidates.height,
        "scored_histories": record_scores.height,
        "scored_aois": aoi_scores.height,
        "score_map": str(AOI_SCORE_JSON),
        "montage": str(montage),
        "absolute_contrast_scale": ABSOLUTE_CONTRAST_SCALE,
        "plume_delta_scale": plume_delta_scale,
        "direction_score_weight": DIRECTION_SCORE_WEIGHT,
        "minimum_histories_per_class": DEFAULT_MINIMUM_PER_CLASS,
        "maximum_histories_per_class": DEFAULT_CANDIDATES_PER_CLASS,
    }
    _write_json_atomic(summary, output_dir / "summary.json")
    _email_montage(montage, str(configuration["run_id"]))
    _write_json_atomic(mapping, Path(AOI_SCORE_JSON))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    """Dispatch the requested orchestration or worker phase."""
    args = parse_args()
    if args.command == "submit":
        submit_pipeline(args.nodes, args.workers_per_node, args.seed)
    elif args.command == "prepare":
        prepare_run(args.run_dir, args.nodes, args.seed)
    elif args.command == "cache-worker":
        run_cache_worker(args.run_dir, args.workers, args.shard_index)
    elif args.command == "finalize":
        finalize_run(args.run_dir, args.workers, args.seed)


if __name__ == "__main__":
    main()
