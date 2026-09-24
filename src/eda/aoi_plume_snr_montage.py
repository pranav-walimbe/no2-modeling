"""Compare current-label sequences from low- and high-plume-SNR AOIs."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from preprocessing.stratify_utils import AOI_ID_COL
from scipy.ndimage import gaussian_filter, label

from config import (
    DATASET_DF,
    DATASET_DIR,
    EMA_DECAY_TIMESCALE_HOURS,
    FULL_DATA_PARQUET,
    IMG_SIZE,
    MODEL_IMAGE_CLIP_ABS,
    NUM_CORES,
    VIS_DIR,
)
from eda.aoi_nox_score_montage import (
    CLASS_COL,
    DELTA_COL,
    RASTER_PATH_COL,
    add_current_labels,
    build_membership,
    calculate_hourly_aoi_nox,
    load_prior_dataset_records,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIMESTEPS = 5
LABEL_TIMESTEPS = 4
CLASS_ORDER = ("decrease", "increase")
STRATUM_ORDER = ("low", "high")
SAMPLES_PER_CLASS_STRATUM = 5
MAX_SCORE_RECORDS_PER_AOI = 32
MIN_AOI_RECORDS = 8
MIN_CLASS_RECORDS = 3
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
DETECTABLE_SNR_THRESHOLD = 0.5
AGREEMENT_LOGISTIC_SLOPE = 2.0
MIN_PLUME_DELTA_SCALE = 0.10
LOCALIZATION_REFERENCE_RATIO = 3.0
BROAD_SIGNAL_DECAY = 3.0
LOW_PERCENTILE_MAX = 0.25
HIGH_PERCENTILE_MIN = 0.75
NORMALIZATION_CENTER = 1868138303979520.0
NORMALIZATION_SCALE = 1199722224989936.2
DEFAULT_SEED = 20260923
FIGURE_DPI = 180
SNR_COL = "record_plume_snr"
PLUME_DELTA_COL = "plume_effective_delta"
PLUME_DELTA_Z_COL = "plume_effective_delta_z"
DIRECTION_AGREEMENT_COL = "direction_agreement"
COMPATIBILITY_COL = "label_aligned_record_score"
AOI_SCORE_COL = "aoi_label_aligned_snr_score"
AOI_PERCENTILE_COL = "aoi_label_aligned_snr_percentile"
SNR_STRATUM_COL = "snr_stratum"


@dataclass(frozen=True)
class PlumeSample:
    """One selected sequence and its SNR metadata."""

    snr_stratum: str
    delta_category: str
    sample_index: int
    aoi_id: int
    aoi_score: float
    percentile: float
    record_snr: float
    plume_delta_z: float
    direction_agreement: float
    delta_nox: float
    hotspot_row: int
    hotspot_column: int
    no2: np.ndarray
    no2_mask: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse plume-SNR montage options."""
    job_id = os.getenv("SLURM_JOB_ID", "latest")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    parser.add_argument("--workers", type=int, default=NUM_CORES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-label-aligned-snr-montage-{job_id}.png",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-label-aligned-snr-montage-{job_id}.csv",
    )
    parser.add_argument(
        "--aoi-score-output",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-label-aligned-snr-scores-{job_id}.csv",
    )
    return parser.parse_args()


def _weighted_mean(values: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> float | None:
    # Return a weighted mean only when enough kernel weight is observed
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
    # Estimate one source-local wind direction in radians from east
    rows, columns = np.indices(wind_u.shape)
    east = columns - hotspot_column
    north = hotspot_row - rows
    source_neighborhood = np.hypot(east, north) <= SOURCE_EXCLUSION_RADIUS_PIXELS
    local_valid = source_neighborhood & np.isfinite(wind_u) & np.isfinite(wind_v)
    if not local_valid.any():
        return np.nan
    local_u = float(np.median(wind_u[local_valid]))
    local_v = float(np.median(wind_v[local_valid]))
    wind_speed = float(np.hypot(local_u, local_v))
    if wind_speed < MIN_WIND_SPEED_MPS:
        return None
    return float(np.arctan2(local_v, local_u))


def _masked_high_pass(standardized: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Remove broad spatial background with mask-normalized Gaussian smoothing
    weights = gaussian_filter(valid.astype(np.float64), BACKGROUND_BLUR_SIGMA_PIXELS, mode="nearest")
    numerator = gaussian_filter(
        np.where(valid, standardized, 0.0),
        BACKGROUND_BLUR_SIGMA_PIXELS,
        mode="nearest",
    )
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
    # Apply a source-anchored plume kernel and crosswind flank control
    rows, columns = np.indices(residual.shape)
    east = columns - hotspot_column
    north = hotspot_row - rows
    unit_u = np.cos(angle)
    unit_v = np.sin(angle)
    along_wind = east * unit_u + north * unit_v
    crosswind = -east * unit_v + north * unit_u
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
    background_values = residual[valid & ~source_neighborhood]
    if background_values.size == 0:
        return np.nan, np.nan
    background_median = float(np.median(background_values))
    background_mad = float(np.median(np.abs(background_values - background_median)))
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
    localization = np.clip(
        (localization_ratio - 1.0) / (LOCALIZATION_REFERENCE_RATIO - 1.0),
        0.0,
        1.0,
    )

    detection_threshold = max(0.5 * noise, NOISE_FLOOR_STANDARDIZED)
    detected, _ = label((residual > detection_threshold) & valid, structure=np.ones((3, 3), dtype=np.int8))
    source_labels = np.unique(detected[source_neighborhood & valid])
    source_labels = source_labels[source_labels > 0]
    anchored = np.isin(detected, source_labels) if source_labels.size else np.zeros_like(valid)
    core_positive_total = float(np.sum(positive[valid] * core_weights[valid]))
    anchored_positive_total = float(np.sum(positive[valid & anchored] * core_weights[valid & anchored]))
    anchored_fraction = anchored_positive_total / core_positive_total if core_positive_total > 0 else 0.0
    broad_fraction = float(np.mean(residual[valid] > detection_threshold))
    broad_penalty = float(np.exp(-BROAD_SIGNAL_DECAY * broad_fraction))
    morphology_weight = np.sqrt(localization * anchored_fraction) * broad_penalty
    return raw_snr * morphology_weight, signed_amplitude_snr * morphology_weight


def _candidate_wind_angles(current: float | None, previous: float | None) -> tuple[float, ...]:
    # Search near current and preceding winds while removing duplicate angles
    bases = [angle for angle in (current, previous) if angle is not None]
    candidates = {
        round(float((base + np.deg2rad(offset)) % (2 * np.pi)), 8)
        for base in bases
        for offset in WIND_SEARCH_OFFSETS_DEGREES
    }
    return tuple(candidates)


def _timestep_snr(
    no2: np.ndarray,
    no2_mask: np.ndarray,
    wind_angles: tuple[float, ...],
    hotspot_row: int,
    hotspot_column: int,
) -> tuple[float, float, float]:
    # Return the strongest constrained matched-filter response and direction
    if not wind_angles:
        return np.nan, np.nan, np.nan
    standardized = (no2 - NORMALIZATION_CENTER) / NORMALIZATION_SCALE
    valid = no2_mask.astype(bool) & np.isfinite(standardized)
    residual, residual_valid = _masked_high_pass(standardized, valid)
    direction_metrics = [
        _direction_score(residual, residual_valid, angle, hotspot_row, hotspot_column) for angle in wind_angles
    ]
    scores = np.asarray([metrics[0] for metrics in direction_metrics])
    amplitudes = np.asarray([metrics[1] for metrics in direction_metrics])
    if not np.isfinite(scores).any():
        return np.nan, np.nan, np.nan
    best_index = int(np.nanargmax(scores))
    return float(scores[best_index]), float(wind_angles[best_index]), float(amplitudes[best_index])


def _score_raster_task(task: tuple[str, int, int, str]) -> dict[str, object]:
    # Load one bundle and summarize its four pre-label plume SNR values
    path_text, hotspot_row, hotspot_column, stored_path = task
    with np.load(path_text, allow_pickle=False) as bundle:
        no2 = np.asarray(bundle["no2"], dtype=np.float64)
        no2_mask = np.asarray(bundle["no2_mask"], dtype=bool)
        wind_u = np.asarray(bundle["wind_u_80m_mps"], dtype=np.float64)
        wind_v = np.asarray(bundle["wind_v_80m_mps"], dtype=np.float64)
    wind_angles = [
        _local_wind_angle(wind_u[index], wind_v[index], hotspot_row, hotspot_column) for index in range(LABEL_TIMESTEPS)
    ]
    score_and_direction = [
        _timestep_snr(
            no2[index],
            no2_mask[index],
            _candidate_wind_angles(wind_angles[index], wind_angles[index - 1] if index > 0 else None),
            hotspot_row,
            hotspot_column,
        )
        for index in range(LABEL_TIMESTEPS)
    ]
    timestep_scores = np.asarray([value[0] for value in score_and_direction])
    directions = np.asarray([value[1] for value in score_and_direction])
    amplitudes = np.asarray([value[2] for value in score_and_direction])
    finite = timestep_scores[np.isfinite(timestep_scores)]
    record_snr = float(np.median(finite)) if finite.size >= MIN_VALID_TIMESTEPS else np.nan
    return {
        RASTER_PATH_COL: stored_path,
        SNR_COL: record_snr,
        "snr_valid_timesteps": int(finite.size),
        **{f"plume_snr_t{index}": float(timestep_scores[index]) for index in range(LABEL_TIMESTEPS)},
        **{f"matched_angle_t{index}": float(directions[index]) for index in range(LABEL_TIMESTEPS)},
        **{f"plume_amplitude_t{index}": float(amplitudes[index]) for index in range(LABEL_TIMESTEPS)},
    }


def sample_score_records(records: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Select a bounded deterministic history sample within each AOI.

    Args:
        records: Full prior-dataset record pool.
        seed: Hash seed controlling the history sample.

    Returns:
        At most 32 records per AOI.
    """
    return (
        records.with_columns(pl.col(RASTER_PATH_COL).hash(seed=seed).alias("_score_sample_order"))
        .sort(AOI_ID_COL, "_score_sample_order")
        .group_by(AOI_ID_COL, maintain_order=True)
        .head(MAX_SCORE_RECORDS_PER_AOI)
        .drop("_score_sample_order")
    )


def score_sampled_rasters(
    records: pl.DataFrame,
    dataset_dir: Path,
    workers: int,
) -> pl.DataFrame:
    """Calculate record plume SNRs using parallel raster reads.

    Args:
        records: Sampled records carrying paths and hotspot coordinates.
        dataset_dir: Root used to resolve stored raster paths.
        workers: Number of parallel raster readers.

    Returns:
        Per-record SNR diagnostics keyed by raster path.
    """
    if workers <= 0:
        raise ValueError("workers must be positive")
    tasks = []
    for row in records.select(RASTER_PATH_COL, "hotspot_row", "hotspot_column").iter_rows(named=True):
        stored_path = str(row[RASTER_PATH_COL])
        path = Path(stored_path)
        absolute_path = path if path.is_absolute() else dataset_dir / path
        tasks.append((str(absolute_path), int(row["hotspot_row"]), int(row["hotspot_column"]), stored_path))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_score_raster_task, tasks, chunksize=32), start=1):
            results.append(result)
            if index % 1_000 == 0 or index == len(tasks):
                print(f"Plume-SNR scan: {index:,}/{len(tasks):,} bundles", flush=True)
    return pl.DataFrame(results)


def add_label_compatibility(records: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Add EMA-aligned plume changes and soft label agreement.

    Args:
        records: Scored records carrying current labels and plume amplitudes.

    Returns:
        Records with compatibility fields and their robust plume-delta scale.
    """
    plume_ema = pl.col("plume_amplitude_t0")
    previous_plume_ema = plume_ema
    for index in range(1, LABEL_TIMESTEPS):
        interval_hours = (
            pl.col(f"timestep_time_t{index}") - pl.col(f"timestep_time_t{index - 1}")
        ).dt.total_seconds() / 3600
        retention = (-interval_hours / EMA_DECAY_TIMESCALE_HOURS).exp()
        previous_plume_ema = plume_ema
        plume_ema = retention * plume_ema + (1 - retention) * pl.col(f"plume_amplitude_t{index}")
    with_delta = records.with_columns((plume_ema - previous_plume_ema).alias(PLUME_DELTA_COL)).filter(
        pl.col(PLUME_DELTA_COL).is_finite() & pl.col(SNR_COL).is_finite() & pl.col(CLASS_COL).is_in(CLASS_ORDER)
    )
    lower, upper = with_delta[PLUME_DELTA_COL].quantile(0.25), with_delta[PLUME_DELTA_COL].quantile(0.75)
    if lower is None or upper is None:
        raise ValueError("Cannot derive a robust plume-delta scale")
    plume_delta_scale = max(float(upper - lower) / 1.349, MIN_PLUME_DELTA_SCALE)
    label_sign = pl.when(pl.col(CLASS_COL) == "increase").then(pl.lit(1.0)).otherwise(pl.lit(-1.0))
    compatible = with_delta.with_columns(
        (pl.col(PLUME_DELTA_COL) / plume_delta_scale).alias(PLUME_DELTA_Z_COL),
        label_sign.alias("_label_sign"),
    ).with_columns(
        (1 / (1 + (-AGREEMENT_LOGISTIC_SLOPE * pl.col("_label_sign") * pl.col(PLUME_DELTA_Z_COL)).exp())).alias(
            DIRECTION_AGREEMENT_COL
        ),
        (pl.col("_label_sign") * pl.col(PLUME_DELTA_Z_COL) > 0).alias("direction_correct"),
    )
    return compatible.with_columns(
        (pl.col(SNR_COL) * pl.col(DIRECTION_AGREEMENT_COL)).alias(COMPATIBILITY_COL)
    ), plume_delta_scale


def aggregate_aoi_scores(records: pl.DataFrame) -> pl.DataFrame:
    """Aggregate class-balanced label compatibility into AOI scores.

    Args:
        records: Sampled records carrying finite compatibility fields.

    Returns:
        AOI scores with empirical percentile ranks and extreme strata.
    """
    by_class = (
        records.group_by(AOI_ID_COL, CLASS_COL)
        .agg(
            pl.col(COMPATIBILITY_COL).quantile(0.75, interpolation="linear").alias("class_compatibility_q75"),
            pl.col("direction_correct").mean().alias("class_direction_correct_fraction"),
            pl.len().alias("class_record_count"),
        )
        .filter(pl.col("class_record_count") >= MIN_CLASS_RECORDS)
    )
    class_tables = {}
    for class_name in CLASS_ORDER:
        class_tables[class_name] = by_class.filter(pl.col(CLASS_COL) == class_name).select(
            AOI_ID_COL,
            *[
                pl.col(column).alias(f"{class_name}_{column}")
                for column in (
                    "class_compatibility_q75",
                    "class_direction_correct_fraction",
                    "class_record_count",
                )
            ],
        )
    scores = (
        class_tables["increase"]
        .join(class_tables["decrease"], on=AOI_ID_COL, how="inner")
        .with_columns(
            (
                (pl.col("increase_class_compatibility_q75") * pl.col("decrease_class_compatibility_q75")).sqrt()
                * (
                    pl.col("increase_class_direction_correct_fraction")
                    * pl.col("decrease_class_direction_correct_fraction")
                ).sqrt()
            ).alias(AOI_SCORE_COL)
        )
        .sort(AOI_SCORE_COL, AOI_ID_COL)
        .with_row_index("_snr_rank", offset=1)
        .with_columns((pl.col("_snr_rank") / pl.len()).alias(AOI_PERCENTILE_COL))
        .with_columns(
            pl.when(pl.col(AOI_PERCENTILE_COL) <= LOW_PERCENTILE_MAX)
            .then(pl.lit("low"))
            .when(pl.col(AOI_PERCENTILE_COL) > HIGH_PERCENTILE_MIN)
            .then(pl.lit("high"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias(SNR_STRATUM_COL)
        )
        .drop("_snr_rank")
    )
    return scores


def select_manifest(
    records: pl.DataFrame,
    aoi_scores: pl.DataFrame,
    seed: int,
) -> pl.DataFrame:
    """Select five representative records per class and SNR extreme.

    Args:
        records: Scored records carrying current-label compatibility.
        aoi_scores: AOI-level plume SNR scores and strata.
        seed: Hash seed controlling equivalent AOI order.

    Returns:
        Twenty selected records ordered for plotting.
    """
    candidates = (
        records.join(aoi_scores, on=AOI_ID_COL, how="inner")
        .filter(pl.col(SNR_STRATUM_COL).is_not_null() & pl.col(CLASS_COL).is_in(CLASS_ORDER))
        .with_columns(
            pl.when(pl.col(CLASS_COL) == "increase")
            .then(pl.col("increase_class_compatibility_q75"))
            .otherwise(pl.col("decrease_class_compatibility_q75"))
            .alias("_target_compatibility_q75"),
            pl.col(AOI_ID_COL).hash(seed=seed).alias("_aoi_order"),
        )
        .with_columns(
            (pl.col(COMPATIBILITY_COL) - pl.col("_target_compatibility_q75")).abs().alias("_representativeness")
        )
        .sort("_representativeness", RASTER_PATH_COL)
        .unique(subset=[AOI_ID_COL, CLASS_COL], keep="first", maintain_order=True)
    )
    selected = []
    for stratum_index, stratum in enumerate(STRATUM_ORDER):
        for class_index, class_name in enumerate(CLASS_ORDER):
            cell = (
                candidates.filter((pl.col(SNR_STRATUM_COL) == stratum) & (pl.col(CLASS_COL) == class_name))
                .sort("_aoi_order", AOI_ID_COL)
                .head(SAMPLES_PER_CLASS_STRATUM)
            )
            if cell.height < SAMPLES_PER_CLASS_STRATUM:
                raise ValueError(
                    f"Need {SAMPLES_PER_CLASS_STRATUM} samples for {stratum}/{class_name}, found {cell.height}"
                )
            selected.append(
                cell.with_columns(
                    pl.lit(stratum_index).alias("_stratum_order"),
                    pl.lit(class_index).alias("_class_order"),
                    pl.int_range(0, pl.len()).alias("sample_index"),
                )
            )
    return pl.concat(selected).sort("_class_order", "sample_index", "_stratum_order")


def _load_plume_sample(row: dict[str, object], dataset_dir: Path) -> PlumeSample:
    # Load physical raster arrays for one selected record
    stored_path = Path(str(row[RASTER_PATH_COL]))
    path = stored_path if stored_path.is_absolute() else dataset_dir / stored_path
    with np.load(path, allow_pickle=False) as bundle:
        no2 = np.asarray(bundle["no2"], dtype=np.float64)
        no2_mask = np.asarray(bundle["no2_mask"], dtype=bool)
        wind_u = np.asarray(bundle["wind_u_80m_mps"], dtype=np.float64)
        wind_v = np.asarray(bundle["wind_v_80m_mps"], dtype=np.float64)
    if no2.shape != (TIMESTEPS, IMG_SIZE, IMG_SIZE) or no2_mask.shape != no2.shape:
        raise ValueError(f"Unexpected raster shape in {path}: {no2.shape}")
    return PlumeSample(
        snr_stratum=str(row[SNR_STRATUM_COL]),
        delta_category=str(row[CLASS_COL]),
        sample_index=int(row["sample_index"]),
        aoi_id=int(row[AOI_ID_COL]),
        aoi_score=float(row[AOI_SCORE_COL]),
        percentile=float(row[AOI_PERCENTILE_COL]),
        record_snr=float(row[SNR_COL]),
        plume_delta_z=float(row[PLUME_DELTA_Z_COL]),
        direction_agreement=float(row[DIRECTION_AGREEMENT_COL]),
        delta_nox=float(row[DELTA_COL]),
        hotspot_row=int(row["hotspot_row"]),
        hotspot_column=int(row["hotspot_column"]),
        no2=no2,
        no2_mask=no2_mask,
        wind_u=wind_u,
        wind_v=wind_v,
    )


def _draw_wind_arrow(axis: plt.Axes, sample: PlumeSample, timestep: int) -> None:
    # Draw the local downwind direction from the source hotspot
    rows, columns = np.indices((IMG_SIZE, IMG_SIZE))
    radius = np.hypot(columns - sample.hotspot_column, sample.hotspot_row - rows)
    valid = (
        (radius <= SOURCE_EXCLUSION_RADIUS_PIXELS)
        & np.isfinite(sample.wind_u[timestep])
        & np.isfinite(sample.wind_v[timestep])
    )
    if not valid.any():
        return
    local_u = float(np.median(sample.wind_u[timestep][valid]))
    local_v = float(np.median(sample.wind_v[timestep][valid]))
    speed = float(np.hypot(local_u, local_v))
    if speed < MIN_WIND_SPEED_MPS:
        return
    arrow_length = 4.0
    axis.annotate(
        "",
        xy=(
            sample.hotspot_column + arrow_length * local_u / speed,
            sample.hotspot_row - arrow_length * local_v / speed,
        ),
        xytext=(sample.hotspot_column, sample.hotspot_row),
        arrowprops={"arrowstyle": "->", "color": "#00ffff", "linewidth": 0.9},
    )


def write_montage(manifest: pl.DataFrame, dataset_dir: Path, output_path: Path) -> Path:
    """Write the low- versus high-AOI-plume-SNR sequence montage.

    Args:
        manifest: Twenty selected records.
        dataset_dir: Root used to resolve raster bundle paths.
        output_path: PNG destination.

    Returns:
        Saved PNG path.
    """
    samples = [_load_plume_sample(row, dataset_dir) for row in manifest.to_dicts()]
    lookup = {(sample.delta_category, sample.snr_stratum, sample.sample_index): sample for sample in samples}
    row_count = len(CLASS_ORDER) * SAMPLES_PER_CLASS_STRATUM
    column_count = len(STRATUM_ORDER) * TIMESTEPS
    figure, axes = plt.subplots(row_count, column_count, figsize=(20, 16), squeeze=False)
    color_map = matplotlib.colormaps["RdBu_r"].copy()
    color_map.set_bad("#d7d7d7")
    image = None
    for class_index, class_name in enumerate(CLASS_ORDER):
        for sample_index in range(SAMPLES_PER_CLASS_STRATUM):
            row_index = class_index * SAMPLES_PER_CLASS_STRATUM + sample_index
            for stratum_index, stratum in enumerate(STRATUM_ORDER):
                sample = lookup[(class_name, stratum, sample_index)]
                standardized = np.clip(
                    (sample.no2 - NORMALIZATION_CENTER) / NORMALIZATION_SCALE,
                    -MODEL_IMAGE_CLIP_ABS,
                    MODEL_IMAGE_CLIP_ABS,
                )
                standardized = np.where(sample.no2_mask, standardized, np.nan)
                for timestep in range(TIMESTEPS):
                    column_index = stratum_index * TIMESTEPS + timestep
                    axis = axes[row_index, column_index]
                    image = axis.imshow(
                        standardized[timestep], cmap=color_map, vmin=-3, vmax=3, interpolation="nearest"
                    )
                    axis.scatter(
                        sample.hotspot_column,
                        sample.hotspot_row,
                        marker="+",
                        color="#00ffff",
                        s=18,
                        linewidths=0.8,
                    )
                    _draw_wind_arrow(axis, sample, timestep)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row_index == 0:
                        suffix = " (post-label)" if timestep == TIMESTEPS - 1 else ""
                        axis.set_title(f"{stratum.upper()}  t{timestep}{suffix}", fontsize=8)
                    is_outer_label = (stratum_index == 0 and timestep == 0) or (
                        stratum_index == len(STRATUM_ORDER) - 1 and timestep == TIMESTEPS - 1
                    )
                    if is_outer_label:
                        if stratum_index == len(STRATUM_ORDER) - 1:
                            axis.yaxis.set_label_position("right")
                        axis.set_ylabel(
                            f"AOI {sample.aoi_id}\nscore {sample.aoi_score:.2f}  "
                            f"p{100 * sample.percentile:.0f}\nrecord SNR {sample.record_snr:.2f}  "
                            f"agree {sample.direction_agreement:.2f}\n"
                            f"plume dz {sample.plume_delta_z:+.2f}  delta {sample.delta_nox:+.0f}",
                            fontsize=6.5,
                            rotation=0,
                            ha="right" if stratum_index == 0 else "left",
                            va="center",
                            labelpad=42,
                        )
            if sample_index == 0:
                axes[row_index, 0].text(
                    -1.12,
                    0.5,
                    class_name.upper(),
                    transform=axes[row_index, 0].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    weight="bold",
                )
    if image is None:
        raise RuntimeError("No raster images were plotted")
    figure.suptitle(
        "Current-label NO2 sequences from low- and high label-aligned plume-SNR AOIs\n"
        "Class-balanced bottom versus top AOI-score quartile; 5 decrease and 5 increase samples per stratum; "
        "cyan arrows show local downwind direction",
        fontsize=14,
        y=0.998,
    )
    figure.subplots_adjust(left=0.085, right=0.915, top=0.955, bottom=0.055, hspace=0.09, wspace=0.04)
    color_axis = figure.add_axes((0.36, 0.018, 0.28, 0.012))
    figure.colorbar(
        image,
        cax=color_axis,
        orientation="horizontal",
        label="NO2 robust z-score from the fixed 10,000-bundle global sample",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    """Calculate label-aligned AOI plume SNR and write a balanced montage."""
    args = parse_args()
    records = load_prior_dataset_records(args.dataframe_dir)
    score_sample = sample_score_records(records, args.seed)
    print(
        f"Sampled {score_sample.height:,} records from {score_sample[AOI_ID_COL].n_unique():,} AOIs for SNR scoring",
        flush=True,
    )
    record_scores = score_sampled_rasters(score_sample, args.dataset_dir, args.workers)

    raw_columns = [
        "facilityId",
        "unitId",
        "lat",
        "lon",
        "emissions_hour_utc",
        "opTime",
        "noxMass",
        "noxMassMeasureFlg",
    ]
    raw_records = pl.scan_parquet(FULL_DATA_PARQUET).select(raw_columns)
    membership = build_membership(records, raw_records)
    labeled = add_current_labels(records, calculate_hourly_aoi_nox(raw_records, membership))
    scored_labeled = labeled.join(record_scores, on=RASTER_PATH_COL, how="inner")
    compatible, plume_delta_scale = add_label_compatibility(scored_labeled)
    print(f"Robust plume-delta scale: {plume_delta_scale:.4f}", flush=True)
    aoi_scores = aggregate_aoi_scores(compatible)
    print(
        f"Scored {aoi_scores.height:,} AOIs with at least {MIN_CLASS_RECORDS} records in each direction",
        flush=True,
    )
    manifest = select_manifest(compatible, aoi_scores, args.seed)
    counts = manifest.group_by(SNR_STRATUM_COL, CLASS_COL).len().sort(SNR_STRATUM_COL, CLASS_COL)
    print(f"Selected manifest:\n{counts}", flush=True)

    for output in (args.output, args.manifest_output, args.aoi_score_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.manifest_output)
    aoi_scores.write_csv(args.aoi_score_output)
    write_montage(manifest, args.dataset_dir, args.output)
    print(f"Saved manifest to {args.manifest_output}", flush=True)
    print(f"Saved AOI scores to {args.aoi_score_output}", flush=True)
    print(f"Saved montage to {args.output}", flush=True)


if __name__ == "__main__":
    main()
