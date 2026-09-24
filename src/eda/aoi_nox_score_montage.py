"""Compare standardized NO2 sequences from low- and high-NOx-score AOIs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    add_ema_targets,
    add_projected_coordinates,
    add_timestep_nox,
    build_aoi_membership,
    build_aoi_spatial_frame,
    filter_usable_nox_measurements,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

from config import (
    DATASET_DF,
    DATASET_DIR,
    EMA_DECAY_TIMESCALE_HOURS,
    EMA_HISTORY_TIMESTEPS,
    FULL_DATA_PARQUET,
    IMG_SIZE,
    LABEL_TIMESTEP_INDEX,
    MODEL_IMAGE_CLIP_ABS,
    STRATIFICATION_EMA_CHANGE_THRESHOLD,
    VIS_DIR,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_ORDER = ("decrease", "steady", "increase")
HALF_ORDER = ("bottom", "top")
TIMESTEPS = 5
SAMPLES_PER_CELL = 10
DEFAULT_SEED = 20260923
DEFAULT_NORMALIZATION_SAMPLE_SIZE = 10_000
ROBUST_STD_NORMALIZER = 1.349
RASTER_PATH_COL = "raster_bundle_path"
SCORE_COL = "aoi_nox_score"
PERCENTILE_COL = "aoi_score_percentile"
HALF_COL = "score_half"
CLASS_COL = "current_delta_category"
DELTA_COL = "current_effective_delta_nox"
MEAN_OP_TIME_COL = "mean_unit_op_time"
MEDIAN_OP_TIME_COL = "median_mean_unit_op_time"
HOURLY_NOX_COL = "total_nox_mass"
FIGURE_DPI = 160


@dataclass(frozen=True)
class RasterSample:
    """One selected record and its standardized masked NO2 sequence."""

    score_half: str
    delta_category: str
    sample_index: int
    aoi_id: int
    score: float
    percentile: float
    delta_nox: float
    hotspot_row: int
    hotspot_column: int
    sequence: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse AOI-score montage options."""
    job_id = os.getenv("SLURM_JOB_ID", "latest")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    parser.add_argument(
        "--normalization-sample-size",
        type=int,
        default=DEFAULT_NORMALIZATION_SAMPLE_SIZE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-nox-score-standardized-montage-{job_id}.png",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-nox-score-standardized-montage-{job_id}.csv",
    )
    parser.add_argument(
        "--normalization-output",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-nox-score-robust-normalization-{job_id}.json",
    )
    return parser.parse_args()


def load_prior_dataset_records(dataframe_dir: Path) -> pl.DataFrame:
    """Load the prior dataset fields needed to relabel and plot records.

    Args:
        dataframe_dir: Directory containing train, validation, and test tables.

    Returns:
        One combined frame with parsed UTC timestep timestamps.
    """
    columns = [
        AOI_ID_COL,
        "lat",
        "lon",
        *(f"timestep_time_t{index}" for index in range(TIMESTEPS)),
        "hotspot_row",
        "hotspot_column",
        RASTER_PATH_COL,
    ]
    frames = []
    for split in ("train", "val", "test"):
        path = dataframe_dir / f"{split}_df.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prior dataset split table not found: {path}")
        frame = pl.read_csv(path, columns=columns).with_columns(pl.lit(split).alias("split"))
        frames.append(frame)
    timestamp_expressions = [
        pl.col(f"timestep_time_t{index}").str.to_datetime(time_zone="UTC") for index in range(TIMESTEPS)
    ]
    return pl.concat(frames).with_columns(timestamp_expressions)


def build_membership(records: pl.DataFrame, raw_records: pl.LazyFrame) -> pl.DataFrame:
    """Map source facilities into prior-dataset AOIs.

    Args:
        records: Prior dataset records carrying AOI centers.
        raw_records: Full emissions history carrying facility coordinates.

    Returns:
        Facility-to-AOI membership rows.
    """
    aois = (
        records.select(AOI_ID_COL, "lat", "lon")
        .unique(subset=AOI_ID_COL, keep="first")
        .sort(AOI_ID_COL)
        .pipe(add_projected_coordinates)
    )
    facilities = raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    return build_aoi_membership(aois, facilities, build_aoi_spatial_frame(aois))


def calculate_aoi_scores(raw_records: pl.LazyFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Score AOIs by mean total NOx during above-median operating hours.

    Args:
        raw_records: Full unit-hour emissions history.
        membership: Facility-to-AOI membership rows.

    Returns:
        One deterministically ranked score row per AOI.
    """
    hourly = (
        raw_records.filter(pl.col("opTime").is_finite() & (pl.col("opTime") >= 0))
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(
            pl.col("opTime").mean().alias(MEAN_OP_TIME_COL),
            pl.col("noxMass")
            .filter(usable_nox_measurement_expr() & pl.col("noxMass").is_finite() & (pl.col("noxMass") >= 0))
            .sum()
            .alias(HOURLY_NOX_COL),
        )
        .collect(engine="streaming")
    )
    medians = hourly.group_by(AOI_ID_COL).agg(pl.col(MEAN_OP_TIME_COL).median().alias(MEDIAN_OP_TIME_COL))
    scores = (
        hourly.join(medians, on=AOI_ID_COL, how="inner")
        .filter(pl.col(MEAN_OP_TIME_COL) >= pl.col(MEDIAN_OP_TIME_COL))
        .group_by(AOI_ID_COL)
        .agg(
            pl.col(HOURLY_NOX_COL).mean().alias(SCORE_COL),
            pl.len().alias("score_history_hours"),
            pl.col(MEDIAN_OP_TIME_COL).first(),
        )
        .filter(pl.col(SCORE_COL).is_finite())
        .sort(SCORE_COL, AOI_ID_COL)
        .with_row_index("_score_rank", offset=1)
        .with_columns((pl.col("_score_rank") / pl.len()).alias(PERCENTILE_COL))
        .with_columns(
            pl.when(pl.col(PERCENTILE_COL) <= 0.5).then(pl.lit("bottom")).otherwise(pl.lit("top")).alias(HALF_COL)
        )
        .drop("_score_rank")
    )
    return scores


def calculate_hourly_aoi_nox(raw_records: pl.LazyFrame, membership: pl.DataFrame) -> pl.DataFrame:
    """Aggregate usable total NOx for current-label interpolation.

    Args:
        raw_records: Full unit-hour emissions history.
        membership: Facility-to-AOI membership rows.

    Returns:
        Valid AOI-hour total NOx rows.
    """
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


def add_current_labels(records: pl.DataFrame, hourly_nox: pl.DataFrame) -> pl.DataFrame:
    """Recompute current overlap-weighted EMA labels for prior raster records.

    Args:
        records: Prior records carrying five consecutive observation times.
        hourly_nox: Valid AOI-hour NOx totals.

    Returns:
        Records carrying the current effective delta and class.
    """
    previous_observations = (
        load_tempo_mapping()
        .select(AOI_ID_COL, "tempo_time")
        .sort(AOI_ID_COL, "tempo_time")
        .with_columns(pl.col("tempo_time").shift(1).over(AOI_ID_COL).alias("_interval_start_t0"))
        .rename({"tempo_time": "timestep_time_t0"})
    )
    aligned = records.join(previous_observations, on=[AOI_ID_COL, "timestep_time_t0"], how="left")
    missing_previous = aligned["_interval_start_t0"].null_count()
    if missing_previous:
        print(f"Dropping {missing_previous:,} records without a mapped pre-t0 observation", flush=True)
        aligned = aligned.filter(pl.col("_interval_start_t0").is_not_null())
    aligned = add_timestep_nox(aligned, hourly_nox, TIMESTEPS)
    aligned = add_ema_targets(
        aligned,
        EMA_HISTORY_TIMESTEPS,
        EMA_DECAY_TIMESCALE_HOURS,
        label_timestep_index=LABEL_TIMESTEP_INDEX,
    )
    delta = pl.col("effective_delta_nox")
    return (
        aligned.filter(delta.is_finite())
        .rename({"effective_delta_nox": DELTA_COL})
        .with_columns(
            pl.when(pl.col(DELTA_COL) < -STRATIFICATION_EMA_CHANGE_THRESHOLD)
            .then(pl.lit("decrease"))
            .when(pl.col(DELTA_COL) > STRATIFICATION_EMA_CHANGE_THRESHOLD)
            .then(pl.lit("increase"))
            .otherwise(pl.lit("steady"))
            .alias(CLASS_COL)
        )
    )


def select_manifest(records: pl.DataFrame, scores: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Select ten records from every score-half and current-class cell.

    Args:
        records: Relabeled prior-dataset records.
        scores: AOI scores and percentile halves.
        seed: Deterministic selection seed.

    Returns:
        Sixty selected records ordered for plotting.
    """
    candidates = records.join(scores, on=AOI_ID_COL, how="inner").with_columns(
        pl.struct(AOI_ID_COL, "timestep_time_t3", RASTER_PATH_COL).hash(seed=seed).alias("_selection_order")
    )
    selected = []
    for class_index, class_name in enumerate(CLASS_ORDER):
        for half_index, score_half in enumerate(HALF_ORDER):
            cell = candidates.filter((pl.col(CLASS_COL) == class_name) & (pl.col(HALF_COL) == score_half)).sort(
                "_selection_order", AOI_ID_COL, "timestep_time_t3"
            )
            distinct = cell.unique(subset=AOI_ID_COL, keep="first", maintain_order=True)
            chosen = distinct.head(SAMPLES_PER_CELL)
            if chosen.height < SAMPLES_PER_CELL:
                used_paths = chosen[RASTER_PATH_COL].to_list()
                chosen = pl.concat(
                    [
                        chosen,
                        cell.filter(~pl.col(RASTER_PATH_COL).is_in(used_paths)).head(SAMPLES_PER_CELL - chosen.height),
                    ]
                )
            if chosen.height < SAMPLES_PER_CELL:
                raise ValueError(
                    f"Need {SAMPLES_PER_CELL} samples for {score_half}/{class_name}, found {chosen.height}"
                )
            selected.append(
                chosen.with_columns(
                    pl.lit(class_index).alias("_class_order"),
                    pl.lit(half_index).alias("_half_order"),
                    pl.int_range(0, pl.len()).alias("sample_index"),
                )
            )
    return pl.concat(selected).sort("_class_order", "sample_index", "_half_order").drop("_selection_order")


def calculate_robust_no2_standardization(
    records: pl.DataFrame,
    dataset_dir: Path,
    sample_size: int,
    seed: int,
) -> dict[str, int | float | str]:
    """Estimate robust NO2 normalization from sampled raster bundles.

    Args:
        records: Full prior-dataset record pool.
        dataset_dir: Root used to resolve raster bundle paths.
        sample_size: Number of five-timestep bundles to sample.
        seed: Deterministic sampling seed.

    Returns:
        Robust center, scale, and sample diagnostics.
    """
    if sample_size <= 0:
        raise ValueError("normalization sample size must be positive")
    paths = (
        records.select(RASTER_PATH_COL)
        .unique()
        .with_columns(pl.col(RASTER_PATH_COL).hash(seed=seed).alias("_sample_order"))
        .sort("_sample_order")
        .head(sample_size)[RASTER_PATH_COL]
        .to_list()
    )
    if len(paths) < sample_size:
        raise ValueError(f"Requested {sample_size:,} normalization bundles but found {len(paths):,}")

    maximum_pixels = len(paths) * TIMESTEPS * IMG_SIZE * IMG_SIZE
    with tempfile.TemporaryDirectory(prefix=".aoi-nox-normalization-") as temporary_dir:
        pooled = np.memmap(
            Path(temporary_dir) / "valid-no2.bin",
            dtype=np.float32,
            mode="w+",
            shape=(maximum_pixels,),
        )
        offset = 0
        for index, serialized_path in enumerate(paths, start=1):
            stored_path = Path(str(serialized_path))
            raster_path = stored_path if stored_path.is_absolute() else dataset_dir / stored_path
            with np.load(raster_path, allow_pickle=False) as bundle:
                no2 = np.asarray(bundle["no2"], dtype=np.float32)
            values = no2[np.isfinite(no2)]
            pooled[offset : offset + values.size] = values
            offset += values.size
            if index % 1_000 == 0 or index == len(paths):
                print(f"Robust-normalization scan: {index:,}/{len(paths):,} bundles", flush=True)
        if offset == 0:
            raise ValueError("Sampled raster bundles contain no finite NO2 pixels")
        lower, median, upper = np.percentile(pooled[:offset], (25, 50, 75), overwrite_input=True)

    scale = float(upper - lower) / ROBUST_STD_NORMALIZER
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sampled NO2 rasters produced an invalid robust scale")
    return {
        "method": "median_and_iqr_over_1.349",
        "sample_seed": seed,
        "sampled_bundles": len(paths),
        "sampled_timestep_rasters": len(paths) * TIMESTEPS,
        "valid_pixels": offset,
        "lower_quartile": float(lower),
        "center": float(median),
        "upper_quartile": float(upper),
        "scale": scale,
    }


def _load_sample(row: dict[str, object], dataset_dir: Path, center: float, scale: float) -> RasterSample:
    # Load one masked sequence and apply the training standardization
    stored_path = Path(str(row[RASTER_PATH_COL]))
    raster_path = stored_path if stored_path.is_absolute() else dataset_dir / stored_path
    with np.load(raster_path, allow_pickle=False) as bundle:
        missing = {"no2", "no2_mask"}.difference(bundle.files)
        if missing:
            raise ValueError(f"Raster bundle {raster_path} is missing arrays: {sorted(missing)}")
        no2 = np.asarray(bundle["no2"], dtype=np.float64)
        mask = np.asarray(bundle["no2_mask"], dtype=bool)
    if no2.shape != (TIMESTEPS, IMG_SIZE, IMG_SIZE) or mask.shape != no2.shape:
        raise ValueError(f"Unexpected NO2 raster shape in {raster_path}: {no2.shape}")
    standardized = np.clip((no2 - center) / scale, -MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_CLIP_ABS)
    return RasterSample(
        score_half=str(row[HALF_COL]),
        delta_category=str(row[CLASS_COL]),
        sample_index=int(row["sample_index"]),
        aoi_id=int(row[AOI_ID_COL]),
        score=float(row[SCORE_COL]),
        percentile=float(row[PERCENTILE_COL]),
        delta_nox=float(row[DELTA_COL]),
        hotspot_row=int(row["hotspot_row"]),
        hotspot_column=int(row["hotspot_column"]),
        sequence=np.where(mask, standardized, np.nan),
    )


def write_montage(
    manifest: pl.DataFrame,
    dataset_dir: Path,
    normalization: dict[str, int | float | str],
    output_path: Path,
) -> Path:
    """Write the six-cell standardized sequence montage.

    Args:
        manifest: Selected records across score halves and classes.
        dataset_dir: Root used to resolve raster bundle paths.
        normalization: Robust sampled NO2 center and scale.
        output_path: PNG destination.

    Returns:
        Saved PNG path.
    """
    center = float(normalization["center"])
    scale = float(normalization["scale"])
    samples = [_load_sample(row, dataset_dir, center, scale) for row in manifest.to_dicts()]
    lookup = {(sample.delta_category, sample.score_half, sample.sample_index): sample for sample in samples}
    row_count = len(CLASS_ORDER) * SAMPLES_PER_CELL
    column_count = len(HALF_ORDER) * TIMESTEPS
    figure, axes = plt.subplots(row_count, column_count, figsize=(20, 43), squeeze=False)
    color_map = matplotlib.colormaps["RdBu_r"].copy()
    color_map.set_bad("#d7d7d7")
    image = None
    for class_index, class_name in enumerate(CLASS_ORDER):
        for sample_index in range(SAMPLES_PER_CELL):
            row_index = class_index * SAMPLES_PER_CELL + sample_index
            for half_index, score_half in enumerate(HALF_ORDER):
                sample = lookup[(class_name, score_half, sample_index)]
                for timestep in range(TIMESTEPS):
                    column_index = half_index * TIMESTEPS + timestep
                    axis = axes[row_index, column_index]
                    image = axis.imshow(
                        sample.sequence[timestep],
                        cmap=color_map,
                        vmin=-3,
                        vmax=3,
                        interpolation="nearest",
                    )
                    axis.scatter(
                        sample.hotspot_column,
                        sample.hotspot_row,
                        marker="+",
                        color="#00ffff",
                        s=18,
                        linewidths=0.8,
                    )
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row_index == 0:
                        suffix = " (post-label)" if timestep == TIMESTEPS - 1 else ""
                        axis.set_title(f"{score_half.upper()}  t{timestep}{suffix}", fontsize=8)
                    is_outer_label = (half_index == 0 and timestep == 0) or (
                        half_index == len(HALF_ORDER) - 1 and timestep == TIMESTEPS - 1
                    )
                    if is_outer_label:
                        if half_index == len(HALF_ORDER) - 1:
                            axis.yaxis.set_label_position("right")
                        axis.set_ylabel(
                            f"AOI {sample.aoi_id}\nscore {sample.score:.1f}\n"
                            f"p{100 * sample.percentile:.0f}  delta {sample.delta_nox:+.0f}",
                            fontsize=6.3,
                            rotation=0,
                            ha="right" if half_index == 0 else "left",
                            va="center",
                            labelpad=31,
                        )
            if sample_index == 0:
                axes[row_index, 0].text(
                    -1.05,
                    0.5,
                    class_name.upper(),
                    transform=axes[row_index, 0].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    weight="bold",
                )
        if class_index < len(CLASS_ORDER) - 1:
            boundary = (class_index + 1) * SAMPLES_PER_CELL - 1
            for axis in axes[boundary]:
                axis.spines["bottom"].set_color("black")
                axis.spines["bottom"].set_linewidth(2.2)
    if image is None:
        raise RuntimeError("No raster images were plotted")
    figure.suptitle(
        "Prior-dataset NO2 sequences by total-NOx AOI score and current EMA-change class\n"
        "10 samples per cell; AOI score = mean hourly total NOx after AOI-median opTime filtering; "
        f"robust normalization from {int(normalization['sampled_bundles']):,} sampled bundles; "
        "cyan + = source hotspot",
        fontsize=14,
        y=0.999,
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.982, bottom=0.018, hspace=0.08, wspace=0.04)
    color_axis = figure.add_axes((0.36, 0.006, 0.28, 0.006))
    figure.colorbar(
        image,
        cax=color_axis,
        orientation="horizontal",
        label="NO2 robust z-score: (value - sampled median) / (sampled IQR / 1.349)",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    """Recompute labels and AOI scores then write a balanced raster montage."""
    args = parse_args()
    records = load_prior_dataset_records(args.dataframe_dir)
    print(f"Loaded {records.height:,} prior-dataset records across {records[AOI_ID_COL].n_unique():,} AOIs", flush=True)
    normalization = calculate_robust_no2_standardization(
        records,
        args.dataset_dir,
        args.normalization_sample_size,
        args.seed,
    )
    args.normalization_output.parent.mkdir(parents=True, exist_ok=True)
    args.normalization_output.write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    print(
        f"Robust NO2 normalization: center={float(normalization['center']):.6g}, "
        f"scale={float(normalization['scale']):.6g}",
        flush=True,
    )
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
    print(f"Built {membership.height:,} facility-to-AOI memberships", flush=True)
    scores = calculate_aoi_scores(raw_records, membership)
    print(
        f"Scored {scores.height:,} AOIs; median cutoff lies between score percentiles 50 and 50+",
        flush=True,
    )
    hourly_nox = calculate_hourly_aoi_nox(raw_records, membership)
    labeled = add_current_labels(records, hourly_nox)
    manifest = select_manifest(labeled, scores, args.seed)
    counts = manifest.group_by(HALF_COL, CLASS_COL).len().sort(HALF_COL, CLASS_COL)
    print(f"Selected manifest:\n{counts}", flush=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.manifest_output)
    write_montage(manifest, args.dataset_dir, normalization, args.output)
    print(f"Saved manifest to {args.manifest_output}", flush=True)
    print(f"Saved robust normalization to {args.normalization_output}", flush=True)
    print(f"Saved montage to {args.output}", flush=True)


if __name__ == "__main__":
    main()
