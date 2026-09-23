"""Partition AOI-hour emission records into train, validation, and test splits."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    LABEL_MODE_COL,
    MAJOR_CITY_DIST_COL,
    add_aoi_bounds,
    add_ema_targets,
    add_major_city_distance,
    add_sequence_weather_paths,
    add_tempo_sequences,
    add_timestep_nox,
    aggregate_aoi_hours,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    calculate_activity_conditioned_aoi_features,
    cluster_aois,
    filter_usable_nox_measurements,
    select_top_coal_aois,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

from config import (
    EMA_DECAY_TIMESCALE_HOURS,
    EMA_HISTORY_TIMESTEPS,
    FULL_DATA_PARQUET,
    LABEL_TIMESTEP_INDEX,
    MIN_COVERAGE_PERCENT,
    SEQUENCE_TIMESTEPS,
    STRAT_BASE_DIR,
    STRATIFICATION_AOI_FRACTION,
    STRATIFICATION_EMA_CHANGE_THRESHOLD,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
    VIS_DIR,
)

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
DELTA_CATEGORY_COL = "delta_category"
EMA_BUCKET_NAMES = ("decrease", "steady", "increase")
TIMESTEP_COLUMNS = [
    column
    for index in range(SEQUENCE_TIMESTEPS)
    for column in (
        f"t{index}_timestamp",
        f"t{index}_nox",
        f"no2_paths_t{index}",
        f"weather_path_t{index}",
    )
]
OUTPUT_COLUMNS = [
    AOI_ID_COL,
    "lat",
    "lon",
    "lat_min",
    "lat_max",
    "lon_min",
    "lon_max",
    MAJOR_CITY_DIST_COL,
    "num_coal_units",
    "num_ng_units",
    "num_units",
    "_source_east_km",
    "_source_north_km",
    "_source_unit_count",
    "date",
    "hour",
    "emissions_hour_utc",
    "cluster",
    *TIMESTEP_COLUMNS,
    "label_delta_mins",
    "coverage_percent",
    "avg_heat_input",
    "avg_pwr_gen",
    "avg_coal_nox",
    "effective_delta_nox",
    DELTA_CATEGORY_COL,
    LABEL_MODE_COL,
]
REQUIRED_COLUMNS = [
    "facilityId",
    "unitId",
    "lat",
    "lon",
    "date",
    "hour",
    "emissions_hour_utc",
    "noxMass",
    "opTime",
    "grossLoad",
    "heatInput",
    "noxMassMeasureFlg",
    "primaryFuelInfo",
    "attributePrimaryFuelInfo",
]

DEFAULT_DIAGNOSTIC_OUTPUT = Path(VIS_DIR) / "stratification_ema_balance.png"
HISTOGRAM_QUANTILES = (0.01, 0.99)


def parse_args() -> argparse.Namespace:
    """Parse stratification command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-output", type=Path, default=DEFAULT_DIAGNOSTIC_OUTPUT)
    return parser.parse_args()


def _split_by_cluster(
    frame: pl.DataFrame,
    category_column: str | None = None,
) -> dict[str, pl.DataFrame]:
    # Greedily assign large clusters against record or class-specific targets
    category_names = EMA_BUCKET_NAMES if category_column is not None else ()
    count_expressions = [pl.len().alias("records")]
    count_expressions.extend(
        (pl.col(category_column) == category).sum().alias(category)
        for category in category_names
    )
    cluster_counts = (
        frame.group_by("cluster")
        .agg(count_expressions)
        .with_columns(pl.col("cluster").hash(seed=SPLIT_SEED).alias("_tie_breaker"))
        .sort(["records", "_tie_breaker"], descending=[True, False])
    )
    if cluster_counts.height < len(SPLIT_FRACTIONS):
        raise ValueError("At least three geographic clusters are required")

    target_columns = category_names or ("records",)
    totals = {column: float(cluster_counts[column].sum()) for column in target_columns}
    targets = {
        split: {column: totals[column] * fraction for column in target_columns}
        for split, fraction in SPLIT_FRACTIONS.items()
    }
    assigned = {
        split: {column: 0.0 for column in target_columns}
        for split in SPLIT_FRACTIONS
    }
    cluster_assignments: list[dict[str, object]] = []
    assigned_cluster_counts = {split: 0 for split in SPLIT_FRACTIONS}
    for index, cluster in enumerate(cluster_counts.iter_rows(named=True)):
        empty_splits = [split for split, count in assigned_cluster_counts.items() if count == 0]
        remaining_clusters = cluster_counts.height - index
        destinations = empty_splits if remaining_clusters == len(empty_splits) else SPLIT_FRACTIONS
        destination = min(
            destinations,
            key=lambda destination: sum(
                ((
                    assigned[split][column]
                    + (float(cluster[column]) if split == destination else 0.0)
                    - targets[split][column]
                ) / max(targets[split][column], 1.0)) ** 2
                for split in SPLIT_FRACTIONS
                for column in target_columns
            ),
        )
        cluster_assignments.append({"cluster": cluster["cluster"], "split": destination})
        assigned_cluster_counts[destination] += 1
        for column in target_columns:
            assigned[destination][column] += float(cluster[column])

    assignments = pl.DataFrame(
        cluster_assignments,
        schema={"cluster": frame.schema["cluster"], "split": pl.String},
    )
    splits = {
        split: frame.join(
            assignments.filter(pl.col("split") == split).select("cluster"),
            on="cluster",
            how="inner",
        )
        for split in SPLIT_FRACTIONS
    }
    for split, split_frame in splits.items():
        category_summary = ""
        if category_column is not None:
            counts = split_frame.group_by(category_column).len()
            by_category = dict(counts.iter_rows())
            category_summary = "; " + ", ".join(
                f"{category}={by_category.get(category, 0):,}" for category in category_names
            )
        print(
            f"[{split}] assigned {split_frame.height:,}/{frame.height:,} eligible records "
            f"({split_frame.height / frame.height:.1%}; target {SPLIT_FRACTIONS[split]:.1%})"
            f"{category_summary}"
        )
    return splits


def filter_stratification_rule(frame: pl.DataFrame) -> pl.DataFrame:
    """Assign classes from the raw effective EMA change.

    Args:
        frame: Eligible records carrying raw effective deltas.

    Returns:
        Records labeled with the raw EMA class.
    """
    raw_delta = pl.col("effective_delta_nox")
    return frame.with_columns(
        pl.when(raw_delta < -STRATIFICATION_EMA_CHANGE_THRESHOLD)
        .then(pl.lit("decrease"))
        .when(raw_delta > STRATIFICATION_EMA_CHANGE_THRESHOLD)
        .then(pl.lit("increase"))
        .otherwise(pl.lit("steady"))
        .alias(DELTA_CATEGORY_COL)
    )


def _filter_metadata_eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    # Apply non-raster candidate quality requirements
    return frame.filter(
        (pl.col("coverage_percent") >= MIN_COVERAGE_PERCENT)
        & pl.col("avg_coal_nox").is_finite()
        & pl.col("avg_heat_input").is_finite()
        & pl.col("avg_pwr_gen").is_finite()
        & pl.col(MAJOR_CITY_DIST_COL).is_finite()
    )


def select_balanced_ema_records(
    frame: pl.DataFrame,
    split: str,
    *,
    seed: int = SPLIT_SEED,
) -> pl.DataFrame:
    """Select equal deterministic samples from three raw EMA-change buckets.

    Args:
        frame: Eligible records carrying the raw effective NOx delta.
        split: Split name used in progress and error messages.
        seed: Deterministic within-class ordering seed.

    Returns:
        All three buckets downsampled to the smallest bucket count.
    """
    counts = frame.group_by(DELTA_CATEGORY_COL).len().sort(DELTA_CATEGORY_COL)
    missing_classes = set(EMA_BUCKET_NAMES).difference(counts[DELTA_CATEGORY_COL].to_list())
    if missing_classes:
        raise ValueError(f"[{split}] EMA buckets have no records: {', '.join(sorted(missing_classes))}")
    records_per_class = int(counts["len"].min())
    selected = []
    for class_name in EMA_BUCKET_NAMES:
        class_records = (
            frame.filter(pl.col(DELTA_CATEGORY_COL) == class_name)
            .with_columns(
                pl.struct(AOI_ID_COL, "emissions_hour_utc")
                .hash(seed=seed)
                .alias("_selection_tie_breaker")
            )
            .sort("_selection_tie_breaker", AOI_ID_COL, "emissions_hour_utc")
            .head(records_per_class)
            .drop("_selection_tie_breaker")
        )
        selected.append(class_records)
    balanced = pl.concat(selected).sort(AOI_ID_COL, "emissions_hour_utc")
    count_summary = ", ".join(
        f"{row[DELTA_CATEGORY_COL]}={row['len']:,}" for row in counts.iter_rows(named=True)
    )
    print(
        f"[{split}] threshold=+/-{STRATIFICATION_EMA_CHANGE_THRESHOLD:.2f}; eligible {count_summary}; "
        f"selected {records_per_class:,} per class ({balanced.height:,} total)"
    )
    return balanced


def _plot_stratification_diagnostics(
    eligible_splits: dict[str, pl.DataFrame],
    balanced_splits: dict[str, pl.DataFrame],
    output_path: Path,
) -> None:
    # Compare raw EMA changes and bucket composition before and after balancing
    split_names = tuple(SPLIT_FRACTIONS)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    combined = np.concatenate(
        [split["effective_delta_nox"].drop_nulls().to_numpy() for split in eligible_splits.values()]
    )
    finite = combined[np.isfinite(combined)]
    lower, upper = np.quantile(finite, HISTOGRAM_QUANTILES)
    limit = max(abs(lower), abs(upper))
    if limit == 0:
        limit = 1.0
    threshold = STRATIFICATION_EMA_CHANGE_THRESHOLD
    total_eligible = sum(split.height for split in eligible_splits.values())
    colors = ("#3977af", "#999999", "#d65f4a")
    for column_index, split_name in enumerate(split_names):
        eligible = eligible_splits[split_name]
        balanced = balanced_splits[split_name]
        values = eligible["effective_delta_nox"].drop_nulls().to_numpy()
        values = values[np.isfinite(values)]
        visible = values[np.abs(values) <= limit]
        aoi_count = eligible[AOI_ID_COL].n_unique()

        histogram_axis = axes[0, column_index]
        histogram_axis.hist(visible, bins=70, range=(-limit, limit), color="#4c78a8", edgecolor="white")
        histogram_axis.axvline(0, color="black", linewidth=1)
        histogram_axis.axvline(-threshold, color="#b22222", linestyle="--", linewidth=1.5)
        histogram_axis.axvline(threshold, color="#b22222", linestyle="--", linewidth=1.5)
        histogram_axis.set_title(
            f"{split_name}: {eligible.height:,} eligible ({eligible.height / total_eligible:.1%}), {aoi_count} AOIs"
        )
        histogram_axis.set_xlabel("Raw effective EMA NOx change")
        histogram_axis.set_ylabel("AOI-hour count")
        histogram_axis.grid(axis="y", alpha=0.2)

        eligible_counts = np.array(
            [
                int((values < -threshold).sum()),
                int((np.abs(values) <= threshold).sum()),
                int((values > threshold).sum()),
            ]
        )
        balanced_count = balanced.height // len(EMA_BUCKET_NAMES)
        balanced_counts = np.full(len(EMA_BUCKET_NAMES), balanced_count)
        positions = np.arange(len(EMA_BUCKET_NAMES))
        width = 0.38
        bucket_axis = axes[1, column_index]
        before_bars = bucket_axis.bar(
            positions - width / 2,
            100 * eligible_counts / eligible_counts.sum(),
            width,
            label="Eligible",
            color=colors,
            alpha=0.55,
        )
        after_bars = bucket_axis.bar(
            positions + width / 2,
            100 * balanced_counts / balanced_counts.sum(),
            width,
            label="Balanced",
            color=colors,
        )
        bucket_axis.set_xticks(positions, EMA_BUCKET_NAMES)
        bucket_axis.set_ylim(0, 105)
        bucket_axis.set_ylabel("Share of records (%)")
        bucket_axis.grid(axis="y", alpha=0.2)
        bucket_axis.set_title(f"Maximum balanced set: {balanced.height:,} ({balanced_count:,} per bucket)")
        bucket_axis.bar_label(before_bars, labels=[f"{count:,}" for count in eligible_counts], fontsize=8)
        bucket_axis.bar_label(after_bars, labels=[f"{count:,}" for count in balanced_counts], fontsize=8)
        if column_index == 0:
            bucket_axis.legend(loc="upper left")
    split_counts = ", ".join(f"{name}={balanced_splits[name].height:,}" for name in split_names)
    total_balanced = sum(split.height for split in balanced_splits.values())
    figure.suptitle(
        f"Raw EMA change threshold +/-{STRATIFICATION_EMA_CHANGE_THRESHOLD:g}\n"
        f"Maximum balanced set: {total_balanced:,} total ({split_counts})"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _serialize_no2_paths(frame: pl.DataFrame) -> pl.DataFrame:
    # Encode every configured TEMPO granule list for CSV output
    return frame.with_columns(
        [
            pl.concat_str(pl.lit('["'), pl.col(f"no2_paths_t{index}").list.join('", "'), pl.lit('"]')).alias(
                f"no2_paths_t{index}"
            )
            for index in range(SEQUENCE_TIMESTEPS)
        ]
    )


def build_stratification_candidates() -> pl.DataFrame:
    """Build eligible AOI-hour records for every AOI.

    Returns:
        Eligible records before geographic splitting.
    """
    source = pl.scan_parquet(FULL_DATA_PARQUET)
    raw_records = source.select(REQUIRED_COLUMNS).with_columns(pl.col("date").cast(pl.Date, strict=False))
    records = raw_records.pipe(filter_usable_nox_measurements).filter(pl.col("noxMass").is_finite())

    facilities = raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    all_aois = build_aois(facilities)
    all_spatial_aois = build_aoi_spatial_frame(all_aois)
    membership = build_aoi_membership(all_aois, facilities, all_spatial_aois)
    aoi_features = calculate_activity_conditioned_aoi_features(raw_records, membership)
    selected_features = select_top_coal_aois(aoi_features, STRATIFICATION_AOI_FRACTION)
    selected_ids = selected_features.select(AOI_ID_COL)
    aois = all_aois.join(selected_ids, on=AOI_ID_COL, how="inner")
    membership = membership.join(selected_ids, on=AOI_ID_COL, how="inner")
    spatial_aois = build_aoi_spatial_frame(aois)
    bounded_aois = add_major_city_distance(add_aoi_bounds(aois))
    observations = load_tempo_mapping()
    print(
        f"Selected {aois.height:,}/{aoi_features.filter(pl.col('num_coal_units') > 0).height:,} "
        "coal-containing AOIs by avg_coal_nox"
    )
    invalid_aoi_hours = (
        raw_records.filter(~usable_nox_measurement_expr() | ~pl.col("noxMass").is_finite())
        .join(membership.lazy(), on="facilityId", how="inner")
        .select(AOI_ID_COL, "emissions_hour_utc")
        .unique()
        .with_columns(pl.lit(True).alias("_has_invalid_nox"))
        .collect(engine="streaming")
    )
    hourly = (
        aggregate_aoi_hours(records, aois, membership, selected_features)
        .join(invalid_aoi_hours, on=[AOI_ID_COL, "emissions_hour_utc"], how="left")
        .filter(pl.col("_has_invalid_nox").is_null())
        .drop("_has_invalid_nox")
    )
    frame = hourly.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = add_tempo_sequences(
        frame,
        observations,
        SEQUENCE_TIMESTEPS,
        label_timestep_index=LABEL_TIMESTEP_INDEX,
    )
    frame = add_timestep_nox(frame, hourly, SEQUENCE_TIMESTEPS)
    frame = frame.filter(pl.col(f"timestep_time_t{LABEL_TIMESTEP_INDEX}").is_not_null())
    frame = add_ema_targets(
        frame,
        hourly,
        EMA_HISTORY_TIMESTEPS,
        EMA_DECAY_TIMESCALE_HOURS,
        label_timestep_index=LABEL_TIMESTEP_INDEX,
    )
    frame = frame.with_columns(pl.lit("causal_ema").alias(LABEL_MODE_COL))
    bounds = bounded_aois.select(
        AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL
    )
    frame = add_sequence_weather_paths(
        frame.join(bounds, on=AOI_ID_COL, how="left"),
        SEQUENCE_TIMESTEPS,
    )
    frame = _filter_metadata_eligibility(frame).filter(pl.col("effective_delta_nox").is_finite())
    return frame.rename(
        {f"timestep_time_t{index}": f"t{index}_timestamp" for index in range(SEQUENCE_TIMESTEPS)}
    )


def main() -> None:
    """Build stratified AOI-hour metadata splits for dataset generation."""
    args = parse_args()
    candidates = build_stratification_candidates()
    frame = filter_stratification_rule(candidates)
    print(
        f"Labeled {frame.height:,} records with raw EMA threshold +/-"
        f"{STRATIFICATION_EMA_CHANGE_THRESHOLD:g}"
    )
    print(f"Using {frame[AOI_ID_COL].n_unique():,} eligible selected AOIs")
    eligible_splits = _split_by_cluster(frame, category_column=DELTA_CATEGORY_COL)
    splits = {
        split: select_balanced_ema_records(split_frame, split)
        for split, split_frame in eligible_splits.items()
    }
    _plot_stratification_diagnostics(eligible_splits, splits, args.diagnostic_output)
    print(f"Saved EMA-balance diagnostics to {args.diagnostic_output}")

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    del frame
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        split = splits.pop(name)
        serialized = _serialize_no2_paths(split)
        serialized.select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
