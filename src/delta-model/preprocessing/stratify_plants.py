"""Partition AOI-hour emission records into train, validation, and test splits."""

import argparse
import math
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
    add_projected_coordinates,
    add_sequence_weather_paths,
    add_tempo_sequences,
    add_timestep_nox,
    aggregate_aoi_hours,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    calculate_activity_conditioned_aoi_features,
    calculate_operating_aoi_characteristics,
    cluster_aois,
    deterministic_weighted_sample,
    filter_groups_by_class_count,
    filter_usable_nox_measurements,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

from config import (
    EMA_DECAY_TIMESCALE_HOURS,
    EMA_HISTORY_TIMESTEPS,
    FULL_DATA_PARQUET,
    LABEL_TIMESTEP_INDEX,
    SEQUENCE_TIMESTEPS,
    STRAT_BASE_DIR,
    STRATIFICATION_AOI_FRACTION,
    STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR,
    STRATIFICATION_INNOVATION_RELATIVE_FLOOR,
    STRATIFICATION_MINIMUM_RECORDS_PER_CLASS,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
    VIS_DIR,
)

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
MIN_STEADY_SELECTION_WEIGHT = 0.01
DELTA_CATEGORY_COL = "delta_category"
AOI_SCORE_COL = "aoi_score"
AOI_SCORE_PERCENTILE_COL = "aoi_score_percentile"
AOI_SCALE_COL = "aoi_active_median_nox"
EMA_UPDATE_ALPHA_COL = "ema_update_alpha"
EMA_INNOVATION_COL = "ema_innovation_nox"
HYBRID_THRESHOLD_COL = "hybrid_innovation_threshold"
EMA_BUCKET_NAMES = ("decrease", "steady", "increase")
TIMESTEP_COLUMNS = [
    column
    for index in range(SEQUENCE_TIMESTEPS)
    for column in (
        f"t{index}_timestamp",
        f"timestep_age_hours_t{index}",
        f"t{index}_nox",
        f"no2_paths_t{index}",
        f"weather_path_t{index}",
    )
]
OUTPUT_COLUMNS = [
    AOI_ID_COL,
    AOI_SCORE_COL,
    AOI_SCORE_PERCENTILE_COL,
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
    "date",
    "hour",
    "emissions_hour_utc",
    "cluster",
    *TIMESTEP_COLUMNS,
    "label_delta_mins",
    "avg_heat_input",
    "avg_pwr_gen",
    "avg_coal_nox",
    "effective_previous_nox",
    "effective_current_nox",
    "effective_delta_nox",
    AOI_SCALE_COL,
    EMA_UPDATE_ALPHA_COL,
    EMA_INNOVATION_COL,
    HYBRID_THRESHOLD_COL,
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
    "facility_nameplate_capacity_mw",
]

DEFAULT_DIAGNOSTIC_OUTPUT = Path(VIS_DIR) / "stratification_ema_balance.png"
DEFAULT_AOI_CHARACTERISTICS_PLOT = Path(VIS_DIR) / "stratification_aoi_characteristics.png"
DEFAULT_AOI_CHARACTERISTICS_OUTPUT = Path(VIS_DIR) / "stratification_aoi_characteristics.csv"
DEFAULT_AOI_SELECTION_OUTPUT = Path(VIS_DIR) / "stratification_aoi_selection.csv"
HISTOGRAM_QUANTILES = (0.01, 0.99)
AOI_CHARACTERISTIC_PANELS = (
    ("active_median_total_nox", "Active median total NOx", "log10(1 + lb/hr)", True),
    ("history_emitting_fraction", "Emitting-hour fraction", "Fraction", False),
    ("facility_count", "Facilities per AOI", "Count", False),
    ("active_mean_operating_units", "Active operating units", "Mean count", False),
    ("max_source_distance_km", "Maximum source distance", "km", False),
    ("largest_facility_capacity_share", "Dominant-facility capacity share", "Fraction", False),
)


def parse_args() -> argparse.Namespace:
    """Parse stratification command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-output", type=Path, default=DEFAULT_DIAGNOSTIC_OUTPUT)
    parser.add_argument(
        "--aoi-characteristics-plot",
        type=Path,
        default=DEFAULT_AOI_CHARACTERISTICS_PLOT,
    )
    parser.add_argument(
        "--aoi-characteristics-output",
        type=Path,
        default=DEFAULT_AOI_CHARACTERISTICS_OUTPUT,
    )
    parser.add_argument("--aoi-selection-output", type=Path, default=DEFAULT_AOI_SELECTION_OUTPUT)
    return parser.parse_args()


def _split_by_cluster(
    frame: pl.DataFrame,
    category_column: str | None = None,
) -> dict[str, pl.DataFrame]:
    # Greedily assign large clusters against record or class-specific targets
    category_names = EMA_BUCKET_NAMES if category_column is not None else ()
    count_expressions = [pl.len().alias("records")]
    count_expressions.extend((pl.col(category_column) == category).sum().alias(category) for category in category_names)
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
    assigned = {split: {column: 0.0 for column in target_columns} for split in SPLIT_FRACTIONS}
    cluster_assignments: list[dict[str, object]] = []
    assigned_cluster_counts = {split: 0 for split in SPLIT_FRACTIONS}
    for index, cluster in enumerate(cluster_counts.iter_rows(named=True)):
        empty_splits = [split for split, count in assigned_cluster_counts.items() if count == 0]
        remaining_clusters = cluster_counts.height - index
        destinations = empty_splits if remaining_clusters == len(empty_splits) else SPLIT_FRACTIONS
        destination = min(
            destinations,
            key=lambda destination: sum(
                (
                    (
                        assigned[split][column]
                        + (float(cluster[column]) if split == destination else 0.0)
                        - targets[split][column]
                    )
                    / max(targets[split][column], 1.0)
                )
                ** 2
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
    """Assign classes from an absolute and AOI-relative EMA innovation.

    Args:
        frame: Eligible records carrying timestep NOx values and raw effective deltas.

    Returns:
        Records carrying the AOI scale, innovation, threshold, and class.
    """
    timestep_columns = [f"t{index}_nox" for index in range(SEQUENCE_TIMESTEPS)]
    scales = (
        frame.select(AOI_ID_COL, pl.concat_list(timestep_columns).alias("_timestep_nox"))
        .explode("_timestep_nox", empty_as_null=True)
        .filter(pl.col("_timestep_nox").is_finite() & (pl.col("_timestep_nox") > 0))
        .group_by(AOI_ID_COL)
        .agg(pl.col("_timestep_nox").median().alias(AOI_SCALE_COL))
        .filter(pl.col(AOI_SCALE_COL).is_finite() & (pl.col(AOI_SCALE_COL) > 0))
    )
    current_time = pl.col(f"t{LABEL_TIMESTEP_INDEX}_timestamp")
    previous_time = pl.col(f"t{LABEL_TIMESTEP_INDEX - 1}_timestamp")
    interval_hours = (current_time - previous_time).dt.total_seconds() / 3600
    alpha = 1 - (-interval_hours / EMA_DECAY_TIMESCALE_HOURS).exp()
    threshold = pl.max_horizontal(
        pl.lit(STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR),
        STRATIFICATION_INNOVATION_RELATIVE_FLOOR * pl.col(AOI_SCALE_COL),
    )
    labeled = (
        frame.join(scales, on=AOI_ID_COL, how="inner")
        .with_columns(alpha.alias(EMA_UPDATE_ALPHA_COL))
        .filter(pl.col(EMA_UPDATE_ALPHA_COL).is_finite() & (pl.col(EMA_UPDATE_ALPHA_COL) > 0))
        .with_columns(
            (pl.col("effective_delta_nox") / pl.col(EMA_UPDATE_ALPHA_COL)).alias(EMA_INNOVATION_COL),
            threshold.alias(HYBRID_THRESHOLD_COL),
        )
        .filter(pl.col(EMA_INNOVATION_COL).is_finite())
    )
    innovation = pl.col(EMA_INNOVATION_COL)
    return labeled.with_columns(
        pl.when(innovation <= -pl.col(HYBRID_THRESHOLD_COL))
        .then(pl.lit("decrease"))
        .when(innovation >= pl.col(HYBRID_THRESHOLD_COL))
        .then(pl.lit("increase"))
        .otherwise(pl.lit("steady"))
        .alias(DELTA_CATEGORY_COL)
    )


def filter_aoi_class_floor(
    frame: pl.DataFrame,
    minimum_per_class: int = STRATIFICATION_MINIMUM_RECORDS_PER_CLASS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Retain AOIs with enough candidate records in every EMA class.

    Args:
        frame: Labeled candidate records.
        minimum_per_class: Required records in each class for one AOI.

    Returns:
        Eligible records and an AOI-level class-count audit.
    """
    return filter_groups_by_class_count(
        frame,
        AOI_ID_COL,
        DELTA_CATEGORY_COL,
        EMA_BUCKET_NAMES,
        minimum_per_class,
    )


def _filter_metadata_eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    # Apply non-raster candidate quality requirements
    return frame.filter(
        pl.all_horizontal([pl.col(f"t{index}_nox").is_finite() for index in range(SEQUENCE_TIMESTEPS)])
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
    """Select equal deterministic samples from three EMA-innovation buckets.

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
    identity_columns = (AOI_ID_COL, "emissions_hour_utc", f"t{LABEL_TIMESTEP_INDEX}_timestamp")
    for class_name in EMA_BUCKET_NAMES:
        class_records = frame.filter(pl.col(DELTA_CATEGORY_COL) == class_name)
        if class_name == "steady":
            weight = (
                1 - pl.col(EMA_INNOVATION_COL).abs() / pl.col(HYBRID_THRESHOLD_COL)
            ).clip(MIN_STEADY_SELECTION_WEIGHT, 1.0)
            class_records = deterministic_weighted_sample(
                class_records,
                records_per_class,
                weight,
                identity_columns,
                seed,
            )
        else:
            class_records = (
                class_records.with_columns(
                    pl.struct(*identity_columns).hash(seed=seed).alias("_selection_tie_breaker")
                )
                .sort("_selection_tie_breaker", *identity_columns)
                .head(records_per_class)
                .drop("_selection_tie_breaker")
            )
        selected.append(class_records)
    balanced = pl.concat(selected).sort(AOI_ID_COL, "emissions_hour_utc")
    count_summary = ", ".join(f"{row[DELTA_CATEGORY_COL]}={row['len']:,}" for row in counts.iter_rows(named=True))
    print(
        f"[{split}] threshold=max({STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR:g}, "
        f"{STRATIFICATION_INNOVATION_RELATIVE_FLOOR:.0%} of AOI scale); eligible {count_summary}; "
        f"selected {records_per_class:,} per class ({balanced.height:,} total)"
    )
    return balanced


def _plot_stratification_diagnostics(
    eligible_splits: dict[str, pl.DataFrame],
    balanced_splits: dict[str, pl.DataFrame],
    output_path: Path,
) -> None:
    # Compare EMA innovations and bucket composition before and after balancing
    split_names = tuple(SPLIT_FRACTIONS)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    combined = np.concatenate([split[EMA_INNOVATION_COL].drop_nulls().to_numpy() for split in eligible_splits.values()])
    finite = combined[np.isfinite(combined)]
    lower, upper = np.quantile(finite, HISTOGRAM_QUANTILES)
    limit = max(abs(lower), abs(upper))
    if limit == 0:
        limit = 1.0
    threshold = STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR
    total_eligible = sum(split.height for split in eligible_splits.values())
    colors = ("#3977af", "#999999", "#d65f4a")
    for column_index, split_name in enumerate(split_names):
        eligible = eligible_splits[split_name]
        balanced = balanced_splits[split_name]
        values = eligible[EMA_INNOVATION_COL].drop_nulls().to_numpy()
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
        histogram_axis.set_xlabel("EMA innovation (lb/hr)")
        histogram_axis.set_ylabel("AOI-hour count")
        histogram_axis.grid(axis="y", alpha=0.2)

        category_counts = dict(eligible.group_by(DELTA_CATEGORY_COL).len().iter_rows())
        eligible_counts = np.array([category_counts.get(category, 0) for category in EMA_BUCKET_NAMES])
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
        "Hybrid EMA innovation threshold: "
        f"max({STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR:g} lb/hr, "
        f"{STRATIFICATION_INNOVATION_RELATIVE_FLOOR:.0%} of AOI scale)\n"
        f"Maximum balanced set: {total_balanced:,} total ({split_counts})"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _rank_aoi_scores(all_aois: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    # Rank scored members of the facility-centered AOI set
    ranked = all_aois.select(AOI_ID_COL).join(scores, on=AOI_ID_COL, how="inner").sort(AOI_SCORE_COL, AOI_ID_COL)
    return ranked.with_row_index("aoi_score_rank", offset=1).with_columns(
        (pl.col("aoi_score_rank") / pl.len()).alias(AOI_SCORE_PERCENTILE_COL)
    )


def select_top_scored_aois(ranked_scores: pl.DataFrame, fraction: float) -> pl.DataFrame:
    """Select the highest-scoring share of mapped AOIs.

    Args:
        ranked_scores: Mapped AOIs carrying score ranks and percentiles.
        fraction: Selected share in the interval ``(0, 1]``.

    Returns:
        Deterministically selected AOI score rows.
    """
    selected_count = math.ceil(ranked_scores.height * fraction)
    return ranked_scores.sort(
        AOI_SCORE_COL,
        AOI_ID_COL,
        descending=[True, False],
    ).head(selected_count)


def _static_aoi_characteristics(relevant: pl.LazyFrame, membership: pl.DataFrame) -> pl.DataFrame:
    # Summarize source counts and facility capacity concentration
    static = (
        relevant.select("facilityId", "unitId")
        .unique()
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("facilityId").n_unique().alias("facility_count"),
            pl.struct("facilityId", "unitId").n_unique().alias("unit_count"),
        )
        .collect(engine="streaming")
    )
    capacity = (
        relevant.group_by("facilityId")
        .agg(pl.col("facility_nameplate_capacity_mw").max().alias("_facility_capacity_mw"))
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("_facility_capacity_mw").sum().alias("_total_capacity_mw"),
            pl.col("_facility_capacity_mw").max().alias("_largest_capacity_mw"),
        )
        .with_columns(
            pl.when(pl.col("_total_capacity_mw") > 0)
            .then(pl.col("_largest_capacity_mw") / pl.col("_total_capacity_mw"))
            .alias("largest_facility_capacity_share")
        )
        .select(AOI_ID_COL, "largest_facility_capacity_share")
        .collect(engine="streaming")
    )
    return static.join(capacity, on=AOI_ID_COL, how="inner")


def _geographic_aoi_characteristics(
    relevant: pl.LazyFrame,
    membership: pl.DataFrame,
    aois: pl.DataFrame,
) -> pl.DataFrame:
    # Measure the farthest member facility from each AOI center
    facility_points = add_projected_coordinates(
        relevant.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    ).select("facilityId", pl.col("x_m").alias("_facility_x_m"), pl.col("y_m").alias("_facility_y_m"))
    return (
        membership.join(facility_points, on="facilityId", how="inner")
        .join(
            aois.select(AOI_ID_COL, pl.col("x_m").alias("_aoi_x_m"), pl.col("y_m").alias("_aoi_y_m")),
            on=AOI_ID_COL,
            how="inner",
        )
        .with_columns(
            (
                (
                    (pl.col("_facility_x_m") - pl.col("_aoi_x_m")).pow(2)
                    + (pl.col("_facility_y_m") - pl.col("_aoi_y_m")).pow(2)
                ).sqrt()
                / 1_000
            ).alias("_source_distance_km")
        )
        .group_by(AOI_ID_COL)
        .agg(pl.col("_source_distance_km").max().alias("max_source_distance_km"))
    )


def calculate_aoi_characteristics(
    raw_records: pl.LazyFrame,
    membership: pl.DataFrame,
    aois: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate source and operating characteristics for one AOI subset.

    Args:
        raw_records: Full unit-hour emissions history.
        membership: Facility-to-AOI membership for the requested subset.
        aois: Requested AOI centers.

    Returns:
        One characteristic row per AOI.
    """
    relevant_facilities = membership["facilityId"].unique()
    relevant = raw_records.filter(pl.col("facilityId").is_in(relevant_facilities.implode()))
    static = _static_aoi_characteristics(relevant, membership)
    operating = calculate_operating_aoi_characteristics(relevant, membership)
    geometry = _geographic_aoi_characteristics(relevant, membership, aois)
    return static.join(
        operating,
        on=AOI_ID_COL,
        how="inner",
    ).join(geometry, on=AOI_ID_COL, how="inner")


def _plot_aoi_characteristics(characteristics: pl.DataFrame, output_path: Path) -> None:
    # Compare each selected AOI characteristic with the scored unselected cohort
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    cohorts = ((False, "Scored, unselected", "#999999"), (True, "Selected", "#3977af"))
    for axis, (column, title, xlabel, log_transform) in zip(
        axes.flat,
        AOI_CHARACTERISTIC_PANELS,
        strict=True,
    ):
        plotted: list[tuple[np.ndarray, str, str, float, float]] = []
        for selected, label, color in cohorts:
            values = characteristics.filter(pl.col("selected_for_stratification") == selected)[column].to_numpy()
            values = values[np.isfinite(values)]
            raw_median = float(np.median(values)) if values.size else np.nan
            if log_transform:
                values = np.log10(1 + np.clip(values, 0, None))
            if values.size:
                plotted.append((values, label, color, float(np.median(values)), raw_median))
        if not plotted:
            axis.set(title=title, xlabel=xlabel)
            axis.text(0.5, 0.5, "No finite values", transform=axis.transAxes, ha="center", va="center")
            continue
        combined = np.concatenate([values for values, _, _, _, _ in plotted])
        lower, upper = np.quantile(combined, HISTOGRAM_QUANTILES)
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 31)
        median_labels = []
        for values, label, color, median, raw_median in plotted:
            visible = values[(values >= lower) & (values <= upper)]
            axis.hist(visible, bins=bins, density=True, alpha=0.45, color=color, label=label)
            axis.axvline(median, color=color, linewidth=1.5, linestyle="--")
            median_labels.append(f"{label}: median {raw_median:,.2f}")
        axis.set(title=title, xlabel=xlabel, ylabel="Density")
        axis.grid(axis="y", alpha=0.2)
        axis.text(0.98, 0.96, "\n".join(median_labels), transform=axis.transAxes, ha="right", va="top", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=2)
    selected_count = characteristics.filter(pl.col("selected_for_stratification")).height
    figure.suptitle(
        f"AOI characteristics after active-median-NOx selection "
        f"({selected_count:,}/{characteristics.height:,} scored AOIs retained)",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
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


def build_stratification_candidates(
    aoi_characteristics_plot: Path = DEFAULT_AOI_CHARACTERISTICS_PLOT,
    aoi_characteristics_output: Path = DEFAULT_AOI_CHARACTERISTICS_OUTPUT,
    aoi_selection_output: Path = DEFAULT_AOI_SELECTION_OUTPUT,
) -> pl.DataFrame:
    """Build eligible AOI-hour records for score-selected AOIs.

    Args:
        aoi_characteristics_plot: Destination for the AOI characteristics dashboard.
        aoi_characteristics_output: Destination for the underlying AOI table.
        aoi_selection_output: Destination for the complete AOI score-selection audit.

    Returns:
        Eligible records before geographic splitting.
    """
    source = pl.scan_parquet(FULL_DATA_PARQUET)
    raw_records = source.select(REQUIRED_COLUMNS).with_columns(pl.col("date").cast(pl.Date, strict=False))
    records = raw_records.pipe(filter_usable_nox_measurements).filter(pl.col("noxMass").is_finite())

    facilities = raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    all_aois = build_aois(facilities)
    all_spatial_aois = build_aoi_spatial_frame(all_aois)
    all_membership = build_aoi_membership(all_aois, facilities, all_spatial_aois)
    characteristics = calculate_aoi_characteristics(raw_records, all_membership, all_aois)
    nox_scores = (
        characteristics.filter(
            pl.col("active_median_total_nox").is_finite() & (pl.col("active_median_total_nox") >= 0)
        )
        .select(AOI_ID_COL, pl.col("active_median_total_nox").alias(AOI_SCORE_COL))
    )
    ranked_scores = _rank_aoi_scores(all_aois, nox_scores)
    selected_scores = select_top_scored_aois(ranked_scores, STRATIFICATION_AOI_FRACTION)
    selected_ids = selected_scores.select(AOI_ID_COL)
    selection_audit = (
        all_aois.select(AOI_ID_COL)
        .join(
            ranked_scores.select(AOI_ID_COL, AOI_SCORE_COL, AOI_SCORE_PERCENTILE_COL),
            on=AOI_ID_COL,
            how="left",
        )
        .join(selected_ids.with_columns(pl.lit(True).alias("selected_for_stratification")), on=AOI_ID_COL, how="left")
        .with_columns(
            pl.col(AOI_SCORE_COL).is_not_null().alias("has_aoi_score"),
            pl.col("selected_for_stratification").fill_null(False),
        )
        .sort(AOI_ID_COL)
    )
    aoi_selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_audit.write_csv(aoi_selection_output)
    aois = all_aois.join(selected_ids, on=AOI_ID_COL, how="inner")
    membership = all_membership.join(selected_ids, on=AOI_ID_COL, how="inner")
    characteristics = (
        characteristics.join(
            selected_ids.with_columns(pl.lit(True).alias("selected_for_stratification")),
            on=AOI_ID_COL,
            how="left",
        )
        .with_columns(pl.col("selected_for_stratification").fill_null(False))
        .sort(AOI_ID_COL)
    )
    aoi_characteristics_output.parent.mkdir(parents=True, exist_ok=True)
    characteristics.write_csv(aoi_characteristics_output)
    _plot_aoi_characteristics(characteristics, aoi_characteristics_plot)
    print(f"Saved AOI characteristics to {aoi_characteristics_plot} and {aoi_characteristics_output}")
    selected_features = calculate_activity_conditioned_aoi_features(raw_records, membership).join(
        selected_scores.select(AOI_ID_COL, AOI_SCORE_COL, AOI_SCORE_PERCENTILE_COL),
        on=AOI_ID_COL,
        how="inner",
    )
    spatial_aois = build_aoi_spatial_frame(aois)
    bounded_aois = add_major_city_distance(add_aoi_bounds(aois))
    observations = load_tempo_mapping()
    print(
        f"Selected {aois.height:,}/{ranked_scores.height:,} scored AOIs "
        f"from {all_aois.height:,} total AOIs by active median NOx; "
        f"{all_aois.height - ranked_scores.height:,} AOIs are unscored"
    )
    print(f"Saved complete AOI selection audit to {aoi_selection_output}")
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
    frame = frame.filter(pl.col(f"timestep_time_t{SEQUENCE_TIMESTEPS - 1}").is_not_null())
    frame = add_timestep_nox(frame, hourly, SEQUENCE_TIMESTEPS)
    frame = add_ema_targets(
        frame,
        EMA_HISTORY_TIMESTEPS,
        EMA_DECAY_TIMESCALE_HOURS,
        label_timestep_index=LABEL_TIMESTEP_INDEX,
    )
    frame = frame.with_columns(pl.lit("linear_interpolated_timestep_ema").alias(LABEL_MODE_COL))
    bounds = bounded_aois.select(AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL)
    frame = add_sequence_weather_paths(
        frame.join(bounds, on=AOI_ID_COL, how="left"),
        SEQUENCE_TIMESTEPS,
    )
    frame = _filter_metadata_eligibility(frame).filter(pl.col("effective_delta_nox").is_finite())
    return frame.rename({f"timestep_time_t{index}": f"t{index}_timestamp" for index in range(SEQUENCE_TIMESTEPS)})


def main() -> None:
    """Build stratified AOI-hour metadata splits for dataset generation."""
    args = parse_args()
    candidates = build_stratification_candidates(
        args.aoi_characteristics_plot,
        args.aoi_characteristics_output,
        args.aoi_selection_output,
    )
    frame = filter_stratification_rule(candidates)
    print(
        f"Labeled {frame.height:,} records with hybrid EMA innovation threshold "
        f"max({STRATIFICATION_INNOVATION_ABSOLUTE_FLOOR:g} lb/hr, "
        f"{STRATIFICATION_INNOVATION_RELATIVE_FLOOR:.0%} of AOI scale)"
    )
    frame, class_audit = filter_aoi_class_floor(frame)
    eligible_aoi_count = class_audit.filter(pl.col("meets_class_floor")).height
    print(
        f"Retained {eligible_aoi_count:,}/{class_audit.height:,} selected AOIs with at least "
        f"{STRATIFICATION_MINIMUM_RECORDS_PER_CLASS} records in every class"
    )
    eligible_splits = _split_by_cluster(frame, category_column=DELTA_CATEGORY_COL)
    splits = {split: select_balanced_ema_records(split_frame, split) for split, split_frame in eligible_splits.items()}
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
