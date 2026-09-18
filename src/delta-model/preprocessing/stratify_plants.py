"""Partition AOI-hour emission records into train, validation, and test splits."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    DELTA_EFFECTIVE_NOX_SCALED_COL,
    DELTA_NOX_COL,
    DELTA_NOX_SCALED_COL,
    EFFECTIVE_CURRENT_NOX_COL,
    LABEL_MODE_COL,
    MAJOR_CITY_DIST_COL,
    NOX_COL,
    PREV_QTR_MED_NOX_COL,
    PREVIOUS_QUARTER_POWER_COL,
    add_aoi_bounds,
    add_ema_targets,
    add_major_city_distance,
    add_scaled_nox_targets,
    add_sequence_weather_paths,
    add_tempo_sequences,
    aggregate_aoi_hours,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    calculate_aoi_scores,
    cluster_aois,
    filter_usable_nox_measurements,
    select_split_records,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

from config import (
    AOI_SELECTION_COUNT,
    EMA_DECAY_TIMESCALE_HOURS,
    FULL_DATA_PARQUET,
    MIN_COVERAGE_PERCENT,
    SEQUENCE_TIMESTEPS,
    STRAT_BASE_DIR,
    TEST_RECORDS,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS,
    VAL_RECORDS_CSV,
    VIS_DIR,
)

SPLIT_RECORD_COUNTS = {"train": TRAIN_RECORDS, "val": VAL_RECORDS, "test": TEST_RECORDS}
TOTAL_RECORDS = sum(SPLIT_RECORD_COUNTS.values())
SPLIT_FRACTIONS = {split: count / TOTAL_RECORDS for split, count in SPLIT_RECORD_COUNTS.items()}
SPLIT_SEED = 42
TIMESTEP_COLUMNS = [
    column
    for index in range(SEQUENCE_TIMESTEPS)
    for column in (
        f"timestep_time_t{index}",
        f"timestep_age_hours_t{index}",
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
    "total_nameplate_capacity_mw",
    "_source_east_km",
    "_source_north_km",
    "_source_unit_count",
    "date",
    "hour",
    "emissions_hour_utc",
    "cluster",
    *TIMESTEP_COLUMNS,
    "tempo_delta_minutes",
    "coverage_percent",
    "avg_heat_input",
    "avg_pwr_gen",
    NOX_COL,
    PREV_QTR_MED_NOX_COL,
    DELTA_NOX_COL,
    DELTA_NOX_SCALED_COL,
    EFFECTIVE_CURRENT_NOX_COL,
    "effective_delta_nox",
    DELTA_EFFECTIVE_NOX_SCALED_COL,
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
    "grossLoad",
    "heatInput",
    "noxMassMeasureFlg",
    "primaryFuelInfo",
    "attributePrimaryFuelInfo",
    "facility_nameplate_capacity_mw",
]

DEFAULT_AOI_RECORD_SHARE_OUTPUT = Path(VIS_DIR) / "stratification_aoi_record_shares.png"
DEFAULT_HISTOGRAM_OUTPUT = Path(VIS_DIR) / "stratification_scaled_label_histograms.png"
HISTOGRAM_QUANTILES = (0.01, 0.99)


def parse_args() -> argparse.Namespace:
    """Parse stratification command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi-record-share-output",
        type=Path,
        default=DEFAULT_AOI_RECORD_SHARE_OUTPUT,
    )
    parser.add_argument("--histogram-output", type=Path, default=DEFAULT_HISTOGRAM_OUTPUT)
    return parser.parse_args()


def _split_by_cluster(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # Greedily assign large clusters against total record targets
    cluster_counts = (
        frame.group_by("cluster")
        .agg(pl.len().alias("records"))
        .with_columns(pl.col("cluster").hash(seed=SPLIT_SEED).alias("_tie_breaker"))
        .sort(["records", "_tie_breaker"], descending=[True, False])
    )
    if cluster_counts.height < len(SPLIT_FRACTIONS):
        raise ValueError("At least three geographic clusters are required")

    total_records = float(cluster_counts["records"].sum())
    targets = {split: total_records * fraction for split, fraction in SPLIT_FRACTIONS.items()}
    assigned = {split: 0.0 for split in SPLIT_FRACTIONS}
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
                    assigned[split]
                    + (float(cluster["records"]) if split == destination else 0.0)
                    - targets[split]
                )
                ** 2
                for split in SPLIT_FRACTIONS
            ),
        )
        cluster_assignments.append({"cluster": cluster["cluster"], "split": destination})
        assigned_cluster_counts[destination] += 1
        assigned[destination] += float(cluster["records"])

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
        print(
            f"[{split}] assigned {split_frame.height:,}/{frame.height:,} eligible records "
            f"({split_frame.height / frame.height:.1%}; target {SPLIT_FRACTIONS[split]:.1%})"
        )
    return splits


def _filter_metadata_eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    # Apply non-raster candidate quality requirements
    return frame.filter(
        (pl.col("coverage_percent") >= MIN_COVERAGE_PERCENT)
        & pl.col("avg_pwr_gen").is_finite()
        & pl.col(MAJOR_CITY_DIST_COL).is_finite()
        & pl.col(PREVIOUS_QUARTER_POWER_COL).is_finite()
        & (pl.col(PREVIOUS_QUARTER_POWER_COL) > 0)
    )


def _plot_aoi_record_shares(
    splits: dict[str, pl.DataFrame],
    output_path: Path,
) -> None:
    # Show each AOI's percentage of the final sampled records by split
    split_names = tuple(SPLIT_FRACTIONS)
    shares_by_split = {
        name: split.group_by(AOI_ID_COL)
        .agg(pl.len().alias("record_count"))
        .with_columns((100 * pl.col("record_count") / split.height).alias("record_share_percent"))
        .sort("record_share_percent", AOI_ID_COL, descending=[True, False])
        for name, split in splits.items()
    }
    max_aois = max(frame.height for frame in shares_by_split.values())
    max_share = max(
        frame["record_share_percent"].max()
        for frame in shares_by_split.values()
        if not frame.is_empty()
    )
    figure, axes = plt.subplots(
        1,
        len(split_names),
        figsize=(24, max(10, max_aois * 0.32)),
        constrained_layout=True,
        sharex=True,
    )
    colors = plt.get_cmap("tab10").colors
    for split_index, (axis, split_name) in enumerate(zip(axes, split_names, strict=True)):
        split_shares = shares_by_split[split_name]
        positions = list(range(split_shares.height))
        percentages = split_shares["record_share_percent"].to_numpy()
        axis.barh(
            positions,
            percentages,
            color=colors[split_index],
        )
        axis.set_yticks(positions, [str(aoi_id) for aoi_id in split_shares[AOI_ID_COL]])
        axis.invert_yaxis()
        axis.set_xlim(0, max_share * 1.15)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_xlabel("Share of split records (%)")
        axis.set_ylabel("AOI ID")
        axis.set_title(f"{split_name}: {split_shares.height} AOIs, {splits[split_name].height:,} records")
        for position, percentage in zip(positions, percentages, strict=True):
            axis.text(
                float(percentage) + max_share * 0.01,
                position,
                f"{float(percentage):.2f}%",
                va="center",
                fontsize=7,
            )
    figure.suptitle("AOI shares of final sampled records by split")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_scaled_label_histograms(splits: dict[str, pl.DataFrame], output_path: Path) -> None:
    # Plot each target on a common robust x-axis across geographic splits
    target_rows = (
        (DELTA_NOX_SCALED_COL, "Scaled hourly NOx delta"),
        (DELTA_EFFECTIVE_NOX_SCALED_COL, "Scaled effective NOx delta"),
    )
    split_names = tuple(SPLIT_FRACTIONS)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row_index, (column, row_title) in enumerate(target_rows):
        combined = np.concatenate([split[column].drop_nulls().to_numpy() for split in splits.values()])
        finite = combined[np.isfinite(combined)]
        lower, upper = np.quantile(finite, HISTOGRAM_QUANTILES)
        limit = max(abs(lower), abs(upper))
        if limit == 0:
            limit = 1.0
        for column_index, split_name in enumerate(split_names):
            axis = axes[row_index, column_index]
            values = splits[split_name][column].drop_nulls().to_numpy()
            values = values[np.isfinite(values)]
            visible = values[np.abs(values) <= limit]
            axis.hist(visible, bins=50, range=(-limit, limit), edgecolor="white")
            axis.axvline(0, color="black", linewidth=1)
            axis.set_title(f"{split_name}: n={len(values):,}")
            axis.set_xlabel("asinh(delta / prior-quarter median NOx)")
            axis.set_ylabel("AOI-hour count")
            if column_index == 0:
                axis.text(-0.2, 0.5, row_title, rotation=90, va="center", transform=axis.transAxes)
            axis.text(
                0.98,
                0.95,
                f"shown: {len(visible):,}\nx range: +/-{limit:.3g}",
                ha="right",
                va="top",
                transform=axis.transAxes,
            )
    split_counts = ", ".join(f"{name}={splits[name].height:,}" for name in split_names)
    figure.suptitle(f"Scaled label distributions by split\nSplit counts: {split_counts}")
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


def main() -> None:
    """Build stratified AOI-hour metadata splits for dataset generation."""
    args = parse_args()
    source = pl.scan_parquet(FULL_DATA_PARQUET)
    raw_records = source.select(REQUIRED_COLUMNS).with_columns(pl.col("date").cast(pl.Date, strict=False))
    records = raw_records.pipe(filter_usable_nox_measurements).filter(pl.col("noxMass").is_finite())

    facilities = raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    aois = build_aois(facilities)
    bounded_aois = add_major_city_distance(add_aoi_bounds(aois))
    observations = load_tempo_mapping()
    spatial_aois = build_aoi_spatial_frame(aois)
    membership = build_aoi_membership(aois, facilities, spatial_aois)
    invalid_aoi_hours = (
        raw_records.filter(~usable_nox_measurement_expr() | ~pl.col("noxMass").is_finite())
        .join(membership.lazy(), on="facilityId", how="inner")
        .select(AOI_ID_COL, "emissions_hour_utc")
        .unique()
        .with_columns(pl.lit(True).alias("_has_invalid_nox"))
        .collect(engine="streaming")
    )
    hourly = (
        aggregate_aoi_hours(records, aois, membership)
        .join(invalid_aoi_hours, on=[AOI_ID_COL, "emissions_hour_utc"], how="left")
        .filter(pl.col("_has_invalid_nox").is_null())
        .drop("_has_invalid_nox")
        .filter(
            pl.col("avg_heat_input").is_not_null()
            & pl.col("avg_pwr_gen").is_not_null()
            & pl.col(PREV_QTR_MED_NOX_COL).is_finite()
            & (pl.col(PREV_QTR_MED_NOX_COL) > 0)
        )
    )
    frame = hourly.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = add_tempo_sequences(frame, observations, SEQUENCE_TIMESTEPS)
    frame = frame.filter(pl.col(f"timestep_time_t{SEQUENCE_TIMESTEPS - 1}").is_not_null())
    frame = add_ema_targets(frame, hourly, SEQUENCE_TIMESTEPS, EMA_DECAY_TIMESCALE_HOURS)
    frame = add_scaled_nox_targets(frame)
    frame = frame.with_columns(pl.lit("causal_ema").alias(LABEL_MODE_COL))
    bounds = bounded_aois.select(
        AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL
    )
    frame = add_sequence_weather_paths(
        frame.join(bounds, on=AOI_ID_COL, how="left"),
        SEQUENCE_TIMESTEPS,
    )
    frame = _filter_metadata_eligibility(frame).filter(pl.col("effective_delta_nox").is_finite())
    aoi_scores = calculate_aoi_scores(raw_records, hourly, frame, bounded_aois, membership)
    if aoi_scores.height < AOI_SELECTION_COUNT:
        raise ValueError(
            f"Requested {AOI_SELECTION_COUNT} AOIs but only {aoi_scores.height} have complete score inputs"
        )
    selected_aoi_scores = aoi_scores.head(AOI_SELECTION_COUNT)
    frame = frame.join(selected_aoi_scores.select(AOI_ID_COL), on=AOI_ID_COL, how="inner")
    print(f"Selected the top {selected_aoi_scores.height} of {aoi_scores.height} scoreable AOIs")
    splits = _split_by_cluster(frame)
    splits = {
        split: select_split_records(
            split_frame,
            split,
            SPLIT_RECORD_COUNTS[split],
            seed=SPLIT_SEED,
        )
        for split, split_frame in splits.items()
    }
    _plot_aoi_record_shares(splits, args.aoi_record_share_output)
    print(f"Saved AOI record-share chart to {args.aoi_record_share_output}")
    _plot_scaled_label_histograms(splits, args.histogram_output)
    print(f"Saved scaled-label histograms to {args.histogram_output}")

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    del frame, hourly
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        split = splits.pop(name)
        serialized = _serialize_no2_paths(split)
        serialized.select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
