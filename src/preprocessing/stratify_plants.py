"""Partition AOI-hour emission records into train, validation, and test splits."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import (
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
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    DELTA_EFFECTIVE_NOX_SCALED_COL,
    DELTA_NOX_COL,
    DELTA_NOX_SCALED_COL,
    EFFECTIVE_CURRENT_NOX_COL,
    EFFECTIVE_DELTA_NOX_COL,
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
    cluster_aois,
    filter_usable_nox_measurements,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

SPLIT_RECORD_COUNTS = {"train": TRAIN_RECORDS, "val": VAL_RECORDS, "test": TEST_RECORDS}
TOTAL_RECORDS = sum(SPLIT_RECORD_COUNTS.values())
SPLIT_FRACTIONS = {split: count / TOTAL_RECORDS for split, count in SPLIT_RECORD_COUNTS.items()}
SPLIT_SEED = 42
TAIL_FRACTION = 0.025
BALANCE_BIN_COUNT = 20
BALANCE_EXPONENT = 0.5
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
    EFFECTIVE_DELTA_NOX_COL,
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

DEFAULT_HISTOGRAM_OUTPUT = Path(VIS_DIR) / "stratification_scaled_label_histograms.png"
HISTOGRAM_QUANTILES = (0.01, 0.99)


def parse_args() -> argparse.Namespace:
    """Parse stratification command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
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


def _tempered_bin_quotas(bin_counts: dict[int, int], target_records: int) -> dict[int, int]:
    # Allocate quotas proportional to the square root of bin population
    weights = {bin_id: count**BALANCE_EXPONENT for bin_id, count in bin_counts.items()}
    lower = 0.0
    upper = max(bin_counts[bin_id] / weights[bin_id] for bin_id in bin_counts)
    for _ in range(64):
        midpoint = (lower + upper) / 2
        allocated = sum(min(bin_counts[bin_id], midpoint * weights[bin_id]) for bin_id in bin_counts)
        if allocated <= target_records:
            lower = midpoint
        else:
            upper = midpoint

    fractional_quotas = {
        bin_id: min(bin_counts[bin_id], lower * weights[bin_id]) for bin_id in bin_counts
    }
    quotas = {bin_id: int(np.floor(quota)) for bin_id, quota in fractional_quotas.items()}
    remaining = target_records - sum(quotas.values())
    candidates = sorted(
        (bin_id for bin_id in bin_counts if quotas[bin_id] < bin_counts[bin_id]),
        key=lambda bin_id: (-(fractional_quotas[bin_id] - quotas[bin_id]), bin_id),
    )
    for bin_id in candidates[:remaining]:
        quotas[bin_id] += 1
    if sum(quotas.values()) != target_records:
        raise RuntimeError(f"Could not allocate {target_records:,} records across scaled-delta bins")
    return quotas


def _select_split_records(frame: pl.DataFrame, split: str, target_records: int) -> pl.DataFrame:
    # Trim raw-delta tails then select the configured split sample
    finite = frame.filter(
        pl.col(DELTA_NOX_COL).is_finite() & pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL).is_finite()
    )
    lower_bound, upper_bound = finite.select(
        pl.col(DELTA_NOX_COL).quantile(TAIL_FRACTION, interpolation="linear").alias("lower"),
        pl.col(DELTA_NOX_COL).quantile(1 - TAIL_FRACTION, interpolation="linear").alias("upper"),
    ).row(0)
    trimmed = finite.filter(pl.col(DELTA_NOX_COL).is_between(lower_bound, upper_bound, closed="both"))
    if trimmed.height < target_records:
        raise ValueError(
            f"[{split}] requested {target_records:,} records but only {trimmed.height:,} remain after tail trimming"
        )

    randomized = trimmed.with_columns(
        pl.struct(AOI_ID_COL, "emissions_hour_utc").hash(seed=SPLIT_SEED).alias("_selection_tie_breaker")
    )
    if split != "train":
        selected = (
            randomized.sort("_selection_tie_breaker", AOI_ID_COL, "emissions_hour_utc")
            .head(target_records)
            .drop("_selection_tie_breaker")
            .sort(AOI_ID_COL, "emissions_hour_utc")
        )
        print(
            f"[{split}] raw delta P2.5-P97.5 [{lower_bound:.6g}, {upper_bound:.6g}] retained "
            f"{trimmed.height:,}/{finite.height:,}; randomly selected {selected.height:,} records"
        )
        return selected

    scaled_min, scaled_max = trimmed.select(
        pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL).min().alias("scaled_min"),
        pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL).max().alias("scaled_max"),
    ).row(0)
    if scaled_min == scaled_max:
        raise ValueError(f"[{split}] cannot balance a constant scaled effective-delta target")

    binned = randomized.with_columns(
        (
            (pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL) - scaled_min)
            / (scaled_max - scaled_min)
            * BALANCE_BIN_COUNT
        )
        .floor()
        .cast(pl.Int32)
        .clip(0, BALANCE_BIN_COUNT - 1)
        .alias("_selection_bin"),
    )
    bin_counts = dict(binned.group_by("_selection_bin").len().iter_rows())
    quotas = _tempered_bin_quotas(bin_counts, target_records)
    quota_frame = pl.DataFrame(
        {
            "_selection_bin": list(quotas),
            "_selection_quota": list(quotas.values()),
        },
        schema={"_selection_bin": pl.Int32, "_selection_quota": pl.UInt32},
    )
    selected = (
        binned.sort("_selection_bin", "_selection_tie_breaker", AOI_ID_COL, "emissions_hour_utc")
        .with_columns(pl.col("_selection_bin").cum_count().over("_selection_bin").alias("_selection_rank"))
        .join(quota_frame, on="_selection_bin", how="left")
        .filter(pl.col("_selection_rank") <= pl.col("_selection_quota"))
        .drop("_selection_bin", "_selection_tie_breaker", "_selection_rank", "_selection_quota")
        .sort(AOI_ID_COL, "emissions_hour_utc")
    )
    if selected.height != target_records:
        raise RuntimeError(f"[{split}] selected {selected.height:,} records instead of {target_records:,}")
    print(
        f"[{split}] raw delta P2.5-P97.5 [{lower_bound:.6g}, {upper_bound:.6g}] retained "
        f"{trimmed.height:,}/{finite.height:,}; selected {selected.height:,} records with square-root balancing "
        f"across {len(bin_counts):,} scaled-delta bins"
    )
    return selected


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
                f"shown: {len(visible):,}\nx range: ±{limit:.3g}",
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
    bounded_aois = add_aoi_bounds(aois)
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
    bounds = add_major_city_distance(bounded_aois).select(
        AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL
    )
    frame = add_sequence_weather_paths(
        frame.join(bounds, on=AOI_ID_COL, how="left"),
        SEQUENCE_TIMESTEPS,
    )
    frame = _filter_metadata_eligibility(frame).filter(pl.col(EFFECTIVE_DELTA_NOX_COL).is_finite())
    splits = _split_by_cluster(frame)
    splits = {
        split: _select_split_records(split_frame, split, SPLIT_RECORD_COUNTS[split])
        for split, split_frame in splits.items()
    }

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    _plot_scaled_label_histograms(splits, args.histogram_output)
    print(f"Saved scaled-label histograms to {args.histogram_output}")
    del frame, hourly
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        split = splits.pop(name)
        serialized = _serialize_no2_paths(split)
        serialized.select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
