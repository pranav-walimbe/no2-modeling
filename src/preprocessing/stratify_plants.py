"""Partition AOI-hour emission records into train, validation, and test splits."""

import os
from pathlib import Path

import polars as pl

from config import (
    DELTA_THRESHOLD,
    FULL_DATA_PARQUET,
    LABEL_COL,
    MIN_COVERAGE_PERCENT,
    MIN_MAJOR_CITY_DISTANCE_KM,
    MIN_PREV_QTR_REL_DELTA,
    NOX_LOWER_PERCENTILE,
    NOX_UPPER_PERCENTILE,
    STRAT_BASE_DIR,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
)
from preprocessing.generate_dataset_utils import write_json_atomic
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    LABEL_MODE_COL,
    MAJOR_CITY_DIST_COL,
    PREV_QTR_AVG_NOX_COL,
    PREV_QTR_REL_DELTA_COL,
    PREVIOUS_QUARTER_POWER_COL,
    add_aoi_bounds,
    add_hrrr_files,
    add_major_city_distance,
    aggregate_aoi_hours,
    apply_binary_target,
    apply_target_label_mode,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    classification_summary,
    cluster_aois,
    filter_usable_nox_measurements,
)
from preprocessing.tempo_mapping import (
    add_tempo_observations,
    load_tempo_mapping,
    serialize_tempo_path_lists,
)

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
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
    "date",
    "hour",
    "emissions_hour_utc",
    "cluster",
    "tempo",
    "prev_tempo",
    "tempo_time",
    "prev_tempo_time",
    "tempo_delta_minutes",
    "coverage_percent",
    "hrrr",
    "avg_heat_input",
    "avg_pwr_gen",
    "nox_mass",
    PREV_QTR_AVG_NOX_COL,
    PREV_QTR_REL_DELTA_COL,
    "delta_nox_mass",
    LABEL_COL,
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


def _filter_nox_mass_percentiles(
    frame: pl.DataFrame,
    lower_percentile: float = NOX_LOWER_PERCENTILE,
    upper_percentile: float = NOX_UPPER_PERCENTILE,
) -> tuple[pl.DataFrame, float, float]:
    # Calculate global bounds from finite aggregate AOI-hour NOx values
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("NOx percentiles must satisfy 0 <= lower < upper <= 100")
    finite = frame.filter(pl.col("nox_mass").is_finite())
    if finite.is_empty():
        raise ValueError("Cannot calculate NOx percentiles without finite nox_mass values")
    lower_bound, upper_bound = finite.select(
        pl.col("nox_mass").quantile(lower_percentile / 100, interpolation="linear").alias("lower"),
        pl.col("nox_mass").quantile(upper_percentile / 100, interpolation="linear").alias("upper"),
    ).row(0)
    filtered = finite.filter(pl.col("nox_mass").is_between(lower_bound, upper_bound, closed="both"))
    print(
        f"Aggregate AOI-hour NOx percentiles retained {filtered.height:,}/{frame.height:,} records; "
        f"P{lower_percentile:g}={lower_bound:.6g}, P{upper_percentile:g}={upper_bound:.6g}"
    )
    return filtered, float(lower_bound), float(upper_bound)


def _split_by_cluster(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # Greedily assign large clusters against total and per-class record targets
    cluster_counts = (
        frame.group_by("cluster")
        .agg(
            pl.len().alias("records"),
            (pl.col(LABEL_COL) == 0).sum().alias("negative_records"),
            (pl.col(LABEL_COL) == 1).sum().alias("positive_records"),
        )
        .with_columns(pl.col("cluster").hash(seed=SPLIT_SEED).alias("_tie_breaker"))
        .sort(["records", "_tie_breaker"], descending=[True, False])
    )
    if cluster_counts.height < len(SPLIT_FRACTIONS):
        raise ValueError("At least three geographic clusters are required")

    count_columns = ("records", "negative_records", "positive_records")
    totals = {column: float(cluster_counts[column].sum()) for column in count_columns}
    targets = {
        split: {column: total * SPLIT_FRACTIONS[split] for column, total in totals.items()}
        for split in SPLIT_FRACTIONS
    }
    assigned = {
        split: {column: 0.0 for column in count_columns}
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
                (
                    (
                        assigned[split][column]
                        + (float(cluster[column]) if split == destination else 0.0)
                        - targets[split][column]
                    )
                    / max(totals[column], 1.0)
                )
                ** 2
                for split in SPLIT_FRACTIONS
                for column in count_columns
            ),
        )
        cluster_assignments.append({"cluster": cluster["cluster"], "split": destination})
        assigned_cluster_counts[destination] += 1
        for column in count_columns:
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
        print(
            f"[{split}] assigned {split_frame.height:,}/{frame.height:,} eligible records "
            f"({split_frame.height / frame.height:.1%}; target {SPLIT_FRACTIONS[split]:.1%})"
        )
    return splits


def _add_prev_qtr_rel_delta(frame: pl.DataFrame) -> pl.DataFrame:
    # Normalize absolute change by the absolute prior-quarter NOx level
    denominator = pl.col(PREV_QTR_AVG_NOX_COL).abs()
    return frame.with_columns(
        pl.when(denominator > 0)
        .then(pl.col("delta_nox_mass").abs() / denominator)
        .alias(PREV_QTR_REL_DELTA_COL)
    )


def _filter_relative_delta(
    splits: dict[str, pl.DataFrame],
    minimum: float = MIN_PREV_QTR_REL_DELTA,
) -> dict[str, pl.DataFrame]:
    # Apply one fixed relative-change floor to every geographic split
    if not 0 <= minimum:
        raise ValueError("Minimum relative delta must be nonnegative")
    filtered = {
        name: split.filter(
            pl.col(PREV_QTR_REL_DELTA_COL).is_finite()
            & (pl.col(PREV_QTR_REL_DELTA_COL) >= minimum)
        )
        for name, split in splits.items()
    }
    for name in splits:
        print(
            f"[{name}] relative-delta filter retained {filtered[name].height:,}/{splits[name].height:,} records"
        )
    print(f"Minimum relative delta: {minimum:.6g}")
    return filtered


def _filter_metadata_eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    # Apply non-raster candidate quality requirements
    return frame.filter(
        (pl.col("coverage_percent") >= MIN_COVERAGE_PERCENT)
        & (pl.col(MAJOR_CITY_DIST_COL) >= MIN_MAJOR_CITY_DISTANCE_KM)
        & pl.col("avg_pwr_gen").is_finite()
        & pl.col(MAJOR_CITY_DIST_COL).is_finite()
        & pl.col(PREVIOUS_QUARTER_POWER_COL).is_finite()
        & (pl.col(PREVIOUS_QUARTER_POWER_COL) > 0)
    )


def main() -> None:
    """Build stratified AOI-hour metadata splits for dataset generation."""
    source = pl.scan_parquet(FULL_DATA_PARQUET)
    records = (
        source.select(REQUIRED_COLUMNS)
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .pipe(filter_usable_nox_measurements)
    )

    facilities = records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    aois = build_aois(facilities)
    bounded_aois = add_aoi_bounds(aois)
    observations = load_tempo_mapping()
    spatial_aois = build_aoi_spatial_frame(aois)
    membership = build_aoi_membership(aois, facilities, spatial_aois)
    frame = aggregate_aoi_hours(records, aois, membership).filter(
        pl.col("avg_heat_input").is_not_null()
        & pl.col("avg_pwr_gen").is_not_null()
        & pl.col(PREV_QTR_AVG_NOX_COL).is_finite()
        & pl.col("delta_nox_mass").is_not_null()
    )
    frame = frame.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = _add_prev_qtr_rel_delta(apply_target_label_mode(add_tempo_observations(frame, observations)))
    frame = frame.filter(pl.col("tempo").is_not_null() & pl.col("prev_tempo").is_not_null())
    bounds = add_major_city_distance(bounded_aois).select(
        AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL
    )
    frame = add_hrrr_files(frame.join(bounds, on=AOI_ID_COL, how="left"))
    frame = _filter_metadata_eligibility(frame)
    frame, nox_lower_bound, nox_upper_bound = _filter_nox_mass_percentiles(frame)
    labeled = apply_binary_target({"all": frame})["all"]
    eligible = _filter_relative_delta({"all": labeled})["all"]
    splits = _split_by_cluster(serialize_tempo_path_lists(eligible))

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    summary = {
        "deadband": {
            "raw_delta_nox_threshold": DELTA_THRESHOLD,
            "retained_rule": "abs(delta_nox_mass) > threshold",
        },
        "relative_delta_filter": {
            "column": PREV_QTR_REL_DELTA_COL,
            "minimum": MIN_PREV_QTR_REL_DELTA,
            "retained_rule": "prev_qtr_rel_delta >= minimum",
        },
        "nox_mass_outlier_filter": {
            "lower_percentile": NOX_LOWER_PERCENTILE,
            "upper_percentile": NOX_UPPER_PERCENTILE,
            "lower_bound": nox_lower_bound,
            "upper_bound": nox_upper_bound,
            "retained_rule": "lower_bound <= nox_mass <= upper_bound",
        },
        "filter_retention": {
            "deadband": classification_summary(frame, labeled),
            "relative_delta_filter": classification_summary(labeled, eligible),
        },
        "split_assignment": {
            "target_fractions": SPLIT_FRACTIONS,
            "achieved_fractions": {name: splits[name].height / eligible.height for name in splits},
        },
        "splits": {
            name: {
                "natural_class_distribution": classification_summary(split, split),
            }
            for name, split in splits.items()
        },
    }
    del frame, labeled, eligible
    write_json_atomic(summary, Path(STRAT_BASE_DIR) / "classification_summary.json")
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        splits.pop(name).select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
