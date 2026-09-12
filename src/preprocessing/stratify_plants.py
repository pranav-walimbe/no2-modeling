"""Partition AOI-hour emission records into train, validation, and test splits."""

import os
from pathlib import Path

import polars as pl

from config import (
    COAL_DOMINANT_POWER_FRACTION,
    DELTA_THRESHOLD,
    FULL_DATA_PARQUET,
    LABEL_COL,
    MIN_COVERAGE_PERCENT,
    MIN_MAJOR_CITY_DISTANCE_KM,
    NOX_LOWER_PERCENTILE,
    NOX_UPPER_PERCENTILE,
    PREV_QTR_REL_DELTA_LOWER_PERCENTILE,
    STRAT_BASE_DIR,
    TEST_RECORDS_CSV,
    TEST_RECORDS_SIZE,
    TRAIN_RECORDS_CSV,
    TRAIN_RECORDS_SIZE,
    VAL_RECORDS_CSV,
    VAL_RECORDS_SIZE,
)
from preprocessing.generate_dataset_utils import write_json_atomic
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    LABEL_MODE_COL,
    MAJOR_CITY_DIST_COL,
    PREV_QTR_AVG_NOX_COL,
    PREV_QTR_REL_DELTA_COL,
    PREVIOUS_QUARTER_COAL_POWER_COL,
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

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
SPLIT_SEED = 42
SPLIT_RECORD_LIMITS = {
    "train": TRAIN_RECORDS_SIZE,
    "val": VAL_RECORDS_SIZE,
    "test": TEST_RECORDS_SIZE,
}

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
    # Assign each geographic cluster to exactly one split
    clusters = frame.select("cluster").unique().sort("cluster").sample(fraction=1.0, shuffle=True, seed=SPLIT_SEED)
    train_count = min(clusters.height - 2, max(1, int(clusters.height * TRAIN_FRACTION)))
    val_count = min(clusters.height - train_count - 1, max(1, int(clusters.height * VAL_FRACTION)))
    cluster_splits = {
        "train": clusters.slice(0, train_count),
        "val": clusters.slice(train_count, val_count),
        "test": clusters.slice(train_count + val_count),
    }
    return {
        name: frame.join(split_clusters, on="cluster", how="inner") for name, split_clusters in cluster_splits.items()
    }


def _add_prev_qtr_rel_delta(frame: pl.DataFrame) -> pl.DataFrame:
    # Normalize absolute change by the absolute prior-quarter NOx level
    denominator = pl.col(PREV_QTR_AVG_NOX_COL).abs()
    return frame.with_columns(
        pl.when(denominator > 0)
        .then(pl.col("delta_nox_mass").abs() / denominator)
        .alias(PREV_QTR_REL_DELTA_COL)
    )


def _filter_relative_delta_percentile(
    splits: dict[str, pl.DataFrame],
    lower_percentile: float = PREV_QTR_REL_DELTA_LOWER_PERCENTILE,
) -> tuple[dict[str, pl.DataFrame], float]:
    # Fit one cutoff across deadband-eligible records from every geographic split
    if not 0 <= lower_percentile < 100:
        raise ValueError("Relative-delta percentile must satisfy 0 <= lower < 100")
    relative_delta = pl.concat(
        [split.select(PREV_QTR_REL_DELTA_COL) for split in splits.values()],
        how="vertical",
    ).filter(pl.col(PREV_QTR_REL_DELTA_COL).is_finite())
    if relative_delta.is_empty():
        raise ValueError("Cannot calculate a relative-delta percentile without finite values")
    lower_bound = relative_delta.select(
        pl.col(PREV_QTR_REL_DELTA_COL).quantile(lower_percentile / 100, interpolation="linear")
    ).item()
    filtered = {
        name: split.filter(
            pl.col(PREV_QTR_REL_DELTA_COL).is_finite()
            & (pl.col(PREV_QTR_REL_DELTA_COL) >= lower_bound)
        )
        for name, split in splits.items()
    }
    for name in splits:
        print(
            f"[{name}] relative-delta filter retained {filtered[name].height:,}/{splits[name].height:,} records"
        )
    print(f"Relative-delta lower bound: P{lower_percentile:g}={lower_bound:.6g}")
    return filtered, float(lower_bound)


def _limit_splits(
    splits: dict[str, pl.DataFrame],
    limits: dict[str, int] = SPLIT_RECORD_LIMITS,
) -> dict[str, pl.DataFrame]:
    # Balance labels while retaining lagged power priority within each class
    limited: dict[str, pl.DataFrame] = {}
    for name, split in splits.items():
        limit = limits[name]
        indexed = split.with_row_index("_priority_row").with_columns(
            split.select(AOI_ID_COL, "date", "hour").hash_rows(seed=SPLIT_SEED).alias("_priority_hash")
        )
        class_limit = limit // 2
        selected_classes = []
        for label in (0, 1):
            class_pool = indexed.filter(pl.col(LABEL_COL) == label)
            if class_pool.height < class_limit:
                raise ValueError(
                    f"{name} class {label} has {class_pool.height:,} records; "
                    f"cannot select the requested {class_limit:,}"
                )
            selected_classes.append(_select_priority_records(class_pool, class_limit))
        selected = pl.concat(selected_classes, how="vertical")
        limited[name] = selected.sort("_priority_row").drop("_priority_row", "_priority_hash", "_priority_round")
        print(f"[{name}] selected {class_limit:,} records per class from {split.height:,} candidates")
    return limited


def _select_priority_records(frame: pl.DataFrame, limit: int) -> pl.DataFrame:
    # Every candidate is coal-dominant before priority selection
    return _rank_priority_pool(frame, PREVIOUS_QUARTER_COAL_POWER_COL).head(limit)


def _rank_priority_pool(frame: pl.DataFrame, priority_column: str) -> pl.DataFrame:
    # Round-robin across AOIs before taking another record from the same AOI
    return (
        frame.sort(
            [AOI_ID_COL, priority_column, PREVIOUS_QUARTER_POWER_COL, "_priority_hash"],
            descending=[False, True, True, False],
            nulls_last=True,
        )
        .with_columns(pl.col(AOI_ID_COL).cum_count().over(AOI_ID_COL).alias("_priority_round"))
        .sort(
            ["_priority_round", priority_column, PREVIOUS_QUARTER_POWER_COL, AOI_ID_COL, "_priority_hash"],
            descending=[False, True, True, False, False],
            nulls_last=True,
        )
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
    frame = frame.filter(
        (pl.col("coverage_percent") >= MIN_COVERAGE_PERCENT)
        & (pl.col(MAJOR_CITY_DIST_COL) >= MIN_MAJOR_CITY_DISTANCE_KM)
        & pl.col("avg_pwr_gen").is_finite()
        & pl.col(MAJOR_CITY_DIST_COL).is_finite()
        & pl.col(PREVIOUS_QUARTER_POWER_COL).is_finite()
        & (pl.col(PREVIOUS_QUARTER_POWER_COL) > 0)
        & pl.col(PREVIOUS_QUARTER_COAL_POWER_COL).is_finite()
        & (pl.col(PREVIOUS_QUARTER_COAL_POWER_COL) / pl.col(PREVIOUS_QUARTER_POWER_COL) > COAL_DOMINANT_POWER_FRACTION)
    )
    frame, nox_lower_bound, nox_upper_bound = _filter_nox_mass_percentiles(frame)
    frame = serialize_tempo_path_lists(frame)
    geographic_splits = _split_by_cluster(frame)
    labeled_splits = apply_binary_target(geographic_splits)
    relative_delta_splits, relative_delta_lower_bound = _filter_relative_delta_percentile(labeled_splits)
    splits = _limit_splits(relative_delta_splits)
    del frame

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    summary = {
        "coal_dominance": {
            "power_period": "previous_quarter",
            "minimum_coal_power_fraction_exclusive": COAL_DOMINANT_POWER_FRACTION,
        },
        "deadband": {
            "raw_delta_nox_threshold": DELTA_THRESHOLD,
            "retained_rule": "abs(delta_nox_mass) > threshold",
        },
        "relative_delta_filter": {
            "column": PREV_QTR_REL_DELTA_COL,
            "lower_percentile": PREV_QTR_REL_DELTA_LOWER_PERCENTILE,
            "lower_bound": relative_delta_lower_bound,
            "retained_rule": "prev_qtr_rel_delta >= lower_bound",
        },
        "nox_mass_outlier_filter": {
            "lower_percentile": NOX_LOWER_PERCENTILE,
            "upper_percentile": NOX_UPPER_PERCENTILE,
            "lower_bound": nox_lower_bound,
            "upper_bound": nox_upper_bound,
            "retained_rule": "lower_bound <= nox_mass <= upper_bound",
        },
        "splits": {
            name: {
                "deadband": classification_summary(geographic_splits[name], labeled_splits[name]),
                "relative_delta_filter": classification_summary(labeled_splits[name], relative_delta_splits[name]),
                "candidate_balance": classification_summary(relative_delta_splits[name], splits[name]),
            }
            for name in splits
        },
    }
    write_json_atomic(summary, Path(STRAT_BASE_DIR) / "classification_summary.json")
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        splits.pop(name).select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
