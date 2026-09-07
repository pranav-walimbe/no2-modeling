"""Partition AOI-hour emission records into train, validation, and test splits."""

import os
from pathlib import Path

import polars as pl

from collection.emissions_schema import (
    EMISSIONS_HOUR_UTC_COL,
    FACILITY_NAMEPLATE_CAPACITY_MW_COL,
    TOTAL_NAMEPLATE_CAPACITY_MW_COL,
)
from config import (
    DEADBAND_THRESHOLD_COL,
    DEADBAND_TRAIN_FRACTION,
    DELTA_NOX_MASS_COL,
    DELTA_NOX_SCALE_COL,
    FULL_DATA_PARQUET,
    LABEL_COL,
    MIN_COVERAGE_PERCENT,
    MIN_MAJOR_CITY_DISTANCE_KM,
    NOX_MASS_COL,
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
    filter_quantitative_outliers,
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
    TOTAL_NAMEPLATE_CAPACITY_MW_COL,
    "date",
    "hour",
    EMISSIONS_HOUR_UTC_COL,
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
    NOX_MASS_COL,
    DELTA_NOX_MASS_COL,
    DELTA_NOX_SCALE_COL,
    DEADBAND_THRESHOLD_COL,
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
    EMISSIONS_HOUR_UTC_COL,
    "noxMass",
    "grossLoad",
    "heatInput",
    "noxMassMeasureFlg",
    "primaryFuelInfo",
    "attributePrimaryFuelInfo",
    FACILITY_NAMEPLATE_CAPACITY_MW_COL,
]


def _split_by_cluster(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # Assign each geographic cluster to exactly one split
    if frame["cluster"].null_count() > 0:
        raise ValueError("Cannot split records with missing cluster assignments")

    clusters = frame.select("cluster").unique().sort("cluster").sample(fraction=1.0, shuffle=True, seed=SPLIT_SEED)
    if clusters.height < 3:
        raise ValueError("At least three geographic clusters are required to create train, validation, and test splits")

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


def _limit_splits(
    splits: dict[str, pl.DataFrame],
    limits: dict[str, int] = SPLIT_RECORD_LIMITS,
) -> dict[str, pl.DataFrame]:
    # Balance labels while retaining lagged power priority within each class
    limited: dict[str, pl.DataFrame] = {}
    required = {
        AOI_ID_COL,
        "date",
        "hour",
        LABEL_COL,
        PREVIOUS_QUARTER_COAL_POWER_COL,
        PREVIOUS_QUARTER_POWER_COL,
    }
    for name, split in splits.items():
        limit = limits[name]
        if limit < 2 or limit % 2:
            raise ValueError(f"{name} split limit must be a positive even integer")
        missing = required.difference(split.columns)
        if missing:
            raise ValueError(f"Cannot prioritize {name} split without columns: {', '.join(sorted(missing))}")
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
    # Exhaust positive coal-output candidates before using the general pool
    coal_pool = frame.filter(
        pl.col(PREVIOUS_QUARTER_COAL_POWER_COL).is_finite()
        & (pl.col(PREVIOUS_QUARTER_COAL_POWER_COL) > 0)
    )
    selected_coal = _rank_priority_pool(coal_pool, PREVIOUS_QUARTER_COAL_POWER_COL).head(limit)
    remaining_count = limit - selected_coal.height
    if not remaining_count:
        return selected_coal
    general_pool = frame.join(selected_coal.select("_priority_row"), on="_priority_row", how="anti")
    selected_general = _rank_priority_pool(general_pool, PREVIOUS_QUARTER_POWER_COL).head(remaining_count)
    return pl.concat((selected_coal, selected_general), how="vertical")


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
    missing_columns = set(REQUIRED_COLUMNS).difference(source.collect_schema().names())
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Full emissions data is missing required AOI columns: {missing}. "
            "Rerun collection.scrape_emissions and collection.scrape_locations."
        )
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
        & pl.col(DELTA_NOX_MASS_COL).is_not_null()
        & pl.col(DELTA_NOX_SCALE_COL).is_not_null()
    )
    frame = frame.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = apply_target_label_mode(add_tempo_observations(frame, observations))
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
    )
    frame = serialize_tempo_path_lists(frame)
    filtered_splits = filter_quantitative_outliers(_split_by_cluster(frame))
    labeled_splits, threshold = apply_binary_target(filtered_splits)
    splits = _limit_splits(labeled_splits)
    del frame

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    summary = {
        "version": 1,
        "deadband": {
            "training_fraction": DEADBAND_TRAIN_FRACTION,
            "raw_delta_nox_threshold": threshold,
            "retained_rule": "abs(delta_nox_mass) > threshold",
        },
        "splits": {
            name: {
                "deadband": classification_summary(filtered_splits[name], labeled_splits[name]),
                "candidate_balance": classification_summary(labeled_splits[name], splits[name]),
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
