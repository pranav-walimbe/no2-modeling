"""Partition AOI-hour emission records into train, validation, and test splits."""

import os

import polars as pl

from config import (
    DELTA_NOX_MASS_COL,
    DELTA_NOX_SCALE_COL,
    FULL_DATA_PARQUET,
    LABEL_COL,
    MIN_CITY_PROXIMITY,
    MIN_COVERAGE_PERCENT,
    NOX_MASS_COL,
    STRAT_BASE_DIR,
    TEST_RECORDS_CSV,
    TEST_RECORDS_SIZE,
    TRAIN_RECORDS_CSV,
    TRAIN_RECORDS_SIZE,
    VAL_RECORDS_CSV,
    VAL_RECORDS_SIZE,
)
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    MAJOR_CITY_DIST_COL,
    add_aoi_bounds,
    add_hrrr_files,
    add_major_city_distance,
    aggregate_aoi_hours,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
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
    "date",
    "hour",
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
    LABEL_COL,
]
REQUIRED_COLUMNS = [
    "facilityId",
    "unitId",
    "lat",
    "lon",
    "date",
    "hour",
    "noxMass",
    "grossLoad",
    "heatInput",
    "noxMassMeasureFlg",
    "primaryFuelInfo",
    "attributePrimaryFuelInfo",
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
    # Shuffle before truncation to avoid retaining only the earliest source rows
    return {
        name: split.sample(n=limits[name], shuffle=True, seed=SPLIT_SEED) if split.height > limits[name] else split
        for name, split in splits.items()
    }


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
        pl.col("avg_heat_input").is_not_null() & pl.col("avg_pwr_gen").is_not_null() & pl.col(LABEL_COL).is_not_null()
    )
    frame = frame.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = add_tempo_observations(frame, observations)
    frame = frame.filter(pl.col("tempo").is_not_null() & pl.col("prev_tempo").is_not_null())
    bounds = add_major_city_distance(bounded_aois).select(
        AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL
    )
    frame = add_hrrr_files(frame.join(bounds, on=AOI_ID_COL, how="left"))
    frame = frame.filter(
        (pl.col("coverage_percent") >= MIN_COVERAGE_PERCENT) & (pl.col(MAJOR_CITY_DIST_COL) >= MIN_CITY_PROXIMITY)
    )
    frame = serialize_tempo_path_lists(frame)
    splits = _limit_splits(filter_quantitative_outliers(_split_by_cluster(frame)))
    del frame

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        splits.pop(name).select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
