"""Partition AOI-hour emission records into train, validation, and test splits."""

import argparse
import os
from collections.abc import Iterable

import polars as pl

from config import (
    DELTA_NOX_MASS_COL,
    DELTA_NOX_MED_COL,
    DELTA_NOX_SCALE_COL,
    FULL_DATA_PARQUET,
    LABEL_COL,
    NOX_MASS_COL,
    STRAT_BASE_DIR,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
)
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    add_aoi_bounds,
    add_tempo_files,
    aggregate_aoi_hours,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    build_tempo_mapping,
    cluster_aois,
    filter_usable_nox_measurements,
    plot_split_distributions,
)

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
SPLIT_SEED = 42

OUTPUT_COLUMNS = [
    AOI_ID_COL,
    "lat",
    "lon",
    "lat_min",
    "lat_max",
    "lon_min",
    "lon_max",
    "num_coal_units",
    "num_ng_units",
    "date",
    "hour",
    "cluster",
    "tempo",
    "prev_tempo",
    "era5",
    "avg_heat_input",
    "avg_pwr_gen",
    NOX_MASS_COL,
    DELTA_NOX_MASS_COL,
    DELTA_NOX_MED_COL,
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
        name: frame.join(split_clusters, on="cluster", how="inner")
        for name, split_clusters in cluster_splits.items()
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse stratification command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute and replace the cached TEMPO mapping",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Build stratified AOI-hour metadata splits for dataset generation."""
    args = parse_args(argv)
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
    spatial_aois = build_aoi_spatial_frame(aois)
    membership = build_aoi_membership(aois, facilities, spatial_aois)
    frame = aggregate_aoi_hours(records, aois, membership).filter(
        pl.col("avg_heat_input").is_not_null()
        & pl.col("avg_pwr_gen").is_not_null()
        & pl.col(LABEL_COL).is_not_null()
    )
    frame = frame.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = add_tempo_files(frame, build_tempo_mapping(overwrite=args.overwrite)).filter(
        pl.col("tempo").is_not_null() & pl.col("prev_tempo").is_not_null()
    )
    bounds = add_aoi_bounds(aois).select(AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max")
    frame = frame.join(bounds, on=AOI_ID_COL, how="left").with_columns(
        pl.concat_str(
            pl.lit("era5_"),
            pl.col("date").dt.year(),
            pl.lit("_"),
            pl.col("date").dt.month().cast(pl.String).str.pad_start(2, "0"),
            pl.lit(".nc"),
        ).alias("era5")
    )

    splits = {name: split.select(OUTPUT_COLUMNS) for name, split in _split_by_cluster(frame).items()}

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    splits["train"].write_csv(TRAIN_RECORDS_CSV)
    splits["val"].write_csv(VAL_RECORDS_CSV)
    splits["test"].write_csv(TEST_RECORDS_CSV)
    plot_split_distributions(splits["train"], splits["val"], splits["test"])


if __name__ == "__main__":
    main()
