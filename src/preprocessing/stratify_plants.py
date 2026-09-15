"""Partition AOI-hour emission records into train, validation, and test splits."""

import os
from pathlib import Path

import polars as pl

from config import (
    EMA_DECAY_TIMESCALE_HOURS,
    EMA_DELTA_THRESHOLD,
    FULL_DATA_PARQUET,
    LABEL_COL,
    MIN_COVERAGE_PERCENT,
    MIN_MAJOR_CITY_DISTANCE_KM,
    SEQUENCE_TIMESTEPS,
    STRAT_BASE_DIR,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
)
from preprocessing.generate_dataset_utils import write_json_atomic
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    EFFECTIVE_CURRENT_NOX_COL,
    EFFECTIVE_DELTA_NOX_COL,
    EFFECTIVE_PREVIOUS_NOX_COL,
    LABEL_MODE_COL,
    MAJOR_CITY_DIST_COL,
    PREV_QTR_AVG_NOX_COL,
    PREVIOUS_QUARTER_POWER_COL,
    add_aoi_bounds,
    add_ema_targets,
    add_major_city_distance,
    add_sequence_weather_paths,
    add_tempo_sequences,
    aggregate_aoi_hours,
    apply_binary_target,
    build_aoi_membership,
    build_aoi_spatial_frame,
    build_aois,
    classification_summary,
    cluster_aois,
    filter_usable_nox_measurements,
    usable_nox_measurement_expr,
)
from preprocessing.tempo_mapping import load_tempo_mapping

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42
TIMESTEP_COLUMNS = [
    column
    for index in range(SEQUENCE_TIMESTEPS)
    for column in (
        f"timestep_time_t{index}",
        f"timestep_age_hours_t{index}",
        f"no2_paths_t{index}",
        f"wind_path_t{index}",
        f"temperature_path_t{index}",
    )
]
EMA_AUDIT_COLUMNS = [
    f"{prefix}_{suffix}"
    for prefix in ("current_ema", "previous_ema")
    for suffix in (
        "component_hours",
        "component_nox_mass",
        "overlap_seconds",
        "age_hours",
        "normalized_weights",
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
    "nox_mass",
    PREV_QTR_AVG_NOX_COL,
    EFFECTIVE_CURRENT_NOX_COL,
    EFFECTIVE_PREVIOUS_NOX_COL,
    EFFECTIVE_DELTA_NOX_COL,
    *EMA_AUDIT_COLUMNS,
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
        split: {column: total * SPLIT_FRACTIONS[split] for column, total in totals.items()} for split in SPLIT_FRACTIONS
    }
    assigned = {split: {column: 0.0 for column in count_columns} for split in SPLIT_FRACTIONS}
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


def _serialize_audit_lists(frame: pl.DataFrame) -> pl.DataFrame:
    # Preserve variable hourly overlap components as JSON-compatible CSV fields
    expressions = []
    for column in EMA_AUDIT_COLUMNS:
        values = pl.col(column).list.eval(pl.element().cast(pl.String))
        if column.endswith("component_hours"):
            serialized = pl.concat_str(pl.lit('["'), values.list.join('","'), pl.lit('"]'))
        else:
            serialized = pl.concat_str(pl.lit("["), values.list.join(","), pl.lit("]"))
        expressions.append(serialized.alias(column))
    return frame.with_columns(expressions)


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
            & pl.col(PREV_QTR_AVG_NOX_COL).is_finite()
        )
    )
    frame = hourly.join(cluster_aois(aois, spatial_aois), on=AOI_ID_COL, how="left")
    frame = add_tempo_sequences(frame, observations, SEQUENCE_TIMESTEPS)
    frame = frame.filter(pl.col(f"timestep_time_t{SEQUENCE_TIMESTEPS - 1}").is_not_null())
    frame = add_ema_targets(frame, hourly, SEQUENCE_TIMESTEPS, EMA_DECAY_TIMESCALE_HOURS)
    frame = frame.with_columns(pl.lit("causal_ema").alias(LABEL_MODE_COL))
    bounds = add_major_city_distance(bounded_aois).select(
        AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max", MAJOR_CITY_DIST_COL
    )
    frame = add_sequence_weather_paths(
        frame.join(bounds, on=AOI_ID_COL, how="left"),
        SEQUENCE_TIMESTEPS,
    )
    frame = _filter_metadata_eligibility(frame)
    eligible = apply_binary_target(
        {"all": frame},
        threshold=EMA_DELTA_THRESHOLD,
        target_column=EFFECTIVE_DELTA_NOX_COL,
    )["all"]
    splits = _split_by_cluster(eligible)

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    summary = {
        "deadband": {
            "ema_delta_nox_threshold": EMA_DELTA_THRESHOLD,
            "retained_rule": f"abs({EFFECTIVE_DELTA_NOX_COL}) > threshold",
        },
        "temporal_contract": {
            "sequence_timesteps": SEQUENCE_TIMESTEPS,
            "ema_decay_timescale_hours": EMA_DECAY_TIMESCALE_HOURS,
            "timestep_order": "oldest_to_newest",
            "missing_timestep_policy": "reject",
        },
        "filter_retention": {
            "deadband": classification_summary(frame, eligible),
        },
        "split_assignment": {
            "target_fractions": SPLIT_FRACTIONS,
            "achieved_fractions": {name: splits[name].height / eligible.height for name in splits},
        },
        "split_sizes": {name: split.height for name, split in splits.items()},
        "splits": {
            name: {
                "natural_class_distribution": classification_summary(split, split),
            }
            for name, split in splits.items()
        },
    }
    del frame, eligible, hourly
    write_json_atomic(summary, Path(STRAT_BASE_DIR) / "classification_summary.json")
    # Project and write one split at a time so the copies never coexist
    for name, destination in (("train", TRAIN_RECORDS_CSV), ("val", VAL_RECORDS_CSV), ("test", TEST_RECORDS_CSV)):
        split = splits.pop(name)
        serialized = _serialize_no2_paths(split)
        serialized = _serialize_audit_lists(serialized)
        serialized.select(OUTPUT_COLUMNS).write_csv(destination)


if __name__ == "__main__":
    main()
