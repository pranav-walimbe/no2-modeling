"""Search AOI scoring rules against held-out directional plume quality."""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from preprocessing.stratify_utils import (
    AOI_ID_COL,
    add_projected_coordinates,
    usable_nox_measurement_expr,
)
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from config import DATASET_DF, DATASET_DIR, FULL_DATA_PARQUET, NUM_CORES, VIS_DIR
from eda.aoi_nox_score_montage import (
    CLASS_COL,
    RASTER_PATH_COL,
    add_current_labels,
    build_membership,
    calculate_hourly_aoi_nox,
    load_prior_dataset_records,
)
from eda.aoi_plume_snr_montage import (
    PLUME_DELTA_Z_COL,
    SNR_COL,
    add_label_compatibility,
    sample_score_records,
    score_sampled_rasters,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_ORDER = ("decrease", "increase")
DEFAULT_SEED = 20260923
RECORD_FOLDS = 3
AOI_FOLDS = 5
MIN_SEARCH_CLASS_RECORDS = 3
MIN_EVALUATION_CLASS_RECORDS = 2
FIXED_SNR_SCALE = 0.5
FIXED_DIRECTION_SCALE = 1.0
FIXED_UNCERTAINTY_PENALTY = 0.5
TOP_FRACTION = 0.25
SNR_SCALES = (0.25, 0.5, 1.0)
DIRECTION_SCALES = (0.5, 1.0, 2.0)
AGGREGATIONS = ("mean", "median", "q75")
CLASS_BALANCE_PENALTIES = (0.0, 0.5, 1.0)
UNCERTAINTY_PENALTIES = (0.0, 0.5, 1.0)
FEATURE_RETAINED_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
FEATURE_DIRECTIONS = ("high", "low")
TARGET_COL = "held_out_directional_quality"
CANDIDATE_RECORD_COL = "candidate_record_quality"
FIXED_RECORD_COL = "fixed_record_quality"
MARGIN_COL = "signed_directional_strength"
FIGURE_DPI = 180


def parse_args() -> argparse.Namespace:
    """Parse AOI-score search options."""
    job_id = os.getenv("SLURM_JOB_ID", "latest")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    parser.add_argument("--workers", type=int, default=NUM_CORES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-score-quality-search-{job_id}",
    )
    return parser.parse_args()


def _label_sign_expr() -> pl.Expr:
    # Map the two directional classes to signed numeric targets
    return pl.when(pl.col(CLASS_COL) == "increase").then(pl.lit(1.0)).otherwise(pl.lit(-1.0))


def add_fixed_record_quality(records: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Add frozen directional quality and deterministic record folds.

    Args:
        records: Records carrying current labels, matched-filter SNR, and plume changes.
        seed: Hash seed controlling record-fold membership.

    Returns:
        Directional records with fixed evaluation quality and fold indices.
    """
    return (
        records.filter(pl.col(CLASS_COL).is_in(CLASS_ORDER))
        .with_columns(
            (_label_sign_expr() * pl.col(PLUME_DELTA_Z_COL)).alias(MARGIN_COL),
            (pl.col(RASTER_PATH_COL).hash(seed=seed) % RECORD_FOLDS).cast(pl.Int8).alias("record_fold"),
        )
        .with_columns(
            ((pl.col(SNR_COL) / FIXED_SNR_SCALE).tanh() * (pl.col(MARGIN_COL) / FIXED_DIRECTION_SCALE).tanh()).alias(
                FIXED_RECORD_COL
            )
        )
    )


def _class_summary(
    records: pl.DataFrame,
    value_column: str,
    aggregation: str,
    minimum_records: int,
) -> pl.DataFrame:
    # Aggregate a record metric separately for both directional classes
    value = pl.col(value_column)
    if aggregation == "mean":
        center = value.mean()
    elif aggregation == "median":
        center = value.median()
    elif aggregation == "q75":
        center = value.quantile(0.75, interpolation="linear")
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    return (
        records.group_by(AOI_ID_COL, CLASS_COL)
        .agg(
            center.alias("class_center"),
            value.std(ddof=1).fill_null(0.0).alias("class_std"),
            pl.len().alias("class_records"),
        )
        .filter(pl.col("class_records") >= minimum_records)
        .with_columns((pl.col("class_std") / pl.col("class_records").cast(pl.Float64).sqrt()).alias("class_se"))
    )


def _wide_class_summary(summary: pl.DataFrame) -> pl.DataFrame:
    # Join decrease and increase summaries onto one row per eligible AOI
    tables = {}
    for class_name in CLASS_ORDER:
        tables[class_name] = summary.filter(pl.col(CLASS_COL) == class_name).select(
            AOI_ID_COL,
            pl.col("class_center").alias(f"{class_name}_center"),
            pl.col("class_se").alias(f"{class_name}_se"),
            pl.col("class_records").alias(f"{class_name}_records"),
        )
    return tables["decrease"].join(tables["increase"], on=AOI_ID_COL, how="inner")


def held_out_aoi_quality(records: pl.DataFrame) -> pl.DataFrame:
    """Calculate the frozen class-balanced AOI quality target.

    Args:
        records: Held-out records carrying fixed directional quality.

    Returns:
        One uncertainty-adjusted target per eligible AOI.
    """
    wide = _wide_class_summary(_class_summary(records, FIXED_RECORD_COL, "mean", MIN_EVALUATION_CLASS_RECORDS))
    return wide.with_columns(
        (
            (pl.col("increase_center") + pl.col("decrease_center")) / 2
            - FIXED_UNCERTAINTY_PENALTY * (pl.col("increase_se") + pl.col("decrease_se")) / 2
        ).alias(TARGET_COL)
    )


def candidate_aoi_scores(
    records: pl.DataFrame,
    snr_scale: float,
    direction_scale: float,
    aggregation: str,
    class_balance_penalty: float,
    uncertainty_penalty: float,
) -> pl.DataFrame:
    """Calculate one candidate AOI score from calibration records.

    Args:
        records: Calibration records with SNR and directional margin.
        snr_scale: Saturation scale for plume detectability.
        direction_scale: Saturation scale for signed directional strength.
        aggregation: Within-class record aggregation.
        class_balance_penalty: Penalty for unequal class-specific quality.
        uncertainty_penalty: Penalty applied to within-class standard errors.

    Returns:
        One candidate score per eligible AOI.
    """
    candidate_records = records.with_columns(
        ((pl.col(SNR_COL) / snr_scale).tanh() * (pl.col(MARGIN_COL) / direction_scale).tanh()).alias(
            CANDIDATE_RECORD_COL
        )
    )
    wide = _wide_class_summary(
        _class_summary(candidate_records, CANDIDATE_RECORD_COL, aggregation, MIN_SEARCH_CLASS_RECORDS)
    )
    return wide.with_columns(
        (
            (pl.col("increase_center") + pl.col("decrease_center")) / 2
            - class_balance_penalty * (pl.col("increase_center") - pl.col("decrease_center")).abs() / 2
            - uncertainty_penalty * (pl.col("increase_se") + pl.col("decrease_se")) / 2
        ).alias("candidate_aoi_score")
    )


def _ranking_metrics(joined: pl.DataFrame) -> dict[str, float | int]:
    # Evaluate deterministic top and bottom AOI subsets
    ordered = joined.sort("candidate_aoi_score", AOI_ID_COL, descending=[True, False])
    count = ordered.height
    if count < 8:
        return {
            "evaluated_aois": count,
            "spearman": np.nan,
            "overall_quality": np.nan,
            "top_quality": np.nan,
            "bottom_quality": np.nan,
            "top_lift": np.nan,
            "top_bottom_separation": np.nan,
        }
    subset_count = max(1, math.ceil(TOP_FRACTION * count))
    score_values = ordered["candidate_aoi_score"].to_numpy()
    quality_values = ordered[TARGET_COL].to_numpy()
    correlation = spearmanr(score_values, quality_values).statistic
    overall = float(np.mean(quality_values))
    top = float(np.mean(quality_values[:subset_count]))
    bottom = float(np.mean(quality_values[-subset_count:]))
    return {
        "evaluated_aois": count,
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "overall_quality": overall,
        "top_quality": top,
        "bottom_quality": bottom,
        "top_lift": top - overall,
        "top_bottom_separation": top - bottom,
    }


def sweep_aoi_heuristics(records: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, object]]:
    """Search candidate aggregation rules across held-out record folds.

    Args:
        records: Scored directional records with deterministic fold indices.

    Returns:
        Ranked parameter results and the best parameter dictionary.
    """
    rows = []
    for fold in range(RECORD_FOLDS):
        calibration = records.filter(pl.col("record_fold") != fold)
        evaluation = held_out_aoi_quality(records.filter(pl.col("record_fold") == fold))
        for snr_scale, direction_scale, aggregation in product(SNR_SCALES, DIRECTION_SCALES, AGGREGATIONS):
            for class_penalty, uncertainty_penalty in product(
                CLASS_BALANCE_PENALTIES,
                UNCERTAINTY_PENALTIES,
            ):
                candidate = candidate_aoi_scores(
                    calibration,
                    snr_scale,
                    direction_scale,
                    aggregation,
                    class_penalty,
                    uncertainty_penalty,
                )
                metrics = _ranking_metrics(
                    candidate.select(AOI_ID_COL, "candidate_aoi_score").join(
                        evaluation.select(AOI_ID_COL, TARGET_COL),
                        on=AOI_ID_COL,
                        how="inner",
                    )
                )
                rows.append(
                    {
                        "fold": fold,
                        "snr_scale": snr_scale,
                        "direction_scale": direction_scale,
                        "aggregation": aggregation,
                        "class_balance_penalty": class_penalty,
                        "uncertainty_penalty": uncertainty_penalty,
                        **metrics,
                    }
                )
    fold_results = pl.DataFrame(rows)
    parameter_columns = [
        "snr_scale",
        "direction_scale",
        "aggregation",
        "class_balance_penalty",
        "uncertainty_penalty",
    ]
    ranked = (
        fold_results.group_by(parameter_columns)
        .agg(
            pl.col("evaluated_aois").mean().alias("mean_evaluated_aois"),
            pl.col("spearman").mean().alias("mean_spearman"),
            pl.col("overall_quality").mean().alias("mean_overall_quality"),
            pl.col("top_quality").mean().alias("mean_top_quality"),
            pl.col("bottom_quality").mean().alias("mean_bottom_quality"),
            pl.col("top_lift").mean().alias("mean_top_lift"),
            pl.col("top_lift").std(ddof=1).fill_null(0.0).alias("std_top_lift"),
            pl.col("top_bottom_separation").mean().alias("mean_top_bottom_separation"),
        )
        .with_columns(
            (pl.col("mean_top_lift") + 0.25 * pl.col("mean_spearman") - 0.10 * pl.col("std_top_lift")).alias(
                "selection_objective"
            )
        )
        .sort("selection_objective", descending=True)
    )
    best = ranked.row(0, named=True)
    return ranked, {column: best[column] for column in parameter_columns}


def cross_fitted_best_scores(records: pl.DataFrame, best: dict[str, object]) -> pl.DataFrame:
    """Build held-out AOI predictions for the selected heuristic.

    Args:
        records: Scored directional records.
        best: Selected heuristic parameters.

    Returns:
        Cross-fitted AOI scores paired with held-out quality.
    """
    frames = []
    for fold in range(RECORD_FOLDS):
        candidate = candidate_aoi_scores(
            records.filter(pl.col("record_fold") != fold),
            float(best["snr_scale"]),
            float(best["direction_scale"]),
            str(best["aggregation"]),
            float(best["class_balance_penalty"]),
            float(best["uncertainty_penalty"]),
        )
        quality = held_out_aoi_quality(records.filter(pl.col("record_fold") == fold))
        frames.append(
            candidate.select(AOI_ID_COL, "candidate_aoi_score")
            .join(quality.select(AOI_ID_COL, TARGET_COL), on=AOI_ID_COL, how="inner")
            .with_columns(pl.lit(fold).alias("fold"))
        )
    return pl.concat(frames)


def _fuel_expressions() -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    # Derive broad unit-level fuel-family flags
    fuel = pl.coalesce("primaryFuelInfo", "attributePrimaryFuelInfo").fill_null("").str.to_lowercase()
    return fuel.str.contains("coal"), fuel.str.contains("natural gas"), fuel.str.contains("oil")


def calculate_emissions_features(
    raw_records: pl.LazyFrame,
    membership: pl.DataFrame,
    raster_records: pl.DataFrame,
) -> pl.DataFrame:
    """Build broad static and activity-conditioned AOI features.

    Args:
        raw_records: Full unit-hour emissions history.
        membership: Facility-to-AOI membership table.
        raster_records: Prior dataset records carrying AOI centers.

    Returns:
        One numeric emissions-feature row per AOI.
    """
    coal, gas, oil = _fuel_expressions()
    tagged = raw_records.with_columns(
        coal.alias("_is_coal"),
        gas.alias("_is_gas"),
        oil.alias("_is_oil"),
        pl.col("unitType").fill_null("").str.to_lowercase().alias("_unit_type"),
    )
    units = (
        tagged.group_by("facilityId", "unitId")
        .agg(
            pl.col("_is_coal").any(),
            pl.col("_is_gas").any(),
            pl.col("_is_oil").any(),
            pl.col("_unit_type").str.contains("boiler").any().alias("_is_boiler"),
            pl.col("_unit_type").str.contains("combined cycle").any().alias("_is_combined_cycle"),
            pl.col("_unit_type").str.contains("combustion turbine").any().alias("_is_turbine"),
            pl.col("maxHourlyHIRate").max().alias("_max_hourly_hi_rate"),
            pl.col("noxControlInfo").is_not_null().any().alias("_has_nox_control"),
        )
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.len().alias("unit_count"),
            pl.col("facilityId").n_unique().alias("facility_count"),
            pl.col("_is_coal").sum().alias("coal_unit_count"),
            pl.col("_is_gas").sum().alias("gas_unit_count"),
            pl.col("_is_oil").sum().alias("oil_unit_count"),
            pl.col("_is_boiler").mean().alias("boiler_unit_fraction"),
            pl.col("_is_combined_cycle").mean().alias("combined_cycle_unit_fraction"),
            pl.col("_is_turbine").mean().alias("turbine_unit_fraction"),
            pl.col("_has_nox_control").mean().alias("nox_control_unit_fraction"),
            pl.col("_max_hourly_hi_rate").sum().alias("total_max_hourly_hi_rate"),
            pl.col("_max_hourly_hi_rate").mean().alias("mean_max_hourly_hi_rate"),
        )
        .collect(engine="streaming")
    )
    facilities = (
        raw_records.group_by("facilityId")
        .agg(pl.col("facility_nameplate_capacity_mw").max().alias("_facility_capacity_mw"))
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("_facility_capacity_mw").sum().alias("total_nameplate_capacity_mw"),
            pl.col("_facility_capacity_mw").max().alias("largest_facility_capacity_mw"),
        )
        .collect(engine="streaming")
    )

    hourly = (
        tagged.filter(pl.col("opTime").is_finite() & (pl.col("opTime") >= 0))
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(
            pl.col("opTime").mean().alias("mean_unit_op_time"),
            (pl.col("opTime") > 0).sum().alias("operating_unit_count"),
            pl.col("noxMass")
            .filter(usable_nox_measurement_expr() & pl.col("noxMass").is_finite() & (pl.col("noxMass") >= 0))
            .sum()
            .alias("total_nox"),
            pl.col("noxMass")
            .filter(
                pl.col("_is_coal")
                & usable_nox_measurement_expr()
                & pl.col("noxMass").is_finite()
                & (pl.col("noxMass") >= 0)
            )
            .sum()
            .alias("coal_nox"),
            pl.col("noxMass")
            .filter(
                pl.col("_is_gas")
                & usable_nox_measurement_expr()
                & pl.col("noxMass").is_finite()
                & (pl.col("noxMass") >= 0)
            )
            .sum()
            .alias("gas_nox"),
            pl.col("heatInput")
            .filter(pl.col("heatInput").is_finite() & (pl.col("heatInput") >= 0))
            .sum()
            .alias("total_heat_input"),
            pl.col("grossLoad")
            .filter(pl.col("grossLoad").is_finite() & (pl.col("grossLoad") >= 0))
            .sum()
            .alias("total_gross_load"),
        )
        .collect(engine="streaming")
        .sort(AOI_ID_COL, "emissions_hour_utc")
        .with_columns(pl.col("total_nox").diff().abs().over(AOI_ID_COL).alias("absolute_hourly_nox_change"))
    )
    medians = hourly.group_by(AOI_ID_COL).agg(pl.col("mean_unit_op_time").median().alias("median_mean_unit_op_time"))
    active = hourly.join(medians, on=AOI_ID_COL, how="inner").filter(
        pl.col("mean_unit_op_time") >= pl.col("median_mean_unit_op_time")
    )
    active_features = active.group_by(AOI_ID_COL).agg(
        pl.len().alias("active_history_hours"),
        pl.col("total_nox").mean().alias("active_mean_total_nox"),
        pl.col("total_nox").median().alias("active_median_total_nox"),
        pl.col("total_nox").quantile(0.75).alias("active_q75_total_nox"),
        pl.col("total_nox").quantile(0.90).alias("active_q90_total_nox"),
        pl.col("total_nox").std(ddof=1).alias("active_std_total_nox"),
        pl.col("coal_nox").mean().alias("active_mean_coal_nox"),
        pl.col("gas_nox").mean().alias("active_mean_gas_nox"),
        pl.col("total_heat_input").mean().alias("active_mean_heat_input"),
        pl.col("total_heat_input").median().alias("active_median_heat_input"),
        pl.col("total_heat_input").quantile(0.90).alias("active_q90_heat_input"),
        pl.col("total_gross_load").mean().alias("active_mean_gross_load"),
        pl.col("total_gross_load").quantile(0.90).alias("active_q90_gross_load"),
        pl.col("operating_unit_count").mean().alias("active_mean_operating_units"),
        pl.col("absolute_hourly_nox_change").mean().alias("active_mean_absolute_nox_change"),
    )
    overall_features = hourly.group_by(AOI_ID_COL).agg(
        pl.len().alias("history_hours"),
        pl.col("total_nox").mean().alias("history_mean_total_nox"),
        pl.col("total_nox").std(ddof=1).alias("history_std_total_nox"),
        (pl.col("total_nox") > 0).mean().alias("history_emitting_fraction"),
        pl.col("mean_unit_op_time").mean().alias("history_mean_unit_op_time"),
        (pl.col("mean_unit_op_time") > 0).mean().alias("history_operating_fraction"),
        pl.col("absolute_hourly_nox_change").mean().alias("history_mean_absolute_nox_change"),
    )

    aoi_centers = (
        raster_records.select(AOI_ID_COL, "lat", "lon")
        .unique(subset=AOI_ID_COL, keep="first")
        .pipe(add_projected_coordinates)
        .select(AOI_ID_COL, pl.col("x_m").alias("_aoi_x_m"), pl.col("y_m").alias("_aoi_y_m"))
    )
    facility_points = add_projected_coordinates(
        raw_records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId").collect()
    ).select("facilityId", pl.col("x_m").alias("_facility_x_m"), pl.col("y_m").alias("_facility_y_m"))
    geometry = (
        membership.join(facility_points, on="facilityId", how="inner")
        .join(aoi_centers, on=AOI_ID_COL, how="inner")
        .with_columns(
            (
                (
                    (pl.col("_facility_x_m") - pl.col("_aoi_x_m")).pow(2)
                    + (pl.col("_facility_y_m") - pl.col("_aoi_y_m")).pow(2)
                ).sqrt()
                / 1000
            ).alias("source_distance_km")
        )
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("source_distance_km").mean().alias("mean_source_distance_km"),
            pl.col("source_distance_km").median().alias("median_source_distance_km"),
            pl.col("source_distance_km").max().alias("max_source_distance_km"),
        )
    )
    return (
        units.join(facilities, on=AOI_ID_COL, how="inner")
        .join(active_features, on=AOI_ID_COL, how="inner")
        .join(overall_features, on=AOI_ID_COL, how="inner")
        .join(geometry, on=AOI_ID_COL, how="inner")
        .with_columns(
            (pl.col("unit_count") / pl.col("facility_count").clip(lower_bound=1)).alias("units_per_facility"),
            (pl.col("active_std_total_nox") / pl.col("active_mean_total_nox").clip(lower_bound=1e-6)).alias(
                "active_total_nox_cv"
            ),
            (pl.col("active_mean_coal_nox") / pl.col("active_mean_total_nox").clip(lower_bound=1e-6)).alias(
                "active_coal_nox_share"
            ),
            (pl.col("active_mean_gas_nox") / pl.col("active_mean_total_nox").clip(lower_bound=1e-6)).alias(
                "active_gas_nox_share"
            ),
            (
                pl.col("largest_facility_capacity_mw") / pl.col("total_nameplate_capacity_mw").clip(lower_bound=1e-6)
            ).alias("largest_facility_capacity_share"),
        )
    )


def complete_aoi_quality(records: pl.DataFrame) -> pl.DataFrame:
    """Calculate one fixed quality target from all sampled records.

    Args:
        records: Directional records carrying fixed quality.

    Returns:
        One target row per AOI with both classes represented.
    """
    return held_out_aoi_quality(records).select(AOI_ID_COL, pl.col(TARGET_COL).alias("aoi_directional_quality"))


def sweep_feature_thresholds(
    feature_quality: pl.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> pl.DataFrame:
    """Cross-validate simple one-feature AOI retention rules.

    Args:
        feature_quality: AOI features joined to fixed quality targets.
        feature_columns: Numeric feature names to evaluate.
        seed: Hash seed controlling AOI folds.

    Returns:
        Ranked feature, direction, and retained-fraction rules.
    """
    frame = feature_quality.with_columns(
        (pl.col(AOI_ID_COL).hash(seed=seed) % AOI_FOLDS).cast(pl.Int8).alias("aoi_fold")
    )
    rows = []
    for feature in feature_columns:
        finite = frame.filter(pl.col(feature).is_finite() & pl.col("aoi_directional_quality").is_finite())
        for fold in range(AOI_FOLDS):
            training = finite.filter(pl.col("aoi_fold") != fold)
            evaluation = finite.filter(pl.col("aoi_fold") == fold)
            if training.is_empty() or evaluation.is_empty():
                continue
            overall_quality = float(evaluation["aoi_directional_quality"].mean())
            for direction, retained_fraction in product(FEATURE_DIRECTIONS, FEATURE_RETAINED_FRACTIONS):
                quantile = 1 - retained_fraction if direction == "high" else retained_fraction
                cutoff = training[feature].quantile(quantile)
                if cutoff is None:
                    continue
                predicate = pl.col(feature) >= cutoff if direction == "high" else pl.col(feature) <= cutoff
                selected = evaluation.filter(predicate)
                if selected.is_empty():
                    continue
                selected_quality = float(selected["aoi_directional_quality"].mean())
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "retained_fraction": retained_fraction,
                        "fold": fold,
                        "training_cutoff": float(cutoff),
                        "evaluation_aois": evaluation.height,
                        "selected_aois": selected.height,
                        "selected_quality": selected_quality,
                        "quality_lift": selected_quality - overall_quality,
                    }
                )
    fold_results = pl.DataFrame(rows)
    return (
        fold_results.group_by("feature", "direction", "retained_fraction")
        .agg(
            pl.col("training_cutoff").median().alias("median_cutoff"),
            pl.col("evaluation_aois").sum(),
            pl.col("selected_aois").sum(),
            pl.col("selected_quality").mean().alias("mean_selected_quality"),
            pl.col("quality_lift").mean().alias("mean_quality_lift"),
            pl.col("quality_lift").std(ddof=1).fill_null(0.0).alias("std_quality_lift"),
        )
        .with_columns((pl.col("mean_quality_lift") - 0.25 * pl.col("std_quality_lift")).alias("selection_objective"))
        .sort("selection_objective", descending=True)
    )


def evaluate_random_forest(
    feature_quality: pl.DataFrame,
    feature_columns: list[str],
    seed: int,
    workers: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Cross-validate a shallow nonlinear emissions-feature model.

    Args:
        feature_quality: AOI features joined to quality targets.
        feature_columns: Numeric predictor names.
        seed: Random forest and fold seed.
        workers: Maximum model threads.

    Returns:
        Fold metrics, cross-fitted predictions, and mean feature importances.
    """
    frame = feature_quality.with_columns(
        (pl.col(AOI_ID_COL).hash(seed=seed) % AOI_FOLDS).cast(pl.Int8).alias("aoi_fold")
    )
    prediction_rows = []
    metric_rows = []
    importance_rows = []
    for fold in range(AOI_FOLDS):
        training = frame.filter(pl.col("aoi_fold") != fold)
        evaluation = frame.filter(pl.col("aoi_fold") == fold)
        train_x = training.select(feature_columns).to_numpy().astype(np.float64)
        eval_x = evaluation.select(feature_columns).to_numpy().astype(np.float64)
        medians = np.nanmedian(np.where(np.isfinite(train_x), train_x, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        train_x = np.where(np.isfinite(train_x), train_x, medians)
        eval_x = np.where(np.isfinite(eval_x), eval_x, medians)
        train_y = training["aoi_directional_quality"].to_numpy()
        eval_y = evaluation["aoi_directional_quality"].to_numpy()
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=20,
            max_features=0.75,
            random_state=seed + fold,
            n_jobs=min(workers, 8),
        )
        model.fit(train_x, train_y)
        predictions = model.predict(eval_x)
        joined = pl.DataFrame(
            {
                AOI_ID_COL: evaluation[AOI_ID_COL],
                "candidate_aoi_score": predictions,
                TARGET_COL: eval_y,
            }
        )
        metric_rows.append({"fold": fold, **_ranking_metrics(joined)})
        prediction_rows.extend(
            {
                "fold": fold,
                AOI_ID_COL: int(aoi_id),
                "predicted_quality": float(prediction),
                "observed_quality": float(observed),
            }
            for aoi_id, prediction, observed in zip(
                evaluation[AOI_ID_COL],
                predictions,
                eval_y,
                strict=True,
            )
        )
        importance_rows.extend(
            {"fold": fold, "feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_columns, model.feature_importances_, strict=True)
        )
    importances = (
        pl.DataFrame(importance_rows)
        .group_by("feature")
        .agg(
            pl.col("importance").mean().alias("mean_importance"),
            pl.col("importance").std(ddof=1).alias("std_importance"),
        )
        .sort("mean_importance", descending=True)
    )
    return pl.DataFrame(metric_rows), pl.DataFrame(prediction_rows), importances


def write_diagnostic_plot(
    heuristic_results: pl.DataFrame,
    cross_fitted_scores: pl.DataFrame,
    feature_results: pl.DataFrame,
    importances: pl.DataFrame,
    output_path: Path,
) -> None:
    """Plot heuristic ranking and emissions-feature results.

    Args:
        heuristic_results: Ranked AOI heuristic sweep.
        cross_fitted_scores: Best-heuristic held-out scores and targets.
        feature_results: Ranked univariate feature rules.
        importances: Cross-validated random forest importances.
        output_path: Destination PNG.
    """
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    top_heuristics = heuristic_results.head(10).with_row_index("rank", offset=1)
    axes[0, 0].barh(
        top_heuristics["rank"].cast(pl.String).to_list()[::-1],
        top_heuristics["selection_objective"].to_numpy()[::-1],
        color="#315f72",
    )
    axes[0, 0].set(title="Top heuristic configurations", xlabel="Cross-fold selection objective", ylabel="Rank")

    ordered = cross_fitted_scores.sort("candidate_aoi_score").with_row_index("_order")
    deciles = (
        ordered.with_columns(
            (pl.col("_order") * 10 / pl.len()).floor().cast(pl.Int8).clip(upper_bound=9).alias("score_decile")
        )
        .group_by("score_decile")
        .agg(pl.col(TARGET_COL).mean())
        .sort("score_decile")
    )
    axes[0, 1].plot(deciles["score_decile"].to_numpy() + 1, deciles[TARGET_COL].to_numpy(), marker="o")
    axes[0, 1].set(
        title="Held-out quality by predicted-score decile",
        xlabel="Candidate AOI-score decile",
        ylabel="Held-out directional quality",
    )

    best_features = (
        feature_results.unique(subset="feature", keep="first", maintain_order=True).head(12).sort("selection_objective")
    )
    axes[1, 0].barh(
        best_features["feature"].to_list(),
        best_features["mean_quality_lift"].to_numpy(),
        color="#5b8e7d",
    )
    axes[1, 0].set(title="Best one-feature filters", xlabel="Held-out quality lift")

    top_importances = importances.head(12).sort("mean_importance")
    axes[1, 1].barh(
        top_importances["feature"].to_list(),
        top_importances["mean_importance"].to_numpy(),
        color="#b24c63",
    )
    axes[1, 1].set(title="Shallow random-forest importance", xlabel="Mean impurity importance")
    figure.suptitle("AOI scoring quality search", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run held-out AOI-score and emissions-feature searches."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raster_records = load_prior_dataset_records(args.dataframe_dir)
    score_sample = sample_score_records(raster_records, args.seed)
    print(f"Scoring {score_sample.height:,} sampled raster bundles", flush=True)
    record_scores = score_sampled_rasters(score_sample, args.dataset_dir, args.workers)

    raw_columns = [
        "facilityId",
        "unitId",
        "lat",
        "lon",
        "emissions_hour_utc",
        "opTime",
        "noxMass",
        "noxMassMeasureFlg",
        "heatInput",
        "grossLoad",
        "primaryFuelInfo",
        "attributePrimaryFuelInfo",
        "unitType",
        "maxHourlyHIRate",
        "noxControlInfo",
        "facility_nameplate_capacity_mw",
    ]
    raw_records = pl.scan_parquet(FULL_DATA_PARQUET).select(raw_columns)
    membership = build_membership(raster_records, raw_records)
    print(f"Built {membership.height:,} facility-to-AOI memberships", flush=True)
    hourly_nox = calculate_hourly_aoi_nox(raw_records, membership)
    labeled = add_current_labels(raster_records, hourly_nox)
    compatible, plume_delta_scale = add_label_compatibility(
        labeled.join(record_scores, on=RASTER_PATH_COL, how="inner")
    )
    directional_records = add_fixed_record_quality(compatible, args.seed)
    directional_records.write_parquet(args.output_dir / "record_directional_quality.parquet")
    print(
        f"Built {directional_records.height:,} current-label directional records; "
        f"plume-delta scale={plume_delta_scale:.4f}",
        flush=True,
    )

    heuristic_results, best_heuristic = sweep_aoi_heuristics(directional_records)
    cross_fitted = cross_fitted_best_scores(directional_records, best_heuristic)
    heuristic_results.write_csv(args.output_dir / "heuristic_sweep.csv")
    cross_fitted.write_csv(args.output_dir / "best_heuristic_cross_fitted.csv")
    print(f"Best heuristic: {best_heuristic}", flush=True)

    emissions_features = calculate_emissions_features(raw_records, membership, raster_records)
    quality = complete_aoi_quality(directional_records)
    feature_quality = emissions_features.join(quality, on=AOI_ID_COL, how="inner")
    feature_columns = [column for column in emissions_features.columns if column != AOI_ID_COL]
    feature_results = sweep_feature_thresholds(feature_quality, feature_columns, args.seed)
    rf_metrics, rf_predictions, rf_importances = evaluate_random_forest(
        feature_quality,
        feature_columns,
        args.seed,
        args.workers,
    )
    feature_quality.write_csv(args.output_dir / "aoi_quality_and_emissions_features.csv")
    feature_results.write_csv(args.output_dir / "feature_threshold_sweep.csv")
    rf_metrics.write_csv(args.output_dir / "random_forest_metrics.csv")
    rf_predictions.write_csv(args.output_dir / "random_forest_predictions.csv")
    rf_importances.write_csv(args.output_dir / "random_forest_importances.csv")

    plot_path = args.output_dir / "aoi_score_quality_search.png"
    write_diagnostic_plot(
        heuristic_results,
        cross_fitted,
        feature_results,
        rf_importances,
        plot_path,
    )
    best_result = heuristic_results.row(0, named=True)
    best_feature_rules = feature_results.unique(subset="feature", keep="first", maintain_order=True).head(15)
    summary = {
        "label_system": "overlap-interpolated four-timestep irregular EMA ending at t3",
        "normalization_center": 1868138303979520.0,
        "normalization_scale": 1199722224989936.2,
        "normalization_sample_bundles": 10000,
        "sampled_raster_records": score_sample.height,
        "directional_records": directional_records.height,
        "quality_aois": feature_quality.height,
        "plume_delta_scale": plume_delta_scale,
        "best_heuristic": best_heuristic,
        "best_heuristic_metrics": best_result,
        "best_feature_rules": best_feature_rules.to_dicts(),
        "random_forest_mean_metrics": rf_metrics.select(pl.exclude("fold")).mean().to_dicts()[0],
        "top_random_forest_features": rf_importances.head(15).to_dicts(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved AOI score search outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
