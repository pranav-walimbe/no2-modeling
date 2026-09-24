"""Refine temporal SNR aggregation using frozen directional-quality records."""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

from config import VIS_DIR
from eda.aoi_nox_score_montage import CLASS_COL
from eda.aoi_plume_snr_montage import PLUME_DELTA_Z_COL, SNR_COL
from eda.aoi_score_quality_search import (
    AOI_ID_COL,
    MIN_SEARCH_CLASS_RECORDS,
    RECORD_FOLDS,
    TARGET_COL,
    _ranking_metrics,
    held_out_aoi_quality,
    sweep_aoi_heuristics,
)

DEFAULT_INPUT = Path(VIS_DIR) / "aoi-score-quality-search-39195305" / "record_directional_quality.parquet"
BASELINE_AGGREGATION = "median_all"
TIMESTEPS = 4
REFINEMENT_SNR_AGGREGATIONS = ("upper_mean_all", "max_all")
REFINEMENT_SNR_SCALES = (0.25, 0.35, 0.50)
REFINEMENT_DIRECTION_SCALES = (0.50, 0.75, 1.00)
REFINEMENT_SHRINKAGE_COUNTS = (0.0, 2.0, 5.0, 10.0)
REFINEMENT_CLASS_PENALTIES = (0.0, 0.5, 1.0)
REFINEMENT_UNCERTAINTY_PENALTIES = (0.25, 0.5, 0.75)
REFINEMENT_DOWNSIDE_PENALTIES = (0.0, 0.25, 0.5)
LOCAL_MAX_WEIGHTS = (0.50, 0.625, 0.75, 0.875, 1.00)
LOCAL_SNR_SCALES = (0.30, 0.35, 0.40, 0.45, 0.50)
LOCAL_DIRECTION_SCALES = (0.60, 0.75, 0.90, 1.05)
LOCAL_SHRINKAGE_COUNTS = (3.0, 5.0, 7.0, 10.0)
LOCAL_UNCERTAINTY_PENALTIES = (0.25, 0.375, 0.50, 0.625)
FINAL_SNR_AGGREGATION = "max_all"
FINAL_SNR_SCALE = 0.50
FINAL_DIRECTION_SCALE = 0.75
FINAL_SHRINKAGE_COUNT = 7.0
FINAL_CLASS_PENALTY = 0.0
FINAL_UNCERTAINTY_PENALTY = 0.25
FINAL_DOWNSIDE_PENALTY = 0.0


def parse_args() -> argparse.Namespace:
    """Parse refinement options."""
    job_id = os.getenv("SLURM_JOB_ID", "latest")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--stage",
        choices=("temporal", "reliability", "local"),
        default="temporal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(VIS_DIR) / f"aoi-score-refinement-{job_id}",
    )
    return parser.parse_args()


def add_snr_aggregations(records: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Add candidate temporal summaries of four timestep SNR values.

    Args:
        records: Frozen directional-quality records from the full raster scan.

    Returns:
        Records with candidate columns and their aggregation names.
    """
    snr_columns = [f"plume_snr_t{index}" for index in range(TIMESTEPS)]
    values = records.select(snr_columns).to_numpy()
    sorted_values = np.sort(values, axis=1)
    candidates = {
        "median_all": np.nanmedian(values, axis=1),
        "mean_all": np.nanmean(values, axis=1),
        "upper_mean_all": np.nanmean(sorted_values[:, -2:], axis=1),
        "max_all": np.nanmax(values, axis=1),
        "mean_label_pair": np.nanmean(values[:, 2:4], axis=1),
        "max_label_pair": np.nanmax(values[:, 2:4], axis=1),
        "current": values[:, 3],
    }
    for weight in LOCAL_MAX_WEIGHTS:
        name = f"top_weight_{str(weight).replace('.', '_')}"
        candidates[name] = (1 - weight) * sorted_values[:, -2] + weight * sorted_values[:, -1]
    expressions = [pl.Series(f"_snr_{name}", value) for name, value in candidates.items()]
    return records.with_columns(expressions), list(candidates)


def run_snr_search(records: pl.DataFrame, aggregation_names: list[str]) -> pl.DataFrame:
    """Run the frozen AOI search for each temporal SNR aggregation.

    Args:
        records: Records carrying every candidate SNR aggregation.
        aggregation_names: Candidate aggregation names.

    Returns:
        All parameter results ranked across temporal aggregations.
    """
    results = []
    for name in aggregation_names:
        candidate_records = records.with_columns(pl.col(f"_snr_{name}").alias(SNR_COL))
        ranked, _ = sweep_aoi_heuristics(candidate_records)
        results.append(ranked.with_columns(pl.lit(name).alias("snr_aggregation")))
        best = ranked.row(0, named=True)
        print(
            f"{name}: objective={best['selection_objective']:.6f}, "
            f"Spearman={best['mean_spearman']:.6f}, top lift={best['mean_top_lift']:.6f}",
            flush=True,
        )
    return pl.concat(results).sort("selection_objective", descending=True)


def summarize(results: pl.DataFrame) -> dict[str, object]:
    """Summarize the best refinement and its change from baseline.

    Args:
        results: Complete temporal aggregation search.

    Returns:
        JSON-serializable result summary.
    """
    best_by_aggregation = results.unique(
        subset="snr_aggregation",
        keep="first",
        maintain_order=True,
    )
    best = best_by_aggregation.row(0, named=True)
    baseline = best_by_aggregation.filter(pl.col("snr_aggregation") == BASELINE_AGGREGATION).row(0, named=True)
    metric_names = [
        "selection_objective",
        "mean_spearman",
        "mean_top_lift",
        "std_top_lift",
        "mean_top_bottom_separation",
    ]
    return {
        "baseline_aggregation": BASELINE_AGGREGATION,
        "best_configuration": best,
        "baseline_configuration": baseline,
        "best_minus_baseline": {name: float(best[name] - baseline[name]) for name in metric_names},
        "best_by_aggregation": best_by_aggregation.to_dicts(),
    }


def _label_sign_expr() -> pl.Expr:
    # Map current directional classes to numeric signs
    return pl.when(pl.col(CLASS_COL) == "increase").then(pl.lit(1.0)).otherwise(pl.lit(-1.0))


def refined_candidate_scores(
    records: pl.DataFrame,
    snr_column: str,
    snr_scale: float,
    direction_scale: float,
    shrinkage_count: float,
    class_penalty: float,
    uncertainty_penalty: float,
    downside_penalty: float,
) -> pl.DataFrame:
    """Calculate reliability-adjusted AOI scores from calibration records.

    Args:
        records: Calibration records from two deterministic folds.
        snr_column: Temporal SNR aggregation to use.
        snr_scale: SNR saturation scale.
        direction_scale: Directional-strength saturation scale.
        shrinkage_count: Neutral pseudo-count applied to each class mean.
        class_penalty: Penalty for unequal class-specific centers.
        uncertainty_penalty: Penalty for class-specific standard errors.
        downside_penalty: Penalty for negative lower-quartile record quality.

    Returns:
        One candidate score per eligible AOI.
    """
    record_quality = (pl.col(snr_column) / snr_scale).tanh() * (
        _label_sign_expr() * pl.col(PLUME_DELTA_Z_COL) / direction_scale
    ).tanh()
    summary = (
        records.with_columns(record_quality.alias("_candidate_quality"))
        .group_by(AOI_ID_COL, CLASS_COL)
        .agg(
            pl.col("_candidate_quality").mean().alias("_mean"),
            pl.col("_candidate_quality").std(ddof=1).fill_null(0.0).alias("_std"),
            pl.col("_candidate_quality").quantile(0.25, interpolation="linear").alias("_q25"),
            pl.len().alias("_records"),
        )
        .filter(pl.col("_records") >= MIN_SEARCH_CLASS_RECORDS)
        .with_columns(
            (pl.col("_mean") * pl.col("_records") / (pl.col("_records") + shrinkage_count)).alias("_center"),
            (pl.col("_std") / pl.col("_records").cast(pl.Float64).sqrt()).alias("_se"),
            (-pl.col("_q25")).clip(lower_bound=0.0).alias("_downside"),
        )
    )
    class_tables = {}
    for class_name in ("decrease", "increase"):
        class_tables[class_name] = summary.filter(pl.col(CLASS_COL) == class_name).select(
            AOI_ID_COL,
            *(pl.col(column).alias(f"{class_name}{column}") for column in ("_center", "_se", "_downside", "_records")),
        )
    return (
        class_tables["decrease"]
        .join(class_tables["increase"], on=AOI_ID_COL, how="inner")
        .with_columns(
            (
                (pl.col("decrease_center") + pl.col("increase_center")) / 2
                - class_penalty * (pl.col("decrease_center") - pl.col("increase_center")).abs() / 2
                - uncertainty_penalty * (pl.col("decrease_se") + pl.col("increase_se")) / 2
                - downside_penalty * (pl.col("decrease_downside") + pl.col("increase_downside")) / 2
            ).alias("candidate_aoi_score")
        )
    )


def calculate_final_aoi_scores(records: pl.DataFrame) -> pl.DataFrame:
    """Calculate the selected label-aware AOI score from frozen record metrics.

    Args:
        records: Directional-quality records carrying four timestep SNRs.

    Returns:
        Final AOI scores and class-specific reliability diagnostics.
    """
    with_aggregations, _ = add_snr_aggregations(records)
    return (
        refined_candidate_scores(
            with_aggregations,
            f"_snr_{FINAL_SNR_AGGREGATION}",
            FINAL_SNR_SCALE,
            FINAL_DIRECTION_SCALE,
            FINAL_SHRINKAGE_COUNT,
            FINAL_CLASS_PENALTY,
            FINAL_UNCERTAINTY_PENALTY,
            FINAL_DOWNSIDE_PENALTY,
        )
        .rename({"candidate_aoi_score": "final_aoi_score"})
        .sort("final_aoi_score", descending=True)
    )


def _run_reliability_grid(
    records: pl.DataFrame,
    snr_aggregations: tuple[str, ...],
    snr_scales: tuple[float, ...],
    direction_scales: tuple[float, ...],
    shrinkage_counts: tuple[float, ...],
    class_penalties: tuple[float, ...],
    uncertainty_penalties: tuple[float, ...],
    downside_penalties: tuple[float, ...],
) -> pl.DataFrame:
    # Evaluate one reliability grid against the frozen held-out target
    rows = []
    configurations = product(
        snr_aggregations,
        snr_scales,
        direction_scales,
        shrinkage_counts,
        class_penalties,
        uncertainty_penalties,
        downside_penalties,
    )
    for configuration in configurations:
        snr_aggregation, snr_scale, direction_scale, shrinkage, class_penalty, uncertainty, downside = configuration
        fold_metrics = []
        for fold in range(RECORD_FOLDS):
            candidates = refined_candidate_scores(
                records.filter(pl.col("record_fold") != fold),
                f"_snr_{snr_aggregation}",
                snr_scale,
                direction_scale,
                shrinkage,
                class_penalty,
                uncertainty,
                downside,
            )
            evaluation = held_out_aoi_quality(records.filter(pl.col("record_fold") == fold))
            fold_metrics.append(
                _ranking_metrics(
                    candidates.select(AOI_ID_COL, "candidate_aoi_score").join(
                        evaluation.select(AOI_ID_COL, TARGET_COL),
                        on=AOI_ID_COL,
                        how="inner",
                    )
                )
            )
        rows.append(
            {
                "snr_aggregation": snr_aggregation,
                "snr_scale": snr_scale,
                "direction_scale": direction_scale,
                "shrinkage_count": shrinkage,
                "class_balance_penalty": class_penalty,
                "uncertainty_penalty": uncertainty,
                "downside_penalty": downside,
                "mean_evaluated_aois": float(np.mean([metric["evaluated_aois"] for metric in fold_metrics])),
                "mean_spearman": float(np.mean([metric["spearman"] for metric in fold_metrics])),
                "mean_top_lift": float(np.mean([metric["top_lift"] for metric in fold_metrics])),
                "std_top_lift": float(np.std([metric["top_lift"] for metric in fold_metrics], ddof=1)),
                "mean_top_bottom_separation": float(
                    np.mean([metric["top_bottom_separation"] for metric in fold_metrics])
                ),
                "minimum_fold_top_lift": float(min(metric["top_lift"] for metric in fold_metrics)),
            }
        )
    return (
        pl.DataFrame(rows)
        .with_columns(
            (pl.col("mean_top_lift") + 0.25 * pl.col("mean_spearman") - 0.25 * pl.col("std_top_lift")).alias(
                "selection_objective"
            )
        )
        .sort("selection_objective", descending=True)
    )


def run_reliability_search(records: pl.DataFrame) -> pl.DataFrame:
    """Search reliability adjustments around the temporal finalists.

    Args:
        records: Frozen records carrying temporal SNR aggregations.

    Returns:
        Ranked cross-fold parameter results.
    """
    return _run_reliability_grid(
        records,
        REFINEMENT_SNR_AGGREGATIONS,
        REFINEMENT_SNR_SCALES,
        REFINEMENT_DIRECTION_SCALES,
        REFINEMENT_SHRINKAGE_COUNTS,
        REFINEMENT_CLASS_PENALTIES,
        REFINEMENT_UNCERTAINTY_PENALTIES,
        REFINEMENT_DOWNSIDE_PENALTIES,
    )


def run_local_search(records: pl.DataFrame) -> pl.DataFrame:
    """Search interpolated top-timestep SNR summaries near the best region.

    Args:
        records: Frozen records carrying temporal SNR aggregations.

    Returns:
        Ranked local parameter results.
    """
    aggregation_names = tuple(f"top_weight_{str(weight).replace('.', '_')}" for weight in LOCAL_MAX_WEIGHTS)
    return _run_reliability_grid(
        records,
        aggregation_names,
        LOCAL_SNR_SCALES,
        LOCAL_DIRECTION_SCALES,
        LOCAL_SHRINKAGE_COUNTS,
        (0.0,),
        LOCAL_UNCERTAINTY_PENALTIES,
        (0.0,),
    )


def summarize_reliability(results: pl.DataFrame) -> dict[str, object]:
    """Summarize the reliability search and its robust finalists.

    Args:
        results: Ranked reliability configurations.

    Returns:
        JSON-serializable result summary.
    """
    best = results.row(0, named=True)
    best_by_temporal = results.unique(
        subset="snr_aggregation",
        keep="first",
        maintain_order=True,
    )
    return {
        "best_configuration": best,
        "best_by_temporal_aggregation": best_by_temporal.to_dicts(),
        "top_configurations": results.head(20).to_dicts(),
    }


def main() -> None:
    """Run the temporal SNR refinement."""
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Directional-quality input not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, aggregation_names = add_snr_aggregations(pl.read_parquet(args.input))
    if args.stage == "temporal":
        results = run_snr_search(records, aggregation_names)
        result_summary = summarize(results)
        output_name = "temporal_snr_sweep.csv"
    elif args.stage == "reliability":
        results = run_reliability_search(records)
        result_summary = summarize_reliability(results)
        output_name = "reliability_sweep.csv"
    else:
        results = run_local_search(records)
        result_summary = summarize_reliability(results)
        output_name = "local_sweep.csv"
        calculate_final_aoi_scores(records).write_csv(args.output_dir / "final_aoi_scores.csv")
    results.write_csv(args.output_dir / output_name)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved refinement outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
