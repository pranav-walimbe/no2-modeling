"""Evaluation utilities for effective NOx-change regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRUE_TARGET_COL = "y_true"
PREDICTION_COL = "y_pred"
RESIDUAL_COL = "residual"
ABSOLUTE_ERROR_COL = "absolute_error"
MODEL_COMPARISON_METRICS = ("mae", "rmse", "r2", "pearson_r", "spearman_r")


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    """Calculate regression errors and associations.

    Args:
        y_true: Ground-truth continuous targets.
        prediction: Continuous model predictions.

    Returns:
        Sample count, error metrics, bias, and correlations.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    residual = predicted - truth
    has_variation = truth.size > 1 and np.std(truth) > 0 and np.std(predicted) > 0
    pearson = float(np.corrcoef(truth, predicted)[0, 1]) if has_variation else None
    spearman = float(pd.Series(truth).corr(pd.Series(predicted), method="spearman")) if has_variation else None
    return {
        "n": int(truth.size),
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)) if truth.size > 1 else None,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "mean_bias": float(residual.mean()),
    }


def plant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate regression metrics for each held-out AOI.

    Args:
        frame: Row-level regression predictions with AOI coordinates.

    Returns:
        One regression summary row per AOI.
    """
    rows = []
    for (aoi_id, lon, lat), group in frame.groupby(["aoi_id", "lon", "lat"], sort=True):
        metrics = regression_metrics(group[TRUE_TARGET_COL].to_numpy(), group[PREDICTION_COL].to_numpy())
        rows.append({"aoi_id": aoi_id, "lon": lon, "lat": lat, **metrics})
    return pd.DataFrame(rows)


def _model_results(split_frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    # Summarize natural splits and fixed equal-count test magnitude slices
    results: dict[str, object] = {
        "splits": {
            name: regression_metrics(frame[TRUE_TARGET_COL].to_numpy(), frame[PREDICTION_COL].to_numpy())
            for name, frame in split_frames.items()
        }
    }
    test = split_frames["test"]
    ordered_indices = np.argsort(np.abs(test[TRUE_TARGET_COL].to_numpy(dtype=np.float64)), kind="stable")
    slices = {
        name: test.iloc[indices]
        for name, indices in zip(("low", "mid", "high"), np.array_split(ordered_indices, 3), strict=True)
    }
    results["test_absolute_target_tertiles"] = {
        name: regression_metrics(subset[TRUE_TARGET_COL].to_numpy(), subset[PREDICTION_COL].to_numpy())
        for name, subset in slices.items()
    }
    return results


def _model_comparison(
    model_results: dict[str, dict[str, object]],
    primary_model_name: str,
    comparison_model_name: str,
) -> dict[str, object]:
    # Place both scores and signed primary-minus-comparison differences together
    primary = model_results[primary_model_name]["splits"]
    comparison = model_results[comparison_model_name]["splits"]

    def metric_values(split: str, metric: str) -> dict[str, float | None]:
        primary_value = primary[split][metric]
        comparison_value = comparison[split][metric]
        difference = None if primary_value is None or comparison_value is None else primary_value - comparison_value
        return {
            primary_model_name: primary_value,
            comparison_model_name: comparison_value,
            f"{primary_model_name}_minus_{comparison_model_name}": difference,
        }

    return {split: {metric: metric_values(split, metric) for metric in MODEL_COMPARISON_METRICS} for split in primary}


def save_results(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
    *,
    primary_model_name: str,
    comparison_model_name: str | None = None,
) -> None:
    """Save regression metrics and row-level predictions.

    Args:
        model_frames: Row-level predictions by model and data split.
        run_dir: Model-run output directory.
        primary_model_name: Model copied into the top-level result summary.
        comparison_model_name: Baseline model used to calculate differences.
    """
    model_results = {name: _model_results(split_frames) for name, split_frames in model_frames.items()}
    primary_results = model_results[primary_model_name]
    results: dict[str, object] = {
        "primary_model": primary_model_name,
        "splits": primary_results["splits"],
        "test_absolute_target_tertiles": primary_results["test_absolute_target_tertiles"],
        "models": model_results,
    }
    if comparison_model_name is not None:
        results["comparison"] = _model_comparison(model_results, primary_model_name, comparison_model_name)

    output_dir = Path(run_dir)
    with (output_dir / "results.json").open("w") as destination:
        json.dump(results, destination, indent=2)
    for model_name, split_frames in model_frames.items():
        for split, frame in split_frames.items():
            filename = (
                f"{split}_predictions.csv"
                if model_name == primary_model_name
                else f"{model_name}_{split}_predictions.csv"
            )
            frame.to_csv(output_dir / filename, index=False)
