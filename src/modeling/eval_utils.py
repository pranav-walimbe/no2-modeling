"""Evaluation utilities for binary NOx-change classification."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRUE_CLASS_COL = "y_true"
PREDICTED_CLASS_COL = "y_pred"
POSITIVE_PROBABILITY_COL = "probability_positive"
LOGIT_COL = "logit"
MODEL_COMPARISON_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
)


def classification_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
) -> dict[str, float | int | None]:
    """Return thresholded and ranking metrics for a binary classifier.

    Args:
        y_true: Ground-truth zero and one labels.
        positive_probability: Predicted probability of class one.

    Returns:
        Counts, class metrics, and ROC AUC.
    """
    truth = np.asarray(y_true)
    probability = np.asarray(positive_probability, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.uint8)
    true_negative = int(np.sum((truth == 0) & (prediction == 0)))
    false_positive = int(np.sum((truth == 0) & (prediction == 1)))
    false_negative = int(np.sum((truth == 1) & (prediction == 0)))
    true_positive = int(np.sum((truth == 1) & (prediction == 1)))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    precision_denominator = true_positive + false_positive
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / positive_count if positive_count else None
    specificity = true_negative / negative_count if negative_count else None
    f1_denominator = 2 * true_positive + false_positive + false_negative
    balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "n": int(truth.size),
        "negative_count": negative_count,
        "positive_count": positive_count,
        "accuracy": float((true_positive + true_negative) / truth.size),
        "balanced_accuracy": float(balanced_accuracy) if balanced_accuracy is not None else None,
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(2 * true_positive / f1_denominator) if f1_denominator else None,
        "roc_auc": float(roc_auc_score(truth, probability)) if positive_count and negative_count else None,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def plant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute classification metrics for each held-out AOI.

    Args:
        frame: Row-level classifications with AOI coordinates.

    Returns:
        One classification summary row per AOI.
    """
    rows = []
    for (aoi_id, lon, lat), group in frame.groupby(["aoi_id", "lon", "lat"], sort=True):
        metrics = classification_metrics(
            group[TRUE_CLASS_COL].to_numpy(),
            group[POSITIVE_PROBABILITY_COL].to_numpy(),
        )
        rows.append({"aoi_id": aoi_id, "lon": lon, "lat": lat, **metrics})
    return pd.DataFrame(rows)


def _model_results(split_frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    # Summarize full splits and fixed equal-count test magnitude slices
    results: dict[str, object] = {
        "splits": {
            name: classification_metrics(
                frame[TRUE_CLASS_COL].to_numpy(),
                frame[POSITIVE_PROBABILITY_COL].to_numpy(),
            )
            for name, frame in split_frames.items()
        }
    }
    test = split_frames["test"]
    magnitude = np.abs(test["delta_nox_mass"].to_numpy(dtype=np.float64))
    ordered_indices = np.argsort(magnitude, kind="stable")
    slices = {
        name: test.iloc[indices]
        for name, indices in zip(("low", "mid", "high"), np.array_split(ordered_indices, 3), strict=True)
    }
    results["test_absolute_delta_tertiles"] = {
        name: classification_metrics(
            subset[TRUE_CLASS_COL].to_numpy(),
            subset[POSITIVE_PROBABILITY_COL].to_numpy(),
        )
        for name, subset in slices.items()
    }
    return results


def _model_comparison(model_results: dict[str, dict[str, object]]) -> dict[str, object]:
    # Place both scores and their deep-learning difference beside each other
    deep_learning = model_results["deep_learning"]["splits"]
    xgboost = model_results["xgboost"]["splits"]
    def metric_values(split: str, metric: str) -> dict[str, float | None]:
        deep_value = deep_learning[split][metric]
        xgboost_value = xgboost[split][metric]
        difference = None if deep_value is None or xgboost_value is None else deep_value - xgboost_value
        return {
            "deep_learning": deep_value,
            "xgboost": xgboost_value,
            "deep_learning_minus_xgboost": difference,
        }

    return {
        split: {metric: metric_values(split, metric) for metric in MODEL_COMPARISON_METRICS}
        for split in deep_learning
    }


def save_results(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    classification_summaries: dict[str, object],
    run_dir: str | Path,
) -> None:
    """Save classification metrics and row-level predictions.

    Args:
        model_frames: Row-level predictions by model and data split.
        classification_summaries: Pre-balancing retention and prevalence data.
        run_dir: Model-run output directory.
    """
    model_results = {name: _model_results(split_frames) for name, split_frames in model_frames.items()}
    deep_learning_results = model_results["deep_learning"]
    results: dict[str, object] = {
        "splits": deep_learning_results["splits"],
        "test_absolute_delta_tertiles": deep_learning_results["test_absolute_delta_tertiles"],
        "models": model_results,
    }
    results["comparison"] = _model_comparison(model_results)
    results["dataset_classification_summaries"] = classification_summaries

    output_dir = Path(run_dir)
    with (output_dir / "results.json").open("w") as destination:
        json.dump(results, destination, indent=2)
    for model_name, split_frames in model_frames.items():
        for split, frame in split_frames.items():
            filename = (
                f"{split}_predictions.csv"
                if model_name == "deep_learning"
                else f"{model_name}_{split}_predictions.csv"
            )
            frame.to_csv(output_dir / filename, index=False)
