"""Evaluation utilities for three-class emissions-change classification."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from config import MODEL_CLASS_NAMES

TRUE_CLASS_COL = "class_true"
PREDICTED_CLASS_COL = "class_predicted"
PROBABILITY_COLUMNS = tuple(f"probability_{name}" for name in MODEL_CLASS_NAMES)
LOGIT_COLUMNS = tuple(f"logit_{name}" for name in MODEL_CLASS_NAMES)
COMPARISON_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def classification_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Calculate multiclass metrics from a prediction frame.

    Args:
        frame: Row-level class labels and probabilities.

    Returns:
        Overall scores, per-class recall, and a confusion matrix.
    """
    truth = frame[TRUE_CLASS_COL].to_numpy(dtype=np.int64)
    predicted = frame[PREDICTED_CLASS_COL].to_numpy(dtype=np.int64)
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    labels = np.arange(len(MODEL_CLASS_NAMES))
    confusion = confusion_matrix(truth, predicted, labels=labels)
    class_totals = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion),
        class_totals,
        out=np.zeros(len(labels), dtype=np.float64),
        where=class_totals > 0,
    )
    unique_classes = np.unique(truth)
    macro_roc_auc = (
        float(roc_auc_score(truth, probabilities, labels=labels, multi_class="ovr", average="macro"))
        if len(unique_classes) == len(labels)
        else None
    )
    return {
        "n": int(len(frame)),
        "accuracy": float(np.mean(truth == predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "macro_ovr_roc_auc": macro_roc_auc,
        "class_counts": {name: int(np.sum(truth == index)) for index, name in enumerate(MODEL_CLASS_NAMES)},
        "class_recall": {name: float(recalls[index]) for index, name in enumerate(MODEL_CLASS_NAMES)},
        "confusion_matrix": confusion.tolist(),
    }


def plant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate classification metrics for each held-out AOI.

    Args:
        frame: Row-level predictions with AOI coordinates.

    Returns:
        One classification summary row per AOI.
    """
    rows = []
    for (aoi_id, lon, lat), group in frame.groupby(["aoi_id", "lon", "lat"], sort=True):
        metrics = classification_metrics(group)
        rows.append({"aoi_id": aoi_id, "lon": lon, "lat": lat, **metrics})
    return pd.DataFrame(rows)


def _model_results(split_frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    # Summarize each geographic split
    return {split: classification_metrics(frame) for split, frame in split_frames.items()}


def _model_comparison(model_results: dict[str, dict[str, object]]) -> dict[str, object]:
    # Place raster and MLP scores together by split
    comparison = {}
    for split in next(iter(model_results.values())):
        comparison[split] = {
            metric: {model_name: results[split][metric] for model_name, results in model_results.items()}
            for metric in COMPARISON_METRICS
        }
    return comparison


def save_results(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
    *,
    primary_model_name: str,
) -> None:
    """Save classification metrics and row-level predictions.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
        primary_model_name: Model copied into the top-level split summary.
    """
    model_results = {name: _model_results(split_frames) for name, split_frames in model_frames.items()}
    results = {
        "primary_model": primary_model_name,
        "class_names": list(MODEL_CLASS_NAMES),
        "splits": model_results[primary_model_name],
        "models": model_results,
        "comparison": _model_comparison(model_results),
    }
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
