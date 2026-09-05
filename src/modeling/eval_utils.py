"""Evaluation utilities for signed normalized and physical NOx-mass changes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

NORMALIZED_TRUE_COL = "y_true"
NORMALIZED_PRED_COL = "y_pred"
MASS_TRUE_COL = "delta_nox_mass_true"
MASS_PRED_COL = "delta_nox_mass_pred"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    """Return stable metrics that remain meaningful for signed targets."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 1 or not truth.size:
        raise ValueError("Regression metric inputs must be non-empty one-dimensional arrays with equal shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression metric inputs must be finite")

    residual = prediction - truth
    sum_squared_error = float(np.square(residual).sum())
    total_sum_squares = float(np.square(truth - truth.mean()).sum())
    correlation = float(np.corrcoef(truth, prediction)[0, 1]) if truth.std() > 0 and prediction.std() > 0 else None
    return {
        "n": int(truth.size),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(sum_squared_error / truth.size)),
        "bias": float(residual.mean()),
        "r2": float(1.0 - sum_squared_error / total_sum_squares) if total_sum_squares > 0 else None,
        "pearson_r": correlation,
    }


def add_mass_change_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Invert the normalized target using each record's historical scale."""
    required = {"delta_nox_scale", "delta_nox_mass", NORMALIZED_TRUE_COL, NORMALIZED_PRED_COL}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing inverse-transform columns: {', '.join(sorted(missing))}")
    output = frame.copy()
    scale = pd.to_numeric(output["delta_nox_scale"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("delta_nox_scale must be finite and positive")
    output[MASS_TRUE_COL] = pd.to_numeric(output["delta_nox_mass"], errors="raise").to_numpy(dtype=np.float64)
    output[MASS_PRED_COL] = scale * np.sinh(output[NORMALIZED_PRED_COL].to_numpy(dtype=np.float64))
    if not np.isfinite(output[[MASS_TRUE_COL, MASS_PRED_COL]].to_numpy()).all():
        raise ValueError("Inverse target transformation produced non-finite NOx mass changes")
    return output


def plant_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized and physical errors for each held-out AOI."""
    required = {"aoi_id", "lon", "lat", NORMALIZED_TRUE_COL, NORMALIZED_PRED_COL, MASS_TRUE_COL, MASS_PRED_COL}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing plant-metric columns: {', '.join(sorted(missing))}")

    rows = []
    for (aoi_id, lon, lat), group in frame.groupby(["aoi_id", "lon", "lat"], sort=True):
        normalized = regression_metrics(group[NORMALIZED_TRUE_COL].to_numpy(), group[NORMALIZED_PRED_COL].to_numpy())
        physical = regression_metrics(group[MASS_TRUE_COL].to_numpy(), group[MASS_PRED_COL].to_numpy())
        rows.append(
            {
                "aoi_id": aoi_id,
                "lon": lon,
                "lat": lat,
                "n": len(group),
                "normalized_mae": normalized["mae"],
                "mass_change_mae": physical["mae"],
                "mass_change_bias": physical["bias"],
            }
        )
    return pd.DataFrame(rows)


def _split_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    return {
        "normalized_target": regression_metrics(
            frame[NORMALIZED_TRUE_COL].to_numpy(), frame[NORMALIZED_PRED_COL].to_numpy()
        ),
        "nox_mass_change": regression_metrics(frame[MASS_TRUE_COL].to_numpy(), frame[MASS_PRED_COL].to_numpy()),
    }


def save_results(
    split_frames: dict[str, pd.DataFrame],
    run_dir: str | Path,
    *,
    train_target_mean: float,
) -> None:
    """Save metrics, magnitude slices, baselines, and row-level predictions."""
    results: dict[str, object] = {"splits": {name: _split_metrics(frame) for name, frame in split_frames.items()}}
    baselines: dict[str, object] = {}
    for split, frame in split_frames.items():
        truth = frame[NORMALIZED_TRUE_COL].to_numpy(dtype=np.float64)
        mass_truth = frame[MASS_TRUE_COL].to_numpy(dtype=np.float64)
        scale = frame["delta_nox_scale"].to_numpy(dtype=np.float64)
        baselines[split] = {}
        for name, normalized_prediction in {
            "zero_change": np.zeros_like(truth),
            "train_mean": np.full_like(truth, train_target_mean),
        }.items():
            baselines[split][name] = {
                "normalized_target": regression_metrics(truth, normalized_prediction),
                "nox_mass_change": regression_metrics(
                    mass_truth,
                    scale * np.sinh(normalized_prediction),
                ),
            }
    results["baselines"] = baselines

    test = split_frames["test"]
    magnitude = np.abs(test[NORMALIZED_TRUE_COL].to_numpy(dtype=np.float64))
    if len(test) < 3:
        raise ValueError("At least three test records are required for magnitude-sliced evaluation")
    ordered_indices = np.argsort(magnitude, kind="stable")
    slices = {
        name: test.iloc[indices] for name, indices in zip(("low", "mid", "high"), np.array_split(ordered_indices, 3))
    }
    results["test_absolute_magnitude_tertiles"] = {
        name: {
            **_split_metrics(subset),
            "absolute_normalized_range": [
                float(np.abs(subset[NORMALIZED_TRUE_COL]).min()),
                float(np.abs(subset[NORMALIZED_TRUE_COL]).max()),
            ],
        }
        for name, subset in slices.items()
    }

    output_dir = Path(run_dir)
    with (output_dir / "results.json").open("w") as destination:
        json.dump(results, destination, indent=2)
    for split, frame in split_frames.items():
        frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)
