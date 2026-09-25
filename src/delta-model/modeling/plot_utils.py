"""Plotting utilities for seasonal and vision-seasonal classification."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from modeling.eval_utils import PREDICTED_CLASS_COL, TRUE_CLASS_COL, classification_metrics

from config import MODEL_CLASS_NAMES

MODEL_ORDER = ("seasonal", "vision_seasonal")
MODEL_DISPLAY_NAMES = {
    "seasonal": "Seasonal",
    "vision_seasonal": "Vision + seasonal",
}
MODEL_COLORS = {
    "seasonal": "#4C78A8",
    "vision_seasonal": "#F58518",
}
SPLIT_ORDER = ("train", "val", "test")
STRATUM_ORDER = ("Low", "Middle", "High")
AOI_STRATA = {
    "aoi_score": "AOI plume score",
    "major_city_dist": "Major-city distance",
    "num_units": "Total unit count",
    "avg_heat_input": "Average heat input",
    "avg_pwr_gen": "Average power generation",
}


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    # Persist and close one completed plot
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _available_models(model_frames: dict[str, dict[str, pd.DataFrame]]) -> tuple[str, ...]:
    # Keep model order consistent across artifacts
    return tuple(name for name in MODEL_ORDER if name in model_frames)


def plot_split_class_accuracy(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot overall and per-class accuracy for each data split.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    models = _available_models(model_frames)
    figure = plt.figure(figsize=(16, 10))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.25))
    overall_axis = figure.add_subplot(grid[0, :])
    class_axes = [figure.add_subplot(grid[1, index]) for index in range(len(SPLIT_ORDER))]

    split_positions = np.arange(len(SPLIT_ORDER), dtype=np.float64)
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, len(models))
    for model_name, offset in zip(models, offsets, strict=True):
        scores = [classification_metrics(model_frames[model_name][split])["accuracy"] for split in SPLIT_ORDER]
        overall_axis.bar(
            split_positions + offset,
            scores,
            width=width,
            color=MODEL_COLORS[model_name],
            label=MODEL_DISPLAY_NAMES[model_name],
        )
    overall_axis.set(
        title="Overall accuracy by split",
        ylabel="Accuracy",
        xticks=split_positions,
        xticklabels=[split.title() for split in SPLIT_ORDER],
        ylim=(0, 1),
    )
    overall_axis.legend(loc="lower right")

    class_positions = np.arange(len(MODEL_CLASS_NAMES), dtype=np.float64)
    for axis, split in zip(class_axes, SPLIT_ORDER, strict=True):
        for model_name, offset in zip(models, offsets, strict=True):
            recalls = classification_metrics(model_frames[model_name][split])["class_recall"]
            scores = [recalls[class_name] for class_name in MODEL_CLASS_NAMES]
            axis.bar(
                class_positions + offset,
                scores,
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_DISPLAY_NAMES[model_name],
            )
        axis.set(
            title=f"{split.title()} per-class accuracy",
            ylabel="Recall",
            xticks=class_positions,
            xticklabels=[name.title() for name in MODEL_CLASS_NAMES],
            ylim=(0, 1),
        )
    figure.suptitle("Seasonal and vision-seasonal accuracy", fontsize=16)
    figure.tight_layout()
    _save(figure, run_dir, "split_class_accuracy")


def plot_training_curves(
    histories: dict[str, tuple[list[float], list[float]]],
    run_dir: str | Path,
) -> None:
    """Plot train and validation loss for both classifiers.

    Args:
        histories: Train and validation losses by model.
        run_dir: Model-run output directory.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    models = tuple(name for name in MODEL_ORDER if name in histories)
    figure, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5.5), squeeze=False)
    for axis, model_name in zip(axes[0], models, strict=True):
        train_losses, validation_losses = histories[model_name]
        axis.plot(range(1, len(train_losses) + 1), train_losses, label="Train", linewidth=2)
        axis.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation", linewidth=2)
        axis.set(
            title=MODEL_DISPLAY_NAMES[model_name],
            xlabel="Epoch",
            ylabel="Cross-entropy",
        )
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axis.legend()
    figure.suptitle("Training and validation loss", fontsize=16)
    figure.tight_layout()
    _save(figure, run_dir, "training_curves")


def _accuracy(frame: pd.DataFrame) -> float:
    # Calculate row-level classification accuracy
    return float((frame[TRUE_CLASS_COL] == frame[PREDICTED_CLASS_COL]).mean())


def _test_strata_table(model_frames: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    # Assign AOIs to characteristic tertiles and compare record-level accuracy
    seasonal = model_frames["seasonal"]["test"].reset_index(drop=True)
    vision = model_frames["vision_seasonal"]["test"].reset_index(drop=True)
    if not seasonal[["aoi_id", TRUE_CLASS_COL]].equals(vision[["aoi_id", TRUE_CLASS_COL]]):
        raise ValueError("Seasonal and vision test predictions are not row-aligned")

    aoi_values = seasonal.groupby("aoi_id", sort=True)[list(AOI_STRATA)].median(numeric_only=True)
    rows: list[dict[str, object]] = []
    for column, display_name in AOI_STRATA.items():
        percentiles = aoi_values[column].rank(method="average", pct=True)
        strata = pd.cut(
            percentiles,
            bins=(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
            labels=STRATUM_ORDER,
            include_lowest=True,
        )
        for stratum in STRATUM_ORDER:
            aoi_ids = strata.index[strata == stratum]
            mask = seasonal["aoi_id"].isin(aoi_ids)
            if not mask.any():
                continue
            seasonal_accuracy = _accuracy(seasonal.loc[mask])
            vision_accuracy = _accuracy(vision.loc[mask])
            values = aoi_values.loc[aoi_ids, column]
            rows.append(
                {
                    "characteristic": column,
                    "characteristic_label": display_name,
                    "stratum": stratum,
                    "aoi_count": int(len(aoi_ids)),
                    "record_count": int(mask.sum()),
                    "value_min": float(values.min()),
                    "value_max": float(values.max()),
                    "seasonal_accuracy": seasonal_accuracy,
                    "vision_seasonal_accuracy": vision_accuracy,
                    "vision_accuracy_gain": vision_accuracy - seasonal_accuracy,
                }
            )
    return pd.DataFrame(rows)


def plot_test_strata_accuracy(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot test accuracy by AOI-characteristic tertile.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    table = _test_strata_table(model_frames)
    table.to_csv(Path(run_dir) / "test_strata_accuracy.csv", index=False)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    flat_axes = axes.ravel()
    positions = np.arange(len(STRATUM_ORDER), dtype=np.float64)
    for axis, (column, display_name) in zip(flat_axes[: len(AOI_STRATA)], AOI_STRATA.items(), strict=True):
        subset = table.loc[table["characteristic"] == column].set_index("stratum").reindex(STRATUM_ORDER)
        seasonal_scores = subset["seasonal_accuracy"].to_numpy(dtype=np.float64)
        vision_scores = subset["vision_seasonal_accuracy"].to_numpy(dtype=np.float64)
        axis.plot(
            positions,
            seasonal_scores,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS["seasonal"],
            label=MODEL_DISPLAY_NAMES["seasonal"],
        )
        axis.plot(
            positions,
            vision_scores,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS["vision_seasonal"],
            label=MODEL_DISPLAY_NAMES["vision_seasonal"],
        )
        for position, seasonal_score, vision_score in zip(
            positions,
            seasonal_scores,
            vision_scores,
            strict=True,
        ):
            if not np.isfinite(seasonal_score) or not np.isfinite(vision_score):
                continue
            gain = vision_score - seasonal_score
            color = "#2E7D32" if gain >= 0 else "#B71C1C"
            axis.annotate(
                f"{gain:+.3f}",
                (position, max(seasonal_score, vision_score)),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=9,
            )
        range_labels = []
        for stratum, row in subset.iterrows():
            if pd.isna(row["value_min"]):
                range_labels.append(str(stratum))
            else:
                range_labels.append(f"{stratum}\n{row['value_min']:.3g}-{row['value_max']:.3g}")
        axis.set(
            title=display_name,
            xlabel="AOI tertile",
            ylabel="Test accuracy",
            xticks=positions,
            xticklabels=range_labels,
            ylim=(0, 1.05),
        )
    flat_axes[0].legend(loc="lower right")
    for axis in flat_axes[len(AOI_STRATA) :]:
        axis.set_visible(False)
    figure.suptitle("Test accuracy by AOI characteristic\nLabels show vision accuracy minus seasonal accuracy", fontsize=16)
    figure.tight_layout()
    _save(figure, run_dir, "test_strata_accuracy")
