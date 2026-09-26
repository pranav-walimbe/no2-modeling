"""Plotting utilities for seasonal and vision-seasonal classification."""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from config import MODEL_CLASS_NAMES
from modeling.dataset import FINAL_TIMESTEP_TIME_COLUMN
from modeling.eval_utils import PREDICTED_CLASS_COL, TRUE_CLASS_COL, classification_metrics

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
    "avg_heat_input": "Average heat input",
    "num_units": "Total unit count",
}
MONTH_CHARACTERISTIC = "month"
CHARACTERISTIC_PANELS = (
    ("avg_heat_input", "test", "Average heat input"),
    ("num_units", "test", "Total unit count"),
    (MONTH_CHARACTERISTIC, "test", "Month (test)"),
    (MONTH_CHARACTERISTIC, "val", "Month (val)"),
)


def _save(figure: plt.Figure, run_dir: str | Path, plot_name: str) -> None:
    # Persist and close one completed plot
    figure.savefig(Path(run_dir) / f"{plot_name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _available_models(model_frames: dict[str, dict[str, pd.DataFrame]]) -> tuple[str, ...]:
    # Keep model order consistent across artifacts
    return tuple(name for name in MODEL_ORDER if name in model_frames)


def _label_bars(axis: Axes, bars: BarContainer) -> None:
    # Display each plotted accuracy above its bar
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)


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
        bars = overall_axis.bar(
            split_positions + offset,
            scores,
            width=width,
            color=MODEL_COLORS[model_name],
            label=MODEL_DISPLAY_NAMES[model_name],
        )
        _label_bars(overall_axis, bars)
    overall_axis.set(
        title="Overall accuracy by split",
        ylabel="Accuracy",
        xticks=split_positions,
        xticklabels=[split.title() for split in SPLIT_ORDER],
        ylim=(0, 1.08),
    )
    overall_axis.legend(loc="lower right")

    class_positions = np.arange(len(MODEL_CLASS_NAMES), dtype=np.float64)
    for axis, split in zip(class_axes, SPLIT_ORDER, strict=True):
        for model_name, offset in zip(models, offsets, strict=True):
            recalls = classification_metrics(model_frames[model_name][split])["class_recall"]
            scores = [recalls[class_name] for class_name in MODEL_CLASS_NAMES]
            bars = axis.bar(
                class_positions + offset,
                scores,
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_DISPLAY_NAMES[model_name],
            )
            _label_bars(axis, bars)
        axis.set(
            title=f"{split.title()} per-class accuracy",
            ylabel="Recall",
            xticks=class_positions,
            xticklabels=[name.title() for name in MODEL_CLASS_NAMES],
            ylim=(0, 1.08),
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


def _tertiles(values: pd.Series) -> pd.Series:
    # Assign equally ranked low, middle, and high groups
    percentiles = values.rank(method="average", pct=True)
    return pd.cut(
        percentiles,
        bins=(0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
        labels=STRATUM_ORDER,
        include_lowest=True,
    )


def _stratum_result(
    seasonal: pd.DataFrame,
    vision: pd.DataFrame,
    *,
    characteristic: str,
    characteristic_label: str,
    split: str,
    stratum: str,
    mask: pd.Series,
    values: pd.Series,
    stratification_unit: str,
) -> dict[str, object]:
    # Summarize one model comparison group
    seasonal_accuracy = _accuracy(seasonal.loc[mask])
    vision_accuracy = _accuracy(vision.loc[mask])
    return {
        "characteristic": characteristic,
        "characteristic_label": characteristic_label,
        "split": split,
        "stratification_unit": stratification_unit,
        "stratum": stratum,
        "aoi_count": int(seasonal.loc[mask, "aoi_id"].nunique()),
        "record_count": int(mask.sum()),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "seasonal_accuracy": seasonal_accuracy,
        "vision_seasonal_accuracy": vision_accuracy,
        "vision_accuracy_gain": vision_accuracy - seasonal_accuracy,
    }


def _aligned_frames(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Return row-aligned predictions for one split
    seasonal = model_frames["seasonal"][split].reset_index(drop=True)
    vision = model_frames["vision_seasonal"][split].reset_index(drop=True)
    if not seasonal[["aoi_id", TRUE_CLASS_COL]].equals(vision[["aoi_id", TRUE_CLASS_COL]]):
        raise ValueError(f"Seasonal and vision {split} predictions are not row-aligned")
    return seasonal, vision


def _characteristic_accuracy_table(model_frames: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    # Compare model accuracy across AOI tertiles and calendar months
    seasonal, vision = _aligned_frames(model_frames, "test")

    aoi_values = seasonal.groupby("aoi_id", sort=True)[list(AOI_STRATA)].median(numeric_only=True)
    rows: list[dict[str, object]] = []
    for column, display_name in AOI_STRATA.items():
        strata = _tertiles(aoi_values[column])
        for stratum in STRATUM_ORDER:
            aoi_ids = strata.index[strata == stratum]
            mask = seasonal["aoi_id"].isin(aoi_ids)
            if not mask.any():
                continue
            rows.append(
                _stratum_result(
                    seasonal,
                    vision,
                    characteristic=column,
                    characteristic_label=display_name,
                    split="test",
                    stratum=stratum,
                    mask=mask,
                    values=aoi_values.loc[aoi_ids, column],
                    stratification_unit="AOI",
                )
            )

    for split in ("test", "val"):
        seasonal, vision = _aligned_frames(model_frames, split)
        months = pd.to_datetime(seasonal[FINAL_TIMESTEP_TIME_COLUMN], utc=True).dt.month
        for month in sorted(months.unique()):
            mask = months == month
            month_values = months.loc[mask]
            rows.append(
                _stratum_result(
                    seasonal,
                    vision,
                    characteristic=MONTH_CHARACTERISTIC,
                    characteristic_label=f"Month ({split})",
                    split=split,
                    stratum=calendar.month_abbr[month],
                    mask=mask,
                    values=month_values,
                    stratification_unit="record",
                )
            )
    return pd.DataFrame(rows)


def plot_accuracy_by_characteristic(
    model_frames: dict[str, dict[str, pd.DataFrame]],
    run_dir: str | Path,
) -> None:
    """Plot accuracy by test AOI characteristics and test or validation month.

    Args:
        model_frames: Row-level predictions by model and split.
        run_dir: Model-run output directory.
    """
    table = _characteristic_accuracy_table(model_frames)
    table.to_csv(Path(run_dir) / "accuracy_by_characteristic.csv", index=False)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    columns = 2
    rows = int(np.ceil(len(CHARACTERISTIC_PANELS) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 5 * rows), sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    for axis, (column, split, display_name) in zip(flat_axes, CHARACTERISTIC_PANELS, strict=True):
        subset = table.loc[(table["characteristic"] == column) & (table["split"] == split)]
        if column in AOI_STRATA:
            subset = subset.set_index("stratum").reindex(STRATUM_ORDER).reset_index()
        else:
            subset = subset.sort_values("value_min")
        positions = np.arange(len(subset), dtype=np.float64)
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
        if column in AOI_STRATA:
            tick_labels = [
                f"{row['stratum']}\n{row['value_min']:.3g}-{row['value_max']:.3g}"
                for _, row in subset.iterrows()
            ]
            x_label = "AOI tertile"
        else:
            tick_labels = subset["stratum"].tolist()
            x_label = "Month"
        axis.set(
            title=display_name,
            xlabel=x_label,
            ylabel=f"{split.title()} accuracy",
            xticks=positions,
            xticklabels=tick_labels,
            ylim=(0, 1.05),
        )
    flat_axes[0].legend(loc="lower right")
    figure.suptitle(
        "Model accuracy by AOI characteristic and calendar month\n"
        "Labels show vision accuracy minus seasonal accuracy",
        fontsize=16,
    )
    figure.tight_layout()
    _save(figure, run_dir, "accuracy_by_characteristic")
