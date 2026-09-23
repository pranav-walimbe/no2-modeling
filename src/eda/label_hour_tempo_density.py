"""Plot label-ending TEMPO scan timing within the matched CAMPD hour."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import LABEL_TIMESTEP_INDEX, TEST_RECORDS_CSV, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, VIS_DIR

SPLIT_PATHS = {
    "Train": Path(TRAIN_RECORDS_CSV),
    "Validation": Path(VAL_RECORDS_CSV),
    "Test": Path(TEST_RECORDS_CSV),
}
SPLIT_COLORS = {"Train": "#2563EB", "Validation": "#F97316", "Test": "#16A34A"}
LABEL_SCAN_COLUMN = f"t{LABEL_TIMESTEP_INDEX}_timestamp"
HOUR_MINUTES = 60.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / "label-hour-tempo-density.png",
    )
    return parser.parse_args()


def load_scan_offsets(paths: dict[str, Path]) -> pd.DataFrame:
    """Load label scan offsets from the stratified split files.

    Args:
        paths: Display split names mapped to stratified CSV files.

    Returns:
        Split labels and scan offsets in minutes from the CAMPD hour start.
    """
    frames = []
    required = ["emissions_hour_utc", LABEL_SCAN_COLUMN]
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Stratified records not found for {split}: {path}")
        frame = pd.read_csv(path, usecols=required)
        hour_start = pd.to_datetime(frame["emissions_hour_utc"], utc=True, errors="raise")
        scan_time = pd.to_datetime(frame[LABEL_SCAN_COLUMN], utc=True, errors="raise")
        offset_minutes = (scan_time - hour_start).dt.total_seconds() / 60.0
        frames.append(pd.DataFrame({"split": split, "offset_minutes": offset_minutes}))
    return pd.concat(frames, ignore_index=True)


def plot_scan_density(records: pd.DataFrame, output: Path) -> None:
    """Plot the label-ending TEMPO scan density over the CAMPD hour.

    Args:
        records: Records carrying split names and scan offsets.
        output: Destination PNG path.
    """
    offsets = records["offset_minutes"]
    if offsets.empty or offsets.isna().any():
        raise ValueError("Label-ending TEMPO scan offsets must be present for every record")

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(12, 7))
    for split in SPLIT_PATHS:
        values = records.loc[records["split"] == split, "offset_minutes"]
        sns.kdeplot(
            x=values,
            ax=axis,
            color=SPLIT_COLORS[split],
            label=f"{split} (n={len(values):,})",
            linewidth=2.2,
            bw_adjust=0.8,
            cut=0,
        )

    within_hour = offsets.between(0.0, HOUR_MINUTES, inclusive="both")
    after_hour = offsets > HOUR_MINUTES
    within_share = 100.0 * within_hour.mean()
    after_share = 100.0 * after_hour.mean()
    median_early_tail = (HOUR_MINUTES - offsets[within_hour]).median()
    axis.axvspan(0.0, HOUR_MINUTES, color="#3B82F6", alpha=0.08, label="CAMPD label hour")
    axis.axvline(HOUR_MINUTES, color="#111827", linestyle="--", linewidth=1.5)
    axis.text(
        0.02,
        0.96,
        f"Ends within hour: {within_share:.1f}%\n"
        f"Ends after hour: {after_share:.1f}%\n"
        f"Median early tail (within-hour): {median_early_tail:.1f} min",
        ha="left",
        va="top",
        fontsize=11,
        color="#111827",
        transform=axis.transAxes,
    )
    axis.set(
        title=(
            f"Label-ending TEMPO scan timing relative to the matched CAMPD hour\n"
            f"t{LABEL_TIMESTEP_INDEX}; {len(records):,} current stratified records"
        ),
        xlabel="Minutes after start of matched CAMPD label hour",
        ylabel="Density",
    )
    tick_max = max(HOUR_MINUTES, offsets.max())
    axis.set_xticks(range(0, int(tick_max // 10 + 2) * 10, 10))
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.25)
    axis.grid(axis="y", alpha=0.15)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Loaded {len(records):,} stratified records")
    print(f"Scans within CAMPD hour: {within_hour.sum():,} ({within_share:.2f}%)")
    print(f"Scans after CAMPD hour: {after_hour.sum():,} ({after_share:.2f}%)")
    print(f"Median early non-overlap among within-hour scans: {median_early_tail:.2f} minutes")
    print(f"Saved density plot to {output}")


def main() -> None:
    """Generate the label-hour TEMPO timing density plot."""
    args = parse_args()
    plot_scan_density(load_scan_offsets(SPLIT_PATHS), args.output)


if __name__ == "__main__":
    main()
