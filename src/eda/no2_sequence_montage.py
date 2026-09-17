"""Plot five-timestep NO2 sequences for balanced emissions-change labels."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from config import DATASET_DF, DATASET_DIR, EFFECTIVE_DELTA_NOX_COL, IMG_SIZE, LABEL_COL, SEQUENCE_TIMESTEPS, VIS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AOI_COL = "aoi_id"
RASTER_PATH_COL = "raster_bundle_path"
TIMESTEP_TIME_COLUMNS = tuple(f"timestep_time_t{index}" for index in range(SEQUENCE_TIMESTEPS))
NO2_KEY = "no2"
NO2_MASK_KEY = "no2_mask"
NO2_DISPLAY_SCALE = 1e15
DEFAULT_SAMPLE_COUNT = 20
DEFAULT_SEED = 20260917
DISPLAY_QUANTILES = (0.02, 0.98)
SEQUENCES_PER_ROW = 2


def parse_args() -> argparse.Namespace:
    """Parse montage settings.

    Returns:
        Dataset and output settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataframe-dir", type=Path, default=Path(DATASET_DF))
    parser.add_argument("--dataset-dir", type=Path, default=Path(DATASET_DIR))
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run_id = os.getenv("SLURM_JOB_ID", "latest")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / f"no2-sequence-montage-{run_id}.png",
    )
    return parser.parse_args()


def load_manifest(dataframe_dir: Path, split: str, sample_count: int, seed: int) -> pd.DataFrame:
    """Stream one split and select balanced unique samples.

    Args:
        dataframe_dir: Directory containing split CSV files.
        split: Dataset split to sample.
        sample_count: Total number of samples.
        seed: Random sampling seed.

    Returns:
        Selected record metadata.
    """
    if sample_count <= 0 or sample_count % 2:
        raise ValueError("Sample count must be a positive even number")
    path = dataframe_dir / f"{split}_df.csv"
    columns = (AOI_COL, LABEL_COL, EFFECTIVE_DELTA_NOX_COL, RASTER_PATH_COL, *TIMESTEP_TIME_COLUMNS)
    per_label = sample_count // 2
    generators = {label: np.random.default_rng(seed + label) for label in (0, 1)}
    reservoirs: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    unique_counts = {0: 0, 1: 0}
    seen_paths: set[str] = set()
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset metadata lacks columns: {sorted(missing)}")
        for record in reader:
            raster_path = record[RASTER_PATH_COL]
            if raster_path in seen_paths:
                continue
            seen_paths.add(raster_path)
            label = int(record[LABEL_COL])
            if label not in reservoirs:
                continue
            unique_counts[label] += 1
            candidate = {name: record[name] for name in columns}
            reservoir = reservoirs[label]
            if len(reservoir) < per_label:
                reservoir.append(candidate)
                continue
            replacement = int(generators[label].integers(unique_counts[label]))
            if replacement < per_label:
                reservoir[replacement] = candidate
    for label, reservoir in reservoirs.items():
        if len(reservoir) < per_label:
            raise ValueError(f"Class {label} contains only {len(reservoir):,} unique records")
    return pd.DataFrame([*reservoirs[0], *reservoirs[1]])


def _resolve_raster_path(serialized_path: object, dataset_dir: Path) -> Path:
    # Resolve shard-relative paths against the dataset root
    path = Path(str(serialized_path))
    return path if path.is_absolute() else dataset_dir / path


def _load_sequence(serialized_path: object, dataset_dir: Path) -> np.ndarray:
    # Load one raw NO2 sequence and mask invalid retrievals
    path = _resolve_raster_path(serialized_path, dataset_dir)
    with np.load(path, allow_pickle=False) as bundle:
        no2 = np.asarray(bundle[NO2_KEY], dtype=np.float64)
        mask = np.asarray(bundle[NO2_MASK_KEY], dtype=bool)
    expected_shape = (SEQUENCE_TIMESTEPS, IMG_SIZE, IMG_SIZE)
    if no2.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(f"Unexpected NO2 sequence shape in {path}: {no2.shape}, {mask.shape}")
    return np.where(mask & np.isfinite(no2), no2 / NO2_DISPLAY_SCALE, np.nan)


def _display_limits(sequences: list[np.ndarray]) -> tuple[float, float]:
    # Use one robust scale for direct comparison across every panel
    values = np.concatenate([sequence[np.isfinite(sequence)] for sequence in sequences])
    if values.size == 0:
        raise ValueError("Selected sequences contain no valid NO2 pixels")
    lower, upper = np.quantile(values, DISPLAY_QUANTILES)
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return float(lower), float(upper)


def _sequence_label(row: object) -> str:
    # Summarize the target and record identity beside one sequence
    label = int(getattr(row, LABEL_COL))
    direction = "increase" if label else "decrease"
    delta = float(getattr(row, EFFECTIVE_DELTA_NOX_COL))
    final_time = pd.Timestamp(getattr(row, TIMESTEP_TIME_COLUMNS[-1])).strftime("%Y-%m-%d %HZ")
    return f"{direction} ({label})\nΔNOx {delta:+.0f} lb\nAOI {int(getattr(row, AOI_COL))}\n{final_time}"


def write_montage(manifest: pd.DataFrame, dataset_dir: Path, output_path: Path) -> Path:
    """Write the balanced NO2 sequence montage.

    Args:
        manifest: Selected record metadata.
        dataset_dir: Dataset root used to resolve raster paths.
        output_path: Destination PNG.

    Returns:
        Saved PNG path.
    """
    rows = list(manifest.itertuples(index=False))
    sequences = [_load_sequence(getattr(row, RASTER_PATH_COL), dataset_dir) for row in rows]
    if len(rows) % SEQUENCES_PER_ROW:
        raise ValueError(f"Sample count must be divisible by {SEQUENCES_PER_ROW}")
    figure_rows = len(rows) // SEQUENCES_PER_ROW
    block_width = SEQUENCE_TIMESTEPS + 1
    figure_columns = block_width * SEQUENCES_PER_ROW
    width_ratios = [0.8, *([1.0] * SEQUENCE_TIMESTEPS)] * SEQUENCES_PER_ROW
    figure, axes = plt.subplots(
        figure_rows,
        figure_columns,
        figsize=(24, 2.25 * figure_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    colormap = matplotlib.colormaps["viridis"].copy()
    colormap.set_bad("#d9d9d9")
    lower, upper = _display_limits(sequences)
    image = None
    for sequence_index, (row, sequence) in enumerate(zip(rows, sequences, strict=True)):
        figure_row = sequence_index % figure_rows
        block = sequence_index // figure_rows
        label_column = block * block_width
        column_offset = label_column + 1
        label_axis = axes[figure_row, label_column]
        label_axis.axis("off")
        label_axis.text(
            1.0,
            0.5,
            _sequence_label(row),
            color="#c0392b" if int(getattr(row, LABEL_COL)) else "#2874a6",
            fontsize=7.5,
            ha="right",
            va="center",
        )
        for timestep in range(SEQUENCE_TIMESTEPS):
            axis = axes[figure_row, column_offset + timestep]
            image = axis.imshow(sequence[timestep], cmap=colormap, vmin=lower, vmax=upper)
            axis.scatter((IMG_SIZE - 1) / 2, (IMG_SIZE - 1) / 2, marker="+", color="white", s=20, linewidths=0.8)
            axis.set_xticks([])
            axis.set_yticks([])
            if figure_row == 0:
                axis.set_title(f"t-{SEQUENCE_TIMESTEPS - 1 - timestep} h", fontsize=9)
    if image is None:
        raise RuntimeError("No NO2 rasters were plotted")
    figure.suptitle(
        "Five-timestep raw NO2 sequences by emissions-change label\n"
        "20 unique training samples; one shared 2nd–98th percentile scale",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.06, hspace=0.18, wspace=0.04)
    color_axis = figure.add_axes((0.34, 0.02, 0.32, 0.012))
    figure.colorbar(image, cax=color_axis, orientation="horizontal", label="NO2 (10¹⁵ molecules/cm²)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    manifest.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved {len(manifest)} samples to {output_path}")
    return output_path


def main() -> None:
    """Select sequences and write the montage."""
    args = parse_args()
    manifest = load_manifest(args.dataframe_dir, args.split, args.samples, args.seed)
    write_montage(manifest, args.dataset_dir, args.output)


if __name__ == "__main__":
    main()
