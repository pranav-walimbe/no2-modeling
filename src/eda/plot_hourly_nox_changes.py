"""Plot one-hour changes in coal-plant NOx emissions from CAMPD records."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EMISSIONS_RECORDS_CSV, VIS_DIR

REQUIRED_COLUMNS = ["facilityId", "date", "hour", "noxMass", "primaryFuelInfo"]
ONE_HOUR = pd.Timedelta(hours=1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot plant-level one-hour NOx changes for coal-fired units.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(EMISSIONS_RECORDS_CSV),
        help="CAMPD hourly-emissions CSV (default: configured raw emissions file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(VIS_DIR) / "coal_plant_hourly_nox_change_histogram.png",
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="CSV rows processed per chunk.",
    )
    parser.add_argument("--bins", type=int, default=160, help="Number of histogram bins.")
    return parser.parse_args()


def load_coal_plant_hours(input_path: Path, chunk_size: int) -> pd.DataFrame:
    """Load coal records and aggregate unit observations to plant-hour totals.

    Args:
        input_path: CAMPD hourly-emissions CSV.
        chunk_size: Number of CSV rows to read at a time.

    Returns:
        DataFrame with one summed NOx value per facility and hour.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    grouped_chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        input_path,
        usecols=REQUIRED_COLUMNS,
        dtype={
            "facilityId": "string",
            "date": "string",
            "hour": "Int8",
            "noxMass": "float64",
            "primaryFuelInfo": "string",
        },
        chunksize=chunk_size,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        fuel = chunk["primaryFuelInfo"].str.strip().str.casefold()
        coal = chunk.loc[fuel.eq("coal"), ["facilityId", "date", "hour", "noxMass"]]
        coal = coal.dropna(subset=["facilityId", "date", "hour", "noxMass"])
        if coal.empty:
            continue

        grouped = (
            coal.groupby(["facilityId", "date", "hour"], as_index=False, sort=False)["noxMass"]
            .sum(min_count=1)
        )
        grouped_chunks.append(grouped)
        print(f"Processed chunk {chunk_number:,}; retained {len(grouped):,} plant-hours")

    if not grouped_chunks:
        raise ValueError("No usable records with primaryFuelInfo='Coal' were found.")

    plant_hours = pd.concat(grouped_chunks, ignore_index=True)
    plant_hours = (
        plant_hours.groupby(["facilityId", "date", "hour"], as_index=False, sort=False)["noxMass"]
        .sum(min_count=1)
    )
    plant_hours["timestamp"] = pd.to_datetime(plant_hours["date"], errors="coerce") + pd.to_timedelta(
        plant_hours["hour"], unit="h"
    )
    return plant_hours.dropna(subset=["timestamp", "noxMass"])


def calculate_one_hour_changes(plant_hours: pd.DataFrame) -> pd.Series:
    """Calculate signed NOx changes across exactly consecutive plant-hours.

    Args:
        plant_hours: Plant-hour observations with facility, timestamp, and NOx columns.

    Returns:
        Signed changes in plant-level hourly NOx mass, in pounds.
    """
    ordered = plant_hours.sort_values(["facilityId", "timestamp"])
    by_plant = ordered.groupby("facilityId", sort=False)
    time_gap = by_plant["timestamp"].diff()
    changes = by_plant["noxMass"].diff().loc[time_gap.eq(ONE_HOUR)].dropna()
    changes.name = "nox_mass_change_lb"

    if changes.empty:
        raise ValueError("No consecutive one-hour coal-plant observations were found.")
    return changes


def plot_changes(changes: pd.Series, plant_count: int, output_path: Path, bins: int) -> None:
    """Write full-range and central-detail histograms to one PNG.

    Args:
        changes: Signed one-hour plant-level NOx changes.
        plant_count: Number of facilities represented.
        output_path: Destination PNG path.
        bins: Number of bins in each histogram.
    """
    values = changes.to_numpy(dtype=float)
    lower, upper = np.quantile(values, [0.01, 0.99])
    central = values[(values >= lower) & (values <= upper)]

    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    color = "#2E6F9E"

    axes[0].hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.25)
    axes[0].set_yscale("log")
    axes[0].set_title("Full distribution (log-scaled count)")

    axes[1].hist(central, bins=bins, color=color, edgecolor="white", linewidth=0.25)
    axes[1].set_title(f"Central 98% detail ({lower:,.1f} to {upper:,.1f} lb)")

    for axis in axes:
        axis.axvline(0, color="#B22222", linestyle="--", linewidth=1)
        axis.set_xlabel("One-hour change in plant NOx mass (lb)")
        axis.set_ylabel("Consecutive plant-hour pairs")
        axis.grid(axis="y", alpha=0.2)

    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    summary = (
        f"Pairs: {len(values):,}   Plants: {plant_count:,}   "
        f"Mean: {values.mean():,.1f} lb   SD: {values.std():,.1f} lb\n"
        f"P05/P25/P50/P75/P95: "
        f"{quantiles[0]:,.1f} / {quantiles[1]:,.1f} / {quantiles[2]:,.1f} / "
        f"{quantiles[3]:,.1f} / {quantiles[4]:,.1f} lb"
    )
    figure.suptitle("Coal-plant hour-to-hour NOx variability\n" + summary, fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate the coal-plant hourly NOx-change histogram."""
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")

    plant_hours = load_coal_plant_hours(args.input, args.chunk_size)
    changes = calculate_one_hour_changes(plant_hours)
    plant_count = plant_hours["facilityId"].nunique()
    plot_changes(changes, plant_count, args.output, args.bins)

    print(f"Plant-hours: {len(plant_hours):,}")
    print(f"Consecutive one-hour pairs: {len(changes):,}")
    print(f"Plants: {plant_count:,}")
    print(f"Mean change: {changes.mean():,.2f} lb; standard deviation: {changes.std():,.2f} lb")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
