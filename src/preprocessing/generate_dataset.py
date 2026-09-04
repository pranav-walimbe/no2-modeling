"""Generate model-ready image arrays and metadata tables."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CITIES_URL,
    COUNTRIES_URL,
    DATASET_DF,
    IMAGES_DIR,
    LABEL_COL,
    MIN_CITY_POPULATION,
    NUM_CORES,
    PLUME_FILTER_PERCENTILE,
    SPLIT_SIZES,
    TEST_RECORDS_CSV,
    TRAIN_RECORDS_CSV,
    VAL_RECORDS_CSV,
    VIS_DIR,
)
from preprocessing.imagery import compute_bounds, extract_image_data, filter_by_city_proximity


def visualize_split(df: pd.DataFrame, split: str) -> None:
    """Visualize the geographic and label distributions for a split."""
    countries = gpd.read_file(COUNTRIES_URL)
    us = countries[countries.NAME == "United States of America"]
    fig, axes = plt.subplots(1, 2, figsize=(24, 5))
    fig.suptitle(f"{split} (n={len(df)})", fontsize=14)

    us.plot(ax=axes[0], color="lightgray", edgecolor="black")
    axes[0].scatter(df["lon"], df["lat"], s=5, alpha=0.3)
    axes[0].set_xlim(-130, -65)
    axes[0].set_ylim(24, 50)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title("Geographic Distribution")

    axes[1].hist(df[LABEL_COL], bins=50, color="#E87B4C", alpha=0.8, edgecolor="none")
    axes[1].set_xlabel(LABEL_COL)
    axes[1].set_ylabel("count")
    axes[1].set_title("Label Histogram")
    plt.tight_layout()
    os.makedirs(VIS_DIR, exist_ok=True)
    plt.savefig(os.path.join(VIS_DIR, f"dataset_vis_{split}.png"), dpi=150)
    plt.close()


def process_split(df: pd.DataFrame, split: str, cities_gdf: gpd.GeoDataFrame) -> None:
    """Extract and persist image and tabular data for one dataset split."""
    df = compute_bounds(filter_by_city_proximity(df, cities_gdf))
    extraction_args = list(zip(df.index, df.to_dict("records")))
    valid = []
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(extract_image_data, args) for args in extraction_args]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                valid.append(result)

    print(f"{len(valid)} / {len(df)} patches passed filtering")
    if not valid:
        return
    valid.sort(key=lambda result: result[0])

    plume_scores = np.array([result[2] for result in valid])
    threshold = np.percentile(plume_scores, PLUME_FILTER_PERCENTILE * 100)
    valid = [result for result, score in zip(valid, plume_scores) if score >= threshold]
    print(f"[{split}] {len(valid)} samples retained after plume score filter (threshold={threshold:.4f})")
    if not valid:
        return

    n_store = min(SPLIT_SIZES[split], len(valid))
    valid = valid[:n_store]
    valid_indices = [result[0] for result in valid]
    patches = np.concatenate([result[1] for result in valid], axis=0)
    np.save(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"), patches)

    output_df = df.loc[valid_indices].drop(columns=["lat_min", "lat_max", "lon_min", "lon_max"]).copy()
    output_df["npy_idx"] = range(n_store)
    output_df["split"] = split
    output_df["plume_score"] = [result[2] for result in valid]
    output_df["u10"] = [result[3] for result in valid]
    output_df["v10"] = [result[4] for result in valid]
    output_df.to_csv(os.path.join(DATASET_DF, f"{split}_df.csv"), index=False)
    visualize_split(output_df, split)


def main() -> None:
    """Generate image and tabular datasets for every split."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATASET_DF, exist_ok=True)
    cities = gpd.read_file(CITIES_URL)
    cities_gdf = cities[cities["pop_max"] >= MIN_CITY_POPULATION].to_crs("EPSG:5070")
    splits = {
        "train": pd.read_csv(TRAIN_RECORDS_CSV),
        "val": pd.read_csv(VAL_RECORDS_CSV),
        "test": pd.read_csv(TEST_RECORDS_CSV),
    }
    for split, df in splits.items():
        df["date"] = pd.to_datetime(df["date"])
        process_split(df, split, cities_gdf)


if __name__ == "__main__":
    main()
