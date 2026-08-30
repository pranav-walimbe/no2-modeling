"""Cluster, resample, and visualize dataset splits."""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from config import COUNTRIES_URL, IMG_RANGE, LABEL_COL, STRAT_VIS_PNG
from preprocessing.plant_features import project_to_meters


def cluster_plants(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster facilities whose image-patch bounding boxes intersect."""
    plants = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)
    x, y = project_to_meters(plants)
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    adjacency = ((dx < IMG_RANGE * 1000) & (dy < IMG_RANGE * 1000)).astype(int)
    _, labels = connected_components(csr_matrix(adjacency))
    plants["cluster"] = labels
    return plants[["facilityId", "cluster"]]


def resample_uniform(df: pd.DataFrame, n: int, bins: int = 20) -> pd.DataFrame:
    """Resample records to a uniform label distribution using oversampling."""
    rng = np.random.default_rng(42)
    bin_labels = pd.cut(df[LABEL_COL], bins=bins, labels=False, include_lowest=True)
    active_bins = bin_labels.dropna().unique()
    n_per_bin = int(np.ceil(n / len(active_bins)))

    chosen = []
    for bin_label in active_bins:
        indices = np.where(bin_labels == bin_label)[0]
        chosen.extend(rng.choice(indices, size=n_per_bin, replace=len(indices) < n_per_bin).tolist())

    rng.shuffle(chosen)
    chosen = chosen[:n]
    total, unique = len(chosen), len(set(chosen))
    print(f"resample_uniform: {total - unique}/{total} oversampled ({100 * (total - unique) / total:.1f}%)")
    return df.iloc[chosen].reset_index(drop=True)


def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Visualize the geographic and label distributions of each split."""
    countries = gpd.read_file(COUNTRIES_URL)
    us = countries[countries.NAME == "United States of America"]
    _, axes = plt.subplots(2, 3, figsize=(20, 12))
    splits = [("Train", train), ("Val", val), ("Test", test)]

    for axis, (label, split_df) in zip(axes[0], splits):
        us.plot(ax=axis, color="lightgray", edgecolor="black")
        axis.scatter(split_df["lon"], split_df["lat"], s=5, alpha=0.3)
        axis.set_title(f"{label} : n = {split_df.shape[0]}")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_xlim(-130, -65)
        axis.set_ylim(24, 50)

    for axis, (label, split_df) in zip(axes[1], splits):
        axis.hist(split_df[LABEL_COL], bins=50, alpha=0.7)
        axis.set_title(label)
        axis.set_xlabel(LABEL_COL)
        axis.set_ylabel("Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(STRAT_VIS_PNG), exist_ok=True)
    plt.savefig(STRAT_VIS_PNG, dpi=150)
    plt.close()
