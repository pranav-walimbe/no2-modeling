"""Partition power-plant emission records into train, validation, and test splits."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    LABEL_COL,
    NUM_CORES,
    PLANT_TYPE,
    SAMPLE_SIZE,
    SPLIT_SIZES,
    STRAT_BASE_DIR,
    STRAT_INPUT_CSV,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)
from preprocessing.matching import build_tempo_mapping, map_chunk
from preprocessing.plant_features import aggregate_units, compute_adj_plants, compute_prev_qtr_mass
from preprocessing.splitting import cluster_plants, plot_split_distributions, resample_uniform


def main() -> None:
    """Build stratified metadata splits for dataset generation."""
    df = pd.read_csv(STRAT_INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["primaryFuelInfo"] == PLANT_TYPE]

    tempo_by_date = build_tempo_mapping()
    chunks = [df.iloc[index] for index in np.array_split(np.arange(len(df)), NUM_CORES) if len(index) > 0]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(map_chunk, [(chunk, tempo_by_date) for chunk in chunks]))
    df = pd.concat(results).reset_index(drop=True)
    df = df[df["tempo"].notna() & df["prev_tempo"].notna()].reset_index(drop=True)

    adjacent_plants = compute_adj_plants(df)
    df = df.merge(adjacent_plants, on="facilityId")
    df = df[df["num_adj_plants"] == 0].reset_index(drop=True)
    df = aggregate_units(df)
    df = compute_prev_qtr_mass(df)

    low = df[LABEL_COL].quantile(0.10)
    high = df[LABEL_COL].quantile(0.90)
    df = df[(df[LABEL_COL] >= low) & (df[LABEL_COL] <= high)]
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")
    clusters = df[["cluster"]].drop_duplicates()
    train_clusters, remaining_clusters = train_test_split(clusters, test_size=0.30, random_state=42)
    test_clusters, val_clusters = train_test_split(remaining_clusters, test_size=0.50, random_state=42)

    train_samples = df[df["cluster"].isin(train_clusters["cluster"])].reset_index(drop=True)
    val_samples = df[df["cluster"].isin(val_clusters["cluster"])].reset_index(drop=True)
    test_samples = df[df["cluster"].isin(test_clusters["cluster"])].reset_index(drop=True)
    train = resample_uniform(train_samples, n=SPLIT_SIZES["train"] * 3)
    val = resample_uniform(val_samples, n=SPLIT_SIZES["val"] * 3)
    test = resample_uniform(test_samples, n=SPLIT_SIZES["test"] * 3)

    for split_df in [train, val, test]:
        split_df["era5"] = split_df["date"].apply(lambda date: f"era5_{date.year}_{date.month:02d}.nc")

    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    plot_split_distributions(train, val, test)


if __name__ == "__main__":
    main()
