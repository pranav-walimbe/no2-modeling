"""
Script to splice satellite images and generate modeling dataset

Output: train/val/test tempo npy arrays and per-split DataFrames in DATASET_DIR
"""

import sys
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

def compute_bounds(df: pd.DataFrame):
    """Compute IMG_RANGE x IMG_RANGE bounds in lat/lon terms for each record"""
    # convert from lat/lon (WGS84) format to meters (NAD83 Conus Albers) format
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    buffered = gdf.to_crs("EPSG:5070").buffer((IMG_RANGE / 2) * 1000).to_crs("EPSG:4326")
    bounds = buffered.bounds
    df = df.copy()
    df["lat_min"] = bounds["miny"].values
    df["lat_max"] = bounds["maxy"].values
    df["lon_min"] = bounds["minx"].values
    df["lon_max"] = bounds["maxx"].values
    return df

def filter_by_city_proximity(df: pd.DataFrame, cities_gdf: gpd.GeoDataFrame):
    """Filter records to meet MIN_CITY_PROXIMITY threshold to minimize ambient emissions"""
    # convert from lat/lon (WGS84) format to meters (NAD83 Conus Albers) format
    plants_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326").to_crs("EPSG:5070")
    plant_coords = np.column_stack([plants_gdf.geometry.x, plants_gdf.geometry.y])
    cities_coords = np.column_stack([cities_gdf.geometry.x, cities_gdf.geometry.y])

    # store locations + determine plant proximities using spatial lookup data structure
    dists_km = cKDTree(cities_coords).query(plant_coords, k=1)[0] / 1000
    mask = dists_km >= MIN_CITY_PROXIMITY
    return df[mask].reset_index(drop=True)

def extract_tempo_patch(args):
    """Perform TEMPO patch extraction with pixel quality filtering; returns 2-channel (VCD_t, delta) patch"""
    orig_idx, row = args
    try:
        # current hour: full quality checks
        fpath = os.path.join(TEMPO_DIR, row["tempo"])
        with nc.Dataset(fpath) as root:
            lats = root["latitude"][:]
            lons = root["longitude"][:]

            lat_idx = np.where((lats >= row["lat_min"]) & (lats <= row["lat_max"]))[0]
            lon_idx = np.where((lons >= row["lon_min"]) & (lons <= row["lon_max"]))[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                print(f"SKIP [{orig_idx}]: {row['tempo']} — no pixels in range around ({row['lat']:.2f}, {row['lon']:.2f})")
                return None
            r0, r1 = lat_idx.min(), lat_idx.max() + 1
            c0, c1 = lon_idx.min(), lon_idx.max() + 1

            # require sufficient fraction of pixels to be below cloud cover threshold
            cloud_ma = root["support_data"]["eff_cloud_fraction"][0, r0:r1, c0:c1]
            valid = ~np.ma.getmaskarray(cloud_ma)
            if not valid.any():
                print(f"SKIP [{orig_idx}]: {row['tempo']} — cloud array is all masked")
                return None
            cloud = np.array(cloud_ma)
            frac_clean = np.mean(cloud[valid] <= MIN_PIXEL_CLOUD)
            if frac_clean < IMG_CLOUD_FILTER:
                print(f"SKIP [{orig_idx}]: {row['tempo']} — {frac_clean:.1%} pixels below cloud threshold")
                return None

            # require sufficient fraction of pixels to meet retrieval quality flag
            qa = np.ma.filled(root["product"]["main_data_quality_flag"][0, r0:r1, c0:c1], fill_value=2)
            frac_good = np.mean(qa == 0)
            if frac_good < IMG_QA_FILTER:
                print(f"SKIP [{orig_idx}]: {row['tempo']} — {frac_good:.1%} pixels have QA == 0")
                return None

            # reject patches with any masked no2 values or outlier concentrations
            no2_ma = root["product"]["vertical_column_troposphere"][0, r0:r1, c0:c1]
            if np.ma.getmaskarray(no2_ma).any():
                print(f"SKIP [{orig_idx}]: {row['tempo']} — no2 patch contains masked pixels")
                return None
            no2 = np.array(no2_ma)

            if no2.min() < MIN_IMG_VAL:
                print(f"SKIP [{orig_idx}]: {row['tempo']} — min pixel value {no2.min():.2e} < {MIN_IMG_VAL:.2e}")
                return None
            frac_clipped = np.mean(no2 >= MAX_IMG_VAL)
            if frac_clipped > IMG_VAL_FILTER:
                print(f"SKIP [{orig_idx}]: {row['tempo']} — {frac_clipped:.1%} pixels at max value")
                return None

        no2 = zoom(no2, (IMG_SIZE / no2.shape[0], IMG_SIZE / no2.shape[1]), order=1)

        # previous hour: masked values and value range checks
        fpath_prev = os.path.join(TEMPO_DIR, row["prev_tempo"])
        with nc.Dataset(fpath_prev) as root:
            lats = root["latitude"][:]
            lons = root["longitude"][:]

            lat_idx = np.where((lats >= row["lat_min"]) & (lats <= row["lat_max"]))[0]
            lon_idx = np.where((lons >= row["lon_min"]) & (lons <= row["lon_max"]))[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                print(f"SKIP [{orig_idx}]: {row['prev_tempo']} — no pixels in range")
                return None
            r0, r1 = lat_idx.min(), lat_idx.max() + 1
            c0, c1 = lon_idx.min(), lon_idx.max() + 1

            # require sufficient fraction of pixels to meet retrieval quality flag (loosened constraint)
            qa = np.ma.filled(root["product"]["main_data_quality_flag"][0, r0:r1, c0:c1], fill_value=2)
            if np.mean(qa == 0) < IMG_QA_FILTER * 0.5:
                print(f"SKIP [{orig_idx}]: {row['prev_tempo']} — prev QA below threshold")
                return None

            # reject patches with any masked no2 values or outlier concentrations
            no2_prev_ma = root["product"]["vertical_column_troposphere"][0, r0:r1, c0:c1]
            if np.ma.getmaskarray(no2_prev_ma).any():
                print(f"SKIP [{orig_idx}]: {row['prev_tempo']} — prev no2 patch contains masked pixels")
                return None
            no2_prev = np.array(no2_prev_ma)

            if no2_prev.min() < MIN_IMG_VAL:
                print(f"SKIP [{orig_idx}]: {row['prev_tempo']} — min pixel value {no2_prev.min():.2e} < {MIN_IMG_VAL:.2e}")
                return None
            frac_clipped = np.mean(no2_prev >= MAX_IMG_VAL)
            if frac_clipped > IMG_VAL_FILTER:
                print(f"SKIP [{orig_idx}]: {row['prev_tempo']} — {frac_clipped:.1%} pixels at max value")
                return None

        no2_prev = zoom(no2_prev, (IMG_SIZE / no2_prev.shape[0], IMG_SIZE / no2_prev.shape[1]), order=1)
        delta_no2 = no2 - no2_prev

        # plume score: ratio of peak spread to background spread as a plume visibility heuristic
        p10, p50, p99 = np.percentile(no2, [10, 50, 99])
        plume_score = (p99 - p50) / (p50 - p10 + 1e-10)

        patch = np.stack([no2, delta_no2], axis=0)[np.newaxis, ...].astype(np.float32)  # (1, 2, H, W)
        return (orig_idx, patch, plume_score)

    except Exception as e:
        print(f"ERROR [{orig_idx}]: {row['tempo']}: {e}")
        return None

def visualize_split(df: pd.DataFrame, split: str):
    """Visualize geographic distribution, label distribution, and log-label distribution for a split"""
    us = gpd.read_file(COUNTRIES_URL)
    us = us[us.NAME == "United States of America"]

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

def process_split(df: pd.DataFrame, split: str, cities_gdf: gpd.GeoDataFrame):
    """Parallel patch extraction, npy write, and dataset DataFrame save for a given split"""
    # prefiltering by city proximity
    df = filter_by_city_proximity(df, cities_gdf)
    df = compute_bounds(df)

    # parallelized tempo patch extraction
    args = [(orig_idx, row) for orig_idx, row in df.iterrows()]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(extract_tempo_patch, args))

    valid = [r for r in results if r is not None]
    n_valid = len(valid)
    print(f"{n_valid} / {len(df)} patches passed filtering")
    if n_valid == 0:
        return

    # filter by plume score — drop samples below percentile threshold
    plume_scores = np.array([r[2] for r in valid])
    threshold = np.percentile(plume_scores, PLUME_FILTER_PERCENTILE * 100)
    valid = [r for r, s in zip(valid, plume_scores) if s >= threshold]
    print(f"[{split}] {len(valid)} samples retained after plume score filter (threshold={threshold:.4f})")
    if not valid:
        return

    # extract desired amount of images for split
    n_store = min(SPLIT_SIZES[split], len(valid))
    valid = valid[:n_store]

    valid_idxs = [orig_idx for orig_idx, _, _ in valid]
    plume_scores = [s for _, _, s in valid]

    # store image data in npy — shape: (N, 2, IMG_SIZE, IMG_SIZE) with channels [no2, delta_no2]
    patches = np.concatenate([patch for _, patch, _ in valid], axis=0)
    np.save(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"), patches)

    # store dataset DataFrame with original columns, label, wind speed, and npy index
    out_df = df.loc[valid_idxs].drop(columns=["lat_min", "lat_max", "lon_min", "lon_max"]).copy()
    out_df["npy_idx"] = range(n_store)
    out_df["split"] = split
    out_df["plume_score"] = plume_scores
    out_df.to_csv(os.path.join(DATASET_DF, f"{split}_df.csv"), index=False)

    # visualize split distribution
    visualize_split(out_df, split)

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATASET_DF, exist_ok=True)

    # city locations dataset for proximity filtering
    CITIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places_simple.zip"
    cities_gdf = gpd.read_file(CITIES_URL).to_crs("EPSG:5070")

    splits = {
        "train": pd.read_csv(TRAIN_CSV),
        "val": pd.read_csv(VAL_CSV),
        "test": pd.read_csv(TEST_CSV),
    }

    for split, df in splits.items():
        df["date"] = pd.to_datetime(df["date"])
        if not all(col in df.columns for col in WIND_COLS):
            sys.exit(f"ERROR: Wind data not present in {split} input csv")
        process_split(df, split, cities_gdf)

if __name__ == "__main__":
    main()