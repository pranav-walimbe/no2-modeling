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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def extract_no2_data(fname, row, orig_idx, is_prev=False):
    """Perform patch extraction and quality filtering for a given tempo image"""
    fpath = os.path.join(TEMPO_DIR, fname)
    with nc.Dataset(fpath) as root:
        lats = root["latitude"][:]
        lons = root["longitude"][:]

        # find pixel window covering image bounds
        lat_idx = np.where((lats >= row["lat_min"]) & (lats <= row["lat_max"]))[0]
        lon_idx = np.where((lons >= row["lon_min"]) & (lons <= row["lon_max"]))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            print(f"SKIP [{orig_idx}]: {fname} — no pixels in range around ({row['lat']:.2f}, {row['lon']:.2f})")
            return None
        r0, r1 = lat_idx.min(), lat_idx.max() + 1
        c0, c1 = lon_idx.min(), lon_idx.max() + 1

        # apply cloud cover filter to current scan only
        if not is_prev:
            cloud_ma = root["support_data"]["eff_cloud_fraction"][0, r0:r1, c0:c1]
            valid = ~np.ma.getmaskarray(cloud_ma)
            if not valid.any():
                print(f"SKIP [{orig_idx}]: {fname} — cloud array is all masked")
                return None
            frac_clean = np.mean(np.array(cloud_ma)[valid] <= MIN_PIXEL_CLOUD)
            if frac_clean < IMG_CLOUD_FILTER:
                print(f"SKIP [{orig_idx}]: {fname} — {frac_clean:.1%} pixels below cloud threshold")
                return None

        # check retrieval quality flag
        qa = np.ma.filled(root["product"]["main_data_quality_flag"][0, r0:r1, c0:c1], fill_value=2)
        frac_good = np.mean(qa == 0)
        if frac_good < IMG_QA_FILTER:
            print(f"SKIP [{orig_idx}]: {fname} — {frac_good:.1%} pixels have QA == 0")
            return None

        # reject patches with masked or out-of-range NO2 values
        no2_ma = root["product"]["vertical_column_troposphere"][0, r0:r1, c0:c1]
        if np.ma.getmaskarray(no2_ma).any():
            print(f"SKIP [{orig_idx}]: {fname} — no2 patch contains masked pixels")
            return None
        no2 = np.array(no2_ma)
        if no2.min() < MIN_IMG_VAL:
            print(f"SKIP [{orig_idx}]: {fname} — min pixel value {no2.min():.2e} < {MIN_IMG_VAL:.2e}")
            return None
        frac_clipped = np.mean(no2 >= MAX_IMG_VAL)
        if frac_clipped > IMG_VAL_FILTER:
            print(f"SKIP [{orig_idx}]: {fname} — {frac_clipped:.1%} pixels at max value")
            return None

    # resize to target image dimensions
    return zoom(no2, (IMG_SIZE / no2.shape[0], IMG_SIZE / no2.shape[1]), order=1)

def extract_wind_scalars(row):
    """Extract ERA5 u10/v10 scalar values at the nearest grid point to the plant location"""
    fpath = os.path.join(ERA5_DIR, row["era5"])
    with nc.Dataset(fpath) as ds:
        ds.set_auto_mask(False)
        lats = ds["latitude"][:]
        lons = ds["longitude"][:]

        lat_idx = int(np.argmin(np.abs(lats - row["lat"])))
        lon_idx = int(np.argmin(np.abs(lons - row["lon"])))

        dt = pd.Timestamp(row["date"])
        time_idx = (dt.day - 1) * 24 + int(row["hour"])

        u10 = float(ds["u10"][time_idx, lat_idx, lon_idx])
        v10 = float(ds["v10"][time_idx, lat_idx, lon_idx])

    return u10, v10

def extract_image_data(args):
    """Extract image channels and wind scalars; returns (orig_idx, patch, plume_score, u10, v10) or None"""
    orig_idx, row = args
    try:
        no2 = extract_no2_data(row["tempo"], row, orig_idx, is_prev=False)
        if no2 is None:
            return None

        no2_prev = extract_no2_data(row["prev_tempo"], row, orig_idx, is_prev=True)
        if no2_prev is None:
            return None

        u10, v10 = extract_wind_scalars(row)

        delta_no2 = no2 - no2_prev
        p10, p50, p99 = np.percentile(no2, [10, 50, 99])
        plume_score = (p99 - p50) / (p50 - p10 + 1e-10)

        patch = np.stack([no2, delta_no2], axis=0)[np.newaxis, ...].astype(np.float32)  # (1, 2, H, W)
        return (orig_idx, patch, plume_score, u10, v10)

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

    # filter records by city proximity and compute image bounds
    df = filter_by_city_proximity(df, cities_gdf)
    df = compute_bounds(df)

    # extract no2 and wind data in parallel
    args = [(orig_idx, row) for orig_idx, row in zip(df.index, df.to_dict("records"))]
    valid = []
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(extract_image_data, arg) for arg in args]
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                valid.append(result)

    n_valid = len(valid)
    print(f"{n_valid} / {len(df)} patches passed filtering")
    if n_valid == 0:
        return

    # drop samples below plume score percentile threshold
    plume_scores = np.array([r[2] for r in valid])
    threshold = np.percentile(plume_scores, PLUME_FILTER_PERCENTILE * 100)
    valid = [r for r, s in zip(valid, plume_scores) if s >= threshold]
    print(f"[{split}] {len(valid)} samples retained after plume score filter (threshold={threshold:.4f})")
    if not valid:
        return

    # trim to target split size
    n_store = min(SPLIT_SIZES[split], len(valid))
    valid = valid[:n_store]

    valid_idxs = [r[0] for r in valid]
    plume_scores = [r[2] for r in valid]
    u10_vals = [r[3] for r in valid]
    v10_vals = [r[4] for r in valid]

    # save image stack with shape: (N, 2, IMG_SIZE, IMG_SIZE)
    patches = np.concatenate([r[1] for r in valid], axis=0)
    np.save(os.path.join(IMAGES_DIR, f"{split}_tempo.npy"), patches)

    # save dataset DataFrame with metadata and npy index
    out_df = df.loc[valid_idxs].drop(columns=["lat_min", "lat_max", "lon_min", "lon_max"]).copy()
    out_df["npy_idx"] = range(n_store)
    out_df["split"] = split
    out_df["plume_score"] = plume_scores
    out_df["u10"] = u10_vals
    out_df["v10"] = v10_vals
    out_df.to_csv(os.path.join(DATASET_DF, f"{split}_df.csv"), index=False)

    # plot geographic and label distributions
    visualize_split(out_df, split)

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATASET_DF, exist_ok=True)

    # load in city locations dataset for proximity filtering
    cities_gdf = gpd.read_file(CITIES_URL).to_crs("EPSG:5070")

    splits = {
        "train": pd.read_csv(TRAIN_CSV),
        "val": pd.read_csv(VAL_CSV),
        "test": pd.read_csv(TEST_CSV),
    }

    for split, df in splits.items():
        df["date"] = pd.to_datetime(df["date"])
        process_split(df, split, cities_gdf)

if __name__ == "__main__":
    main()