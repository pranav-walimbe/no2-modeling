"""                                                                                                                                             
Script to stratify power plant emission records into train/test/val splits        

Output: train.csv, test.csv, val.csv located in output directory                                                                                
"""                                                                                                                                             

import sys                                                                                                                                      
import os                                                                                                                                                                                                                                                                 
from sklearn.model_selection import train_test_split                                                                                            
import geopandas as gpd
import matplotlib.pyplot as plt                                                                                                                 
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
from datetime import datetime
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # access config                                                                                                                                                                                                
from config import *  

# helper functions
def parse_tile(fname: str):
    """Return (dt, fname) if tile meets duration requirement"""
    try:                                    
        fpath = os.path.join(TEMPO_DIR, fname)
        ds = xr.open_dataset(fpath, engine="netcdf4")
        start = pd.Timestamp(ds.attrs["time_coverage_start"])                                                                                   
        end = pd.Timestamp(ds.attrs["time_coverage_end"])
        ds.close()                                                                                                                              
        if (end - start).total_seconds() / 60 < MIN_TEMPO_DURATION:
            return None
        dt = pd.to_datetime(fname.split("_")[4], format="%Y%m%dT%H%M%SZ")
        return (dt, fname)
    except Exception as e:
        print(f"WARNING: skipping {fname}: {e}")
        return None

def build_tempo_mapping():
    """Return dict mapping dates to (timestamp, fname) for tiles meeting duration requirement"""
    fnames = [f for f in os.listdir(TEMPO_DIR) if f.endswith(".nc")]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(parse_tile, fnames))
    tempo_by_date = defaultdict(list)
    for result in results:
        if result is not None:
            dt, fname = result
            tempo_by_date[dt.date()].append((dt, fname))
    return tempo_by_date

def map_to_tempo(row: pd.Series, tempo_by_date: dict):
    """Return the first TEMPO filename captured within MINS_FILTER minutes before the record"""
    target_dt = datetime(year=row['date'].year, month=row['date'].month, day=row['date'].day, hour=row['hour'])
    for curr_dt, fname in tempo_by_date.get(row['date'].date(), []):
        dt_diff = curr_dt - target_dt
        if pd.Timedelta(0) < dt_diff <= pd.Timedelta(minutes=MINS_FILTER):
            return fname
    return None

def map_chunk(args):
    """pickleable helper function for parallelized record-tempo mapping"""
    chunk, tempo_by_date = args
    return chunk.assign(tempo=chunk.apply(lambda row: map_to_tempo(row, tempo_by_date), axis=1))

def cluster_plants(df: pd.DataFrame):
    """check for intersecting IMG_RANGE x IMG_RANGE bounding boxes before stratifying"""
    plants = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)

    # convert from lat/lon (WGS84) format to meters format (NAD83 Conus Albers)
    gdf = gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants["lon"], plants["lat"]), crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:5070")
    x = np.array([geom.x for geom in gdf_proj.geometry])
    y = np.array([geom.y for geom in gdf_proj.geometry])

    # construct adjacency matrix to derive cluster values
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    adjacency = ((dx < IMG_RANGE * 1000) & (dy < IMG_RANGE * 1000)).astype(int)
    n_components, labels = connected_components(csr_matrix(adjacency))
    plants["cluster"] = labels
    return plants[["facilityId", "cluster"]]

def rebalance_splits(val: pd.DataFrame, test: pd.DataFrame, other_df: pd.DataFrame):
    """Rebalance val and test split to equal sizes using weighted sampling from unused rows"""
    target_size = max(len(val), len(test))

    if len(other_df) < target_size:
        sys.exit(f"ERROR: other_df (size={len(other_df)}) is smaller than target_size ({target_size}), cannot rebalance")

    region_freq = other_df["epaRegion"].value_counts(normalize=True)
    weights = other_df["epaRegion"].map(lambda r: 1.0 / region_freq[r]).fillna(1.0 / region_freq.mean())

    if len(val) < target_size:
        extra = other_df.sample(target_size - len(val), random_state=42, weights=weights)
        other_df = other_df.drop(extra.index)
        val = pd.concat([val, extra]).reset_index(drop=True)

    if len(test) < target_size:
        extra = other_df.sample(target_size - len(test), random_state=42, weights=weights)
        other_df = other_df.drop(extra.index)
        test = pd.concat([test, extra]).reset_index(drop=True)

    return val, test

def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Visualize stratification in terms of geography and nox emissions"""

    # get US outline for geographic visualization
    us = gpd.read_file(COUNTRIES_URL)
    us = us[us.NAME == "United States of America"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    splits = [("Train", train), ("Val", val), ("Test", test)]

    # spatial visualization of stratification
    for ax, (label, split_df) in zip(axes[0], splits):
        us.plot(ax=ax, color="lightgray", edgecolor="black")
        ax.scatter(split_df["lon"], split_df["lat"], s=5, alpha=0.3)
        ax.set_title(f"{label} : n = {split_df.shape[0]}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)

    # histogram of log-scale noxMass distribution
    for ax, (label, split_df) in zip(axes[1], splits):
        ax.hist(np.log1p(split_df["noxMass"]), bins=50, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("log(noxMass + 1)")
        ax.set_ylabel("Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(STRAT_VIS_PNG), exist_ok=True)
    plt.savefig(STRAT_VIS_PNG, dpi=150)
    plt.close()

def main():
    df = pd.read_csv(STRAT_INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])

    # build TEMPO mapping and run parallelized record-tempo mapping
    tempo_by_date = build_tempo_mapping()
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), NUM_CORES) if len(idx) > 0]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(map_chunk, [(c, tempo_by_date) for c in chunks]))
    df = pd.concat(results).reset_index(drop=True)
    df = df[df['tempo'].notna()].reset_index(drop=True)

    # sample desired dataset size with weighted sampling (inversely proportional to EPA region)
    region_freq = df["epaRegion"].value_counts(normalize=True)
    weights = (df["epaRegion"].map(lambda r: 1.0 / region_freq[r]))
    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42, weights=weights).reset_index(drop=True)

    # stratify clusters and map back to split dataframes
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")
    cluster_ids = pd.Series(df["cluster"].unique())

    strat_clusters, other_clusters = train_test_split(cluster_ids, test_size=0.50, random_state=42)
    train_c, temp_c = train_test_split(cluster_ids, test_size=0.30, random_state=42)
    val_c, test_c = train_test_split(temp_c, test_size=0.50, random_state=42)

    train = df[df["cluster"].isin(train_c)]
    val = df[df["cluster"].isin(val_c)]
    test = df[df["cluster"].isin(test_c)]
    other_df = df[df["cluster"].isin(other_clusters)]

    # rebalance val and test to equal sizes
    val, test = rebalance_splits(val, test, other_df)

    # save splits and generate visualization
    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    plot_split_distributions(train, val, test)

if __name__ == "__main__":
    main()