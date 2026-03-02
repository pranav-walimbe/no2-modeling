"""                              
Script to stratify power plant emission records into train/test/val splits
                                                                                                                                
Output: train.csv, test.csv, val.csv located in output directory
"""                                                                                                                             
                
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
from datetime import datetime
import pickle
import numpy as np
import xarray as xr
import os

# configuration
VIS_PATH = "/global/home/users/pranavwalimbe/vis/strat_vis.png" # visualization of plant partitioning 
TEMPO_PATH = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V03/tmp" # directory of tempo tiles
TEMPO_MAPPING = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/tempo_mapping/tempo.pkl" # OPTIONAL: mapping of tempo tiles to dates/ranges
BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe"
INPUT_CSV = os.path.join(BASE_DIR, "nox_emissions_1", "nox_emissions_full2.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "test.csv")
VAL_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "val.csv")
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK"))
NUM_SAMPLES = 10000 # desired dataset size
MINS_FILTER = 30 # filtering parameter for time difference between emissions record and tempo tile

# helper functions
def read_tempo_file(fname: str):
    """parse spatial range from tempo tile"""
    if not fname.endswith(".nc"):
        return None
    try:
        ds = xr.open_dataset(os.path.join(TEMPO_PATH, fname), engine="netcdf4")
        lat = ds.latitude.values
        lon = ds.longitude.values
        ds.close()
        dt = pd.to_datetime(fname.split("_")[4], format="%Y%m%dT%H%M%SZ")
        return dt, {
            'lat': (float(lat.min()), float(lat.max())),
            'lon': (float(lon.min()), float(lon.max())),
            'fname': fname
        }
    except OSError as e:
        print(f"WARNING: skipping {fname}: {e}")
        return None

def validate_record(row: pd.Series, tempo_by_date: dict):
    """check whether a given emissions record has valid corresponding tempo data"""
    target_lat = row['lat']
    target_lon = row['lon']
    target_dt = datetime(
        year=row['date'].year,
        month=row['date'].month,
        day=row['date'].day,
        hour=row['hour']
    )
    dlat = 24 / 111.0
    dlon = 24 / (111.0 * np.cos(np.radians(target_lat)))

    # iterate through tempo mapping to find first valid tile
    for curr_dt, locations in tempo_by_date.get(target_dt.date(), []):
        dt_diff = abs(target_dt - curr_dt)
        if dt_diff <= pd.Timedelta(minutes=MINS_FILTER):
            for loc in locations:
                lat_min, lat_max = loc['lat']
                lon_min, lon_max = loc['lon']

                # require valid spatial range + ability to derive 48kmx48km slice for modeling 
                if (lat_min <= target_lat - dlat and
                    lat_max >= target_lat + dlat and
                    lon_min <= target_lon - dlon and
                    lon_max >= target_lon + dlon):
                    return loc['fname']
    return None

def validate_chunk(chunk: pd.DataFrame, tempo_by_date: dict):
    """pickleable helper function to validate a chunk of records against the tempo mapping"""
    return chunk.apply(lambda row: validate_record(row, tempo_by_date), axis=1)

def build_tempo_mapping():
    """build or load date-keyed mapping of tempo tile spatial coverage"""
    if os.path.exists(TEMPO_MAPPING):
        with open(TEMPO_MAPPING, "rb") as f:
            return pickle.load(f)

    # parallelized tempo file parsing
    fnames = os.listdir(TEMPO_PATH)
    tempo_mapping = {}
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        for result in executor.map(read_tempo_file, fnames, chunksize=125):
            if result is not None:
                dt, entry = result
                tempo_mapping.setdefault(dt, []).append(entry)

    # use date as first layer of filtering for record validation
    tempo_by_date = defaultdict(list)
    for dt, locs in tempo_mapping.items():
        tempo_by_date[dt.date()].append((dt, locs))

    with open(TEMPO_MAPPING, "wb") as f:
        pickle.dump(tempo_by_date, f)
    return tempo_by_date

def filter_plant_data(df: pd.DataFrame):
    """filter power plant records based on tempo tile availability"""
    tempo_by_date = build_tempo_mapping()

    # parallelized record validation
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), NUM_CORES) if len(idx) > 0]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(partial(validate_chunk, tempo_by_date=tempo_by_date), chunks))

    df['tempo'] = pd.concat(results)
    return df[df['tempo'].notna()].copy()

def cluster_plants(df: pd.DataFrame):
    """assign plants with overlapping 48km x 48km boxes to same cluster"""
    plants = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)

    # apply lat, lon -> km conversion
    lat_km = plants["lat"].values * 111.0
    lon_km = plants["lon"].values * 111.0 * np.cos(np.radians(plants["lat"].values))

    # construct adjacency matrix to derive distinct clusters
    dx = np.abs(lon_km[:, None] - lon_km[None, :])
    dy = np.abs(lat_km[:, None] - lat_km[None, :])
    adjacency = ((dx < 48) & (dy < 48)).astype(int)
    n_components, labels = connected_components(csr_matrix(adjacency))
    plants["cluster"] = labels
    return plants[["facilityId", "cluster"]]

def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """visualize stratification in terms of geography and nox emissions"""
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    us = gpd.read_file(url)
    us = us[us.NAME == "United States of America"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    splits = [("Train", train), ("Val", val), ("Test", test)]

    # spatial visualization of partitioning
    for ax, (label, split_df) in zip(axes[0], splits):
        us.plot(ax=ax, color="lightgray", edgecolor="black")
        ax.scatter(split_df["lon"], split_df["lat"], s=5, alpha=0.3)
        ax.set_title(label)
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
    plt.savefig(VIS_PATH, dpi=150)
    plt.close()

def main():
    # filter emissions records
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = filter_plant_data(df)
    print(f"post-filter dataset size: {df.shape}")
    df = df.sample(min(NUM_SAMPLES, len(df)), random_state=42)

    # apply clustering and stratify on EPA region
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")
    cluster_features = df.groupby("cluster").agg(epaRegion=("epaRegion", "first")).reset_index()
    cluster_features["strat_key"] = cluster_features["epaRegion"].astype(str)
    train_c, temp_c = train_test_split(cluster_features, test_size=0.30, random_state=42, stratify=cluster_features["strat_key"])
    val_c, test_c = train_test_split(temp_c, test_size=0.50, random_state=42, stratify=temp_c["strat_key"])
    train = df[df["cluster"].isin(train_c["cluster"])].drop(columns=["cluster"])
    val = df[df["cluster"].isin(val_c["cluster"])].drop(columns=["cluster"])
    test = df[df["cluster"].isin(test_c["cluster"])].drop(columns=["cluster"])

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    plot_split_distributions(train, val, test)

if __name__ == "__main__":
    main()