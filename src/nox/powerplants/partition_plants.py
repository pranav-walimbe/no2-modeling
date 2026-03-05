"""                                                                                                                                             
Script to stratify power plant emission records into train/test/val splits      

Output: train.csv, test.csv, val.csv located in output directory                                                                                
"""

from sklearn.model_selection import train_test_split                                                                                            
import geopandas as gpd                                                                                                                         
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components                                                                                           
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
from datetime import datetime
import numpy as np
import os

# configuration
VIS_PATH = "/global/home/users/pranavwalimbe/vis/strat_vis.png"
TEMPO_PATH = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V03/tmp"
BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe"
INPUT_CSV = os.path.join(BASE_DIR, "nox_emissions_1", "nox_emissions_full2.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "test.csv")
VAL_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "val.csv")
NUM_SAMPLES = 10000
MINS_FILTER = 60
PATCH_SIZE = 48

# helper functions
def build_tempo_mapping():
    """generate dict mapping dates to (timestamp, file)"""
    tempo_by_date = defaultdict(list)
    for fname in os.listdir(TEMPO_PATH):
        if not fname.endswith(".nc"):
            continue
        try:
            dt = pd.to_datetime(fname.split("_")[4], format="%Y%m%dT%H%M%SZ")
            tempo_by_date[dt.date()].append((dt, fname))
        except Exception as e:
            print(f"WARNING: skipping {fname}: {e}")
    return tempo_by_date

def map_to_tempo(row: pd.Series, tempo_by_date: dict):
    """Return the first TEMPO filename captured within MINS_FILTER minutes after the record"""
    target_dt = datetime(year=row['date'].year, month=row['date'].month,
                        day=row['date'].day, hour=row['hour'])
    for curr_dt, fname in tempo_by_date.get(row['date'].date(), []):
        dt_diff = curr_dt - target_dt
        if pd.Timedelta(0) < dt_diff <= pd.Timedelta(minutes=MINS_FILTER):
            return fname
    return None

def cluster_plants(df: pd.DataFrame):
    """check for intersecting PATCH_SIZE x PATCH_SIZE bounding boxes before stratifying"""
    plants = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)

    # apply lat, lon -> km conversion
    gdf = gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants["lon"], plants["lat"]), crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:5070")
    x = np.array([geom.x for geom in gdf_proj.geometry])
    y = np.array([geom.y for geom in gdf_proj.geometry])

    # construct adjacency matrix to derive cluster values
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    adjacency = ((dx < PATCH_SIZE * 1000) & (dy < PATCH_SIZE * 1000)).astype(int)
    n_components, labels = connected_components(csr_matrix(adjacency))
    plants["cluster"] = labels
    return plants[["facilityId", "cluster"]]

def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Visualize stratification in terms of geography and nox emissions"""
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    us = gpd.read_file(url)                                                                                                                     
    us = us[us.NAME == "United States of America"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))                                                                                            
    splits = [("Train", train), ("Val", val), ("Test", test)]

    # spatial visualization of stratification
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
    os.makedirs(os.path.dirname(VIS_PATH), exist_ok=True)
    plt.savefig(VIS_PATH, dpi=150)
    plt.close()

def main():         
    # sample desired dataset size from all records                                                                                                           
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])                                                                                 
    df = df.sample(NUM_SAMPLES, random_state=42).reset_index(drop=True)

    # build TEMPO mapping and map each record to a tile
    tempo_by_date = build_tempo_mapping()
    df['tempo'] = df.apply(lambda row: map_to_tempo(row, tempo_by_date), axis=1)
    df = df[df['tempo'].notna()] 

    # cluster spatially-proximate plants and stratify using EPA region
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")
    cluster_features = df.groupby("cluster").agg(epaRegion=("epaRegion", "first")).reset_index()
    cluster_features["strat_key"] = cluster_features["epaRegion"].astype(str)
    train_c, temp_c = train_test_split(cluster_features, test_size=0.30, random_state=42, stratify=cluster_features["strat_key"])
    val_c, test_c = train_test_split(temp_c, test_size=0.50, random_state=42, stratify=temp_c["strat_key"])
    train = df[df["cluster"].isin(train_c["cluster"])].drop(columns=["cluster"])
    val = df[df["cluster"].isin(val_c["cluster"])].drop(columns=["cluster"])
    test = df[df["cluster"].isin(test_c["cluster"])].drop(columns=["cluster"])

    # save splits and generate visualization
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    plot_split_distributions(train, val, test)

if __name__ == "__main__":
    main()