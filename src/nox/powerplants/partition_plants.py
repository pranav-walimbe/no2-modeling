"""
Script to stratify power plant emission records into train/test/val splits

Output: train.csv, test.csv, val.csv located in output directory
"""

from sklearn.model_selection import train_test_split
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components                                                                           
from scipy.sparse import csr_matrix
import pandas as pd 
import numpy as np 
import os

# configuration
VIS_PATH = "/global/home/users/pranavwalimbe/vis/strat_vis.png" # stratification visualization
BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe"   
INPUT_CSV = os.path.join(BASE_DIR, "nox_emissions_1", "nox_emissions_full2.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "test.csv")     
VAL_PATH = os.path.join(BASE_DIR, "nox_powerplant_data", "val.csv")
NUM_SAMPLES = 100000 # total dataset size to sample

# helper functions
def cluster_plants(df: pd.DataFrame):
    """assign plants with overlapping 48km x 48km boxes to same cluster"""
    # apply lat, lon -> km conversion logic
    plants = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)
    lat_km = plants["lat"].values * 111.0                                                                                       
    lon_km = plants["lon"].values * 111.0 * np.cos(np.radians(plants["lat"].values))                                            

    # construct adjacency matrix for power plants and return cluster values
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

    # geographic plots of power plant locations
    for ax, (label, split_df) in zip(axes[0], splits):
        us.plot(ax=ax, color="lightgray", edgecolor="black")
        ax.scatter(split_df["lon"], split_df["lat"], s=5, alpha=0.3)
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)

    # noxMass histograms using log scale on x axis
    for ax, (label, split_df) in zip(axes[1], splits):
        ax.hist(np.log1p(split_df["noxMass"]), bins=50, alpha=0.7)
        ax.set_title(label)                                                                      
        ax.set_xlabel("log(noxMass + 1)") 
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(VIS_PATH, dpi=150)
    plt.close()

def main():
    # sample desired number of records and assign clusters
    df = pd.read_csv(INPUT_CSV).sample(NUM_SAMPLES, random_state=42)
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")

    # aggregate clusters using epa region field
    cluster_features = df.groupby("cluster").agg(epaRegion=("epaRegion", "first")).reset_index()
    cluster_features["strat_key"] = cluster_features["epaRegion"].astype(str)

    # split clusters using 70/15/15 split
    train_c, temp_c = train_test_split(cluster_features, test_size=0.30, random_state=42, stratify=cluster_features["strat_key"])
    val_c, test_c = train_test_split(temp_c, test_size=0.50, random_state=42, stratify=temp_c["strat_key"])

    # map split back to original dataframe
    train = df[df["cluster"].isin(train_c["cluster"])].drop(columns=["cluster"])
    val = df[df["cluster"].isin(val_c["cluster"])].drop(columns=["cluster"])
    test = df[df["cluster"].isin(test_c["cluster"])].drop(columns=["cluster"])

    # save to output files and plot visualization
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    plot_split_distributions(train, val, test)

if __name__ == "__main__":
    main()