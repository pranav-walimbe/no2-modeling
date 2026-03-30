"""                                                                                                                         
Script to splice satellite images and generate modeling dataset                                                             
                                                                                                                            
Output: train/val/test tempo zarr stores and per-split DataFrames in DATASET_DIR                                            
"""                                                                                                                         
                                                                                                                                                                                                                                                
import sys
import os                                                                                                                   
import zarr     
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
    """Perform TEMPO patch extraction with pixel quality filtering"""
    orig_idx, row = args                                                                                                    
    fname = row["tempo"]
    try:
        fpath = os.path.join(TEMPO_DIR, fname)                                                                              
        with nc.Dataset(fpath) as root:
            lats = root["latitude"][:]                                                                                      
            lons = root["longitude"][:]                                                                                     

            # compute patch bound index values                                                                              
            lat_idx = np.where((lats >= row["lat_min"]) & (lats <= row["lat_max"]))[0]                                      
            lon_idx = np.where((lons >= row["lon_min"]) & (lons <= row["lon_max"]))[0]                                                                                                  
            if lat_idx.size == 0 or lon_idx.size == 0:
                print(f"SKIP [{orig_idx}]: {fname} — no pixels in range around ({row['lat']:.2f}, {row['lon']:.2f})")       
                return None                                                                                                 
            r0, r1 = lat_idx.min(), lat_idx.max() + 1
            c0, c1 = lon_idx.min(), lon_idx.max() + 1                                                                       
                
            # require cloud quality to meet quality checks                                                                  
            cloud = root["support_data"]["eff_cloud_fraction"][0, r0:r1, c0:c1]
            valid = ~np.isnan(cloud)                                                                                        
            if not valid.any():
                print(f"SKIP [{orig_idx}]: {fname} — cloud array is all NaN")                                               
                return None                                                                                                 
            frac_clean = np.mean(cloud[valid] <= MIN_PIXEL_CLOUD)
            if frac_clean < MIN_IMG_CLOUD:                                                                                  
                print(f"SKIP [{orig_idx}]: {fname} — only {frac_clean:.1%} pixels below cloud threshold")                   
                return None
                                                                                                                            
            # require QA values = 0                                                                                         
            qa = root["product"]["main_data_quality_flag"][0, r0:r1, c0:c1]
            if not np.all(qa == 0):                                                                                         
                frac_bad = np.mean(qa != 0)
                print(f"SKIP [{orig_idx}]: {fname} — {frac_bad:.1%} pixels have QA != 0")                                   
                return None
                                                                                                                            
            no2 = root["product"]["vertical_column_troposphere"][0, r0:r1, c0:c1].astype(np.float32)                        

        # resize patch to IMG_SIZE x IMG_SIZE                                                                               
        no2 = zoom(no2, (IMG_SIZE / no2.shape[0], IMG_SIZE / no2.shape[1]), order=1)                                        
        return (orig_idx, no2[np.newaxis, ...].astype(np.float32))
                                                                                                                            
    except Exception as e:                                                                                                  
        print(f"ERROR [{orig_idx}]: {fname}: {e}")                                                                          
        return None                                                                                                         
                
def visualize_split(df: pd.DataFrame, valid_idxs: list, split: str):                                                        
    """Visualize geographic distribution and label distribution for a split"""
    valid_df = df.loc[valid_idxs]                                                                                           
                
    # get US outline for geographic visualization                                                                                                                              
    us = gpd.read_file(COUNTRIES_URL)                                                                                           
    us = us[us.NAME == "United States of America"]                                                                             
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))                                                                         
    fig.suptitle(f"{split} (n={len(valid_df)})", fontsize=14)
                                                                                                                            
    # geographic distribution                                                                                               
    us.plot(ax=axes[0], color="lightgray", edgecolor="black")
    axes[0].scatter(valid_df["lon"], valid_df["lat"], s=5, alpha=0.3)                                                       
    axes[0].set_xlim(-130, -65)                                                                                             
    axes[0].set_ylim(24, 50)
    axes[0].set_xlabel("Longitude")                                                                                         
    axes[0].set_ylabel("Latitude")                                                                                          
    axes[0].set_title("Geographic Distribution")
                                                                                                                            
    # log label distribution                                                                                                
    axes[1].hist(np.log1p(valid_df[LABEL_COL]), bins=50, alpha=0.7)
    axes[1].set_xlabel(f"log({LABEL_COL} + 1)")                                                                             
    axes[1].set_ylabel("Count")
    axes[1].set_title("Label Distribution (log scale)")                                                                     
                                                                                                                            
    plt.tight_layout()
    os.makedirs(VIS_DIR, exist_ok=True)                                                                                     
    plt.savefig(os.path.join(VIS_DIR, f"dataset_vis_{split}.png"), dpi=150)
    plt.close()                                                                                                             

def process_split(df: pd.DataFrame, split: str, cities_gdf: gpd.GeoDataFrame):                                              
    """Parallel patch extraction, zarr write, and dataset DataFrame save for a given split"""
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

    # undersample image count for given split                                                                               
    n_store = min(SPLIT_SIZES[split], n_valid)
    valid = valid[:n_store]                                                                                                 

    # store image data in zarr                                                                                              
    tempo_path = os.path.join(IMAGES_DIR, f"{split}_tempo.zarr")
    tempo_store = zarr.open(tempo_path, mode="w",
        shape=(n_store, 1, IMG_SIZE, IMG_SIZE),                                                                             
        chunks=(1, 1, IMG_SIZE, IMG_SIZE),
        dtype="float32"                                                                                                     
    )           

    valid_idxs = []
    for i, (orig_idx, patch) in enumerate(valid):
        tempo_store[i] = patch                                                                                              
        valid_idxs.append(orig_idx)
                                                                                                                            
    # store dataset DataFrame with original columns, label, wind, and zarr index                                            
    out_df = df.loc[valid_idxs].drop(columns=["lat_min", "lat_max", "lon_min", "lon_max"]).copy()
    out_df["zarr_idx"] = range(n_store)                                                                                     
    out_df["split"] = split
    out_df.to_csv(os.path.join(DATASET_DF, f"{split}_df.csv"), index=False)                                                 
                
    # visualize split distribution                                                                                          
    visualize_split(df, valid_idxs, split)
                                                                                                                            
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