"""                           
Script to splice satellite images and generate modeling dataset
                                            
Output: train/val/test tempo zarr stores, label numpy arrays, and wind numpy arrays in DATASET_DIR
"""
                                                                                                                                                
import sys
import os                                                                                                                                       
import zarr     
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
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
    plants_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs("EPSG:5070")
    plant_coords = np.column_stack([plants_gdf.geometry.x, plants_gdf.geometry.y])
    cities_coords = np.column_stack([cities_gdf.geometry.x, cities_gdf.geometry.y])

    # store locations + determine plant proximities using spatial lookup data structure
    dists_km = cKDTree(cities_coords).query(plant_coords, k=1)[0] / 1000
    mask = dists_km >= MIN_CITY_PROXIMITY
    return df[mask].reset_index(drop=True)

def extract_tempo_patch(args):
    """perform TEMPO patch extraction with pixel quality filtering"""
    orig_idx, row = args
    fname = row["tempo"]
    try:
        fpath = os.path.join(TEMPO_DIR, fname)
        ds = xr.open_dataset(fpath, engine="netcdf4")
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        ds.close()

        # find IMG_RANGE x IMG_RANGE patch bounds 
        lat_idx = np.where((lats >= row["lat_min"]) & (lats <= row["lat_max"]))[0]
        lon_idx = np.where((lons >= row["lon_min"]) & (lons <= row["lon_max"]))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            print(f"SKIP [{orig_idx}]: {fname} — no pixels in range around ({row['lat']:.2f}, {row['lon']:.2f})")
            return None
        r0, r1 = lat_idx.min(), lat_idx.max() + 1
        c0, c1 = lon_idx.min(), lon_idx.max() + 1

        # access tempo data: cloud fraction, pixel quality, no2 concentration
        ds = xr.open_dataset(fpath, engine="netcdf4", group="support_data")                                                                             
        cloud = ds["eff_cloud_fraction"].squeeze().isel(latitude=slice(r0, r1), longitude=slice(c0, c1)).values                                       
        ds.close()                                                                
        ds = xr.open_dataset(fpath, engine="netcdf4", group="product")
        qa  = ds["main_data_quality_flag"].squeeze().isel(latitude=slice(r0, r1), longitude=slice(c0, c1)).values                                       
        no2 = ds["vertical_column_troposphere"].squeeze().isel(latitude=slice(r0, r1), longitude=slice(c0, c1)).values.astype(np.float32)
        ds.close()   

        # require valid cloud fraction threshold
        valid = ~np.isnan(cloud)
        if not valid.any():
            print(f"SKIP [{orig_idx}]: {fname} — cloud array is all NaN")
            return None                         
        if np.mean(cloud[valid] <= MIN_PIXEL_CLOUD) < MIN_IMG_CLOUD:
            frac_clean = np.mean(cloud[valid] <= MIN_PIXEL_CLOUD)                                                                                       
            print(f"SKIP [{orig_idx}]: {fname} — only {frac_clean:.1%} pixels below cloud threshold")
            return None     

        # require pixel quality = 0                                                                                                                            
        if not np.all(qa == 0):
            frac_bad = np.mean(qa != 0)
            print(f"SKIP [{orig_idx}]: {fname} — {frac_bad:.1%} pixels have QA != 0")
            return None

        # resize image to IMG_SIZE x IMG_SIZE
        no2 = zoom(no2, (IMG_SIZE / no2.shape[0], IMG_SIZE / no2.shape[1]), order=1)
        return (orig_idx, no2[np.newaxis, ...].astype(np.float32))

    except Exception as e:
        print(f"ERROR [{orig_idx}]: {fname}: {e}")
        return None

def process_split(df: pd.DataFrame, split: str, cities_gdf: gpd.GeoDataFrame):
    """Parallel patch extraction, zarr write, and label save for a given split"""
    df = filter_by_city_proximity(df, cities_gdf)
    df = compute_bounds(df)

    # parallelized tempo patch extraction
    args = [(orig_idx, row) for orig_idx, row in df.iterrows()]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(extract_tempo_patch, args))

    valid = [r for r in results if r is not None]
    n_valid = len(valid)
    print(f"  {n_valid} / {len(df)} patches passed filtering")
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

    # store labels in npy file
    labels_path = os.path.join(LABELS_DIR, f"{split}_labels.npy")
    labels = df.loc[valid_idxs, LABEL_COL].values.astype(np.float32)
    np.save(labels_path, labels)

    # store wind data (tabular) in npy file
    wind_path = os.path.join(WIND_DIR, f"{split}_wind.npy")                                                                              
    wind = df.loc[valid_idxs, WIND_COLS].values.astype(np.float32)
    np.save(wind_path, wind) 

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(WIND_DIR, exist_ok=True)

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