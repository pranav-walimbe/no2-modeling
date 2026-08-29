"""Extract and quality-filter TEMPO image patches and ERA5 wind features."""

import os

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from no2_modeling.config import (
    ERA5_DIR,
    IMG_CLOUD_FILTER,
    IMG_QA_FILTER,
    IMG_RANGE,
    IMG_SIZE,
    IMG_VAL_FILTER,
    MAX_IMG_VAL,
    MIN_CITY_PROXIMITY,
    MIN_IMG_VAL,
    MIN_PIXEL_CLOUD,
    TEMPO_DIR,
)


def compute_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute image-patch bounds in latitude and longitude for each record."""
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    buffered = gdf.to_crs("EPSG:5070").buffer((IMG_RANGE / 2) * 1000).to_crs("EPSG:4326")
    bounds = buffered.bounds
    df = df.copy()
    df["lat_min"] = bounds["miny"].values
    df["lat_max"] = bounds["maxy"].values
    df["lon_min"] = bounds["minx"].values
    df["lon_max"] = bounds["maxx"].values
    return df


def filter_by_city_proximity(df: pd.DataFrame, cities_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Remove records too close to a major city."""
    plants_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    plant_coords = np.column_stack([plants_gdf.geometry.x, plants_gdf.geometry.y])
    city_coords = np.column_stack([cities_gdf.geometry.x, cities_gdf.geometry.y])
    distances_km = cKDTree(city_coords).query(plant_coords, k=1)[0] / 1000
    return df[distances_km >= MIN_CITY_PROXIMITY].reset_index(drop=True)


def extract_no2_data(fname: str, row: dict, orig_idx: int, is_prev: bool = False) -> np.ndarray | None:
    """Extract and quality-filter a TEMPO NO2 patch."""
    path = os.path.join(TEMPO_DIR, fname)
    with nc.Dataset(path) as root:
        latitudes = root["latitude"][:]
        longitudes = root["longitude"][:]
        lat_idx = np.where((latitudes >= row["lat_min"]) & (latitudes <= row["lat_max"]))[0]
        lon_idx = np.where((longitudes >= row["lon_min"]) & (longitudes <= row["lon_max"]))[0]
        if lat_idx.size == 0 or lon_idx.size == 0:
            print(f"SKIP [{orig_idx}]: {fname} - no pixels around ({row['lat']:.2f}, {row['lon']:.2f})")
            return None
        row_start, row_end = lat_idx.min(), lat_idx.max() + 1
        col_start, col_end = lon_idx.min(), lon_idx.max() + 1

        if not is_prev:
            cloud = root["support_data"]["eff_cloud_fraction"][0, row_start:row_end, col_start:col_end]
            valid = ~np.ma.getmaskarray(cloud)
            if not valid.any():
                print(f"SKIP [{orig_idx}]: {fname} - cloud array is all masked")
                return None
            fraction_clean = np.mean(np.array(cloud)[valid] <= MIN_PIXEL_CLOUD)
            if fraction_clean < IMG_CLOUD_FILTER:
                print(f"SKIP [{orig_idx}]: {fname} - {fraction_clean:.1%} pixels below cloud threshold")
                return None

        qa = np.ma.filled(
            root["product"]["main_data_quality_flag"][0, row_start:row_end, col_start:col_end],
            fill_value=2,
        )
        fraction_good = np.mean(qa == 0)
        if fraction_good < IMG_QA_FILTER:
            print(f"SKIP [{orig_idx}]: {fname} - {fraction_good:.1%} pixels have QA == 0")
            return None

        no2_masked = root["product"]["vertical_column_troposphere"][0, row_start:row_end, col_start:col_end]
        if np.ma.getmaskarray(no2_masked).any():
            print(f"SKIP [{orig_idx}]: {fname} - NO2 patch contains masked pixels")
            return None
        no2 = np.array(no2_masked)
        if no2.min() < MIN_IMG_VAL:
            print(f"SKIP [{orig_idx}]: {fname} - min pixel value {no2.min():.2e} < {MIN_IMG_VAL:.2e}")
            return None
        fraction_clipped = np.mean(no2 >= MAX_IMG_VAL)
        if fraction_clipped > IMG_VAL_FILTER:
            print(f"SKIP [{orig_idx}]: {fname} - {fraction_clipped:.1%} pixels at max value")
            return None

    return zoom(no2, (IMG_SIZE / no2.shape[0], IMG_SIZE / no2.shape[1]), order=1)


def extract_wind_scalars(row: dict) -> tuple[float, float]:
    """Extract ERA5 wind values nearest to a plant location and hour."""
    path = os.path.join(ERA5_DIR, row["era5"])
    with nc.Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        latitudes = dataset["latitude"][:]
        longitudes = dataset["longitude"][:]
        lat_idx = int(np.argmin(np.abs(latitudes - row["lat"])))
        lon_idx = int(np.argmin(np.abs(longitudes - row["lon"])))
        timestamp = pd.Timestamp(row["date"])
        time_idx = (timestamp.day - 1) * 24 + int(row["hour"])
        u10 = float(dataset["u10"][time_idx, lat_idx, lon_idx])
        v10 = float(dataset["v10"][time_idx, lat_idx, lon_idx])
    return u10, v10


def extract_image_data(args: tuple[int, dict]) -> tuple[int, np.ndarray, float, float, float] | None:
    """Extract two image channels, a plume score, and wind features."""
    orig_idx, row = args
    try:
        no2 = extract_no2_data(row["tempo"], row, orig_idx)
        if no2 is None:
            return None
        previous_no2 = extract_no2_data(row["prev_tempo"], row, orig_idx, is_prev=True)
        if previous_no2 is None:
            return None
        u10, v10 = extract_wind_scalars(row)
        delta_no2 = no2 - previous_no2
        p10, p50, p99 = np.percentile(no2, [10, 50, 99])
        plume_score = (p99 - p50) / (p50 - p10 + 1e-10)
        patch = np.stack([no2, delta_no2], axis=0)[np.newaxis, ...].astype(np.float32)
        return orig_idx, patch, plume_score, u10, v10
    except (IndexError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR [{orig_idx}]: {row['tempo']}: {error}")
        return None
