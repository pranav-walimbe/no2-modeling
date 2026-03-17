"""                                                                                                                                             
Extract ERA5 u10/v10 wind components at each power plant location and hour                                                                     
                                                                                                                                                
Output: wind columns appended to train.csv, val.csv, test.csv in STRAT_BASE_DIR                                                                 
"""                                                                                                                                             
                                                                                                                                                
import sys      
import os                                                                                                                                       
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

_era5_cache = {} # worker-level cache
def get_era5_ds(year, month):
    """access era5 files using a cache to minimize I/O demand"""
    key = (year, month)
    if key not in _era5_cache:
        _era5_cache.clear()
        path = os.path.join(ERA5_DIR, f"era5_{year}_{month:02d}.nc")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            return None
        _era5_cache[key] = xr.open_dataset(path, engine="netcdf4")
    return _era5_cache[key]

def extract_era5_row(lat, lon, date_str, hour):
    """Extract ERA5 u10/v10 at nearest grid point and hour"""
    try:
        dt = pd.to_datetime(date_str)
        ds = get_era5_ds(dt.year, dt.month)
        if ds is None:
            raise FileNotFoundError
        target_time = pd.Timestamp(dt.year, dt.month, dt.day, int(hour))
        val = ds.sel(latitude=lat, longitude=lon, valid_time=target_time, method="nearest")
        u10 = float(val["u10"].values)
        v10 = float(val["v10"].values)
        return {
            "era5_u10":        u10,
            "era5_v10":        v10,
            "era5_wind_speed": float(np.sqrt(u10**2 + v10**2)),
            "era5_wind_dir":   float(np.degrees(np.arctan2(-u10, -v10)) % 360),
        }
    except Exception as e:
        print(f"ERROR: {date_str} h{hour} lat={lat} lon={lon}: {e}")
        return {"era5_u10": np.nan, "era5_v10": np.nan,
                "era5_wind_speed": np.nan, "era5_wind_dir": np.nan}

def map_chunk(chunk):
    """pickleable helper function for parallelized wind value extraction"""
    records = [
        extract_era5_row(row["lat"], row["lon"], row["date"], row["hour"])
        for _, row in chunk.iterrows()
    ]
    return pd.concat([chunk, pd.DataFrame(records, index=chunk.index)], axis=1)

def process_split(csv_path, split):
    """extract wind values for a single split csv"""
    df = pd.read_csv(csv_path)
    df = df.sort_values("date") # optimize worker-level cache hits by sorting by date

    # parallelized wind value extraction
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), NUM_CORES) if len(idx) > 0]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(map_chunk, chunks))

    # shuffle order so csv splits do not remain date-ordered
    out_df = pd.concat(results).sample(frac=1, random_state=42).reset_index(drop=True)
    out_df.to_csv(csv_path, index=False)

def main():
    process_split(TRAIN_CSV, "train")
    process_split(VAL_CSV,   "val")
    process_split(TEST_CSV,  "test")

if __name__ == "__main__":
    main()