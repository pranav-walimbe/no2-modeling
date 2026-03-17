"""
Extracts ERA5 u10/v10 wind components at each power plant location and hour.
Outputs train_era5.csv, val_era5.csv, test_era5.csv to ERA5_OUT_DIR (defined in config).
Each output file contains the original split columns plus era5_u10, era5_v10,
era5_wind_speed, and era5_wind_dir for each (plant, date, hour) observation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.nox.config import TRAIN_CSV, VAL_CSV, TEST_CSV, NUM_CORES, ERA5_DIR, ERA5_OUT_DIR
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

SPLITS = {
    "train": TRAIN_CSV,
    "val":   VAL_CSV,
    "test":  TEST_CSV,
}

def extract_era5_row(lat, lon, date_str, hour):
    """Extract ERA5 u10/v10 at nearest grid point and hour for a single observation."""
    try:
        dt = pd.to_datetime(date_str)
        path = Path(ERA5_DIR) / f"era5_{dt.year}_{dt.month:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        ds = xr.open_dataset(path, engine='netcdf4')
        target_time = pd.Timestamp(dt.year, dt.month, dt.day, int(hour))
        val = ds.sel(latitude=lat, longitude=lon, valid_time=target_time, method='nearest')
        u10 = float(val['u10'].values)
        v10 = float(val['v10'].values)
        ds.close()

        return {
            'era5_u10':        u10,
            'era5_v10':        v10,
            'era5_wind_speed': float(np.sqrt(u10**2 + v10**2)),
            'era5_wind_dir':   float(np.degrees(np.arctan2(-u10, -v10)) % 360),
        }
    except Exception as e:
        print(f"ERROR: {date_str} h{hour} lat{lat} lon{lon}: {e}")
        return {'era5_u10': np.nan, 'era5_v10': np.nan,
                'era5_wind_speed': np.nan, 'era5_wind_dir': np.nan}

def map_chunk(chunk):
    """Process one chunk of rows in parallel."""
    records = [
        extract_era5_row(row['lat'], row['lon'], row['date'], row['hour'])
        for _, row in chunk.iterrows()
    ]
    return pd.concat([chunk, pd.DataFrame(records, index=chunk.index)], axis=1)

def main():
    """Extract ERA5 wind variables for all splits and save to ERA5_OUT_DIR."""
    for split, in_csv in SPLITS.items():
        out_path = Path(ERA5_OUT_DIR) / f"{split}_era5.csv"

        df = pd.read_csv(in_csv)
        chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), NUM_CORES)
                  if len(idx) > 0]

        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            results = list(executor.map(map_chunk, chunks))

        out_df = pd.concat(results).reset_index(drop=True)
        out_df.to_csv(out_path, index=False)

        n_valid = out_df['era5_u10'].notna().sum()
        print(f"{split}: {n_valid}/{len(df)} valid extractions -> {out_path}")

if __name__ == '__main__':
    main()
