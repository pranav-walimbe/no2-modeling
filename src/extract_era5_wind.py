"""
extract_era5_wind.py

For each row in train/val/test.csv, extract ERA5 u10/v10 at the
plant's lat/lon and hour, then save enriched CSV.

Usage:
    python extract_era5_wind.py --split train
    python extract_era5_wind.py --split val
    python extract_era5_wind.py --split test
"""

import argparse
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
NOX_DIR  = Path("/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_powerplant_data")
ERA5_DIR = Path("/global/scratch/projects/fc_nitrates/pranavwalimbe/era5")
OUT_DIR  = Path("/global/scratch/users/ryuto/fc_nitrates/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ERA5 cache (keep one month open at a time) ─────────────────────────────────
_era5_cache = {}

def get_era5_ds(year, month):
    key = (year, month)
    if key not in _era5_cache:
        _era5_cache.clear()  # only hold one month in memory at a time
        path = ERA5_DIR / f"era5_{year}_{month:02d}.nc"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            return None
        _era5_cache[key] = xr.open_dataset(path, engine='netcdf4')
    return _era5_cache[key]

def extract_era5(lat, lon, date_str, hour):
    """Return dict of ERA5 wind variables at nearest grid point and hour."""
    try:
        dt = pd.to_datetime(date_str)
        ds = get_era5_ds(dt.year, dt.month)
        if ds is None:
            return {'era5_u10': np.nan, 'era5_v10': np.nan,
                    'era5_wind_speed': np.nan, 'era5_wind_dir': np.nan}

        target_time = pd.Timestamp(dt.year, dt.month, dt.day, int(hour))

        val = ds.sel(
            latitude=lat,
            longitude=lon,
            valid_time=target_time,
            method='nearest'
        )
        u10 = float(val['u10'].values)
        v10 = float(val['v10'].values)
        speed = float(np.sqrt(u10**2 + v10**2))
        direction = float(np.degrees(np.arctan2(-u10, -v10)) % 360)

        return {
            'era5_u10': u10,
            'era5_v10': v10,
            'era5_wind_speed': speed,
            'era5_wind_dir': direction,
        }
    except Exception as e:
        print(f"  ERA5 error {date_str} h{hour} lat{lat} lon{lon}: {e}")
        return {'era5_u10': np.nan, 'era5_v10': np.nan,
                'era5_wind_speed': np.nan, 'era5_wind_dir': np.nan}

# ── Main ───────────────────────────────────────────────────────────────────────
def main(split):
    in_path  = NOX_DIR / f"{split}.csv"
    out_path = OUT_DIR / f"{split}_era5.csv"

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path)
    print(f"  {len(df)} rows")

    records = []
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  Row {i}/{len(df)} ...")
        records.append(extract_era5(row['lat'], row['lon'], row['date'], row['hour']))

    era5_df = pd.DataFrame(records, index=df.index)
    out_df  = pd.concat([df, era5_df], axis=1)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Shape: {out_df.shape}")
    n_valid = era5_df['era5_u10'].notna().sum()
    print(f"Valid ERA5 extractions: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    main(args.split)
