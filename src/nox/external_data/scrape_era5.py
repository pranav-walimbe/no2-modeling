"""
Script to collect ERA5 wind data tiles for a provided date range

Output: era5 monthly files in ERA5_DIR
"""

import cdsapi
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

def main():
    """loop through valid date range and download hourly files"""
    os.makedirs(ERA5_DIR, exist_ok=True)
    client = cdsapi.Client()

    for year in range(WIND_START_YEAR, WIND_END_YEAR + 1):
        m_start = WIND_START_MONTH if year == WIND_START_YEAR else 1
        m_end = WIND_END_MONTH if year == WIND_END_YEAR else 12
        for month in range(m_start, m_end + 1):
            output_file = os.path.join(ERA5_DIR, f"era5_{year}_{month:02d}.nc")
            # check if file already present
            if os.path.exists(output_file):
                continue
            try:
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            # "2m_temperature",
                            # "boundary_layer_height",
                        ],
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": [50, -130, 26, -67],
                        "format": "netcdf",
                    },
                    output_file)
            except Exception as e:
                print(f"ERROR downloading {year}-{month:02d}: {e}")

if __name__ == "__main__":
    main()