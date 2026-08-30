"""
Script to collect ERA5 wind data tiles for a provided date range

Output: era5 monthly files in ERA5_DIR
"""

import os

import cdsapi

from config import (
    ERA5_DIR,
    WIND_END_MONTH,
    WIND_END_YEAR,
    WIND_START_MONTH,
    WIND_START_YEAR,
)
from prerequisites import require_cds_credentials


def main():
    """loop through valid date range and download hourly files"""
    require_cds_credentials()
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
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [50, -130, 24, -65],
                    "format": "netcdf",
                },
                output_file,
            )


if __name__ == "__main__":
    main()
