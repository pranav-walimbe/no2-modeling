"""
Script to augment hourly emissions data with location data from EPA facilities API

Output: nox_emissions_full.csv in output directory
"""

import pandas as pd 
import os 
from dotenv import load_dotenv
import requests
import time
import sys

# configuration
load_dotenv()
API_KEY = os.getenv("CAMPD_API_KEY")
if not API_KEY:
    sys.exit("ERROR: CAMPD_API_KEY not found in .env file.")
BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_emissions_1"
INPUT_PATH = os.path.join(BASE_DIR, "nox_emissions_all.csv")  
OUTPUT_PATH = os.path.join(BASE_DIR, "nox_emissions_full.csv")

# helper functions
def get_facility_location(facility_id: int, year: int = 2023, retries: int = 3):
    """collect lat, lon, epaRegion from EPA API"""
    url = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
    params = {
        "api_key": API_KEY,
        "facilityId": facility_id,
        "year": year,
        "page": 1,
        "perPage": 1,
    }
    # use exponential backoff with retries
    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching {facility_id} data (attempt {attempt}/{retries})")
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    return None
                return (data[0]["latitude"], data[0]["longitude"], data[0]["epaRegion"])
            print(f"ERROR: API {resp.status_code} for {facility_id}: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed for facility {facility_id} (attempt {attempt}): {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None

def main():
    # collect unique facility ids
    facility_ids = set()
    for chunk in pd.read_csv(INPUT_PATH, usecols=["facilityId"], chunksize=500000):
        facility_ids.update(chunk["facilityId"].unique())

    # fetch lat, lon, epaRegion for each facility
    facility_info = {}
    for fid in facility_ids:
        result = get_facility_location(fid)
        if result:
            facility_info[fid] = result
    info_df = pd.DataFrame.from_dict(facility_info, orient="index", columns=["lat", "lon", "epaRegion"])
    info_df.index.name = "facilityId"

    # replace existing file if exists
    if os.path.exists(OUTPUT_PATH):                                                                                                 
        os.remove(OUTPUT_PATH)

    # process input CSV in chunks + apply opTime / NaN filtering
    header_written = False
    for chunk in pd.read_csv(INPUT_PATH, chunksize=500000):
        chunk = chunk[chunk["opTime"] >= 1]
        chunk = chunk.merge(info_df, on="facilityId", how="left")
        chunk = chunk.dropna()
        if chunk.empty:
            continue
        chunk.to_csv(OUTPUT_PATH, mode="a", index=False, header=not header_written)
        header_written = True

if __name__ == "__main__":
    main()