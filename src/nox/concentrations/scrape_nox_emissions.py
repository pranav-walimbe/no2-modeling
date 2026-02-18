"""
Scrape hourly NOx emissions data from the EPA CAMPD/EASEY API
(sequential version for benchmarking against the parallelized script).

Output: campd_nox_all_states.csv
"""

import os
import sys
import csv
import time
from datetime import date, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv

# Configuration
load_dotenv()
API_KEY = os.getenv("CAMPD_API_KEY")
if not API_KEY:
    sys.exit("ERROR: CAMPD_API_KEY not found in .env file.")
STREAMING_URL = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly"
OUTPUT_CSV = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_emissions_1/nox_emissions_all.csv"
START_DATE = date(2023, 8, 1)
END_DATE = date(2025, 12, 30)

STATE_CODES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
]

NOX_COLS = [
    "stateCode",
    "facilityName",
    "facilityId",
    "unitId",
    "date",
    "hour",
    "opTime",
    "noxMass",
    "noxMassUom",
    "noxRate",
    "noxRateUom",
    "grossLoad",
    "grossLoadUom",
    "primaryFuelInfo",
    "unitType",
]

def month_ranges(start: date, end: date):
    """Return list of (begin_date, end_date) strings for each month in the range."""
    ranges = []                                                                                   
    current = start.replace(day=1)                                                                
    while current <= end:                                                                         
        if current.month == 12:                                                               
            month_end = current.replace(day=31)                                               
        else:                                                                                 
            month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)   
        if month_end > end:                                                                   
            month_end = end                                                                   
        ranges.append((current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))         
        if current.month == 12:                                                               
            current = date(current.year + 1, 1, 1)                                            
        else:                                                                                 
            current = date(current.year, current.month + 1, 1)                                
    return ranges 

def fetch_chunk(state: str, begin: str, end: str, retries: int = 3):
    """Fetch one state/month chunk from the API with exponential backoff."""
    params = {
        "api_key": API_KEY,
        "beginDate": begin,
        "endDate": end,
        "stateCode": state,
        "operatingHoursOnly": True,
    }
    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching {state} {begin} to {end} (attempt {attempt}/{retries})")
            resp = requests.get(STREAMING_URL, params=params, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            print(f"ERROR: API {resp.status_code} for {state} {begin}: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed for {state} {begin} (attempt {attempt}/{retries}): {e}")
        if attempt < retries:
            time.sleep(2 ** attempt)
    return None

def main():
    # replace existing file if present
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    months = list(month_ranges(START_DATE, END_DATE))
    header_written = False

    for state in STATE_CODES:
        for begin, end in months:
            rows = fetch_chunk(state, begin, end)
            if rows is None:
                print(f"ERROR: Failed to fetch {state} {begin} to {end}")
                continue
            
            # remove rows with NaN / missing data
            df = pd.DataFrame(rows)
            cols = [c for c in NOX_COLS if c in df.columns]
            df = df[cols]
            required_cols = {"noxMass", "noxRate", "grossLoad", "opTime"}
            if not required_cols.issubset(df.columns):
                print(f"ERROR: {state} {begin} missing columns")
                continue
            df = df.dropna(subset=required_cols)
            if df.empty:
                continue

            df.to_csv(OUTPUT_CSV, mode="a", index=False, header=not header_written,
                       quoting=csv.QUOTE_NONNUMERIC)
            header_written = True

if __name__ == "__main__":
    main()