"""
Scrape hourly NOx emissions data from the EPA CAMPD/EASEY API

Output: nox_emissions_all.csv
"""

import sys
import os                                                                                                                                       
import csv
import time
from datetime import timedelta
import requests
import pandas as pd
import calendar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # access config                                                                                                                                             
from config import *

# states to include in requests
STATE_CODES = [
    "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID", 
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", 
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", 
    "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", 
    "VA", "WA", "WV", "WI", "WY",
]

# fields to keep from API response
NOX_COLS = [
    "stateCode", # state abbreviation
    "facilityName", # power plant name
    "facilityId", # EPA facility ID (ORISPL code)
    "unitId", # generating unit ID within the facility
    "date", # date of measurement (YYYY-MM-DD)
    "hour", # hour of measurement (0–23)
    "opTime", # % of hour in active operation 
    "noxMass", # NOx emissions mass
    "noxMassUom", # emissions amount measurement unit
    "noxRate", # NOx emissions rate
    "noxRateUom", # emissions rate measurement unit
    "grossLoad", # gross electrical output
    "grossLoadUom", # grossload measurement unit
    "primaryFuelInfo", # fuel type (e.g. natural gas, coal)
    "unitType", # generating unit type (e.g. tangentially-fired, combined cycle)
]

def month_ranges(start: date, end: date):
    """Return list of (begin_date, end_date) strings for each month in the range."""
    ranges = []                                                                                                                                 
    current = start.replace(day=1)                                                                                                              
    while current <= end:                                                                                                                       
        last_day = calendar.monthrange(current.year, current.month)[1]                                                                          
        month_end = min(current.replace(day=last_day), end)                                                                                     
        ranges.append((current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))                                                           
        current = month_end + timedelta(days=1)                                                                                                 
    return ranges

def fetch_chunk(state: str, begin: str, end: str, retries: int = 3):
    """Fetch one state/month chunk from the API"""
    url = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly"
    params = {
        "api_key": API_KEY,
        "beginDate": begin,
        "endDate": end,
        "stateCode": state,
        "operatingHoursOnly": True,
    }
    # make requests with exponential backoff
    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching {state} {begin} to {end} (attempt {attempt}/{retries})")
            resp = requests.get(url, params=params, timeout=120)
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
    if os.path.exists(EMISSIONS_RECORDS_CSV):
        os.remove(EMISSIONS_RECORDS_CSV)
    months = list(month_ranges(EMISSIONS_START_DATE, EMISSIONS_END_DATE))
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

            df.to_csv(EMISSIONS_RECORDS_CSV, mode="a", index=False, header=not header_written,
                       quoting=csv.QUOTE_NONNUMERIC)
            header_written = True

if __name__ == "__main__":
    main()