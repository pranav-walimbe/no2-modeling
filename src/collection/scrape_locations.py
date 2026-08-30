"""Add current facility locations to the raw hourly emissions records."""

import os
import time
from datetime import date

import pandas as pd
import requests

from config import EMISSIONS_RECORDS_CSV, EMISSIONS_START_DATE, FULL_DATA_CSV
from prerequisites import require_campd_credentials

API_URL = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
CHUNK_SIZE = 500_000
MAX_RETRIES = 3


def get_facility_location(
    facility_id: int,
    latest_year: int,
    earliest_year: int,
) -> tuple[float, float, int] | None:
    """Return coordinates from the newest available facility attributes.

    Args:
        facility_id: EPA facility identifier.
        latest_year: First facility-attribute year to query.
        earliest_year: Oldest facility-attribute year to query.

    Returns:
        Latitude, longitude, and EPA region when attributes are available.
    """
    for year in range(latest_year, earliest_year - 1, -1):
        params = {
            "api_key": require_campd_credentials(),
            "facilityId": int(facility_id),
            "year": year,
            "page": 1,
            "perPage": 1,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"Fetching facility {facility_id} for {year} (attempt {attempt}/{MAX_RETRIES})")
                response = requests.get(API_URL, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                records = payload.get("items", []) if isinstance(payload, dict) else payload
                if records:
                    record = records[0]
                    return (record["latitude"], record["longitude"], record["epaRegion"])
                break
            except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as error:
                if attempt == MAX_RETRIES:
                    print(f"WARNING: facility {facility_id} failed for {year}: {error}")
                    break
                time.sleep(2**attempt)

    return None


def main() -> None:
    """Write emissions records augmented with current facility locations."""
    require_campd_credentials()

    facility_ids: set[int] = set()
    for chunk in pd.read_csv(EMISSIONS_RECORDS_CSV, usecols=["facilityId"], chunksize=CHUNK_SIZE):
        facility_ids.update(chunk["facilityId"].dropna().astype(int).unique())

    facility_info: dict[int, tuple[float, float, int]] = {}
    for facility_id in sorted(facility_ids):
        location = get_facility_location(
            facility_id,
            latest_year=date.today().year,
            earliest_year=EMISSIONS_START_DATE.year,
        )
        if location is not None:
            facility_info[facility_id] = location

    info_frame = pd.DataFrame.from_dict(
        facility_info,
        orient="index",
        columns=["lat", "lon", "epaRegion"],
    )
    info_frame.index.name = "facilityId"

    if os.path.exists(FULL_DATA_CSV):
        os.remove(FULL_DATA_CSV)

    header_written = False
    for chunk in pd.read_csv(EMISSIONS_RECORDS_CSV, chunksize=CHUNK_SIZE):
        augmented = chunk.merge(info_frame, on="facilityId", how="left")
        augmented = augmented.dropna(subset=["lat", "lon", "epaRegion"])
        if augmented.empty:
            continue
        augmented.to_csv(FULL_DATA_CSV, mode="a", index=False, header=not header_written)
        header_written = True


if __name__ == "__main__":
    main()
