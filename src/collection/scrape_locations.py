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
RECORDS_PER_PAGE = 500

FACILITY_ATTRIBUTE_COLUMNS = {
    "year": "facilityAttributeYear",
    "latitude": "lat",
    "longitude": "lon",
    "epaRegion": "epaRegion",
    "county": "county",
    "countyCode": "countyCode",
    "fipsCode": "fipsCode",
    "nercRegion": "nercRegion",
    "sourceCategory": "sourceCategory",
    "ownerOperator": "ownerOperator",
}

UNIT_ATTRIBUTE_COLUMNS = {
    "operatingStatus": "unitOperatingStatus",
    "commercialOperationDate": "commercialOperationDate",
    "associatedGeneratorsAndNameplateCapacity": "generatorAndNameplateCapacity",
    "associatedStacks": "associatedStacks",
    "primaryFuelInfo": "attributePrimaryFuelInfo",
    "secondaryFuelInfo": "secondaryFuelInfo",
    "unitType": "attributeUnitType",
    "maxHourlyHIRate": "maxHourlyHIRate",
    "noxControlInfo": "noxControlInfo",
    "so2ControlInfo": "so2ControlInfo",
    "pmControlInfo": "pmControlInfo",
    "hgControlInfo": "hgControlInfo",
    "programCodeInfo": "programCodeInfo",
    "noxPhase": "noxPhase",
    "so2Phase": "so2Phase",
}


def _fetch_attribute_page(
    facility_id: int,
    year: int,
    page: int,
) -> list[dict[str, object]] | None:
    # Fetch one page while distinguishing an empty result from a failed request
    params = {
        "api_key": require_campd_credentials(),
        "facilityId": facility_id,
        "year": year,
        "page": page,
        "perPage": RECORDS_PER_PAGE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching facility {facility_id} for {year}, page {page} (attempt {attempt}/{MAX_RETRIES})")
            response = requests.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            records = payload.get("items", []) if isinstance(payload, dict) else payload
            if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                raise TypeError("unexpected facility-attribute response")
            return records
        except (requests.exceptions.RequestException, ValueError, TypeError) as error:
            if attempt == MAX_RETRIES:
                print(f"WARNING: facility {facility_id} failed for {year}, page {page}: {error}")
                return None
            time.sleep(2**attempt)

    raise AssertionError("Retry loop exited unexpectedly")


def get_facility_attributes(
    facility_id: int,
    latest_year: int,
    earliest_year: int,
) -> list[dict[str, object]]:
    """Return all unit records from the newest available attribute year.

    Args:
        facility_id: EPA facility identifier.
        latest_year: First facility-attribute year to query.
        earliest_year: Oldest facility-attribute year to query.

    Returns:
        Facility and unit attribute records from the newest available year.
    """
    for year in range(latest_year, earliest_year - 1, -1):
        records: list[dict[str, object]] = []
        page = 1
        while True:
            page_records = _fetch_attribute_page(facility_id, year, page)
            if page_records is None:
                return []
            if not page_records:
                break
            records.extend(page_records)
            if len(page_records) < RECORDS_PER_PAGE:
                return records
            page += 1

        if records:
            return records

    return []


def _build_attribute_frames(
    records: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Separate facility-level fields from unit-level fields before the hourly join
    attributes = pd.DataFrame.from_records(records)
    required_columns = {"facilityId", "unitId", *FACILITY_ATTRIBUTE_COLUMNS, *UNIT_ATTRIBUTE_COLUMNS}
    attributes = attributes.reindex(columns=sorted(required_columns))
    attributes["facilityId"] = pd.to_numeric(attributes["facilityId"], errors="coerce").astype("Int64")
    attributes["unitIdKey"] = attributes["unitId"].astype("string").str.strip()

    facility_columns = ["facilityId", *FACILITY_ATTRIBUTE_COLUMNS]
    facility_attributes = (
        attributes[facility_columns]
        .rename(columns=FACILITY_ATTRIBUTE_COLUMNS)
        .drop_duplicates(subset=["facilityId"], keep="first")
    )

    unit_columns = ["facilityId", "unitIdKey", *UNIT_ATTRIBUTE_COLUMNS]
    unit_attributes = (
        attributes[unit_columns]
        .rename(columns=UNIT_ATTRIBUTE_COLUMNS)
        .drop_duplicates(subset=["facilityId", "unitIdKey"], keep="first")
    )
    return facility_attributes, unit_attributes


def main() -> None:
    """Write emissions records augmented with current facility locations."""
    require_campd_credentials()

    facility_ids: set[int] = set()
    for chunk in pd.read_csv(EMISSIONS_RECORDS_CSV, usecols=["facilityId"], chunksize=CHUNK_SIZE):
        facility_ids.update(chunk["facilityId"].dropna().astype(int).unique())

    attribute_records: list[dict[str, object]] = []
    for facility_id in sorted(facility_ids):
        attribute_records.extend(
            get_facility_attributes(
                facility_id,
                latest_year=date.today().year,
                earliest_year=EMISSIONS_START_DATE.year,
            )
        )
    if not attribute_records:
        raise RuntimeError("CAMPD returned no facility attributes")

    facility_attributes, unit_attributes = _build_attribute_frames(attribute_records)

    if os.path.exists(FULL_DATA_CSV):
        os.remove(FULL_DATA_CSV)

    header_written = False
    for chunk in pd.read_csv(EMISSIONS_RECORDS_CSV, chunksize=CHUNK_SIZE):
        chunk["facilityId"] = pd.to_numeric(chunk["facilityId"], errors="coerce").astype("Int64")
        chunk["unitIdKey"] = chunk["unitId"].astype("string").str.strip()
        augmented = chunk.merge(facility_attributes, on="facilityId", how="left")
        augmented = augmented.merge(unit_attributes, on=["facilityId", "unitIdKey"], how="left")
        augmented = augmented.drop(columns="unitIdKey")
        augmented = augmented.dropna(subset=["lat", "lon", "epaRegion"])
        if augmented.empty:
            continue
        augmented.to_csv(FULL_DATA_CSV, mode="a", index=False, header=not header_written)
        header_written = True


if __name__ == "__main__":
    main()
