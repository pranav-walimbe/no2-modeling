"""Add current facility attributes to hourly emissions in compressed Parquet."""

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from config import EMISSIONS_RECORDS_CSV, EMISSIONS_START_DATE, FULL_DATA_PARQUET
from prerequisites import require_campd_credentials

API_URL = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
CHUNK_SIZE = 500_000
MAX_RETRIES = 3
RECORDS_PER_PAGE = 500
ROW_GROUP_SIZE = 250_000

EMISSIONS_DTYPES = {
    "stateCode": "string",
    "facilityName": "string",
    "facilityId": "Int64",
    "unitId": "string",
    "date": "string",
    "hour": "Int8",
    "opTime": "Float64",
    "noxMass": "Float64",
    "noxRate": "Float64",
    "grossLoad": "Float64",
    "primaryFuelInfo": "string",
    "unitType": "string",
}

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

ATTRIBUTE_STRING_COLUMNS = [
    "county",
    "countyCode",
    "fipsCode",
    "nercRegion",
    "sourceCategory",
    "ownerOperator",
    *(column for column in UNIT_ATTRIBUTE_COLUMNS.values() if column != "maxHourlyHIRate"),
]


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
        except requests.exceptions.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            detail = type(error).__name__ if status is None else f"{type(error).__name__} (HTTP {status})"
            if attempt == MAX_RETRIES:
                print(f"WARNING: facility {facility_id} failed for {year}, page {page}: {detail}")
                return None
            time.sleep(2**attempt)
        except (ValueError, TypeError) as error:
            if attempt == MAX_RETRIES:
                print(f"WARNING: facility {facility_id} failed for {year}, page {page}: {type(error).__name__}")
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


def _augment_chunk(
    chunk: pd.DataFrame,
    facility_attributes: pd.DataFrame,
    unit_attributes: pd.DataFrame,
) -> pd.DataFrame:
    # Join approved attributes and stabilize Parquet column types
    data = chunk.copy()
    data["facilityId"] = pd.to_numeric(data["facilityId"], errors="coerce").astype("Int64")
    data["unitId"] = data["unitId"].astype("string").str.strip()
    data["unitIdKey"] = data["unitId"]
    augmented = data.merge(facility_attributes, on="facilityId", how="left")
    augmented = augmented.merge(unit_attributes, on=["facilityId", "unitIdKey"], how="left")
    augmented = augmented.drop(columns="unitIdKey")

    augmented["facilityAttributeYear"] = pd.to_numeric(
        augmented["facilityAttributeYear"],
        errors="coerce",
    ).astype("Int64")
    for column in ("lat", "lon", "epaRegion", "maxHourlyHIRate"):
        augmented[column] = pd.to_numeric(augmented[column], errors="coerce").astype("Float64")
    for column in ATTRIBUTE_STRING_COLUMNS:
        augmented[column] = augmented[column].astype("string")
    return augmented.dropna(subset=["lat", "lon", "epaRegion"])


def write_augmented_parquet(
    input_path: Path,
    output_path: Path,
    facility_attributes: pd.DataFrame,
    unit_attributes: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Stream enriched emissions into one atomic Zstd Parquet file.

    Args:
        input_path: Raw hourly emissions CSV.
        output_path: Final compressed Parquet file.
        facility_attributes: One row of facility attributes per facility.
        unit_attributes: One row of unit attributes per facility and unit.
        chunk_size: Number of CSV rows read per batch.

    Returns:
        Number of enriched rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    row_count = 0

    try:
        reader = pd.read_csv(input_path, dtype=EMISSIONS_DTYPES, chunksize=chunk_size)
        for chunk in reader:
            augmented = _augment_chunk(chunk, facility_attributes, unit_attributes)
            if augmented.empty:
                continue
            table = pa.Table.from_pandas(augmented, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    temporary_path,
                    schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            elif schema is not None:
                table = table.cast(schema)
            writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            row_count += len(augmented)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()

    if row_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("No emissions rows had complete facility location attributes")
    os.replace(temporary_path, output_path)
    return row_count


def main() -> None:
    """Write emissions records augmented with current facility attributes."""
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

    row_count = write_augmented_parquet(
        input_path=Path(EMISSIONS_RECORDS_CSV),
        output_path=Path(FULL_DATA_PARQUET),
        facility_attributes=facility_attributes,
        unit_attributes=unit_attributes,
    )
    print(f"Wrote {row_count:,} rows to {FULL_DATA_PARQUET}")


if __name__ == "__main__":
    main()
