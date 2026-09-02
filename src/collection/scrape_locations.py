"""Add current facility attributes to hourly emissions with Polars."""

import os
import time
from datetime import date
from pathlib import Path

import polars as pl
import requests

from config import EMISSIONS_RECORDS_PARQUET, EMISSIONS_START_DATE, FULL_DATA_PARQUET
from prerequisites import require_campd_credentials

API_URL = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
MAX_RETRIES = 3
RECORDS_PER_PAGE = 500
ROW_GROUP_SIZE = 250_000

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

ATTRIBUTE_SCHEMA = {
    "facilityId": pl.Int64,
    "unitId": pl.String,
    "year": pl.Int64,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "epaRegion": pl.Float64,
    **{
        column: pl.String
        for column in (
            (set(FACILITY_ATTRIBUTE_COLUMNS) | set(UNIT_ATTRIBUTE_COLUMNS))
            - {"year", "latitude", "longitude", "epaRegion", "maxHourlyHIRate"}
        )
    },
    "maxHourlyHIRate": pl.Float64,
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


def _build_attribute_frames(records: list[dict[str, object]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Separate facility-level fields from unit-level fields before the hourly join
    attributes = pl.DataFrame(records, schema=ATTRIBUTE_SCHEMA, strict=False).with_columns(
        pl.col("unitId").str.strip_chars()
    )
    facility_attributes = (
        attributes.select(
            "facilityId",
            *(pl.col(source).alias(target) for source, target in FACILITY_ATTRIBUTE_COLUMNS.items()),
        )
        .unique(subset="facilityId", keep="first", maintain_order=True)
    )
    unit_attributes = (
        attributes.select(
            "facilityId",
            pl.col("unitId").alias("unitIdKey"),
            *(pl.col(source).alias(target) for source, target in UNIT_ATTRIBUTE_COLUMNS.items()),
        )
        .unique(subset=["facilityId", "unitIdKey"], keep="first", maintain_order=True)
    )
    return facility_attributes, unit_attributes


def write_augmented_parquet(
    input_path: Path,
    output_path: Path,
    facility_attributes: pl.DataFrame,
    unit_attributes: pl.DataFrame,
) -> int:
    """Stream enriched emissions into one atomic Zstd Parquet file.

    Args:
        input_path: Raw hourly emissions Parquet file.
        output_path: Final compressed Parquet file.
        facility_attributes: One row of facility attributes per facility.
        unit_attributes: One row of unit attributes per facility and unit.

    Returns:
        Number of enriched rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)

    source = pl.scan_parquet(input_path)
    missing_columns = {"facilityId", "unitId"}.difference(source.collect_schema().names())
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Raw hourly emissions Parquet is missing required columns: {missing}")

    unit_id = pl.col("unitId").cast(pl.String, strict=False).str.strip_chars()
    augmented = (
        source.with_columns(
            pl.col("facilityId").cast(pl.Int64, strict=False),
            unit_id.alias("unitId"),
            unit_id.alias("unitIdKey"),
        )
        .join(facility_attributes.lazy(), on="facilityId", how="left")
        .join(unit_attributes.lazy(), on=["facilityId", "unitIdKey"], how="left")
        .drop("unitIdKey")
        .drop_nulls(["lat", "lon", "epaRegion"])
    )

    try:
        augmented.sink_parquet(
            temporary_path,
            compression="zstd",
            statistics=True,
            row_group_size=ROW_GROUP_SIZE,
        )
        row_count = pl.scan_parquet(temporary_path).select(pl.len()).collect().item()
        if row_count == 0:
            raise RuntimeError("No emissions rows had complete facility location attributes")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count


def main() -> None:
    """Write hourly emissions enriched with current facility attributes."""
    require_campd_credentials()

    input_path = Path(EMISSIONS_RECORDS_PARQUET)
    facility_ids = (
        pl.scan_parquet(input_path)
        .select(pl.col("facilityId").cast(pl.Int64, strict=False))
        .drop_nulls()
        .unique()
        .sort("facilityId")
        .collect()["facilityId"]
        .to_list()
    )

    attribute_records: list[dict[str, object]] = []
    for facility_id in facility_ids:
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
        input_path=input_path,
        output_path=Path(FULL_DATA_PARQUET),
        facility_attributes=facility_attributes,
        unit_attributes=unit_attributes,
    )
    print(f"Wrote {row_count:,} rows to {FULL_DATA_PARQUET}")


if __name__ == "__main__":
    main()
