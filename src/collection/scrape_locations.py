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
REQUEST_INTERVAL_SECONDS = 4
INITIAL_RETRY_DELAY_SECONDS = 30
MAX_RETRY_DELAY_SECONDS = 300
RATE_LIMIT_WAIT_SECONDS = 3_600
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
    year: int,
    page: int,
) -> list[dict[str, object]]:
    # Fetch one nationwide page and fail after bounded retries
    params = {
        "api_key": require_campd_credentials(),
        "year": year,
        "page": page,
        "perPage": RECORDS_PER_PAGE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching all facilities for {year}, page {page} (attempt {attempt}/{MAX_RETRIES})")
            response = requests.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            records = payload.get("items", []) if isinstance(payload, dict) else payload
            if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
                raise TypeError("unexpected facility-attribute response")
            time.sleep(REQUEST_INTERVAL_SECONDS)
            return records
        except requests.exceptions.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            detail = type(error).__name__ if status is None else f"{type(error).__name__} (HTTP {status})"
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Facility attributes failed for {year}, page {page}: {detail}") from error
            if status == 429:
                retry_after = error.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else RATE_LIMIT_WAIT_SECONDS
            else:
                delay = min(INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
            print(f"WARNING: facility attributes failed for {year}, page {page}: {detail}; retrying in {delay:.0f}s")
            time.sleep(delay)
        except (ValueError, TypeError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Facility attributes returned invalid data for {year}, page {page}: {type(error).__name__}"
                ) from error
            delay = min(INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
            print(
                f"WARNING: facility attributes returned invalid data for {year}, page {page}: "
                f"{type(error).__name__}; retrying in {delay:.0f}s"
            )
            time.sleep(delay)

    raise AssertionError("Retry loop exited unexpectedly")


def _fetch_attribute_year(year: int) -> list[dict[str, object]]:
    # Collect every nationwide page for one year
    records: list[dict[str, object]] = []
    page = 1
    while True:
        page_records = _fetch_attribute_page(year, page)
        records.extend(page_records)
        if len(page_records) < RECORDS_PER_PAGE:
            return records
        page += 1


def get_facility_attributes(
    facility_ids: list[int],
    latest_year: int,
    earliest_year: int,
) -> list[dict[str, object]]:
    """Return unit records from each facility's newest available year.

    Args:
        facility_ids: EPA facility identifiers required by the emissions data.
        latest_year: First facility-attribute year to query.
        earliest_year: Oldest facility-attribute year to query.

    Returns:
        Facility and unit records from the newest available year per facility.
    """
    remaining_facility_ids = set(facility_ids)
    selected_records: list[dict[str, object]] = []

    for year in range(latest_year, earliest_year - 1, -1):
        year_records = _fetch_attribute_year(year)
        matched_records: list[dict[str, object]] = []
        matched_facility_ids: set[int] = set()
        for record in year_records:
            try:
                facility_id = int(record["facilityId"])
            except (KeyError, TypeError, ValueError):
                continue
            if facility_id in remaining_facility_ids:
                matched_records.append(record)
                matched_facility_ids.add(facility_id)

        selected_records.extend(matched_records)
        remaining_facility_ids.difference_update(matched_facility_ids)
        print(
            f"Matched {len(matched_facility_ids):,} facilities for {year}; "
            f"{len(remaining_facility_ids):,} still need attributes"
        )
        if not remaining_facility_ids:
            return selected_records

    missing = ", ".join(str(facility_id) for facility_id in sorted(remaining_facility_ids))
    raise RuntimeError(f"CAMPD returned no facility attributes for required facilities: {missing}")


def _build_attribute_frames(records: list[dict[str, object]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Separate facility-level fields from unit-level fields before the hourly join
    attributes = pl.DataFrame(records, schema=ATTRIBUTE_SCHEMA, strict=False).with_columns(
        pl.col("unitId").str.strip_chars()
    )
    facility_attributes = attributes.select(
        "facilityId",
        *(pl.col(source).alias(target) for source, target in FACILITY_ATTRIBUTE_COLUMNS.items()),
    ).unique(subset="facilityId", keep="first", maintain_order=True)
    unit_attributes = attributes.select(
        "facilityId",
        pl.col("unitId").alias("unitIdKey"),
        *(pl.col(source).alias(target) for source, target in UNIT_ATTRIBUTE_COLUMNS.items()),
    ).unique(subset=["facilityId", "unitIdKey"], keep="first", maintain_order=True)
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
    source_row_count = source.select(pl.len()).collect().item()
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
        if row_count < source_row_count:
            raise RuntimeError(
                f"Location enrichment dropped {source_row_count - row_count:,} of {source_row_count:,} emissions rows"
            )
        if row_count > source_row_count:
            raise RuntimeError(f"Location enrichment added {row_count - source_row_count:,} duplicate emissions rows")
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

    attribute_records = get_facility_attributes(
        facility_ids=facility_ids,
        latest_year=date.today().year,
        earliest_year=EMISSIONS_START_DATE.year,
    )

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
