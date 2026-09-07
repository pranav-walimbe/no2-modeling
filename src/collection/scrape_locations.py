"""Add prediction-date facility attributes to hourly emissions with Polars."""

import os
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
from timezonefinder import TimezoneFinder

from collection.emissions_schema import (
    EMISSIONS_HOUR_UTC_COL,
    FACILITY_NAMEPLATE_CAPACITY_MW_COL,
    LOCAL_STANDARD_DATE_COL,
    LOCAL_STANDARD_HOUR_COL,
    TIME_ZONE_COL,
    UTC_STANDARD_OFFSET_HOURS_COL,
)
from config import EMISSIONS_RECORDS_PARQUET, FULL_DATA_PARQUET
from prerequisites import require_campd_credentials

API_URL = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
MAX_RETRIES = 3
RECORDS_PER_PAGE = 500
REQUEST_INTERVAL_SECONDS = 4
INITIAL_RETRY_DELAY_SECONDS = 30
MAX_RETRY_DELAY_SECONDS = 300
RATE_LIMIT_WAIT_SECONDS = 3_600
ROW_GROUP_SIZE = 250_000
STANDARD_OFFSET_REFERENCE = datetime(2025, 1, 1, 12)
FACILITY_ATTRIBUTE_YEAR_COL = "facilityAttributeYear"
UNIT_ATTRIBUTE_YEAR_COL = "unitAttributeYear"
PREDICTION_YEAR_COL = "_predictionYear"
GENERATOR_CAPACITY_PATTERN = re.compile(
    r"\s*(?P<generator>[^(),]+?)\s*\(\s*(?P<capacity>(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*"
)
CAPACITY_ATTRIBUTE_SCHEMA = {
    "facilityId": pl.Int64,
    FACILITY_ATTRIBUTE_YEAR_COL: pl.Int64,
    FACILITY_NAMEPLATE_CAPACITY_MW_COL: pl.Float64,
}

FacilityYear = tuple[int, int]
GeneratorValues = dict[FacilityYear, dict[str, set[float]]]

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
    """Return matching unit records for every requested attribute year.

    Args:
        facility_ids: EPA facility identifiers required by the emissions data.
        latest_year: Newest facility-attribute year to query.
        earliest_year: Oldest facility-attribute year to query.

    Returns:
        Facility and unit records from all requested years.
    """
    if earliest_year > latest_year:
        raise ValueError("earliest_year must not exceed latest_year")
    required_facility_ids = set(facility_ids)
    matched_facility_ids: set[int] = set()
    selected_records: list[dict[str, object]] = []

    for year in range(latest_year, earliest_year - 1, -1):
        year_records = _fetch_attribute_year(year)
        matched_records: list[dict[str, object]] = []
        matched_this_year: set[int] = set()
        for record in year_records:
            try:
                facility_id = int(record["facilityId"])
            except (KeyError, TypeError, ValueError):
                continue
            if facility_id in required_facility_ids:
                matched_records.append(record)
                matched_this_year.add(facility_id)

        selected_records.extend(matched_records)
        matched_facility_ids.update(matched_this_year)
        print(f"Matched {len(matched_this_year):,}/{len(required_facility_ids):,} required facilities for {year}")

    missing_facility_ids = required_facility_ids.difference(matched_facility_ids)
    if not missing_facility_ids:
        return selected_records
    missing = ", ".join(str(facility_id) for facility_id in sorted(missing_facility_ids))
    raise RuntimeError(f"CAMPD returned no facility attributes for required facilities: {missing}")


def _parse_generator_capacities(value: object) -> tuple[tuple[str, float], ...]:
    # Parse the CAMPD comma-separated generator and capacity field
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ValueError(f"Generator nameplate-capacity value must be text, found {type(value).__name__}")
    if not value.strip():
        return ()
    entries: list[tuple[str, float]] = []
    for part in value.split(","):
        match = GENERATOR_CAPACITY_PATTERN.fullmatch(part)
        if match is None:
            raise ValueError(f"Invalid generator nameplate-capacity entry: {part.strip()!r}")
        generator_id = match.group("generator").strip().upper()
        capacity_mw = float(match.group("capacity"))
        if not generator_id or capacity_mw <= 0:
            raise ValueError(f"Invalid generator nameplate-capacity entry: {part.strip()!r}")
        entries.append((generator_id, capacity_mw))
    return tuple(entries)


def _collect_capacity_values(attributes: pl.DataFrame) -> tuple[GeneratorValues, set[FacilityYear]]:
    # Parse every capacity field into facility-year generator groups
    generator_values: GeneratorValues = {}
    facility_years: set[FacilityYear] = set()
    for facility_id, year, serialized in attributes.select(
        "facilityId", "year", "associatedGeneratorsAndNameplateCapacity"
    ).iter_rows():
        if facility_id is None or year is None:
            raise ValueError("CAMPD capacity attributes require facility and year identifiers")
        facility_key = (int(facility_id), int(year))
        facility_years.add(facility_key)
        try:
            entries = _parse_generator_capacities(serialized)
        except ValueError:
            continue
        facility_generators = generator_values.setdefault(facility_key, {})
        for generator_id, capacity_mw in entries:
            facility_generators.setdefault(generator_id, set()).add(capacity_mw)
    return generator_values, facility_years


def _summarize_capacity(
    facility_key: FacilityYear,
    generator_values: GeneratorValues,
) -> dict[str, int | float]:
    # Exclude generators that have conflicting capacity values
    facility_generators = generator_values.get(facility_key, {})
    conflict_generators = {
        generator_id for generator_id, capacities in facility_generators.items() if len(capacities) > 1
    }
    resolved_capacity_mw = sum(
        next(iter(capacities))
        for generator_id, capacities in facility_generators.items()
        if generator_id not in conflict_generators
    )
    return {
        "facilityId": facility_key[0],
        FACILITY_ATTRIBUTE_YEAR_COL: facility_key[1],
        FACILITY_NAMEPLATE_CAPACITY_MW_COL: float(resolved_capacity_mw),
    }


def _build_capacity_attributes(attributes: pl.DataFrame) -> pl.DataFrame:
    # Resolve generators once per facility and attribute year
    generator_values, facility_years = _collect_capacity_values(attributes)
    rows: list[dict[str, int | float]] = []
    for facility_key in sorted(facility_years):
        rows.append(_summarize_capacity(facility_key, generator_values))
    return pl.DataFrame(rows, schema=CAPACITY_ATTRIBUTE_SCHEMA)


def _build_attribute_frames(records: list[dict[str, object]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Separate time-varying facility and unit fields before the hourly join
    attributes = pl.DataFrame(records, schema=ATTRIBUTE_SCHEMA, strict=False).with_columns(
        pl.col("unitId").str.strip_chars()
    )
    capacity_attributes = _build_capacity_attributes(attributes)
    facility_attributes = (
        attributes.select(
            "facilityId",
            *(pl.col(source).alias(target) for source, target in FACILITY_ATTRIBUTE_COLUMNS.items()),
        )
        .unique(subset=["facilityId", FACILITY_ATTRIBUTE_YEAR_COL], keep="first", maintain_order=True)
        .join(capacity_attributes, on=["facilityId", FACILITY_ATTRIBUTE_YEAR_COL], how="left")
    )
    unit_attributes = attributes.select(
        "facilityId",
        pl.col("year").alias(UNIT_ATTRIBUTE_YEAR_COL),
        pl.col("unitId").alias("unitIdKey"),
        *(pl.col(source).alias(target) for source, target in UNIT_ATTRIBUTE_COLUMNS.items()),
    ).unique(subset=["facilityId", UNIT_ATTRIBUTE_YEAR_COL, "unitIdKey"], keep="first", maintain_order=True)
    return facility_attributes, unit_attributes


@lru_cache(maxsize=1)
def _timezone_finder() -> TimezoneFinder:
    # Load timezone boundaries once per enrichment process
    return TimezoneFinder(in_memory=True)


def _standard_utc_offset_hours(time_zone_name: str) -> int:
    # Remove daylight saving time because Part 75 hours use local standard time
    local_time = STANDARD_OFFSET_REFERENCE.replace(tzinfo=ZoneInfo(time_zone_name))
    standard_offset = local_time.utcoffset() - local_time.dst()
    offset_hours, remainder = divmod(int(standard_offset.total_seconds()), 3_600)
    if remainder:
        raise ValueError(f"Timezone {time_zone_name} does not have a whole-hour standard UTC offset")
    return offset_hours


def _add_facility_time_zones(facility_attributes: pl.DataFrame) -> pl.DataFrame:
    # Resolve each facility coordinate before joining the large hourly table
    finder = _timezone_finder()
    time_zone_names: list[str] = []
    standard_offsets: list[int] = []
    for facility_id, latitude, longitude in facility_attributes.select("facilityId", "lat", "lon").iter_rows():
        if latitude is None or longitude is None:
            raise ValueError(f"Facility {facility_id} is missing coordinates for timezone lookup")
        time_zone_name = finder.timezone_at_land(lng=float(longitude), lat=float(latitude))
        if time_zone_name is None:
            raise ValueError(f"Facility {facility_id} does not map to a land timezone")
        time_zone_names.append(time_zone_name)
        standard_offsets.append(_standard_utc_offset_hours(time_zone_name))
    return facility_attributes.with_columns(
        pl.Series(TIME_ZONE_COL, time_zone_names, dtype=pl.String),
        pl.Series(UTC_STANDARD_OFFSET_HOURS_COL, standard_offsets, dtype=pl.Int8),
    )


def _convert_local_standard_hours_to_utc(frame: pl.LazyFrame) -> pl.LazyFrame:
    # Preserve source clock fields and expose one unambiguous UTC timestamp
    local_hour_start = pl.col("date").cast(pl.Datetime) + pl.duration(hours=pl.col("hour"))
    utc_hour_start = local_hour_start - pl.duration(hours=pl.col(UTC_STANDARD_OFFSET_HOURS_COL))
    return (
        frame.with_columns(
            pl.col("date").alias(LOCAL_STANDARD_DATE_COL),
            pl.col("hour").alias(LOCAL_STANDARD_HOUR_COL),
            utc_hour_start.dt.replace_time_zone("UTC").alias(EMISSIONS_HOUR_UTC_COL),
        )
        .with_columns(
            pl.col(EMISSIONS_HOUR_UTC_COL).dt.date().alias("date"),
            pl.col(EMISSIONS_HOUR_UTC_COL).dt.hour().cast(pl.Int8).alias("hour"),
        )
    )


def _build_prediction_year_attribute_lookups(
    prediction_years: pl.DataFrame,
    facility_attributes: pl.DataFrame,
    unit_attributes: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Resolve temporal attributes on small lookup tables before the hourly joins
    years = prediction_years.select(PREDICTION_YEAR_COL).unique().sort(PREDICTION_YEAR_COL)
    located_facilities = _add_facility_time_zones(facility_attributes)
    facility_lookup = (
        located_facilities.select("facilityId")
        .unique()
        .join(years, how="cross")
        .sort("facilityId", PREDICTION_YEAR_COL)
        .join_asof(
            located_facilities.sort("facilityId", FACILITY_ATTRIBUTE_YEAR_COL),
            left_on=PREDICTION_YEAR_COL,
            right_on=FACILITY_ATTRIBUTE_YEAR_COL,
            by="facilityId",
            strategy="backward",
            check_sortedness=False,
        )
    )
    unit_lookup = (
        unit_attributes.select("facilityId", "unitIdKey")
        .unique()
        .join(years, how="cross")
        .sort("facilityId", "unitIdKey", PREDICTION_YEAR_COL)
        .join_asof(
            unit_attributes.sort("facilityId", "unitIdKey", UNIT_ATTRIBUTE_YEAR_COL),
            left_on=PREDICTION_YEAR_COL,
            right_on=UNIT_ATTRIBUTE_YEAR_COL,
            by=["facilityId", "unitIdKey"],
            strategy="backward",
            check_sortedness=False,
        )
    )
    return facility_lookup, unit_lookup


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
        facility_attributes: Annual facility attributes and capacity summaries.
        unit_attributes: Annual unit attributes per facility and unit.

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
    prediction_years = (
        source.select(pl.col("date").cast(pl.Date, strict=False).dt.year().alias(PREDICTION_YEAR_COL))
        .drop_nulls()
        .unique()
        .collect()
    )
    facility_lookup, unit_lookup = _build_prediction_year_attribute_lookups(
        prediction_years,
        facility_attributes,
        unit_attributes,
    )
    augmented = (
        source.with_columns(
            pl.col("facilityId").cast(pl.Int64, strict=False),
            unit_id.alias("unitId"),
            unit_id.alias("unitIdKey"),
            pl.col("date").cast(pl.Date, strict=False).dt.year().alias(PREDICTION_YEAR_COL),
        )
        .join(
            facility_lookup.lazy(),
            on=["facilityId", PREDICTION_YEAR_COL],
            how="left",
        )
        .join(
            unit_lookup.lazy(),
            on=["facilityId", "unitIdKey", PREDICTION_YEAR_COL],
            how="left",
        )
        .drop("unitIdKey", PREDICTION_YEAR_COL)
        .drop_nulls(["lat", "lon", "epaRegion"])
        .pipe(_convert_local_standard_hours_to_utc)
    )

    try:
        augmented.sink_parquet(
            temporary_path,
            compression="zstd",
            statistics=True,
            row_group_size=ROW_GROUP_SIZE,
        )
        row_count = pl.scan_parquet(temporary_path).select(pl.len()).collect().item()
        if row_count < source_row_count:
            raise RuntimeError(
                f"Location enrichment dropped {source_row_count - row_count:,} of {source_row_count:,} emissions rows"
            )
        if row_count == 0:
            raise RuntimeError("No emissions rows were available for location enrichment")
        if row_count > source_row_count:
            raise RuntimeError(f"Location enrichment added {row_count - source_row_count:,} duplicate emissions rows")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count


def main() -> None:
    """Write hourly emissions enriched with prediction-date attributes."""
    require_campd_credentials()

    input_path = Path(EMISSIONS_RECORDS_PARQUET)
    source = pl.scan_parquet(input_path)
    facility_ids = (
        source
        .select(pl.col("facilityId").cast(pl.Int64, strict=False))
        .drop_nulls()
        .unique()
        .sort("facilityId")
        .collect()["facilityId"]
        .to_list()
    )
    source_years = source.select(
        pl.col("date").cast(pl.Date, strict=False).dt.year().min().alias("earliest"),
        pl.col("date").cast(pl.Date, strict=False).dt.year().max().alias("latest"),
    ).collect().row(0, named=True)
    if source_years["earliest"] is None or source_years["latest"] is None:
        raise ValueError("Raw hourly emissions contain no valid prediction dates")

    attribute_records = get_facility_attributes(
        facility_ids=facility_ids,
        latest_year=int(source_years["latest"]),
        earliest_year=int(source_years["earliest"]),
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
