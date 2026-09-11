"""Add prediction-date facility attributes to hourly emissions with Polars."""

import os
import re
import time
from collections.abc import Iterable
from datetime import date, datetime
from functools import lru_cache
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests
from timezonefinder import TimezoneFinder

from config import EMISSIONS_RECORDS_PARQUET, FULL_DATA_PARQUET
from prerequisites import require_campd_credentials

API_URL = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
CONFIGURATIONS_URL = "https://api.epa.gov/easey/monitor-plan-mgmt/configurations"
PLAN_EXPORT_URL = "https://api.epa.gov/easey/monitor-plan-mgmt/plans/export"
MAX_RETRIES = 3
RECORDS_PER_PAGE = 500
CONFIGURATION_BATCH_SIZE = 100
REQUEST_INTERVAL_SECONDS = 4
INITIAL_RETRY_DELAY_SECONDS = 30
MAX_RETRY_DELAY_SECONDS = 300
RATE_LIMIT_WAIT_SECONDS = 3_600
ROW_GROUP_SIZE = 250_000
STANDARD_OFFSET_REFERENCE = datetime(2025, 1, 1, 12)
FACILITY_ATTRIBUTE_YEAR_COL = "facilityAttributeYear"
UNIT_ATTRIBUTE_YEAR_COL = "unitAttributeYear"
STACK_ATTRIBUTE_YEAR_COL = "stackAttributeYear"
PREDICTION_YEAR_COL = "_predictionYear"
GENERATOR_CAPACITY_PATTERN = re.compile(
    r"\s*(?P<generator>[^(),]+?)\s*\(\s*(?P<capacity>(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*"
)
CAPACITY_ATTRIBUTE_SCHEMA = {
    "facilityId": pl.Int64,
    FACILITY_ATTRIBUTE_YEAR_COL: pl.Int64,
    "facility_nameplate_capacity_mw": pl.Float64,
}
STACK_ATTRIBUTE_SCHEMA = {
    "facilityId": pl.Int64,
    "unitIdKey": pl.String,
    STACK_ATTRIBUTE_YEAR_COL: pl.Int64,
    "stack_pipe_id": pl.String,
    "stack_height_ft": pl.Float64,
    "ground_elevation_ft": pl.Float64,
    "associated_stack_count": pl.Int64,
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


def _fetch_json(url: str, params: dict[str, object], description: str) -> dict[str, object]:
    # Fetch one EPA resource and fail after bounded retries
    headers = {"x-api-key": require_campd_credentials()}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Fetching {description} (attempt {attempt}/{MAX_RETRIES})")
            response = requests.get(url, params=params, headers=headers, timeout=120)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("unexpected monitoring-plan response")
            time.sleep(REQUEST_INTERVAL_SECONDS)
            return payload
        except requests.exceptions.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            detail = type(error).__name__ if status is None else f"{type(error).__name__} (HTTP {status})"
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"EPA request failed for {description}: {detail}") from error
            if status == 429:
                retry_after = error.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else RATE_LIMIT_WAIT_SECONDS
            else:
                delay = min(INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
            print(f"WARNING: EPA request failed for {description}: {detail}; retrying in {delay:.0f}s")
            time.sleep(delay)
        except (TypeError, ValueError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"EPA returned invalid data for {description}") from error
            delay = min(INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
            print(
                f"WARNING: EPA returned invalid data for {description}: "
                f"{type(error).__name__}; retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    raise AssertionError("retry loop exhausted without returning or raising")


def _batched(values: list[int], size: int) -> Iterable[list[int]]:
    # Yield stable API-sized chunks
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _fetch_stack_plan_ids(facility_ids: list[int]) -> list[str]:
    # Discover only plans containing unit-to-stack relationships
    plan_ids: set[str] = set()
    for batch in _batched(sorted(set(facility_ids)), CONFIGURATION_BATCH_SIZE):
        payload = _fetch_json(
            CONFIGURATIONS_URL,
            {"orisCodes": "|".join(str(facility_id) for facility_id in batch)},
            f"monitoring configurations for {len(batch)} facilities",
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise TypeError("monitoring configurations are missing items")
        for plan in items:
            if not isinstance(plan, dict) or not plan.get("unitStackConfigurationData"):
                continue
            plan_id = plan.get("id")
            if isinstance(plan_id, str) and plan_id:
                plan_ids.add(plan_id)
    print(f"Found {len(plan_ids):,} stack-bearing monitoring plans")
    return sorted(plan_ids)


def _fetch_stack_plans(facility_ids: list[int]) -> list[dict[str, object]]:
    # Export reported values only for stack-bearing plans
    plans = []
    for plan_id in _fetch_stack_plan_ids(facility_ids):
        payload = _fetch_json(
            PLAN_EXPORT_URL,
            {"planId": plan_id, "reportedValuesOnly": True},
            f"monitoring plan {plan_id}",
        )
        plans.append(payload)
    return plans


def _parse_date(value: object) -> date | None:
    # Parse EPA ISO dates while accepting null interval endpoints
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"expected an ISO date string, got {type(value).__name__}")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _is_effective(record: dict[str, object], snapshot: date) -> bool:
    # Apply inclusive EPA effective-date intervals
    begin = _parse_date(record.get("beginDate"))
    end = _parse_date(record.get("endDate"))
    return (begin is None or begin <= snapshot) and (end is None or snapshot <= end)


def _latest_effective_attribute(
    attributes: object,
    snapshot: date,
) -> dict[str, object] | None:
    # Select the latest effective physical-attribute record
    if not isinstance(attributes, list):
        return None
    applicable = [item for item in attributes if isinstance(item, dict) and _is_effective(item, snapshot)]
    if not applicable:
        return None
    return max(applicable, key=lambda item: _parse_date(item.get("beginDate")) or date.min)


def _number_or_none(value: object) -> float | None:
    # Normalize optional numeric API values
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _plan_stack_candidates(
    plan: dict[str, object],
    prediction_year: int,
) -> list[dict[str, object]]:
    # Resolve effective unit-stack relationships and physical attributes
    facility_id = plan.get("facilityId", plan.get("orisCode"))
    if facility_id is None:
        return []
    snapshot = date(prediction_year, 1, 1)
    locations = plan.get("monitoringLocationData")
    relationships = plan.get("unitStackConfigurationData")
    if not isinstance(locations, list) or not isinstance(relationships, list):
        return []
    stack_locations = {
        str(location["stackPipeId"]).strip(): location
        for location in locations
        if isinstance(location, dict) and location.get("stackPipeId") is not None
    }
    candidates = []
    for relationship in relationships:
        if not isinstance(relationship, dict) or not _is_effective(relationship, snapshot):
            continue
        unit_id = relationship.get("unitId")
        stack_id = relationship.get("stackPipeId")
        if unit_id is None or stack_id is None:
            continue
        normalized_stack_id = str(stack_id).strip()
        location = stack_locations.get(normalized_stack_id)
        if location is None:
            continue
        attribute = _latest_effective_attribute(location.get("monitoringLocationAttribData"), snapshot)
        if attribute is None:
            continue
        candidates.append(
            {
                "facilityId": int(facility_id),
                "unitIdKey": str(unit_id).strip(),
                STACK_ATTRIBUTE_YEAR_COL: prediction_year,
                "stack_pipe_id": normalized_stack_id,
                "stack_height_ft": _number_or_none(attribute.get("stackHeight")),
                "ground_elevation_ft": _number_or_none(attribute.get("groundElevation")),
            }
        )
    return candidates


def _select_unit_stack(rows: list[dict[str, object]]) -> dict[str, object]:
    # Choose the tallest stack with stable ID tie-breaking
    unique = {
        (
            str(row["stack_pipe_id"]),
            row["stack_height_ft"],
            row["ground_elevation_ft"],
        ): row
        for row in rows
    }
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            -(row["stack_height_ft"] if isinstance(row["stack_height_ft"], float) else float("-inf")),
            str(row["stack_pipe_id"]),
        ),
    )
    selected = dict(ordered[0])
    selected["associated_stack_count"] = len({str(row["stack_pipe_id"]) for row in unique.values()})
    return selected


def build_unit_stack_attributes(
    plans: list[dict[str, object]],
    prediction_years: list[int],
) -> pl.DataFrame:
    """Build one prediction-safe stack characteristic row per unit-year.

    Args:
        plans: Exported EPA monitoring plans.
        prediction_years: Prediction years required by the emissions data.

    Returns:
        Unit-year rows using the tallest effective associated stack.
    """
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = {}
    for prediction_year in sorted(set(prediction_years)):
        for plan in plans:
            for candidate in _plan_stack_candidates(plan, prediction_year):
                key = (
                    int(candidate["facilityId"]),
                    str(candidate["unitIdKey"]),
                    prediction_year,
                )
                grouped.setdefault(key, []).append(candidate)
    rows = [_select_unit_stack(grouped[key]) for key in sorted(grouped)]
    return pl.DataFrame(rows, schema=STACK_ATTRIBUTE_SCHEMA)


def get_unit_stack_attributes(
    facility_ids: list[int],
    prediction_years: list[int],
) -> pl.DataFrame:
    """Fetch EPA monitoring plans and build annual unit stack characteristics.

    Args:
        facility_ids: EPA facility identifiers required by the emissions data.
        prediction_years: Prediction years required by the emissions data.

    Returns:
        Unit-year stack heights and paired ground elevations in feet.
    """
    return build_unit_stack_attributes(_fetch_stack_plans(facility_ids), prediction_years)


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
    if not isinstance(value, str):
        return ()
    if not value.strip():
        return ()
    entries: list[tuple[str, float]] = []
    for part in value.split(","):
        match = GENERATOR_CAPACITY_PATTERN.fullmatch(part)
        if match is None:
            continue
        generator_id = match.group("generator").strip().upper()
        capacity_mw = float(match.group("capacity"))
        if not generator_id or capacity_mw <= 0:
            continue
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
            continue
        facility_key = (int(facility_id), int(year))
        facility_years.add(facility_key)
        entries = _parse_generator_capacities(serialized)
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
        "facility_nameplate_capacity_mw": float(resolved_capacity_mw),
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
        pl.Series("time_zone", time_zone_names, dtype=pl.String),
        pl.Series("utc_standard_offset_hours", standard_offsets, dtype=pl.Int8),
    )


def _convert_local_standard_hours_to_utc(frame: pl.LazyFrame) -> pl.LazyFrame:
    # Preserve source clock fields and expose one unambiguous UTC timestamp
    local_hour_start = pl.col("date").cast(pl.Datetime) + pl.duration(hours=pl.col("hour"))
    utc_hour_start = local_hour_start - pl.duration(hours=pl.col("utc_standard_offset_hours"))
    return frame.with_columns(
        pl.col("date").alias("local_standard_date"),
        pl.col("hour").alias("local_standard_hour"),
        utc_hour_start.dt.replace_time_zone("UTC").alias("emissions_hour_utc"),
    ).with_columns(
        pl.col("emissions_hour_utc").dt.date().alias("date"),
        pl.col("emissions_hour_utc").dt.hour().cast(pl.Int8).alias("hour"),
    )


def _build_prediction_year_attribute_lookups(
    prediction_years: pl.DataFrame,
    facility_attributes: pl.DataFrame,
    unit_attributes: pl.DataFrame,
    stack_attributes: pl.DataFrame | None,
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
    if stack_attributes is not None:
        unit_lookup = unit_lookup.join(
            stack_attributes,
            left_on=["facilityId", "unitIdKey", PREDICTION_YEAR_COL],
            right_on=["facilityId", "unitIdKey", STACK_ATTRIBUTE_YEAR_COL],
            how="left",
        )
    return facility_lookup, unit_lookup


def write_augmented_parquet(
    input_path: Path,
    output_path: Path,
    facility_attributes: pl.DataFrame,
    unit_attributes: pl.DataFrame,
    stack_attributes: pl.DataFrame | None = None,
) -> int:
    """Stream enriched emissions into one atomic Zstd Parquet file.

    Args:
        input_path: Raw hourly emissions Parquet file.
        output_path: Final compressed Parquet file.
        facility_attributes: Annual facility attributes and capacity summaries.
        unit_attributes: Annual unit attributes per facility and unit.
        stack_attributes: Annual unit stack characteristics from monitoring plans.

    Returns:
        Number of enriched rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)

    source = pl.scan_parquet(input_path)
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
        stack_attributes,
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
        source.select(pl.col("facilityId").cast(pl.Int64, strict=False))
        .drop_nulls()
        .unique()
        .sort("facilityId")
        .collect()["facilityId"]
        .to_list()
    )
    source_years = (
        source.select(
            pl.col("date").cast(pl.Date, strict=False).dt.year().min().alias("earliest"),
            pl.col("date").cast(pl.Date, strict=False).dt.year().max().alias("latest"),
        )
        .collect()
        .row(0, named=True)
    )
    prediction_years = list(range(int(source_years["earliest"]), int(source_years["latest"]) + 1))
    attribute_records = get_facility_attributes(
        facility_ids=facility_ids,
        latest_year=prediction_years[-1],
        earliest_year=prediction_years[0],
    )
    stack_attributes = get_unit_stack_attributes(facility_ids, prediction_years)

    facility_attributes, unit_attributes = _build_attribute_frames(attribute_records)
    row_count = write_augmented_parquet(
        input_path=input_path,
        output_path=Path(FULL_DATA_PARQUET),
        facility_attributes=facility_attributes,
        unit_attributes=unit_attributes,
        stack_attributes=stack_attributes,
    )
    print(f"Wrote {row_count:,} rows to {FULL_DATA_PARQUET}")


if __name__ == "__main__":
    main()
