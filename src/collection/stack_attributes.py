"""Collect prediction-year unit stack characteristics from EPA monitoring plans."""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import date, datetime
from math import isfinite

import polars as pl
import requests

from collection.emissions_schema import (
    ASSOCIATED_STACK_COUNT_COL,
    GROUND_ELEVATION_FT_COL,
    STACK_HEIGHT_FT_COL,
    STACK_PIPE_ID_COL,
)
from prerequisites import require_campd_credentials

CONFIGURATIONS_URL = "https://api.epa.gov/easey/monitor-plan-mgmt/configurations"
PLAN_EXPORT_URL = "https://api.epa.gov/easey/monitor-plan-mgmt/plans/export"
CONFIGURATION_BATCH_SIZE = 100
MAX_RETRIES = 3
REQUEST_INTERVAL_SECONDS = 4
INITIAL_RETRY_DELAY_SECONDS = 30
MAX_RETRY_DELAY_SECONDS = 300
RATE_LIMIT_WAIT_SECONDS = 3_600
STACK_ATTRIBUTE_YEAR_COL = "stackAttributeYear"
STACK_ATTRIBUTE_SCHEMA = {
    "facilityId": pl.Int64,
    "unitIdKey": pl.String,
    STACK_ATTRIBUTE_YEAR_COL: pl.Int64,
    STACK_PIPE_ID_COL: pl.String,
    STACK_HEIGHT_FT_COL: pl.Float64,
    GROUND_ELEVATION_FT_COL: pl.Float64,
    ASSOCIATED_STACK_COUNT_COL: pl.Int64,
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
            print(f"WARNING: EPA returned invalid data for {description}; retrying in {delay:.0f}s")
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
                STACK_PIPE_ID_COL: normalized_stack_id,
                STACK_HEIGHT_FT_COL: _number_or_none(attribute.get("stackHeight")),
                GROUND_ELEVATION_FT_COL: _number_or_none(attribute.get("groundElevation")),
            }
        )
    return candidates


def _select_unit_stack(rows: list[dict[str, object]]) -> dict[str, object]:
    # Choose the tallest stack with stable ID tie-breaking
    unique = {
        (
            str(row[STACK_PIPE_ID_COL]),
            row[STACK_HEIGHT_FT_COL],
            row[GROUND_ELEVATION_FT_COL],
        ): row
        for row in rows
    }
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            -(row[STACK_HEIGHT_FT_COL] if isinstance(row[STACK_HEIGHT_FT_COL], float) else float("-inf")),
            str(row[STACK_PIPE_ID_COL]),
        ),
    )
    selected = dict(ordered[0])
    selected[ASSOCIATED_STACK_COUNT_COL] = len({str(row[STACK_PIPE_ID_COL]) for row in unique.values()})
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
