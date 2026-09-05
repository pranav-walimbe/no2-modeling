"""Download CAMPD local-standard-time hourly records into compressed Parquet."""

import calendar
import json
import os
import random
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

from config import EMISSIONS_END_DATE, EMISSIONS_RECORDS_PARQUET, EMISSIONS_START_DATE
from prerequisites import require_campd_credentials

STATE_CODES = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

HOURLY_EMISSIONS_COLUMNS = [
    "stateCode",
    "facilityName",
    "facilityId",
    "unitId",
    "date",
    "hour",
    "opTime",
    "grossLoad",
    "noxMass",
    "noxMassMeasureFlg",
    "noxRate",
    "noxRateMeasureFlg",
    "so2Mass",
    "so2MassMeasureFlg",
    "so2Rate",
    "so2RateMeasureFlg",
    "co2Mass",
    "co2MassMeasureFlg",
    "co2Rate",
    "co2RateMeasureFlg",
    "heatInput",
    "heatInputMeasureFlg",
    "primaryFuelInfo",
    "unitType",
]
HOURLY_STRING_COLUMNS = [
    "stateCode",
    "facilityName",
    "unitId",
    "noxMassMeasureFlg",
    "noxRateMeasureFlg",
    "so2MassMeasureFlg",
    "so2RateMeasureFlg",
    "co2MassMeasureFlg",
    "co2RateMeasureFlg",
    "heatInputMeasureFlg",
    "primaryFuelInfo",
    "unitType",
]
HOURLY_FLOAT_COLUMNS = [
    "opTime",
    "grossLoad",
    "noxMass",
    "noxRate",
    "so2Mass",
    "so2Rate",
    "co2Mass",
    "co2Rate",
    "heatInput",
]
REQUIRED_RESPONSE_COLUMNS = {"facilityId", "unitId", "date", "hour", "opTime"}

API_URL = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly"
REQUEST_INTERVAL_SECONDS = 5
MAX_RETRIES = 8
INITIAL_BACKOFF_SECONDS = 30
MAX_BACKOFF_SECONDS = 300
RETRYABLE_STATUS_CODES = {408, 429}


def latest_completed_quarter_end(today: date) -> date:
    """Return the last day of the latest completed calendar quarter.

    Args:
        today: Date used to determine the current calendar quarter.

    Returns:
        Final day of the preceding calendar quarter.
    """
    quarter_start_month = 3 * ((today.month - 1) // 3) + 1
    current_quarter_start = date(today.year, quarter_start_month, 1)
    return current_quarter_start - timedelta(days=1)


def month_ranges(start: date, end: date) -> list[tuple[str, str]]:
    """Build inclusive monthly request ranges.

    Args:
        start: First date to request.
        end: Last date to request.

    Returns:
        Inclusive begin and end date strings for each request month.
    """
    ranges: list[tuple[str, str]] = []
    current = start.replace(day=1)
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = min(current.replace(day=last_day), end)
        month_start = max(current, start)
        ranges.append((month_start.isoformat(), month_end.isoformat()))
        current = month_end + timedelta(days=1)
    return ranges


def _response_detail(response: requests.Response) -> str:
    # Extract a bounded API message without including the credential-bearing URL
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    return json.dumps(payload)[:500]


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    # Prefer CAMPD's delay when provided
    retry_after = response.headers.get("Retry-After") if response is not None else None
    exponential_delay = min(INITIAL_BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
    delay = float(retry_after) if retry_after and retry_after.isdigit() else exponential_delay
    return delay + random.uniform(0, 5)


def fetch_chunk(state: str, begin: str, end: str) -> list[dict[str, object]]:
    """Fetch one state-month without filtering non-operating records.

    Args:
        state: Two-letter state code.
        begin: Inclusive request start date.
        end: Inclusive request end date.

    Returns:
        CAMPD hourly records for the requested state-month.
    """
    params = {
        "api_key": require_campd_credentials(),
        "beginDate": begin,
        "endDate": end,
        "stateCode": state,
        "operatingHoursOnly": "false",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Fetching {state} {begin} to {end} (attempt {attempt}/{MAX_RETRIES})")
        try:
            response = requests.get(API_URL, params=params, timeout=180)
        except requests.exceptions.RequestException as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"CAMPD request failed for {state} {begin} to {end}") from error
            delay = _retry_delay(None, attempt)
            print(f"WARNING: CAMPD network error; retrying in {delay:.1f} seconds")
            time.sleep(delay)
            continue

        if response.status_code >= 400:
            detail = _response_detail(response)
            message = f"CAMPD rejected {state} {begin} to {end} with HTTP {response.status_code}: {detail}"
            is_retryable = response.status_code in RETRYABLE_STATUS_CODES or response.status_code >= 500
            if not is_retryable or attempt == MAX_RETRIES:
                raise RuntimeError(message)
            delay = _retry_delay(response, attempt)
            print(f"WARNING: {message}; retrying in {delay:.1f} seconds")
            time.sleep(delay)
            continue

        try:
            records = response.json()
        except ValueError as error:
            raise RuntimeError(f"CAMPD returned invalid JSON for {state} {begin} to {end}") from error
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise RuntimeError(f"CAMPD returned an unexpected response for {state} {begin} to {end}")
        return records

    raise AssertionError("Retry loop exited unexpectedly")


def normalize_hourly_records(records: list[dict[str, object]]) -> pl.DataFrame:
    """Normalize one API response to the stable source-time Parquet schema.

    Args:
        records: Hourly records returned by the CAMPD API.

    Returns:
        Records with stable column order and data types. The ``date`` and
        ``hour`` fields remain in CAMPD local standard time until facility
        coordinates are attached during location enrichment.
    """
    frame = pl.DataFrame(records, infer_schema_length=None)
    missing_required = REQUIRED_RESPONSE_COLUMNS.difference(frame.columns)
    if missing_required:
        missing = ", ".join(sorted(missing_required))
        raise ValueError(f"CAMPD response is missing required columns: {missing}")

    missing_optional = set(HOURLY_EMISSIONS_COLUMNS).difference(frame.columns)
    if missing_optional:
        frame = frame.with_columns(pl.lit(None).alias(column) for column in sorted(missing_optional))
    return frame.select(HOURLY_EMISSIONS_COLUMNS).with_columns(
        pl.col("facilityId").cast(pl.Int64, strict=False),
        pl.col("hour").cast(pl.Int8, strict=False),
        pl.col("date").cast(pl.String).str.to_date(strict=False),
        *(pl.col(column).cast(pl.String, strict=False) for column in HOURLY_STRING_COLUMNS),
        *(pl.col(column).cast(pl.Float64, strict=False) for column in HOURLY_FLOAT_COLUMNS),
    )


def main() -> None:
    """Download the configured period and atomically replace raw Parquet."""
    require_campd_credentials()

    output_path = Path(EMISSIONS_RECORDS_PARQUET)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.unlink(missing_ok=True)

    row_count = 0
    latest_date: date | None = None
    available_end = latest_completed_quarter_end(date.today())
    effective_end = min(EMISSIONS_END_DATE, available_end)
    if EMISSIONS_START_DATE > effective_end:
        raise ValueError(f"Emissions start date {EMISSIONS_START_DATE} is after available end date {effective_end}")
    if effective_end < EMISSIONS_END_DATE:
        print(f"Capping CAMPD end date at latest completed quarter: {effective_end}")

    try:
        with tempfile.TemporaryDirectory(prefix=".hourly-parts-", dir=output_path.parent) as directory:
            parts_directory = Path(directory)
            part_number = 0
            for state in STATE_CODES:
                for begin, end in month_ranges(EMISSIONS_START_DATE, effective_end):
                    rows = fetch_chunk(state, begin, end)
                    if not rows:
                        time.sleep(REQUEST_INTERVAL_SECONDS)
                        continue

                    frame = normalize_hourly_records(rows)
                    part_path = parts_directory / f"part-{part_number:05d}.parquet"
                    frame.write_parquet(part_path, compression="zstd", statistics=True)
                    part_number += 1
                    row_count += frame.height
                    chunk_latest_date = frame["date"].drop_nulls().max()
                    if chunk_latest_date is not None:
                        latest_date = max(latest_date or chunk_latest_date, chunk_latest_date)
                    print(f"Wrote {frame.height:,} rows; cumulative rows: {row_count:,}")
                    time.sleep(REQUEST_INTERVAL_SECONDS)

            if part_number == 0:
                raise RuntimeError("CAMPD returned no hourly records")

            pl.scan_parquet(parts_directory / "*.parquet").sink_parquet(
                temporary_path,
                compression="zstd",
                statistics=True,
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Replaced {output_path} with {row_count:,} rows through {latest_date}")


if __name__ == "__main__":
    main()
