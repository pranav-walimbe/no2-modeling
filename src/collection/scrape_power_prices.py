"""Download normalized hourly regional power prices to partitioned Parquet."""

import argparse
import json
import os
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TypeVar
from urllib.error import HTTPError as UrlHTTPError

import pandas as pd
import requests
from pyarrow import ArrowException

from collection.power_price_sources import DEFAULT_ISOS, SOURCE_FACTORIES, MarketName, create_source
from config import POWER_PRICE_BASE_DIR, POWER_PRICE_METADATA_DIR, POWER_PRICE_START_DATE, POWER_PRICE_TEMP_DIR

MARKETS: tuple[MarketName, ...] = ("day_ahead", "real_time")
STANDARD_COLUMNS = [
    "iso",
    "market",
    "location_id",
    "location_name",
    "location_type",
    "interval_start_utc",
    "interval_end_utc",
    "price_usd_mwh",
    "energy",
    "congestion",
    "loss",
    "settlement_status",
    "source_interval_minutes",
    "interval_count",
    "retrieved_at_utc",
    "source",
]
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 30
MAX_BACKOFF_SECONDS = 300
REQUEST_INTERVAL_SECONDS = 2
RETRYABLE_STATUS_CODES = {408, 429}
T = TypeVar("T")


@dataclass(frozen=True)
class MonthWindow:
    """UTC bounds and partition labels for one requested month."""

    year: int
    month: int
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp


def latest_completed_quarter_end(today: date) -> date:
    """Return the latest date expected to overlap completed CAMPD data.

    Args:
        today: Date used to determine the active calendar quarter.

    Returns:
        Last date of the preceding calendar quarter.
    """
    quarter_start_month = 3 * ((today.month - 1) // 3) + 1
    return date(today.year, quarter_start_month, 1) - timedelta(days=1)


def month_windows(start: date, end: date) -> list[MonthWindow]:
    """Build UTC monthly partitions intersecting an inclusive date range.

    Args:
        start: Inclusive first UTC date.
        end: Inclusive final UTC date.

    Returns:
        Monthly half-open UTC windows.
    """
    if start > end:
        raise ValueError(f"Start date {start} is after end date {end}")
    requested_start = pd.Timestamp(start, tz="UTC")
    requested_end = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    cursor = requested_start.replace(day=1)
    windows: list[MonthWindow] = []
    while cursor < requested_end:
        next_month = cursor + pd.DateOffset(months=1)
        windows.append(
            MonthWindow(
                year=cursor.year,
                month=cursor.month,
                start_utc=max(cursor, requested_start),
                end_utc=min(next_month, requested_end),
            ),
        )
        cursor = next_month
    return windows


def local_query_bounds(window: MonthWindow, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Build buffered local dates that fully cover a UTC partition.

    Args:
        window: Target UTC month partition.
        timezone: GridStatus timezone for the source ISO.

    Returns:
        Inclusive local start and exclusive local end query timestamps.
    """
    local_start = window.start_utc.tz_convert(timezone).normalize() - pd.DateOffset(days=1)
    local_end = window.end_utc.tz_convert(timezone).normalize() + pd.DateOffset(days=2)
    return local_start, local_end


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    # Resolve schema differences between GridStatus ISO adapters
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
    choices = ", ".join(candidates)
    raise ValueError(f"GridStatus response is missing all candidate columns: {choices}")


def _canonical_location_type(value: object) -> str:
    # Collapse market-specific labels into the retained location types
    normalized = str(value).strip().casefold().replace("_", " ")
    if normalized in {"hub", "trading hub"}:
        return "hub"
    if normalized in {"zone", "load zone", "loadzone", "dlap"}:
        return "load_zone"
    if normalized == "settlement location":
        return "settlement_location"
    raise ValueError(f"Unsupported regional location type: {value!r}")


def normalize_prices(
    frame: pd.DataFrame,
    iso: str,
    market: MarketName,
    settlement_status: str,
    retrieved_at: pd.Timestamp,
) -> pd.DataFrame:
    """Normalize a GridStatus price response before hourly aggregation.

    Args:
        frame: GridStatus response at its native interval.
        iso: ISO identifier.
        market: Normalized market name.
        settlement_status: Publication or settlement status of the source.
        retrieved_at: UTC retrieval timestamp.

    Returns:
        Native-interval prices with a common schema.
    """
    if frame.empty:
        raise ValueError(f"{iso} returned no {market} regional price rows")
    location_id = _first_column(frame, ("Location Id", "Location"))
    location_name = _first_column(frame, ("Location Name", "Location"))
    price = _first_column(frame, ("LMP", "SPP"))
    normalized = pd.DataFrame(
        {
            "iso": iso,
            "market": market,
            "location_id": location_id.astype("string"),
            "location_name": location_name.astype("string"),
            "location_type": frame["Location Type"].astype("string").map(_canonical_location_type),
            "interval_start_utc": pd.to_datetime(frame["Interval Start"], utc=True),
            "interval_end_utc": pd.to_datetime(frame["Interval End"], utc=True),
            "price_usd_mwh": pd.to_numeric(price, errors="coerce"),
            "settlement_status": settlement_status,
            "retrieved_at_utc": retrieved_at,
            "source": f"gridstatus:{iso.lower()}",
        },
    )
    for output_column, source_column in (
        ("energy", "Energy"),
        ("congestion", "Congestion"),
        ("loss", "Loss"),
    ):
        normalized[output_column] = (
            pd.to_numeric(frame[source_column], errors="coerce") if source_column in frame else pd.NA
        )
    if normalized["price_usd_mwh"].isna().any():
        raise ValueError(f"{iso} {market} response contains missing or nonnumeric prices")
    duplicate_key = ["location_id", "interval_start_utc"]
    if normalized.duplicated(duplicate_key).any():
        raise ValueError(f"{iso} {market} response contains duplicate location intervals")
    return normalized


def aggregate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate duration-weighted hourly prices in UTC.

    Args:
        frame: Normalized prices at their native source interval.

    Returns:
        One complete hourly record per market location.
    """
    data = frame.copy()
    data["duration_minutes"] = (data["interval_end_utc"] - data["interval_start_utc"]).dt.total_seconds() / 60
    if (data["duration_minutes"] <= 0).any():
        raise ValueError("Price response contains a nonpositive interval duration")
    data["hour_utc"] = data["interval_start_utc"].dt.floor("h")
    group_columns = ["iso", "market", "location_id", "location_name", "location_type", "hour_utc"]
    grouped = data.groupby(group_columns, sort=True, observed=True)
    hourly = grouped.agg(
        coverage_minutes=("duration_minutes", "sum"),
        source_interval_minutes=("duration_minutes", "median"),
        interval_count=("duration_minutes", "size"),
        settlement_status=("settlement_status", "first"),
        retrieved_at_utc=("retrieved_at_utc", "max"),
        source=("source", "first"),
    ).reset_index()
    for column in ("price_usd_mwh", "energy", "congestion", "loss"):
        valid_duration = data["duration_minutes"].where(data[column].notna())
        numerator = (
            (data[column] * valid_duration)
            .groupby(
                [data[group_column] for group_column in group_columns],
                observed=True,
            )
            .sum(min_count=1)
        )
        denominator = valid_duration.groupby(
            [data[group_column] for group_column in group_columns],
            observed=True,
        ).sum(min_count=1)
        hourly[column] = (numerator / denominator).to_numpy()
    incomplete = ~hourly["coverage_minutes"].round(6).eq(60)
    if incomplete.any():
        count = int(incomplete.sum())
        raise ValueError(f"Price response contains {count:,} incomplete hourly location records")
    hourly = hourly.rename(columns={"hour_utc": "interval_start_utc"})
    hourly["interval_end_utc"] = hourly["interval_start_utc"] + pd.Timedelta(hours=1)
    return hourly[STANDARD_COLUMNS]


def _status_code(error: BaseException) -> int | None:
    # Extract HTTP status without including a potentially credential-bearing URL
    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
        return error.response.status_code
    if isinstance(error, UrlHTTPError):
        return error.code
    return None


def call_with_retries(operation: Callable[[], T], description: str) -> T:
    """Run one source operation with bounded transient-error retries.

    Args:
        operation: Zero-argument source call.
        description: Credential-free operation description for logs.

    Returns:
        Result returned by the operation.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except (requests.exceptions.RequestException, UrlHTTPError) as error:
            status = _status_code(error)
            retryable = status is None or status in RETRYABLE_STATUS_CODES or status >= 500
            if not retryable or attempt == MAX_RETRIES:
                status_text = f"HTTP {status}" if status is not None else "network error"
                raise RuntimeError(f"{description} failed with {status_text}") from None
            exponential = min(INITIAL_BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
            delay = exponential + random.uniform(0, 5)
            print(f"WARNING: {description} failed transiently; retrying in {delay:.1f} seconds")
            time.sleep(delay)
    raise AssertionError("Retry loop exited unexpectedly")


def _settlement_status(iso: str, market: MarketName) -> str:
    # Describe the specific historical series selected by each adapter
    if market == "day_ahead":
        return "published"
    return {
        "ISONE": "final",
        "MISO": "final",
        "PJM": "verified",
    }.get(iso, "published")


def partition_path(base_dir: Path, iso: str, window: MonthWindow) -> Path:
    """Return the provider directory path for a monthly Parquet file.

    Args:
        base_dir: Root power-price directory.
        iso: ISO identifier.
        window: Target month partition.

    Returns:
        Final Parquet path containing both markets for the month.
    """
    return base_dir / iso / f"{window.year:04d}-{window.month:02d}.parquet"


def write_partition(frame: pd.DataFrame, output_path: Path, temporary_dir: Path) -> None:
    """Validate and atomically write an hourly Parquet partition.

    Args:
        frame: Complete normalized hourly partition.
        output_path: Final provider and month Parquet path.
        temporary_dir: Scratch directory for atomic writes.
    """
    if frame.empty:
        raise ValueError(f"Refusing to write empty partition {output_path}")
    unique_key = ["iso", "market", "location_id", "interval_start_utc"]
    if frame.duplicated(unique_key).any():
        raise ValueError(f"Refusing to write duplicate rows to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / f"prices-{uuid.uuid4().hex}.parquet.part"
    frame.to_parquet(temporary_path, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary_path, output_path)


def _partition_covers_window(output_path: Path, iso: str, window: MonthWindow) -> bool:
    # Inspect stored keys before deciding whether a source request is needed
    if not output_path.exists():
        return False
    try:
        stored = pd.read_parquet(
            output_path,
            columns=["iso", "market", "location_id", "interval_start_utc", "source"],
        )
    except (ArrowException, KeyError, OSError, ValueError):
        return False
    if stored.empty:
        return False
    if set(stored["iso"].dropna().astype(str)) != {iso}:
        return False
    if set(stored["source"].dropna().astype(str)) != {f"gridstatus:{iso.lower()}"}:
        return False
    stored["interval_start_utc"] = pd.to_datetime(stored["interval_start_utc"], utc=True, errors="coerce")
    if stored["interval_start_utc"].isna().any():
        return False
    key = ["market", "location_id", "interval_start_utc"]
    if stored.duplicated(key).any():
        return False

    expected_hours = pd.date_range(
        window.start_utc,
        window.end_utc,
        freq="h",
        inclusive="left",
    )
    for market in MARKETS:
        market_rows = stored.loc[stored["market"] == market]
        if market_rows.empty:
            return False
        for _, location_rows in market_rows.groupby("location_id", observed=True):
            observed_hours = pd.DatetimeIndex(location_rows["interval_start_utc"].unique())
            if not expected_hours.difference(observed_hours).empty:
                return False
    return True


def _update_locations(frame: pd.DataFrame, iso: str, metadata_dir: Path, temporary_dir: Path) -> None:
    # Each ISO task owns its metadata file and cannot conflict with other array tasks
    output_path = metadata_dir / "locations" / f"iso={iso}" / "locations.parquet"
    locations = frame[["iso", "location_id", "location_name", "location_type"]].drop_duplicates()
    if output_path.exists():
        locations = pd.concat([pd.read_parquet(output_path), locations], ignore_index=True).drop_duplicates()
    write_path = temporary_dir / f"locations-{iso}-{os.getpid()}.parquet.part"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    locations.sort_values(["location_type", "location_name"]).to_parquet(
        write_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    os.replace(write_path, output_path)


def _update_manifest(
    frame: pd.DataFrame,
    iso: str,
    window: MonthWindow,
    metadata_dir: Path,
    temporary_dir: Path,
) -> None:
    # Persist completion metadata separately for each ISO array task
    output_path = metadata_dir / "manifests" / f"{iso}.json"
    manifest = json.loads(output_path.read_text()) if output_path.exists() else {"iso": iso, "partitions": {}}
    partition_key = f"{window.year:04d}-{window.month:02d}"
    manifest["partitions"][partition_key] = {
        "rows": len(frame),
        "first_interval_utc": frame["interval_start_utc"].min().isoformat(),
        "last_interval_utc": frame["interval_start_utc"].max().isoformat(),
        "retrieved_at_utc": frame["retrieved_at_utc"].max().isoformat(),
        "markets": {market: int(len(market_frame)) for market, market_frame in frame.groupby("market", observed=True)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / f"manifest-{iso}-{os.getpid()}.json.part"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, output_path)


def scrape_iso(
    iso: str,
    start: date,
    end: date,
    base_dir: Path,
    metadata_dir: Path,
    temporary_dir: Path,
    overwrite: bool = False,
) -> None:
    """Download monthly DA and RT files for one ISO.

    Args:
        iso: ISO identifier.
        start: Inclusive first UTC date.
        end: Inclusive final UTC date.
        base_dir: Root output directory with one subdirectory per ISO.
        metadata_dir: Root output directory for location and manifest metadata.
        temporary_dir: Scratch directory for atomic writes.
        overwrite: Replace existing complete monthly partitions.
    """
    source = None
    for window in month_windows(start, end):
        output_path = partition_path(base_dir, iso, window)
        if not overwrite and _partition_covers_window(output_path, iso, window):
            print(f"Skipping complete partition {output_path}")
            continue
        if output_path.exists() and not overwrite:
            print(f"Refetching incomplete partition {output_path}")
        if source is None:
            source = create_source(iso)
        local_start, local_end = local_query_bounds(window, source.timezone)
        monthly_frames: list[pd.DataFrame] = []
        for market in MARKETS:
            description = f"{iso} {market} {window.year:04d}-{window.month:02d}"
            print(f"Fetching {description}")
            raw = call_with_retries(
                lambda: source.fetch(market, local_start, local_end),
                description,
            )
            retrieved_at = pd.Timestamp.now(tz="UTC")
            normalized = normalize_prices(
                raw,
                iso=iso,
                market=market,
                settlement_status=_settlement_status(iso, market),
                retrieved_at=retrieved_at,
            )
            hourly = aggregate_hourly(normalized)
            hourly = hourly.loc[
                (hourly["interval_start_utc"] >= window.start_utc) & (hourly["interval_start_utc"] < window.end_utc)
            ].reset_index(drop=True)
            monthly_frames.append(hourly)
            time.sleep(REQUEST_INTERVAL_SECONDS)
        monthly = pd.concat(monthly_frames, ignore_index=True).sort_values(
            ["market", "location_id", "interval_start_utc"],
        )
        write_partition(monthly, output_path, temporary_dir)
        _update_locations(monthly, iso, metadata_dir, temporary_dir)
        _update_manifest(monthly, iso, window, metadata_dir, temporary_dir)
        print(f"Wrote {len(monthly):,} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the hourly price backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", required=True, choices=[*SOURCE_FACTORIES, "all"])
    parser.add_argument("--start", type=date.fromisoformat, default=POWER_PRICE_START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=latest_completed_quarter_end(date.today()))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested ISO backfill with resumable monthly partitions."""
    args = parse_args()
    isos = list(DEFAULT_ISOS) if args.iso == "all" else [args.iso]
    for iso in isos:
        scrape_iso(
            iso=iso,
            start=args.start,
            end=args.end,
            base_dir=Path(POWER_PRICE_BASE_DIR),
            metadata_dir=Path(POWER_PRICE_METADATA_DIR),
            temporary_dir=Path(POWER_PRICE_TEMP_DIR),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
