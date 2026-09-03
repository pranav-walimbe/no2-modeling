"""Download selected hourly HRRR analysis fields from NOAA's public archive."""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from datetime import time as datetime_time
from pathlib import Path
from typing import Iterator

import requests

from config import HRRR_DIR, HRRR_END_DATE, HRRR_MAX_WORKERS, HRRR_START_DATE

ARCHIVE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
PRODUCT = "wrfsfcf00"
MAX_RETRIES = 5
INITIAL_RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 120
COPY_CHUNK_SIZE = 1024 * 1024

FIELD_PATTERNS = {
    "temperature_2m": ":TMP:2 m above ground:",
    "wind_u_10m": ":UGRD:10 m above ground:",
    "wind_v_10m": ":VGRD:10 m above ground:",
    "boundary_layer_height": ":HPBL:surface:",
}


@dataclass(frozen=True)
class GribByteRange:
    """Byte range for one field in a remote GRIB2 file."""

    field: str
    start: int
    end: int | None


def hourly_times(start: date, end: date) -> Iterator[datetime]:
    """Yield UTC hours for an inclusive date range.

    Args:
        start: First UTC calendar date.
        end: Last UTC calendar date.

    Yields:
        Consecutive timezone-aware UTC hours.
    """
    if end < start:
        raise ValueError(f"HRRR end date {end} precedes start date {start}")

    current = datetime.combine(start, datetime_time.min, tzinfo=timezone.utc)
    final = datetime.combine(end, datetime_time(hour=23), tzinfo=timezone.utc)
    while current <= final:
        yield current
        current += timedelta(hours=1)


def source_urls(valid_time: datetime) -> tuple[str, str]:
    """Build the HRRR surface-analysis and index URLs for one UTC hour.

    Args:
        valid_time: UTC analysis time.

    Returns:
        GRIB2 URL followed by its text-index URL.
    """
    day = valid_time.strftime("%Y%m%d")
    cycle = valid_time.strftime("%H")
    grib_url = f"{ARCHIVE_URL}/hrrr.{day}/conus/hrrr.t{cycle}z.{PRODUCT}.grib2"
    return grib_url, f"{grib_url}.idx"


def output_path(root: Path, valid_time: datetime) -> Path:
    """Return the repository-standard path for one hourly HRRR subset.

    Args:
        root: HRRR storage root.
        valid_time: UTC analysis time.

    Returns:
        Final GRIB2 subset path.
    """
    filename = f"hrrr_{valid_time:%Y%m%d_%H}z_{PRODUCT}_wind-temp-blh.grib2"
    return root / "raw" / f"{valid_time:%Y}" / f"{valid_time:%m}" / f"{valid_time:%d}" / filename


def parse_index(index_text: str) -> list[GribByteRange]:
    """Locate required fields in a NOAA GRIB2 text index.

    Args:
        index_text: Colon-delimited contents of a GRIB2 index.

    Returns:
        Ordered byte ranges for the configured fields.
    """
    indexed_lines: list[tuple[int, str]] = []
    for line in index_text.splitlines():
        columns = line.split(":", maxsplit=2)
        if len(columns) < 3:
            continue
        try:
            indexed_lines.append((int(columns[1]), line))
        except ValueError:
            continue

    matches: dict[str, GribByteRange] = {}
    for position, (start, line) in enumerate(indexed_lines):
        next_start = indexed_lines[position + 1][0] if position + 1 < len(indexed_lines) else None
        for field, pattern in FIELD_PATTERNS.items():
            if pattern not in line:
                continue
            if field in matches:
                raise ValueError(f"HRRR index contains duplicate {field} fields")
            matches[field] = GribByteRange(
                field=field,
                start=start,
                end=next_start - 1 if next_start is not None else None,
            )

    missing_fields = set(FIELD_PATTERNS).difference(matches)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"HRRR index is missing required fields: {missing}")
    return [matches[field] for field in FIELD_PATTERNS]


def _request(url: str, headers: dict[str, str] | None = None, stream: bool = False) -> requests.Response:
    # Retry transient archive and network failures
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                stream=stream,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 404:
                raise FileNotFoundError(f"HRRR archive object not found: {url}")
            response.raise_for_status()
            return response
        except FileNotFoundError:
            raise
        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"HRRR request failed after {MAX_RETRIES} attempts: {url}") from error
            delay = INITIAL_RETRY_DELAY_SECONDS * 2 ** (attempt - 1)
            print(f"WARNING: HRRR request failed on attempt {attempt}/{MAX_RETRIES}; retrying in {delay}s")
            time.sleep(delay)

    raise AssertionError("Retry loop exited unexpectedly")


def download_hour(valid_time: datetime, root: Path, overwrite: bool = False) -> tuple[Path, bool]:
    """Download four selected HRRR fields into one atomic GRIB2 file.

    Args:
        valid_time: UTC analysis time.
        root: HRRR storage root.
        overwrite: Replace an existing complete file.

    Returns:
        Output path and whether it was downloaded.
    """
    destination = output_path(root, valid_time)
    if destination.exists() and not overwrite:
        return destination, False

    grib_url, index_url = source_urls(valid_time)
    with _request(index_url) as index_response:
        byte_ranges = parse_index(index_response.text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)

    try:
        with temporary_path.open("wb") as output:
            for byte_range in byte_ranges:
                end = "" if byte_range.end is None else str(byte_range.end)
                with _request(
                    grib_url,
                    headers={"Range": f"bytes={byte_range.start}-{end}"},
                    stream=True,
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"HRRR archive ignored byte range for {byte_range.field}")
                    start_position = output.tell()
                    shutil.copyfileobj(response.raw, output, length=COPY_CHUNK_SIZE)
                    if byte_range.end is not None:
                        expected_bytes = byte_range.end - byte_range.start + 1
                        downloaded_bytes = output.tell() - start_position
                        if downloaded_bytes != expected_bytes:
                            raise RuntimeError(
                                f"HRRR field {byte_range.field} has {downloaded_bytes:,} bytes; "
                                f"expected {expected_bytes:,}"
                            )

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(f"HRRR subset is empty for {valid_time:%Y-%m-%d %H:%MZ}")
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination, True


def download_date(day: date, root: Path, workers: int, overwrite: bool = False) -> tuple[int, int]:
    """Download all HRRR analysis hours for one UTC date.

    Args:
        day: UTC date to download.
        root: HRRR storage root.
        workers: Maximum concurrent hourly downloads.
        overwrite: Replace existing complete files.

    Returns:
        Downloaded and skipped file counts.
    """
    hours = list(hourly_times(day, day))
    downloaded = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_hour, valid_time, root, overwrite): valid_time for valid_time in hours}
        for future in as_completed(futures):
            valid_time = futures[future]
            path, was_downloaded = future.result()
            if was_downloaded:
                downloaded += 1
                print(f"Downloaded {valid_time:%Y-%m-%d %HZ}: {path}")
            else:
                skipped += 1
                print(f"Skipping existing {valid_time:%Y-%m-%d %HZ}: {path}")
    return downloaded, skipped


def parse_args() -> argparse.Namespace:
    """Parse command-line options for an HRRR collection run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, default=HRRR_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=HRRR_END_DATE)
    parser.add_argument("--output-dir", type=Path, default=Path(HRRR_DIR))
    parser.add_argument("--workers", type=int, default=HRRR_MAX_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Download configured hourly HRRR analysis fields by UTC date."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("HRRR workers must be at least one")
    if args.end_date < args.start_date:
        raise ValueError(f"HRRR end date {args.end_date} precedes start date {args.start_date}")

    latest_complete_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    effective_end_date = min(args.end_date, latest_complete_date)
    if effective_end_date < args.end_date:
        print(f"Capping HRRR end date at latest complete UTC date: {effective_end_date}")
    if args.start_date > effective_end_date:
        raise ValueError(f"No complete HRRR dates are available from {args.start_date} through {args.end_date}")

    downloaded = 0
    skipped = 0
    current_day = args.start_date
    while current_day <= effective_end_date:
        day_downloaded, day_skipped = download_date(
            day=current_day,
            root=args.output_dir,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        downloaded += day_downloaded
        skipped += day_skipped
        current_day += timedelta(days=1)

    print(f"HRRR collection complete: downloaded {downloaded:,}, skipped {skipped:,}")


if __name__ == "__main__":
    main()
