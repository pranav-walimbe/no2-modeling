"""Download configured TEMPO NO2 granules in bounded batches."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import earthaccess

from config import (
    TEMPO_DIR,
    TEMPO_DOWNLOAD_BATCH_SIZE,
    TEMPO_END_DATE,
    TEMPO_LEVEL,
    TEMPO_PRODUCT,
    TEMPO_START_DATE,
    TEMPO_VERSION,
)
from prerequisites import require_earthdata_credentials

_SUPPORTED_LEVELS = {"L2", "L3"}
_SUPPORTED_VERSIONS = {"V03", "V04"}
_GRANULE_PATTERN = re.compile(r"^TEMPO_NO2_(L2|L3)_(V\d{2})_(\d{8})T(\d{6})Z_S(\d{3})(?:G(\d{2}))?\.nc$")


@dataclass(frozen=True)
class _TempoGranuleName:
    level: str
    version: str
    observed_at: datetime


class _Granule(Protocol):
    def data_links(self) -> list[str]:
        """Return downloadable links for this search result."""


def _parse_utc(value: str) -> datetime:
    # Accept the configured format and ISO timestamps
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _month_windows(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    # Yield closed windows without searching more than one month
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1, day=1, hour=0, minute=0, second=0)
        else:
            next_month = cursor.replace(month=cursor.month + 1, day=1, hour=0, minute=0, second=0)
        yield cursor, min(end, next_month - timedelta(seconds=1))
        cursor = next_month


def _granule_filename(result: _Granule) -> str:
    # Prefer the NetCDF link when a result exposes multiple assets
    for link in result.data_links():
        filename = Path(urlparse(link).path).name
        if filename.endswith(".nc"):
            return filename
    raise ValueError(f"TEMPO granule has no downloadable NetCDF link: {result}")


def _parse_tempo_granule_name(filename: str) -> _TempoGranuleName:
    # Parse storage metadata and validate the level-specific suffix
    match = _GRANULE_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unrecognized TEMPO NO2 filename: {filename}")

    level, version, date_text, time_text, _, granule_text = match.groups()
    if level == "L2" and granule_text is None:
        raise ValueError(f"Level 2 filename lacks a granule number: {filename}")
    if level == "L3" and granule_text is not None:
        raise ValueError(f"Level 3 filename contains a granule number: {filename}")

    observed_at = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return _TempoGranuleName(level=level, version=version, observed_at=observed_at)


def _monthly_granule_directory(base_dir: str | Path, granule: _TempoGranuleName) -> Path:
    # Build the storage directory from parsed observation time
    return Path(base_dir, f"{granule.observed_at.year:04d}", f"{granule.observed_at.month:02d}")


def _validate_config() -> None:
    # Reject inconsistent selections before authentication or network access
    if TEMPO_LEVEL not in _SUPPORTED_LEVELS:
        raise ValueError(f"Unsupported TEMPO_LEVEL {TEMPO_LEVEL!r}; choose L2 or L3")
    if TEMPO_VERSION not in _SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported TEMPO_VERSION {TEMPO_VERSION!r}; choose V03 or V04")
    if TEMPO_PRODUCT != f"TEMPO_NO2_{TEMPO_LEVEL}":
        raise ValueError("TEMPO_PRODUCT must match TEMPO_LEVEL")
    if TEMPO_DOWNLOAD_BATCH_SIZE < 1:
        raise ValueError("TEMPO_DOWNLOAD_BATCH_SIZE must be positive")


def _download_window(start: datetime, end: datetime) -> tuple[int, int, int]:
    # Search one month then flush each directory in bounded batches
    results = earthaccess.search_data(
        short_name=TEMPO_PRODUCT,
        version=TEMPO_VERSION,
        temporal=(start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
    )
    existing_by_directory: dict[Path, set[str]] = {}
    pending_by_directory: dict[Path, list[_Granule]] = {}
    downloaded = 0
    skipped = 0

    for result in results:
        filename = _granule_filename(result)
        granule = _parse_tempo_granule_name(filename)
        if granule.level != TEMPO_LEVEL or granule.version != TEMPO_VERSION:
            raise ValueError(f"Search returned a granule outside the configured collection: {filename}")

        output_dir = _monthly_granule_directory(TEMPO_DIR, granule)
        if output_dir not in existing_by_directory:
            existing_by_directory[output_dir] = {path.name for path in output_dir.glob("*.nc") if path.is_file()}
        if filename in existing_by_directory[output_dir]:
            skipped += 1
            continue

        pending = pending_by_directory.setdefault(output_dir, [])
        pending.append(result)
        if len(pending) >= TEMPO_DOWNLOAD_BATCH_SIZE:
            output_dir.mkdir(parents=True, exist_ok=True)
            earthaccess.download(pending, output_dir)
            existing_by_directory[output_dir].update(_granule_filename(item) for item in pending)
            downloaded += len(pending)
            pending.clear()

    for output_dir, pending in pending_by_directory.items():
        if not pending:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        earthaccess.download(pending, output_dir)
        downloaded += len(pending)

    return len(results), downloaded, skipped


def main() -> None:
    """Download missing TEMPO files using monthly searches and bounded batches."""
    _validate_config()
    start = _parse_utc(TEMPO_START_DATE)
    end = _parse_utc(TEMPO_END_DATE)
    if end < start:
        raise ValueError("TEMPO_END_DATE must not precede TEMPO_START_DATE")

    require_earthdata_credentials()
    earthaccess.login(strategy="environment")
    total_found = 0
    total_downloaded = 0
    total_skipped = 0
    for window_start, window_end in _month_windows(start, end):
        found, downloaded, skipped = _download_window(window_start, window_end)
        total_found += found
        total_downloaded += downloaded
        total_skipped += skipped
        print(f"{window_start:%Y-%m}: found {found}, downloaded {downloaded}, already present {skipped}")

    print(f"Complete: found {total_found}, downloaded {total_downloaded}, already present {total_skipped}")


if __name__ == "__main__":
    main()
