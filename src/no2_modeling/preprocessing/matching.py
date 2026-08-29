"""Match emissions records to qualifying TEMPO satellite tiles."""

import os
import pickle
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import netCDF4 as nc
import pandas as pd

from no2_modeling.config import MIN_TEMPO_DURATION, MINS_FILTER, NUM_CORES, TEMPO_DIR, TEMPO_MAPPING


def select_tempo_after(
    target_dt: datetime,
    candidates: Iterable[tuple[datetime, str]],
    window_minutes: int,
) -> str | None:
    """Return the closest candidate after a target within the given window."""
    window = timedelta(minutes=window_minutes)
    eligible = [
        (curr_dt - target_dt, fname) for curr_dt, fname in candidates if timedelta(0) < curr_dt - target_dt <= window
    ]
    return min(eligible, default=(None, None), key=lambda item: item[0])[1]


def parse_tile(fname: str) -> tuple[datetime, str] | None:
    """Return the timestamp and filename when a tile meets the duration requirement."""
    try:
        with nc.Dataset(os.path.join(TEMPO_DIR, fname)) as dataset:
            start = pd.Timestamp(dataset.time_coverage_start)
            end = pd.Timestamp(dataset.time_coverage_end)
        if (end - start).total_seconds() / 60 < MIN_TEMPO_DURATION:
            return None
        timestamp = pd.to_datetime(fname.split("_")[4], format="%Y%m%dT%H%M%SZ").to_pydatetime()
        return timestamp.replace(tzinfo=timezone.utc), fname
    except (IndexError, OSError, RuntimeError, ValueError) as error:
        print(f"WARNING: skipping {fname}: {error}")
        return None


def normalize_tempo_mapping(mapping: dict) -> dict:
    """Normalize cached timestamps to UTC and sort each date's tiles."""
    normalized = {}
    for scan_date, tiles in mapping.items():
        normalized_tiles = []
        for timestamp, fname in tiles:
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            normalized_tiles.append((timestamp, fname))
        normalized[scan_date] = sorted(normalized_tiles, key=lambda tile: tile[0])
    return normalized


def build_tempo_mapping() -> dict:
    """Map dates to timestamps and filenames for qualifying TEMPO tiles."""
    if os.path.exists(TEMPO_MAPPING):
        with open(TEMPO_MAPPING, "rb") as file:
            return normalize_tempo_mapping(pickle.load(file))

    fnames = [fname for fname in os.listdir(TEMPO_DIR) if fname.endswith(".nc")]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(parse_tile, fnames))

    tempo_by_date = defaultdict(list)
    for result in results:
        if result is not None:
            timestamp, fname = result
            tempo_by_date[timestamp.date()].append((timestamp, fname))

    tempo_by_date = normalize_tempo_mapping(tempo_by_date)
    os.makedirs(os.path.dirname(TEMPO_MAPPING), exist_ok=True)
    with open(TEMPO_MAPPING, "wb") as file:
        pickle.dump(dict(tempo_by_date), file)
    return tempo_by_date


def map_to_tempo(row: pd.Series, tempo_by_date: dict) -> tuple[str | None, str | None]:
    """Return the current and previous TEMPO filenames for an emissions record."""
    target_dt = datetime(
        year=row["date"].year,
        month=row["date"].month,
        day=row["date"].day,
        hour=row["hour"],
        tzinfo=timezone.utc,
    )
    prev_dt = target_dt - pd.Timedelta(hours=1)
    tempo = select_tempo_after(target_dt, tempo_by_date.get(target_dt.date(), []), MINS_FILTER)
    prev_tempo = select_tempo_after(prev_dt, tempo_by_date.get(prev_dt.date(), []), MINS_FILTER)
    return tempo, prev_tempo


def map_chunk(args: tuple[pd.DataFrame, dict]) -> pd.DataFrame:
    """Map a DataFrame chunk to current and previous TEMPO files."""
    chunk, tempo_by_date = args
    results = chunk.apply(lambda row: map_to_tempo(row, tempo_by_date), axis=1)
    chunk = chunk.copy()
    chunk["tempo"], chunk["prev_tempo"] = zip(*results)
    return chunk
