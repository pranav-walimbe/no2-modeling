"""Build TEMPO metadata mappings and match observations to emissions hours."""

import json
import multiprocessing
import os
import pickle
import re
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import polars as pl
import shapely
from pycanopy import SpatialFrame

from config import (
    MIN_TEMPO_DURATION,
    MINS_FILTER,
    NUM_CORES,
    TEMPO_AOI_OBSERVATION_CACHE,
    TEMPO_DIR,
    TEMPO_GEOLOCATION_STRIDE,
    TEMPO_GRANULE_CACHE,
    TEMPO_MAPPING,
    TEMPO_MAX_DELTA_MINUTES,
    TEMPO_MIN_DELTA_MINUTES,
)

AOI_ID_COL = "aoi_id"
SCAN_BREAK_MINUTES = 30
COVERAGE_TOLERANCE_DEGREES = 1e-6
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
GRANULE_PATTERN = re.compile(r"_S(?P<scan>\d+)G(?P<granule>\d+)")
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
LEAP_SECOND_DATES = (
    datetime(1981, 7, 1, tzinfo=timezone.utc),
    datetime(1982, 7, 1, tzinfo=timezone.utc),
    datetime(1983, 7, 1, tzinfo=timezone.utc),
    datetime(1985, 7, 1, tzinfo=timezone.utc),
    datetime(1988, 1, 1, tzinfo=timezone.utc),
    datetime(1990, 1, 1, tzinfo=timezone.utc),
    datetime(1991, 1, 1, tzinfo=timezone.utc),
    datetime(1992, 7, 1, tzinfo=timezone.utc),
    datetime(1993, 7, 1, tzinfo=timezone.utc),
    datetime(1994, 7, 1, tzinfo=timezone.utc),
    datetime(1996, 1, 1, tzinfo=timezone.utc),
    datetime(1997, 7, 1, tzinfo=timezone.utc),
    datetime(1999, 1, 1, tzinfo=timezone.utc),
    datetime(2006, 1, 1, tzinfo=timezone.utc),
    datetime(2009, 1, 1, tzinfo=timezone.utc),
    datetime(2012, 7, 1, tzinfo=timezone.utc),
    datetime(2015, 7, 1, tzinfo=timezone.utc),
    datetime(2017, 1, 1, tzinfo=timezone.utc),
)

GRANULE_SCHEMA = {
    "relative_path": pl.String,
    "scan_num": pl.Int32,
    "granule_num": pl.Int16,
    "time_coverage_start": pl.Datetime(time_zone="UTC"),
    "time_coverage_end": pl.Datetime(time_zone="UTC"),
    "footprint_wkb": pl.Binary,
    "year": pl.Int16,
    "month": pl.Int8,
}
OBSERVATION_SCHEMA = {
    AOI_ID_COL: pl.Int64,
    "scan_date": pl.Date,
    "scan_num": pl.Int32,
    "scan_start": pl.Datetime(time_zone="UTC"),
    "scan_midpoint": pl.Datetime(time_zone="UTC"),
    "scan_end": pl.Datetime(time_zone="UTC"),
    "tempo_time": pl.Datetime(time_zone="UTC"),
    "granule_paths": pl.List(pl.String),
    "mirror_step_starts": pl.List(pl.Int32),
    "mirror_step_ends": pl.List(pl.Int32),
    "sampled_pixel_count": pl.Int64,
}


def select_tempo_after(
    target_dt: datetime,
    candidates: Iterable[tuple[datetime, str]],
    window_minutes: int,
) -> str | None:
    """Return the closest L3 candidate after a target within a time window."""
    window = timedelta(minutes=window_minutes)
    eligible = [
        (current_dt - target_dt, filename)
        for current_dt, filename in candidates
        if timedelta(0) < current_dt - target_dt <= window
    ]
    return min(eligible, default=(None, None), key=lambda item: item[0])[1]


def parse_tile(paths: tuple[str, str]) -> tuple[datetime, str] | None:
    """Return the timestamp and relative path for a qualifying L3 tile."""
    absolute_path, relative_path = paths
    try:
        with nc.Dataset(absolute_path) as dataset:
            start = datetime.fromisoformat(dataset.time_coverage_start.replace("Z", "+00:00"))
            end = datetime.fromisoformat(dataset.time_coverage_end.replace("Z", "+00:00"))
        if (end - start).total_seconds() / SECONDS_PER_MINUTE < MIN_TEMPO_DURATION:
            return None
        filename = Path(relative_path).name
        timestamp = datetime.strptime(filename.split("_")[4], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return timestamp, relative_path
    except (IndexError, OSError, RuntimeError, ValueError) as error:
        print(f"WARNING: skipping {relative_path}: {error}")
        return None


def _as_utc(timestamp: object) -> datetime:
    # Support pandas timestamps stored by older L3 mapping caches
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if not isinstance(timestamp, datetime):
        raise TypeError(f"Unsupported TEMPO timestamp type: {type(timestamp).__name__}")
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def normalize_tempo_mapping(mapping: dict) -> dict[date, list[tuple[datetime, str]]]:
    """Normalize cached L3 timestamps to UTC and sort each date's tiles."""
    normalized = {}
    for scan_date, tiles in mapping.items():
        normalized[scan_date] = sorted(
            [(_as_utc(timestamp), filename) for timestamp, filename in tiles],
            key=lambda tile: tile[0],
        )
    return normalized


def build_tempo_mapping(overwrite: bool = False) -> dict[date, list[tuple[datetime, str]]]:
    """Build or load the legacy L3 timestamp mapping."""
    if not overwrite and os.path.exists(TEMPO_MAPPING):
        with open(TEMPO_MAPPING, "rb") as file:
            return normalize_tempo_mapping(pickle.load(file))

    tempo_root = Path(TEMPO_DIR)
    paths = ((str(path), str(path.relative_to(tempo_root))) for path in tempo_root.rglob("*.nc") if path.is_file())
    with ProcessPoolExecutor(
        max_workers=NUM_CORES,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        tempo_by_date = defaultdict(list)
        for result in executor.map(parse_tile, paths, chunksize=32):
            if result is not None:
                timestamp, filename = result
                tempo_by_date[timestamp.date()].append((timestamp, filename))

    normalized = normalize_tempo_mapping(tempo_by_date)
    os.makedirs(os.path.dirname(TEMPO_MAPPING), exist_ok=True)
    with open(TEMPO_MAPPING, "wb") as file:
        pickle.dump(normalized, file)
    return normalized


def add_tempo_files(
    frame: pl.DataFrame,
    tempo_by_date: dict[date, list[tuple[datetime, str]]],
) -> pl.DataFrame:
    """Attach current and preceding legacy L3 filenames to AOI-hour rows."""
    candidates = pl.DataFrame(
        [
            {"scan_datetime": timestamp, "tempo": filename}
            for tiles in tempo_by_date.values()
            for timestamp, filename in tiles
        ],
        schema={"scan_datetime": pl.Datetime(time_zone="UTC"), "tempo": pl.String},
    ).sort("scan_datetime")
    target_hours = frame.select("date", "hour").unique()
    targets = target_hours.with_columns(
        (pl.col("date").cast(pl.Datetime(time_zone="UTC")) + pl.duration(hours=pl.col("hour"))).alias("target_datetime")
    )
    tolerance = timedelta(minutes=MINS_FILTER)
    current = targets.sort("target_datetime").join_asof(
        candidates,
        left_on="target_datetime",
        right_on="scan_datetime",
        strategy="forward",
        tolerance=tolerance,
        allow_exact_matches=False,
    )
    previous_candidates = candidates.rename({"scan_datetime": "prev_scan_datetime", "tempo": "prev_tempo"})
    matches = (
        current.with_columns((pl.col("target_datetime") - pl.duration(hours=1)).alias("prev_target_datetime"))
        .sort("prev_target_datetime")
        .join_asof(
            previous_candidates,
            left_on="prev_target_datetime",
            right_on="prev_scan_datetime",
            strategy="forward",
            tolerance=tolerance,
            allow_exact_matches=False,
        )
        .drop("target_datetime", "scan_datetime", "prev_target_datetime", "prev_scan_datetime")
    )
    return frame.join(matches, on=["date", "hour"], how="left")


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    # Preserve stable Parquet types when a date has no usable rows
    return pl.DataFrame(schema=schema)


def _parse_utc(value: object) -> datetime:
    # Normalize NetCDF timestamp attributes to aware UTC datetimes
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _granule_numbers(dataset: nc.Dataset, path: Path) -> tuple[int, int]:
    # Prefer global attributes and fall back to the standard filename
    try:
        return int(dataset.getncattr("scan_num")), int(dataset.getncattr("granule_num"))
    except (AttributeError, TypeError, ValueError):
        match = GRANULE_PATTERN.search(path.name)
        if match is None:
            raise ValueError(f"Cannot identify scan and granule numbers for {path.name}")
        return int(match.group("scan")), int(match.group("granule"))


def _parse_granule(task: tuple[str, str, int, int]) -> dict[str, object] | None:
    # Read global attributes only for one Cache A row
    absolute_path_text, relative_path, year, month = task
    absolute_path = Path(absolute_path_text)
    try:
        with nc.Dataset(absolute_path) as dataset:
            scan_num, granule_num = _granule_numbers(dataset, absolute_path)
            start = _parse_utc(dataset.getncattr("time_coverage_start"))
            end = _parse_utc(dataset.getncattr("time_coverage_end"))
            footprint = shapely.from_wkt(str(dataset.getncattr("geospatial_bounds")))
        if footprint.is_empty or not footprint.is_valid:
            raise ValueError("invalid geospatial_bounds polygon")
        return {
            "relative_path": relative_path,
            "scan_num": scan_num,
            "granule_num": granule_num,
            "time_coverage_start": start,
            "time_coverage_end": end,
            "footprint_wkb": shapely.to_wkb(footprint),
            "year": year,
            "month": month,
        }
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"WARNING: skipping {relative_path}: {error}")
        return None


def _write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    # Keep incomplete cache files invisible to resumable readers
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.{os.getpid()}.tmp")
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_parquet_files(paths: Iterable[Path], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    # Read explicit paths to avoid failures from unmatched glob patterns
    frames = [pl.read_parquet(path) for path in sorted(paths)]
    return pl.concat(frames, how="vertical_relaxed") if frames else _empty_frame(schema)


def build_granule_cache(
    tempo_dir: str | Path = TEMPO_DIR,
    cache_dir: str | Path = TEMPO_GRANULE_CACHE,
    *,
    overwrite: bool = False,
    workers: int = NUM_CORES,
) -> set[date]:
    """Incrementally build monthly Cache A Parquet files.

    Args:
        tempo_dir: Root containing ``year/month`` raw granule directories.
        cache_dir: Root for partitioned granule-index Parquet files.
        overwrite: Reparse every discovered granule.
        workers: Number of independent NetCDF readers.

    Returns:
        UTC dates affected by newly parsed granules.
    """
    tempo_root = Path(tempo_dir)
    cache_root = Path(cache_dir)
    if overwrite:
        for cache_path in cache_root.rglob("granules.parquet"):
            cache_path.unlink()
    discovered: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in tempo_root.rglob("*.nc"):
        if not path.is_file():
            continue
        relative = path.relative_to(tempo_root)
        try:
            year, month = int(relative.parts[0]), int(relative.parts[1])
        except (IndexError, ValueError):
            print(f"WARNING: skipping file outside year/month layout: {relative}")
            continue
        discovered[(year, month)].append(path)

    existing_by_month: dict[tuple[int, int], pl.DataFrame] = {}
    tasks: list[tuple[str, str, int, int]] = []
    for (year, month), paths in sorted(discovered.items()):
        cache_path = cache_root / f"year={year:04d}" / f"month={month:02d}" / "granules.parquet"
        existing = _empty_frame(GRANULE_SCHEMA) if overwrite or not cache_path.exists() else pl.read_parquet(cache_path)
        existing_by_month[(year, month)] = existing
        cached_paths = set(existing["relative_path"].to_list())
        tasks.extend(
            (str(path), str(path.relative_to(tempo_root)), year, month)
            for path in sorted(paths)
            if str(path.relative_to(tempo_root)) not in cached_paths
        )

    if workers < 1:
        raise ValueError("workers must be at least 1")
    print(f"Cache A: {len(tasks)} new granules across {len(discovered)} monthly partitions")
    if not tasks:
        parsed = []
    elif workers == 1:
        parsed = [_parse_granule(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            parsed = list(executor.map(_parse_granule, tasks, chunksize=32))

    new_by_month: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    changed_dates: set[date] = set()
    for row in parsed:
        if row is None:
            continue
        new_by_month[(int(row["year"]), int(row["month"]))].append(row)
        start_date = row["time_coverage_start"].date()
        changed_dates.update((start_date, start_date - timedelta(days=1)))

    months_to_write = set(new_by_month)
    if overwrite:
        months_to_write.update(existing_by_month)
    for year_month in months_to_write:
        rows = new_by_month[year_month]
        combined = (
            pl.concat(
                [existing_by_month[year_month], pl.DataFrame(rows, schema=GRANULE_SCHEMA)],
                how="vertical_relaxed",
            )
            .unique(subset="relative_path", keep="last")
            .sort("time_coverage_start", "granule_num")
        )
        year, month = year_month
        destination = cache_root / f"year={year:04d}" / f"month={month:02d}" / "granules.parquet"
        _write_parquet_atomic(combined, destination)

    return changed_dates


def read_granule_cache(cache_dir: str | Path = TEMPO_GRANULE_CACHE) -> pl.DataFrame:
    """Read all Cache A partitions into one Polars frame."""
    return _read_parquet_files(Path(cache_dir).rglob("granules.parquet"), GRANULE_SCHEMA)


def _group_scan_occurrences(index: pl.DataFrame) -> list[dict[str, object]]:
    # Split reused scan numbers when their granules are not temporally adjacent
    scans: list[dict[str, object]] = []
    max_gap = timedelta(minutes=SCAN_BREAK_MINUTES)
    for scan_num_frame in index.partition_by("scan_num", maintain_order=False):
        ordered = scan_num_frame.sort("time_coverage_start", "granule_num").to_dicts()
        current: list[dict[str, object]] = []
        current_end: datetime | None = None
        for row in ordered:
            start = row["time_coverage_start"]
            if current and current_end is not None and start - current_end > max_gap:
                scans.append(_combine_scan_rows(current))
                current = []
                current_end = None
            current.append(row)
            current_end = (
                max(current_end, row["time_coverage_end"]) if current_end is not None else row["time_coverage_end"]
            )
        if current:
            scans.append(_combine_scan_rows(current))
    return sorted(scans, key=lambda scan: scan["scan_start"])


def _combine_scan_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    # Union every granule before testing AOI coverage
    ordered = sorted(rows, key=lambda row: (row["time_coverage_start"], row["granule_num"]))
    scan_start = min(row["time_coverage_start"] for row in ordered)
    scan_end = max(row["time_coverage_end"] for row in ordered)
    footprints = shapely.from_wkb([row["footprint_wkb"] for row in ordered])
    union = shapely.union_all(footprints)
    return {
        "scan_id": f"{scan_start.isoformat()}_S{int(ordered[0]['scan_num']):03d}",
        "scan_num": int(ordered[0]["scan_num"]),
        "scan_start": scan_start,
        "scan_midpoint": scan_start + (scan_end - scan_start) / 2,
        "scan_end": scan_end,
        "footprint": union,
        "granules": ordered,
    }


def _normalize_aois(aois: pl.DataFrame) -> pl.DataFrame:
    # Retain only the stable AOI definition needed by Cache B
    columns = [AOI_ID_COL, "lat_min", "lat_max", "lon_min", "lon_max"]
    return aois.select(columns).cast({AOI_ID_COL: pl.Int64}).sort(AOI_ID_COL)


def _covered_aoi_scans(aois: pl.DataFrame, scans: list[dict[str, object]]) -> list[dict[str, object]]:
    # Use PyCanopy for candidates and exact union coverage for acceptance
    if aois.is_empty() or not scans:
        return []
    normalized = _normalize_aois(aois)
    aoi_geometries = {
        int(row[AOI_ID_COL]): shapely.box(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])
        for row in normalized.iter_rows(named=True)
    }
    covered: list[dict[str, object]] = []
    scans_by_date: dict[date, list[dict[str, object]]] = defaultdict(list)
    for scan in scans:
        scans_by_date[scan["scan_start"].date()].append(scan)
    for date_scans in scans_by_date.values():
        scan_by_id = {scan["scan_id"]: scan for scan in date_scans}
        spatial_rows = [
            {
                "spatial_id": f"aoi:{aoi_id}",
                "geometry": shapely.to_wkb(geometry),
            }
            for aoi_id, geometry in aoi_geometries.items()
        ]
        spatial_rows.extend(
            {
                "spatial_id": f"scan:{scan_id}",
                "geometry": shapely.to_wkb(scan["footprint"]),
            }
            for scan_id, scan in scan_by_id.items()
        )
        spatial = SpatialFrame.from_wkb_polygons(pl.DataFrame(spatial_rows), wkb_col="geometry", index_mode="eager")
        pairs = spatial.intersects_pairs(key_col="spatial_id").select("spatial_id_1", "spatial_id_2")
        for left, right in pairs.iter_rows():
            aoi_key, scan_key = (left, right) if left.startswith("aoi:") else (right, left)
            if not aoi_key.startswith("aoi:") or not scan_key.startswith("scan:"):
                continue
            aoi_id = int(aoi_key.removeprefix("aoi:"))
            scan = scan_by_id[scan_key.removeprefix("scan:")]
            aoi_geometry = aoi_geometries[aoi_id]
            if not scan["footprint"].buffer(COVERAGE_TOLERANCE_DEGREES).covers(aoi_geometry):
                continue
            covering_granules = [
                granule
                for granule in scan["granules"]
                if shapely.from_wkb(granule["footprint_wkb"]).intersection(aoi_geometry).area > 0
            ]
            covered.append(
                {
                    AOI_ID_COL: aoi_id,
                    "bounds": aoi_geometry.bounds,
                    "scan_num": scan["scan_num"],
                    "scan_start": scan["scan_start"],
                    "scan_midpoint": scan["scan_midpoint"],
                    "scan_end": scan["scan_end"],
                    "granules": [{"relative_path": granule["relative_path"]} for granule in covering_granules],
                }
            )
    return covered


def gps_seconds_to_utc(seconds: float) -> datetime:
    """Convert seconds from the GPS epoch to an aware UTC datetime."""
    leap_seconds = 0
    for offset, effective_utc in enumerate(LEAP_SECOND_DATES, start=1):
        threshold = (effective_utc - GPS_EPOCH).total_seconds() + offset
        if seconds < threshold:
            break
        leap_seconds = offset
    return GPS_EPOCH + timedelta(seconds=float(seconds) - leap_seconds)


def _sample_granule_steps(
    path: Path,
    bounds: tuple[float, float, float, float],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    # Count sampled AOI pixels for each mirror-step time
    with nc.Dataset(path) as dataset:
        geolocation = dataset.groups["geolocation"]
        times = np.asarray(np.ma.filled(geolocation.variables["time"][:], np.nan), dtype=np.float64)
        latitudes = np.asarray(np.ma.filled(geolocation.variables["latitude"][:], np.nan), dtype=np.float64)
        longitudes = np.asarray(np.ma.filled(geolocation.variables["longitude"][:], np.nan), dtype=np.float64)
    if times.ndim != 1 or latitudes.ndim != 2 or longitudes.shape != latitudes.shape:
        raise ValueError(f"Unexpected geolocation shapes in {path}")
    if latitudes.shape[0] == times.size:
        latitudes, longitudes = latitudes[:, ::stride], longitudes[:, ::stride]
    elif latitudes.shape[1] == times.size:
        latitudes, longitudes = latitudes[::stride, :].T, longitudes[::stride, :].T
    else:
        raise ValueError(f"No geolocation dimension matches time in {path}")

    lon_min, lat_min, lon_max, lat_max = bounds
    valid = np.isfinite(latitudes) & np.isfinite(longitudes)
    inside = valid & (latitudes >= lat_min) & (latitudes <= lat_max) & (longitudes >= lon_min) & (longitudes <= lon_max)
    counts = inside.sum(axis=1, dtype=np.int64)
    counts = np.where(np.isfinite(times), counts, 0)
    valid_steps = np.flatnonzero((counts > 0) & np.isfinite(times))
    if valid_steps.size == 0:
        return None
    return times, counts, int(valid_steps.min()), int(valid_steps.max())


def _compute_observation(candidate: dict[str, object], tempo_root: Path, stride: int) -> dict[str, object] | None:
    # Combine time samples from every granule covering one AOI scan
    weighted_time = 0.0
    total_count = 0
    paths: list[str] = []
    starts: list[int | None] = []
    ends: list[int | None] = []
    for granule in candidate["granules"]:
        relative_path = str(granule["relative_path"])
        sample = _sample_granule_steps(tempo_root / relative_path, candidate["bounds"], stride)
        if sample is None and stride > 1:
            sample = _sample_granule_steps(tempo_root / relative_path, candidate["bounds"], 1)
        paths.append(relative_path)
        if sample is None:
            starts.append(None)
            ends.append(None)
            continue
        times, counts, step_start, step_end = sample
        weighted_time += float(np.dot(times, counts))
        total_count += int(counts.sum())
        starts.append(step_start)
        ends.append(step_end)
    if total_count == 0:
        return None
    observation_time = gps_seconds_to_utc(weighted_time / total_count)
    scan_start = candidate["scan_start"]
    return {
        AOI_ID_COL: candidate[AOI_ID_COL],
        "scan_date": scan_start.date(),
        "scan_num": candidate["scan_num"],
        "scan_start": scan_start,
        "scan_midpoint": candidate["scan_midpoint"],
        "scan_end": candidate["scan_end"],
        "tempo_time": observation_time,
        "granule_paths": paths,
        "mirror_step_starts": starts,
        "mirror_step_ends": ends,
        "sampled_pixel_count": total_count,
    }


def _write_observation_day(task: tuple[str, str, int, list[dict[str, object]]]) -> str:
    # Compute one date independently and publish its shard atomically
    tempo_dir, destination_text, stride, candidates = task
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            result = _compute_observation(candidate, Path(tempo_dir), stride)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"WARNING: skipping AOI {candidate[AOI_ID_COL]} scan {candidate['scan_start'].isoformat()}: {error}")
            continue
        if result is not None:
            rows.append(result)
    frame = pl.DataFrame(rows, schema=OBSERVATION_SCHEMA) if rows else _empty_frame(OBSERVATION_SCHEMA)
    _write_parquet_atomic(frame.sort(AOI_ID_COL, "tempo_time"), Path(destination_text))
    return destination_text


def _prepare_aoi_snapshot(aois: pl.DataFrame, cache_root: Path, overwrite: bool) -> None:
    # Refuse to mix Cache B rows made for different AOI bounds
    current = _normalize_aois(aois)
    snapshot = cache_root / "aois.parquet"
    if snapshot.exists() and not overwrite:
        cached = pl.read_parquet(snapshot).sort(AOI_ID_COL)
        if not cached.equals(current):
            raise ValueError("AOI definitions changed; rerun stratify_plants with --overwrite")
        return
    _write_parquet_atomic(current, snapshot)


def build_aoi_observation_cache(
    aois: pl.DataFrame,
    granules: pl.DataFrame,
    changed_dates: set[date],
    tempo_dir: str | Path = TEMPO_DIR,
    cache_dir: str | Path = TEMPO_AOI_OBSERVATION_CACHE,
    *,
    overwrite: bool = False,
    workers: int = NUM_CORES,
    stride: int = TEMPO_GEOLOCATION_STRIDE,
) -> None:
    """Build resumable daily Cache B shards for covered AOI scans.

    Args:
        aois: AOI rows carrying identifiers and WGS84 bounds.
        granules: Cache A rows.
        changed_dates: Dates affected by Cache A additions.
        tempo_dir: Root used to resolve relative granule paths.
        cache_dir: Root for Cache B Parquet files.
        overwrite: Replace all existing daily shards.
        workers: Number of independent date workers.
        stride: Cross-track geolocation sampling stride.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    cache_root = Path(cache_dir)
    _prepare_aoi_snapshot(aois, cache_root, overwrite)
    if overwrite:
        for path in cache_root.rglob("date=*.parquet"):
            path.unlink()

    scans = _group_scan_occurrences(granules)
    scans_by_date: dict[date, list[dict[str, object]]] = defaultdict(list)
    for scan in scans:
        scans_by_date[scan["scan_start"].date()].append(scan)

    dates_to_build: list[date] = []
    scans_to_build: list[dict[str, object]] = []
    for scan_date, date_scans in sorted(scans_by_date.items()):
        destination = (
            cache_root
            / f"year={scan_date.year:04d}"
            / f"month={scan_date.month:02d}"
            / f"date={scan_date.isoformat()}.parquet"
        )
        if destination.exists() and not overwrite and scan_date not in changed_dates:
            continue
        dates_to_build.append(scan_date)
        scans_to_build.extend(date_scans)

    candidates = _covered_aoi_scans(aois, scans_to_build)
    candidates_by_date: dict[date, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_date[candidate["scan_start"].date()].append(candidate)

    tasks: list[tuple[str, str, int, list[dict[str, object]]]] = []
    for scan_date in dates_to_build:
        destination = (
            cache_root
            / f"year={scan_date.year:04d}"
            / f"month={scan_date.month:02d}"
            / f"date={scan_date.isoformat()}.parquet"
        )
        tasks.append((str(tempo_dir), str(destination), stride, candidates_by_date[scan_date]))

    print(f"Cache B: {len(tasks)} scan dates using {workers} workers and geolocation stride {stride}")
    if not tasks:
        return
    if workers == 1:
        for task in tasks:
            _write_observation_day(task)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        list(executor.map(_write_observation_day, tasks, chunksize=1))


def read_aoi_observation_cache(cache_dir: str | Path = TEMPO_AOI_OBSERVATION_CACHE) -> pl.DataFrame:
    """Read all daily Cache B shards into one Polars frame."""
    return _read_parquet_files(Path(cache_dir).rglob("date=*.parquet"), OBSERVATION_SCHEMA)


def build_tempo_l2_caches(
    aois: pl.DataFrame,
    *,
    overwrite: bool = False,
    workers: int = NUM_CORES,
) -> pl.DataFrame:
    """Build both TEMPO L2 caches and return AOI observations."""
    changed_dates = build_granule_cache(overwrite=overwrite, workers=workers)
    granules = read_granule_cache()
    build_aoi_observation_cache(
        aois,
        granules,
        changed_dates,
        overwrite=overwrite,
        workers=workers,
    )
    return read_aoi_observation_cache()


def add_tempo_l2_observations(frame: pl.DataFrame, observations: pl.DataFrame) -> pl.DataFrame:
    """Match valid consecutive AOI observations to emissions clock hours."""
    if observations.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.List(pl.String)).alias("tempo"),
            pl.lit(None, dtype=pl.List(pl.String)).alias("prev_tempo"),
            pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("tempo_time"),
            pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("prev_tempo_time"),
            pl.lit(None, dtype=pl.Float64).alias("tempo_delta_minutes"),
            pl.lit(None, dtype=pl.Float64).alias("coverage_percent"),
        )
    windows = (
        observations.sort(AOI_ID_COL, "tempo_time")
        .with_columns(
            pl.col("tempo_time").shift(1).over(AOI_ID_COL).alias("prev_tempo_time"),
            pl.col("granule_paths").shift(1).over(AOI_ID_COL).alias("prev_tempo"),
        )
        .rename({"granule_paths": "tempo"})
        .with_columns(
            ((pl.col("tempo_time") - pl.col("prev_tempo_time")).dt.total_seconds() / SECONDS_PER_MINUTE).alias(
                "tempo_delta_minutes"
            )
        )
        .filter(pl.col("tempo_delta_minutes").is_between(TEMPO_MIN_DELTA_MINUTES, TEMPO_MAX_DELTA_MINUTES))
        .with_columns(
            pl.datetime_ranges(
                pl.col("prev_tempo_time").dt.truncate("1h"),
                (pl.col("tempo_time") - pl.duration(microseconds=1)).dt.truncate("1h"),
                interval="1h",
                time_zone="UTC",
            ).alias("_emissions_hour")
        )
        .explode("_emissions_hour", empty_as_null=True)
        .with_columns(
            pl.col("_emissions_hour").dt.date().alias("date"),
            pl.col("_emissions_hour").dt.hour().alias("hour"),
            (pl.col("_emissions_hour") + pl.duration(hours=1)).alias("_emissions_hour_end"),
        )
        .with_columns(
            (
                pl.min_horizontal("tempo_time", "_emissions_hour_end")
                - pl.max_horizontal("prev_tempo_time", "_emissions_hour")
            ).alias("_overlap")
        )
        .filter(pl.col("_overlap") > pl.duration(microseconds=0))
        .with_columns((pl.col("_overlap").dt.total_seconds() * 100 / SECONDS_PER_HOUR).alias("coverage_percent"))
        .sort(
            [AOI_ID_COL, "date", "hour", "_overlap", "prev_tempo_time"],
            descending=[False, False, False, True, False],
        )
        .unique(subset=[AOI_ID_COL, "date", "hour"], keep="first", maintain_order=True)
        .select(
            AOI_ID_COL,
            "date",
            "hour",
            "tempo",
            "prev_tempo",
            "tempo_time",
            "prev_tempo_time",
            "tempo_delta_minutes",
            "coverage_percent",
        )
    )
    return frame.join(windows, on=[AOI_ID_COL, "date", "hour"], how="left")


def serialize_tempo_path_lists(frame: pl.DataFrame) -> pl.DataFrame:
    """Encode stitched granule path lists as JSON for CSV output."""
    return frame.with_columns(
        [
            pl.col(column).map_elements(lambda paths: json.dumps(paths.to_list()), return_dtype=pl.String).alias(column)
            for column in ("tempo", "prev_tempo")
        ]
    )
