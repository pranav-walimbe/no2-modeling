"""Utilities for building and splitting AOI-hour records."""

import os
import pickle
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import polars as pl
import shapely
from pycanopy import SpatialFrame
from pyproj import Transformer

from config import (
    IMG_RANGE,
    LABEL_COL,
    MIN_TEMPO_DURATION,
    MINS_FILTER,
    NUM_CORES,
    STRAT_VIS_PNG,
    TEMPO_DIR,
    TEMPO_LEVEL,
    TEMPO_MAPPING,
)

AOI_ID_COL = "aoi_id"
WGS84_TO_CONUS = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
CONUS_TO_WGS84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def select_tempo_after(
    target_dt: datetime,
    candidates: Iterable[tuple[datetime, str]],
    window_minutes: int,
) -> str | None:
    """Return the closest candidate after a target within the given window."""
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
        if (end - start).total_seconds() / 60 < MIN_TEMPO_DURATION:
            return None
        filename = Path(relative_path).name
        timestamp = datetime.strptime(filename.split("_")[4], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return timestamp, relative_path
    except (IndexError, OSError, RuntimeError, ValueError) as error:
        print(f"WARNING: skipping {relative_path}: {error}")
        return None


def _as_utc(timestamp: object) -> datetime:
    # Support pandas timestamps stored by older mapping caches
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if not isinstance(timestamp, datetime):
        raise TypeError(f"Unsupported TEMPO timestamp type: {type(timestamp).__name__}")
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def normalize_tempo_mapping(mapping: dict) -> dict[date, list[tuple[datetime, str]]]:
    """Normalize cached timestamps to UTC and sort each date's tiles."""
    normalized = {}
    for scan_date, tiles in mapping.items():
        normalized[scan_date] = sorted(
            [(_as_utc(timestamp), filename) for timestamp, filename in tiles],
            key=lambda tile: tile[0],
        )
    return normalized


def build_tempo_mapping(overwrite: bool = False) -> dict[date, list[tuple[datetime, str]]]:
    """Map dates to qualifying TEMPO tiles with optional cache replacement."""
    if TEMPO_LEVEL != "L3":
        raise RuntimeError("Raw TEMPO L2 granules must be gridded before stratification")
    if not overwrite and os.path.exists(TEMPO_MAPPING):
        with open(TEMPO_MAPPING, "rb") as file:
            return normalize_tempo_mapping(pickle.load(file))

    tempo_root = Path(TEMPO_DIR)
    paths = ((str(path), str(path.relative_to(tempo_root))) for path in tempo_root.rglob("*.nc") if path.is_file())
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
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


def _record_datetime(record_date: date | datetime, hour: int) -> datetime:
    # Build the UTC start of a CAMPD reporting hour
    if isinstance(record_date, datetime):
        record_date = record_date.date()
    return datetime.combine(record_date, datetime.min.time(), tzinfo=timezone.utc).replace(hour=int(hour))


def add_tempo_files(
    frame: pl.DataFrame,
    tempo_by_date: dict[date, list[tuple[datetime, str]]],
) -> pl.DataFrame:
    """Attach current and preceding TEMPO filenames to AOI-hour rows."""
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


def add_projected_coordinates(frame: pl.DataFrame) -> pl.DataFrame:
    """Add NAD83 Conus Albers coordinates to longitude-latitude points."""
    x_m, y_m = WGS84_TO_CONUS.transform(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    return frame.with_columns(
        pl.Series("x_m", x_m, dtype=pl.Float64),
        pl.Series("y_m", y_m, dtype=pl.Float64),
    )


def build_aois(records: pl.DataFrame) -> pl.DataFrame:
    """Build one 72 km square AOI centered on each existing facility."""
    centers = (
        records.select("facilityId", "lat", "lon")
        .drop_nulls()
        .unique(subset="facilityId", keep="first")
        .sort("facilityId")
        .rename({"facilityId": AOI_ID_COL})
    )
    return add_projected_coordinates(centers)


def build_aoi_spatial_frame(aois: pl.DataFrame) -> SpatialFrame:
    """Build an indexed PyCanopy polygon frame for 72 km AOIs."""
    half_width_m = IMG_RANGE * 500
    x_m = aois["x_m"].to_numpy()
    y_m = aois["y_m"].to_numpy()
    polygons = shapely.box(
        np.nextafter(x_m - half_width_m, -np.inf),
        np.nextafter(y_m - half_width_m, -np.inf),
        np.nextafter(x_m + half_width_m, np.inf),
        np.nextafter(y_m + half_width_m, np.inf),
    )
    polygon_frame = aois.select(AOI_ID_COL).with_columns(
        pl.Series("_geometry", shapely.to_wkb(polygons).tolist(), dtype=pl.Binary)
    )
    return SpatialFrame.from_wkb_polygons(polygon_frame, wkb_col="_geometry")


def build_aoi_membership(
    aois: pl.DataFrame,
    records: pl.DataFrame,
    spatial_aois: SpatialFrame | None = None,
) -> pl.DataFrame:
    """Map facilities to every 72 km AOI containing their location."""
    facilities = add_projected_coordinates(
        records.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId", keep="first")
    )
    if aois.is_empty() or facilities.is_empty():
        return pl.DataFrame(schema={AOI_ID_COL: aois.schema[AOI_ID_COL], "facilityId": facilities.schema["facilityId"]})
    indexed_aois = spatial_aois or build_aoi_spatial_frame(aois)
    return (
        indexed_aois.lazy()
        .within_join(facilities.select("facilityId", "x_m", "y_m"), x_col="x_m", y_col="y_m")
        .select(AOI_ID_COL, "facilityId")
        .collect()
        .sort(AOI_ID_COL, "facilityId")
    )


def _fuel_flags() -> tuple[pl.Expr, pl.Expr]:
    # Prefer hourly fuel metadata and fall back to facility attributes
    fuel = pl.coalesce("primaryFuelInfo", "attributePrimaryFuelInfo").fill_null("").str.to_lowercase()
    return fuel.str.contains("coal"), fuel.str.contains("natural gas")


def filter_usable_nox_measurements(
    records: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Remove records whose NOx mass measurement is invalid or unavailable."""
    unusable = (
        pl.col("noxMassMeasureFlg")
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase()
        .str.contains(r"invalid|unavailable")
    )
    return records.filter(~unusable)


def add_previous_quarter_same_hour_averages(hourly: pl.LazyFrame) -> pl.LazyFrame:
    """Replace hourly heat and power means with prior-quarter same-hour means."""
    quarter_columns = hourly.with_columns(
        pl.col("date").dt.year().alias("_year"),
        pl.col("date").dt.quarter().alias("_quarter"),
    )
    previous_quarter = (
        quarter_columns.group_by(AOI_ID_COL, "_year", "_quarter", "hour")
        .agg(
            pl.col("_hourly_avg_heat_input").mean().alias("avg_heat_input"),
            pl.col("_hourly_avg_pwr_gen").mean().alias("avg_pwr_gen"),
        )
        .with_columns(
            pl.when(pl.col("_quarter") == 4).then(pl.col("_year") + 1).otherwise(pl.col("_year")).alias("_year"),
            pl.when(pl.col("_quarter") == 4).then(1).otherwise(pl.col("_quarter") + 1).alias("_quarter"),
        )
    )
    return (
        quarter_columns.drop("_hourly_avg_heat_input", "_hourly_avg_pwr_gen")
        .join(previous_quarter, on=[AOI_ID_COL, "_year", "_quarter", "hour"], how="left")
        .drop("_year", "_quarter")
    )


def aggregate_aoi_hours(
    records: pl.DataFrame | pl.LazyFrame,
    aois: pl.DataFrame,
    membership: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate unit observations and static unit counts to AOI-hour rows."""
    records_lazy = records.lazy() if isinstance(records, pl.DataFrame) else records
    coal, natural_gas = _fuel_flags()
    unit_counts = (
        records_lazy.with_columns(coal.alias("is_coal"), natural_gas.alias("is_ng"))
        .group_by("facilityId", "unitId")
        .agg(pl.col("is_coal").any(), pl.col("is_ng").any())
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("is_coal").sum().cast(pl.UInt32).alias("num_coal_units"),
            pl.col("is_ng").sum().cast(pl.UInt32).alias("num_ng_units"),
        )
    )
    hourly = (
        records_lazy.join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "date", "hour")
        .agg(
            pl.col(LABEL_COL).sum().alias(LABEL_COL),
            pl.col("heatInput").mean().alias("_hourly_avg_heat_input"),
            pl.col("grossLoad").mean().alias("_hourly_avg_pwr_gen"),
        )
    )
    return (
        add_previous_quarter_same_hour_averages(hourly)
        .join(unit_counts, on=AOI_ID_COL, how="left")
        .join(aois.select(AOI_ID_COL, "lat", "lon", "x_m", "y_m").lazy(), on=AOI_ID_COL, how="left")
        .sort(AOI_ID_COL, "date", "hour")
        .collect(engine="streaming")
    )


def cluster_aois(
    aois: pl.DataFrame,
    spatial_aois: SpatialFrame | None = None,
) -> pl.DataFrame:
    """Cluster AOIs whose 72 km bounding boxes overlap."""
    parents = {aoi_id: aoi_id for aoi_id in aois[AOI_ID_COL].to_list()}

    def find(aoi_id: int) -> int:
        while parents[aoi_id] != aoi_id:
            parents[aoi_id] = parents[parents[aoi_id]]
            aoi_id = parents[aoi_id]
        return aoi_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    if parents:
        indexed_aois = spatial_aois or build_aoi_spatial_frame(aois)
        pairs = indexed_aois.intersects_pairs(key_col=AOI_ID_COL)
        for left, right in pairs.select(f"{AOI_ID_COL}_1", f"{AOI_ID_COL}_2").iter_rows():
            union(left, right)

    roots = sorted({find(aoi_id) for aoi_id in parents})
    cluster_by_root = {root: cluster for cluster, root in enumerate(roots)}
    return pl.DataFrame(
        {
            AOI_ID_COL: list(parents),
            "cluster": [cluster_by_root[find(aoi_id)] for aoi_id in parents],
        }
    )


def add_aoi_bounds(frame: pl.DataFrame) -> pl.DataFrame:
    """Add WGS84 bounds for each 72 km AOI square."""
    half_width_m = IMG_RANGE * 500
    x_m = frame["x_m"].to_numpy()
    y_m = frame["y_m"].to_numpy()
    corners = [
        CONUS_TO_WGS84.transform(x_m + x_offset, y_m + y_offset)
        for x_offset in (-half_width_m, half_width_m)
        for y_offset in (-half_width_m, half_width_m)
    ]
    longitudes = np.stack([corner[0] for corner in corners])
    latitudes = np.stack([corner[1] for corner in corners])
    return frame.with_columns(
        pl.Series("lat_min", latitudes.min(axis=0), dtype=pl.Float64),
        pl.Series("lat_max", latitudes.max(axis=0), dtype=pl.Float64),
        pl.Series("lon_min", longitudes.min(axis=0), dtype=pl.Float64),
        pl.Series("lon_max", longitudes.max(axis=0), dtype=pl.Float64),
    )


def plot_split_distributions(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame) -> None:
    """Visualize geographic and label distributions for each split."""
    _, axes = plt.subplots(2, 3, figsize=(20, 12))
    splits = [("Train", train), ("Val", val), ("Test", test)]

    for axis, (label, split_frame) in zip(axes[0], splits):
        axis.scatter(split_frame["lon"], split_frame["lat"], s=5, alpha=0.3)
        axis.set_title(f"{label} : n = {split_frame.height}")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_xlim(-130, -65)
        axis.set_ylim(24, 50)

    for axis, (label, split_frame) in zip(axes[1], splits):
        axis.hist(split_frame[LABEL_COL], bins=50, alpha=0.7)
        axis.set_title(label)
        axis.set_xlabel(LABEL_COL)
        axis.set_ylabel("Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(STRAT_VIS_PNG), exist_ok=True)
    plt.savefig(STRAT_VIS_PNG, dpi=150)
    plt.close()
