"""Utilities for building and splitting AOI-hour records."""

import geopandas as gpd
import numpy as np
import polars as pl
import shapely
from pycanopy import SpatialFrame, distance_to_point
from pyproj import Transformer

from collection.emissions_schema import EMISSIONS_HOUR_UTC_COL
from config import (
    CITIES_URL,
    DELTA_NOX_MASS_COL,
    DELTA_NOX_SCALE_COL,
    DELTA_SCALE_LEVEL_FRACTION,
    IMG_RANGE,
    LABEL_COL,
    MIN_CITY_POPULATION,
    MIN_DELTA_HISTORY,
    NOX_MASS_COL,
    OUTLIER_FILTER_COLUMNS,
    OUTLIER_LOWER_QUANTILE,
    OUTLIER_UPPER_QUANTILE,
)

AOI_ID_COL = "aoi_id"
MAJOR_CITY_DIST_COL = "major_city_dist"
METERS_PER_KM = 1000.0
MAD_NORMAL_SCALE = 1.4826  # puts MAD on a standard-deviation scale under normality
HRRR_PRODUCT = "wrfsfcf00"  # hourly surface analysis product named in every HRRR filename
HRRR_FIELD_SLUG = "wind-temp-blh"  # field subset named in every HRRR filename
WGS84_TO_CONUS = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
CONUS_TO_WGS84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def filter_quantitative_outliers(
    splits: dict[str, pl.DataFrame],
    columns: tuple[str, ...] = OUTLIER_FILTER_COLUMNS,
) -> dict[str, pl.DataFrame]:
    """Apply training-derived 1st/99th percentile bounds to every split.

    Coordinates, integer counts, clock fields, and bounded quality measures are
    intentionally excluded. Their tails are meaningful populations rather
    than obvious continuous-variable anomalies.
    """
    train = splits["train"]
    missing = set(columns).difference(train.columns)
    if missing:
        raise ValueError(f"Cannot filter missing quantitative columns: {', '.join(sorted(missing))}")

    statistics = train.select(
        *(
            expression
            for column in columns
            for expression in (
                pl.col(column)
                .filter(pl.col(column).is_finite())
                .quantile(OUTLIER_LOWER_QUANTILE, interpolation="linear")
                .alias(f"{column}_lower"),
                pl.col(column)
                .filter(pl.col(column).is_finite())
                .quantile(OUTLIER_UPPER_QUANTILE, interpolation="linear")
                .alias(f"{column}_upper"),
            )
        )
    ).row(0, named=True)
    bounds = {
        column: (statistics[f"{column}_lower"], statistics[f"{column}_upper"])
        for column in columns
    }
    empty = [column for column, (lower, upper) in bounds.items() if lower is None or upper is None]
    if empty:
        raise ValueError(f"Cannot calculate outlier bounds without finite values for: {', '.join(empty)}")
    print("Training-derived quantitative outlier bounds:")
    for column, (lower, upper) in bounds.items():
        print(f"  {column}: [{lower:.6g}, {upper:.6g}]")

    keep = pl.all_horizontal(
        pl.col(column).is_between(lower, upper, closed="both")
        for column, (lower, upper) in bounds.items()
    )
    filtered = {name: split.filter(keep) for name, split in splits.items()}
    for name, split in splits.items():
        print(f"[{name}] quantitative outliers retained {filtered[name].height:,}/{split.height:,} records")
    return filtered


def load_major_cities(url: str = CITIES_URL) -> pl.DataFrame:
    """Load centroids of populated places meeting the major-city population threshold.

    Args:
        url: Source of the populated-places shapefile.

    Returns:
        One row per major city carrying WGS84 ``lon`` and ``lat``.
    """
    cities = gpd.read_file(url).to_crs("EPSG:4326")
    cities = cities[cities["pop_max"] >= MIN_CITY_POPULATION]
    return pl.DataFrame(
        {
            "lon": cities.geometry.x.to_numpy().astype(np.float64),
            "lat": cities.geometry.y.to_numpy().astype(np.float64),
        }
    )


def add_major_city_distance(frame: pl.DataFrame, cities: pl.DataFrame | None = None) -> pl.DataFrame:
    """Add the great-circle distance in km from each AOI centroid to the nearest major city.

    Args:
        frame: AOI rows carrying centroid ``lat`` and ``lon`` columns.
        cities: City centroids defaulting to the configured populated-places shapefile.

    Returns:
        The input frame with a ``major_city_dist`` column.
    """
    city_frame = load_major_cities() if cities is None else cities
    if frame.is_empty() or city_frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(MAJOR_CITY_DIST_COL))
    longitudes = frame["lon"].to_numpy().astype(np.float64)
    latitudes = frame["lat"].to_numpy().astype(np.float64)
    nearest_m = np.full(longitudes.shape, np.inf, dtype=np.float64)
    # Haversine metres to one city per pass keeps the running minimum exact
    for city_lon, city_lat in zip(city_frame["lon"], city_frame["lat"], strict=True):
        distances_m = distance_to_point(longitudes, latitudes, city_lon, city_lat, coordinate_system="geographic")
        np.minimum(nearest_m, distances_m, out=nearest_m)
    return frame.with_columns(pl.Series(MAJOR_CITY_DIST_COL, nearest_m / METERS_PER_KM, dtype=pl.Float64))


def add_hrrr_files(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the HRRR storage-root-relative GRIB2 path to AOI-hour rows.

    Args:
        frame: AOI-hour rows carrying UTC date and hour columns.

    Returns:
        The frame with an added HRRR path column.
    """
    return frame.with_columns(
        pl.format(
            "raw/{}/hrrr_{}_{}z_{}_{}.grib2",
            pl.col("date").dt.strftime("%Y/%m/%d"),
            pl.col("date").dt.strftime("%Y%m%d"),
            pl.col("hour").cast(pl.String).str.pad_start(2, "0"),
            pl.lit(HRRR_PRODUCT),
            pl.lit(HRRR_FIELD_SLUG),
        ).alias("hrrr")
    )


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


def add_delta_nox_targets(hourly: pl.LazyFrame) -> pl.LazyFrame:
    """Add hourly NOx changes normalized by the prior completed quarter.

    Args:
        hourly: AOI-hour rows containing a UTC timestamp, date, hour, and
            aggregate NOx mass.

    Returns:
        Rows with raw NOx changes, lagged scales, and transformed labels.
    """
    with_deltas = (
        hourly.with_columns(pl.col(EMISSIONS_HOUR_UTC_COL).alias("_hour_start"))
        .sort(AOI_ID_COL, "_hour_start")
        .with_columns(
            pl.col(NOX_MASS_COL).shift(1).over(AOI_ID_COL).alias("_previous_nox_mass"),
            pl.col("_hour_start").shift(1).over(AOI_ID_COL).alias("_previous_hour_start"),
        )
        .with_columns(
            pl.when(pl.col("_hour_start") - pl.col("_previous_hour_start") == pl.duration(hours=1))
            .then(pl.col(NOX_MASS_COL) - pl.col("_previous_nox_mass"))
            .alias(DELTA_NOX_MASS_COL),
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.quarter().alias("_quarter"),
        )
    )
    quarter_medians = (
        with_deltas.filter(pl.col(DELTA_NOX_MASS_COL).is_not_null())
        .group_by(AOI_ID_COL, "_year", "_quarter")
        .agg(
            pl.col(DELTA_NOX_MASS_COL).median().alias("_delta_nox_med"),
            pl.col(NOX_MASS_COL).median().alias("_median_nox_mass"),
            pl.len().alias("_delta_history_count"),
        )
    )
    quarter_stats = (
        with_deltas.filter(pl.col(DELTA_NOX_MASS_COL).is_not_null())
        .join(quarter_medians, on=[AOI_ID_COL, "_year", "_quarter"], how="inner")
        .group_by(AOI_ID_COL, "_year", "_quarter")
        .agg(
            pl.col("_delta_nox_med").first(),
            pl.col("_median_nox_mass").first(),
            pl.col("_delta_history_count").first(),
            (pl.col(DELTA_NOX_MASS_COL) - pl.col("_delta_nox_med")).abs().median().alias("_delta_nox_mad"),
        )
        .with_columns(
            (
                pl.col("_delta_nox_mad") * MAD_NORMAL_SCALE
                + pl.col("_median_nox_mass").abs() * DELTA_SCALE_LEVEL_FRACTION
            ).alias(DELTA_NOX_SCALE_COL),
            pl.when(pl.col("_quarter") == 4).then(pl.col("_year") + 1).otherwise(pl.col("_year")).alias("_year"),
            pl.when(pl.col("_quarter") == 4).then(1).otherwise(pl.col("_quarter") + 1).alias("_quarter"),
        )
    )
    return (
        with_deltas.join(quarter_stats, on=[AOI_ID_COL, "_year", "_quarter"], how="left")
        .with_columns(
            pl.when((pl.col("_delta_history_count") >= MIN_DELTA_HISTORY) & (pl.col(DELTA_NOX_SCALE_COL) > 0))
            .then((pl.col(DELTA_NOX_MASS_COL) / pl.col(DELTA_NOX_SCALE_COL)).arcsinh())
            .alias(LABEL_COL)
        )
        .drop(
            "_hour_start",
            "_previous_nox_mass",
            "_previous_hour_start",
            "_year",
            "_quarter",
            "_delta_nox_med",
            "_delta_nox_mad",
            "_median_nox_mass",
            "_delta_history_count",
        )
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
        .group_by(AOI_ID_COL, EMISSIONS_HOUR_UTC_COL)
        .agg(
            pl.col("noxMass").sum().alias(NOX_MASS_COL),
            pl.col("heatInput").mean().alias("_hourly_avg_heat_input"),
            pl.col("grossLoad").mean().alias("_hourly_avg_pwr_gen"),
        )
        .with_columns(
            pl.col(EMISSIONS_HOUR_UTC_COL).dt.date().alias("date"),
            pl.col(EMISSIONS_HOUR_UTC_COL).dt.hour().cast(pl.Int8).alias("hour"),
        )
    )
    return (
        add_delta_nox_targets(add_previous_quarter_same_hour_averages(hourly))
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
