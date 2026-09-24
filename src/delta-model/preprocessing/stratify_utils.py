"""Utilities for building and splitting AOI-hour records."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import shapely
from pycanopy import SpatialFrame, distance_to_point
from pyproj import Transformer

from config import (
    EMA_DECAY_TIMESCALE_HOURS,
    IMG_RANGE,
    MIN_CITY_POPULATION,
    SEQUENCE_TIMESTEPS,
    TARGET_LABEL_MODE,
    TEMPO_MAX_DELTA_MINUTES,
    TEMPO_MIN_DELTA_MINUTES,
)

AOI_ID_COL = "aoi_id"
MAJOR_CITY_DIST_COL = "major_city_dist"
LABEL_MODE_COL = "label_mode"
METERS_PER_KM = 1000.0
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
HRRR_PRODUCT = "wrfsfcf00"  # hourly surface analysis product named in every HRRR filename
HRRR_FIELD_SLUG = "wind-temp-blh"  # existing HRRR files include BLH but the dataset ignores it
WGS84_TO_CONUS = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
CONUS_TO_WGS84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
POPULATED_PLACES_PATH = Path(
    "/global/scratch/projects/fc_nitrates/ddp/nox/reference/ne_10m_populated_places_simple.zip"
)


def load_major_cities(path: Path = POPULATED_PLACES_PATH) -> pl.DataFrame:
    """Load centroids of populated places meeting the major-city population threshold.

    Args:
        path: Required local populated-places shapefile archive.

    Returns:
        One row per major city carrying WGS84 ``lon`` and ``lat``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required populated-places archive is missing: {path}")
    cities = gpd.read_file(path).to_crs("EPSG:4326")
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
        cities: City centroids defaulting to the required local archive.

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


def add_sequence_weather_paths(frame: pl.DataFrame, timesteps: int = SEQUENCE_TIMESTEPS) -> pl.DataFrame:
    """Add the weather source path for every sequence timestep.

    Args:
        frame: Records carrying oldest-to-newest timestep timestamps.
        timesteps: Configured sequence length.

    Returns:
        Records with one weather path column per timestep.
    """
    expressions = []
    for index in range(timesteps):
        matched_time = (pl.col(f"timestep_time_t{index}") + pl.duration(minutes=30)).dt.truncate("1h")
        path = pl.concat_str(
            pl.lit("raw/"),
            matched_time.dt.strftime("%Y/%m/%d/hrrr_%Y%m%d_%H"),
            pl.lit(f"z_{HRRR_PRODUCT}_{HRRR_FIELD_SLUG}.grib2"),
        )
        expressions.append(path.alias(f"weather_path_t{index}"))
    return frame.with_columns(expressions)


def add_timestep_nox(
    frame: pl.DataFrame,
    hourly: pl.DataFrame,
    timesteps: int = SEQUENCE_TIMESTEPS,
) -> pl.DataFrame:
    """Linearly interpolate AOI NOx rates at every TEMPO timestamp.

    Args:
        frame: Records carrying one scan timestamp per timestep.
        hourly: Valid AOI-hour emissions values.
        timesteps: Number of scan timestamps to align.

    Returns:
        Records with one point-interpolated NOx value per timestep.
    """
    indexed = frame.with_row_index("_timestep_nox_row")
    result = indexed
    hourly_lookup = hourly.select(AOI_ID_COL, "emissions_hour_utc", "nox_mass")
    for index in range(timesteps):
        time_column = f"timestep_time_t{index}"
        nox_column = f"t{index}_nox"
        interpolated = (
            indexed.select("_timestep_nox_row", AOI_ID_COL, time_column)
            .with_columns(
                pl.col(time_column).dt.truncate("1h").alias("_lower_hour"),
            )
            .with_columns(
                (
                    (pl.col(time_column) - pl.col("_lower_hour")).dt.total_seconds()
                    / SECONDS_PER_HOUR
                )
                .alias("_upper_weight"),
                (pl.col("_lower_hour") + pl.duration(hours=1)).alias("_upper_hour"),
            )
            .join(
                hourly_lookup.rename(
                    {"emissions_hour_utc": "_lower_hour", "nox_mass": "_lower_nox"}
                ),
                on=[AOI_ID_COL, "_lower_hour"],
                how="left",
            )
            .join(
                hourly_lookup.rename(
                    {"emissions_hour_utc": "_upper_hour", "nox_mass": "_upper_nox"}
                ),
                on=[AOI_ID_COL, "_upper_hour"],
                how="left",
            )
            .with_columns(
                pl.when(pl.col("_upper_weight") == 0)
                .then(pl.col("_lower_nox"))
                .otherwise(
                    pl.col("_lower_nox")
                    + pl.col("_upper_weight")
                    * (pl.col("_upper_nox") - pl.col("_lower_nox"))
                )
                .alias(nox_column)
            )
            .select("_timestep_nox_row", nox_column)
        )
        result = result.join(interpolated, on="_timestep_nox_row", how="left")
    return result.drop("_timestep_nox_row")


def add_tempo_sequences(
    frame: pl.DataFrame,
    observations: pl.DataFrame,
    timesteps: int = SEQUENCE_TIMESTEPS,
    label_timestep_index: int | None = None,
) -> pl.DataFrame:
    """Match consecutive TEMPO sequences to the label scan's clock hour.

    Args:
        frame: AOI-hour rows eligible for observation matching.
        observations: AOI scans with timestamps and source path lists.
        timesteps: Number of consecutive scans per record.
        label_timestep_index: Zero-based scan ending the label interval.

    Returns:
        Rows carrying oldest-to-newest scan timestamps and paths.
    """
    label_index = timesteps - 1 if label_timestep_index is None else label_timestep_index
    if timesteps < 2:
        raise ValueError("TEMPO sequences require at least two timesteps")
    if not 0 < label_index < timesteps:
        raise ValueError("label_timestep_index must select a timestep after t0")
    time_columns = [f"timestep_time_t{index}" for index in range(timesteps)]
    path_columns = [f"no2_paths_t{index}" for index in range(timesteps)]
    sequences = (
        observations.lazy()
        .sort(AOI_ID_COL, "tempo_time")
        .with_columns(
            [
                pl.col("tempo_time").shift(timesteps - index - 1).over(AOI_ID_COL).alias(time_columns[index])
                for index in range(timesteps)
            ]
            + [
                pl.col("granule_paths").shift(timesteps - index - 1).over(AOI_ID_COL).alias(path_columns[index])
                for index in range(timesteps)
            ]
        )
    )
    interval_columns = []
    for index in range(1, timesteps):
        interval_column = f"_interval_minutes_{index}"
        interval_columns.append(interval_column)
        sequences = sequences.with_columns(
            (
                (pl.col(time_columns[index]) - pl.col(time_columns[index - 1])).dt.total_seconds()
                / SECONDS_PER_MINUTE
            ).alias(interval_column)
        )
    sequences = (
        sequences.filter(
            pl.all_horizontal(
                [
                    pl.col(column).is_between(TEMPO_MIN_DELTA_MINUTES, TEMPO_MAX_DELTA_MINUTES)
                    for column in interval_columns
                ]
            )
        )
        .with_columns(
            pl.col(time_columns[label_index]).dt.date().alias("date"),
            pl.col(time_columns[label_index]).dt.hour().alias("hour"),
            pl.col(f"_interval_minutes_{label_index}").alias("label_delta_mins"),
        )
        .select(
            AOI_ID_COL,
            "date",
            "hour",
            *time_columns,
            *path_columns,
            "label_delta_mins",
        )
        .collect()
    )
    return frame.join(sequences, on=[AOI_ID_COL, "date", "hour"], how="left")


def add_ema_targets(
    frame: pl.DataFrame,
    timesteps: int = SEQUENCE_TIMESTEPS,
    decay_timescale_hours: float = EMA_DECAY_TIMESCALE_HOURS,
    label_timestep_index: int | None = None,
) -> pl.DataFrame:
    """Add an irregular-time EMA over interpolated timestep NOx values.

    Args:
        frame: TEMPO-matched records carrying configured timestep timestamps.
        timesteps: Number of interpolated timestep values in the EMA.
        decay_timescale_hours: Positive exponential e-folding time in hours.
        label_timestep_index: Zero-based scan ending the current EMA window.

    Returns:
        Records with previous and current EMA states plus their difference.
    """
    label_index = timesteps - 1 if label_timestep_index is None else label_timestep_index
    first_index = label_index - timesteps + 1
    ema = pl.col(f"t{first_index}_nox")
    previous_ema = ema
    for index in range(first_index + 1, label_index + 1):
        interval_hours = (
            pl.col(f"timestep_time_t{index}") - pl.col(f"timestep_time_t{index - 1}")
        ).dt.total_seconds() / SECONDS_PER_HOUR
        retention = (-interval_hours / decay_timescale_hours).exp()
        previous_ema = ema
        ema = retention * ema + (1 - retention) * pl.col(f"t{index}_nox")

    age_columns = [
        (
            (
                pl.col(f"timestep_time_t{label_index}")
                - pl.col(f"timestep_time_t{index}")
            ).dt.total_seconds()
            / SECONDS_PER_HOUR
        ).alias(f"timestep_age_hours_t{index}")
        for index in range(first_index, label_index + 1)
    ]
    return frame.with_columns(
        *age_columns,
        previous_ema.alias("effective_previous_nox"),
        ema.alias("effective_current_nox"),
    ).with_columns(
        (pl.col("effective_current_nox") - pl.col("effective_previous_nox")).alias("effective_delta_nox")
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


def calculate_activity_conditioned_aoi_features(
    records: pl.DataFrame | pl.LazyFrame,
    membership: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate static AOI features over higher-activity hours.

    Args:
        records: Full unit-hour emissions history.
        membership: Facility-to-AOI membership table.

    Returns:
        One row per AOI with unit counts and activity-conditioned averages.
    """
    records_lazy = records.lazy() if isinstance(records, pl.DataFrame) else records
    coal, natural_gas = _fuel_flags()
    tagged = records_lazy.with_columns(coal.alias("_is_coal"), natural_gas.alias("_is_ng"))
    unit_counts = (
        tagged.join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "facilityId", "unitId")
        .agg(pl.col("_is_coal").any(), pl.col("_is_ng").any())
        .group_by(AOI_ID_COL)
        .agg(
            pl.len().cast(pl.UInt32).alias("num_units"),
            pl.col("_is_coal").sum().cast(pl.UInt32).alias("num_coal_units"),
            pl.col("_is_ng").sum().cast(pl.UInt32).alias("num_ng_units"),
        )
    )
    hourly = (
        tagged.filter(pl.col("opTime").is_finite() & (pl.col("opTime") >= 0))
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(
            pl.col("opTime").mean().alias("_mean_unit_operating_time"),
            pl.col("noxMass")
            .filter(
                pl.col("_is_coal")
                & usable_nox_measurement_expr()
                & pl.col("noxMass").is_finite()
                & (pl.col("noxMass") >= 0)
            )
            .sum()
            .alias("_coal_nox_mass"),
        )
    )
    activity_cutoffs = hourly.group_by(AOI_ID_COL).agg(
        pl.col("_mean_unit_operating_time").median().alias("_median_avg_op_time")
    )
    selected_hours = (
        hourly.join(activity_cutoffs, on=AOI_ID_COL, how="inner")
        .filter(pl.col("_mean_unit_operating_time") >= pl.col("_median_avg_op_time"))
        .select(AOI_ID_COL, "emissions_hour_utc", "_coal_nox_mass")
    )
    selected_record_averages = (
        tagged.join(membership.lazy(), on="facilityId", how="inner")
        .join(
            selected_hours.select(AOI_ID_COL, "emissions_hour_utc"),
            on=[AOI_ID_COL, "emissions_hour_utc"],
            how="inner",
        )
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("heatInput").filter(pl.col("heatInput").is_finite()).mean().alias("avg_heat_input"),
            pl.col("grossLoad").filter(pl.col("grossLoad").is_finite()).mean().alias("avg_pwr_gen"),
        )
    )
    activity_features = (
        selected_hours.group_by(AOI_ID_COL)
        .agg(pl.col("_coal_nox_mass").mean().alias("avg_coal_nox"))
        .join(selected_record_averages, on=AOI_ID_COL, how="inner")
    )
    return (
        unit_counts.join(activity_features, on=AOI_ID_COL, how="inner")
        .filter(pl.col("avg_coal_nox").is_finite())
        .collect(engine="streaming")
    )


def select_top_coal_aois(aoi_features: pl.DataFrame, fraction: float) -> pl.DataFrame:
    """Select the highest coal-NOx-ranked share of coal-containing AOIs.

    Args:
        aoi_features: Static AOI features carrying coal-unit counts and coal NOx.
        fraction: Selected share in the interval ``(0, 1]``.

    Returns:
        Deterministically ranked and selected AOI feature rows.
    """
    ranked = aoi_features.filter(pl.col("num_coal_units") > 0).sort(
        "avg_coal_nox",
        AOI_ID_COL,
        descending=[True, False],
    )
    selected_count = math.ceil(ranked.height * fraction)
    return ranked.with_row_index("coal_nox_rank", offset=1).head(selected_count)


def filter_usable_nox_measurements(
    records: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Remove records whose NOx mass measurement is invalid or unavailable."""
    return records.filter(usable_nox_measurement_expr())


def usable_nox_measurement_expr() -> pl.Expr:
    """Return the CAMPD flag expression for usable NOx measurements.

    Returns:
        Boolean expression rejecting explicit invalid and unavailable flags.
    """
    unusable = (
        pl.col("noxMassMeasureFlg")
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase()
        .str.contains(r"invalid|unavailable")
    )
    return ~unusable


def add_delta_nox_targets(hourly: pl.LazyFrame) -> pl.LazyFrame:
    """Add consecutive-hour raw NOx changes.

    Args:
        hourly: AOI-hour rows containing a UTC timestamp, date, hour, and
            aggregate NOx mass.

    Returns:
        Rows with raw NOx changes when the preceding AOI hour is available.
    """
    return (
        hourly.with_columns(pl.col("emissions_hour_utc").alias("_hour_start"))
        .sort(AOI_ID_COL, "_hour_start")
        .with_columns(
            pl.col("nox_mass").shift(1).over(AOI_ID_COL).alias("_previous_nox_mass"),
            pl.col("_hour_start").shift(1).over(AOI_ID_COL).alias("_previous_hour_start"),
        )
        .with_columns(
            pl.when(pl.col("_hour_start") - pl.col("_previous_hour_start") == pl.duration(hours=1))
            .then(pl.col("nox_mass") - pl.col("_previous_nox_mass"))
            .alias("delta_nox_mass"),
        )
        .drop("_hour_start", "_previous_nox_mass", "_previous_hour_start")
    )


def apply_target_label_mode(
    frame: pl.DataFrame,
    mode: str = TARGET_LABEL_MODE,
) -> pl.DataFrame:
    """Select hard-hour or scan-overlap-weighted raw NOx changes.

    Args:
        frame: AOI-hour rows after TEMPO observation pairing.
        mode: Configured target construction method.

    Returns:
        Rows with the selected raw target and its mode.
    """
    if mode == "hard_hour":
        return frame.with_columns(pl.lit(mode).alias(LABEL_MODE_COL))

    indexed = frame.with_row_index("_label_row")
    label_lookup = indexed.select(
        AOI_ID_COL,
        pl.col("emissions_hour_utc").alias("_contribution_hour"),
        pl.col("delta_nox_mass").alias("_contribution_delta"),
    )
    contributions = (
        indexed.filter(pl.col("tempo_time").is_not_null() & pl.col("prev_tempo_time").is_not_null())
        .select("_label_row", AOI_ID_COL, "tempo_time", "prev_tempo_time")
        .with_columns(
            pl.datetime_ranges(
                pl.col("prev_tempo_time").dt.truncate("1h"),
                (pl.col("tempo_time") - pl.duration(microseconds=1)).dt.truncate("1h"),
                interval="1h",
                time_zone="UTC",
            ).alias("_contribution_hour")
        )
        .explode("_contribution_hour", empty_as_null=True)
        .with_columns(
            (
                pl.min_horizontal("tempo_time", pl.col("_contribution_hour") + pl.duration(hours=1))
                - pl.max_horizontal("prev_tempo_time", "_contribution_hour")
            )
            .dt.total_seconds()
            .alias("_overlap_seconds")
        )
        .join(label_lookup, on=[AOI_ID_COL, "_contribution_hour"], how="left")
        .group_by("_label_row")
        .agg(
            pl.col("_overlap_seconds").sum().alias("_total_overlap_seconds"),
            pl.col("_overlap_seconds")
            .filter(pl.col("_contribution_delta").is_finite())
            .sum()
            .alias("_valid_overlap_seconds"),
            (pl.col("_contribution_delta") * pl.col("_overlap_seconds"))
            .filter(pl.col("_contribution_delta").is_finite())
            .sum()
            .alias("_weighted_delta_sum"),
        )
        .with_columns(
            pl.when(pl.col("_valid_overlap_seconds") == pl.col("_total_overlap_seconds"))
            .then(pl.col("_weighted_delta_sum") / pl.col("_total_overlap_seconds"))
            .alias("_weighted_delta_nox_mass")
        )
        .select("_label_row", "_weighted_delta_nox_mass")
    )
    return (
        indexed.join(contributions, on="_label_row", how="left")
        .with_columns(
            pl.col("_weighted_delta_nox_mass").alias("delta_nox_mass"),
            pl.lit(mode).alias(LABEL_MODE_COL),
        )
        .drop("_label_row", "_weighted_delta_nox_mass")
    )


def aggregate_aoi_hours(
    records: pl.DataFrame | pl.LazyFrame,
    aois: pl.DataFrame,
    membership: pl.DataFrame,
    aoi_features: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate unit observations and attach static AOI features.

    Args:
        records: Usable unit-hour emissions records.
        aois: Selected AOI centroids and projected coordinates.
        membership: Facility-to-selected-AOI membership table.
        aoi_features: Static activity-conditioned AOI features.

    Returns:
        One row per selected AOI and emissions hour.
    """
    records_lazy = records.lazy() if isinstance(records, pl.DataFrame) else records
    facility_units = records_lazy.select("facilityId", "unitId").unique()
    facility_unit_counts = facility_units.group_by("facilityId").agg(
        pl.len().cast(pl.UInt32).alias("_source_unit_count")
    )
    facility_locations = add_projected_coordinates(
        records_lazy.select("facilityId", "lat", "lon").drop_nulls().unique(subset="facilityId", keep="first").collect()
    )
    source_locations = (
        facility_locations.lazy()
        .join(facility_unit_counts, on="facilityId", how="inner")
        .join(membership.lazy(), on="facilityId", how="inner")
        .join(
            aois.select(AOI_ID_COL, "x_m", "y_m").lazy().rename({"x_m": "_aoi_x_m", "y_m": "_aoi_y_m"}),
            on=AOI_ID_COL,
            how="inner",
        )
        .with_columns(
            ((pl.col("x_m") - pl.col("_aoi_x_m")) / 1_000).alias("_source_east_km"),
            ((pl.col("y_m") - pl.col("_aoi_y_m")) / 1_000).alias("_source_north_km"),
        )
        .sort(AOI_ID_COL, "facilityId")
        .group_by(AOI_ID_COL, maintain_order=True)
        .agg(
            pl.col("_source_east_km").cast(pl.String).str.join(",").alias("_source_east_km"),
            pl.col("_source_north_km").cast(pl.String).str.join(",").alias("_source_north_km"),
            pl.col("_source_unit_count").cast(pl.String).str.join(",").alias("_source_unit_count"),
        )
    )
    hourly = (
        records_lazy.join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(pl.col("noxMass").sum().alias("nox_mass"))
        .with_columns(
            pl.col("emissions_hour_utc").dt.date().alias("date"),
            pl.col("emissions_hour_utc").dt.hour().cast(pl.Int8).alias("hour"),
        )
    )
    return (
        hourly.join(aoi_features.lazy(), on=AOI_ID_COL, how="inner")
        .join(source_locations, on=AOI_ID_COL, how="left")
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
