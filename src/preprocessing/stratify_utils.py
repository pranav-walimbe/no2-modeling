"""Utilities for building and splitting AOI-hour records."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import shapely
from pycanopy import SpatialFrame, distance_to_point
from pyproj import Transformer

from config import (
    EFFECTIVE_DELTA_NOX_COL,
    EMA_DECAY_TIMESCALE_HOURS,
    IMG_RANGE,
    LABEL_COL,
    MIN_CITY_POPULATION,
    SEQUENCE_TIMESTEPS,
    TARGET_LABEL_MODE,
    TEMPO_MAX_DELTA_MINUTES,
    TEMPO_MIN_DELTA_MINUTES,
)

AOI_ID_COL = "aoi_id"
MAJOR_CITY_DIST_COL = "major_city_dist"
LABEL_MODE_COL = "label_mode"
PREVIOUS_QUARTER_COAL_POWER_COL = "_previous_quarter_coal_power"
PREVIOUS_QUARTER_POWER_COL = "_previous_quarter_power"
PREV_QTR_MED_NOX_COL = "prev_qtr_med_nox"
EFFECTIVE_CURRENT_NOX_COL = "effective_current_nox"
EFFECTIVE_PREVIOUS_NOX_COL = "effective_previous_nox"
NOX_COL = "nox"
DELTA_NOX_COL = "delta_nox"
DELTA_NOX_SCALED_COL = "delta_nox_scaled"
DELTA_EFFECTIVE_NOX_SCALED_COL = "delta_effective_nox_scaled"
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


def _tempered_bin_quotas(
    bin_counts: dict[int, int],
    target_records: int,
    balance_exponent: float,
) -> dict[int, int]:
    # Allocate quotas in proportion to tempered bin populations
    weights = {bin_id: count**balance_exponent for bin_id, count in bin_counts.items()}
    lower = 0.0
    upper = max(bin_counts[bin_id] / weights[bin_id] for bin_id in bin_counts)
    for _ in range(64):
        midpoint = (lower + upper) / 2
        allocated = sum(min(bin_counts[bin_id], midpoint * weights[bin_id]) for bin_id in bin_counts)
        if allocated <= target_records:
            lower = midpoint
        else:
            upper = midpoint

    fractional_quotas = {
        bin_id: min(bin_counts[bin_id], lower * weights[bin_id]) for bin_id in bin_counts
    }
    quotas = {bin_id: int(np.floor(quota)) for bin_id, quota in fractional_quotas.items()}
    remaining = target_records - sum(quotas.values())
    candidates = sorted(
        (bin_id for bin_id in bin_counts if quotas[bin_id] < bin_counts[bin_id]),
        key=lambda bin_id: (-(fractional_quotas[bin_id] - quotas[bin_id]), bin_id),
    )
    for bin_id in candidates[:remaining]:
        quotas[bin_id] += 1
    if sum(quotas.values()) != target_records:
        raise RuntimeError(f"Could not allocate {target_records:,} records across scaled-delta bins")
    return quotas


def select_split_records(
    frame: pl.DataFrame,
    split: str,
    target_records: int,
    *,
    tail_fraction: float,
    balance_scaled_delta: bool,
    balance_bin_count: int,
    balance_exponent: float,
    seed: int,
) -> pl.DataFrame:
    """Trim raw-delta tails and select a deterministic split sample.

    Args:
        frame: Eligible records carrying raw and scaled effective NOx deltas.
        split: Split name used in progress and error messages.
        target_records: Exact number of records to return.
        tail_fraction: Fraction removed from each raw-delta tail.
        balance_scaled_delta: Whether to temper the scaled-delta distribution.
        balance_bin_count: Equal-width bins used for tempered selection.
        balance_exponent: Exponent applied to source bin populations.
        seed: Deterministic record-ordering seed.

    Returns:
        Selected records in AOI and emissions-time order.
    """
    if target_records <= 0:
        raise ValueError("target_records must be positive")
    if not 0 <= tail_fraction < 0.5:
        raise ValueError("tail_fraction must satisfy 0 <= tail_fraction < 0.5")
    if balance_scaled_delta and balance_bin_count <= 0:
        raise ValueError("balance_bin_count must be positive")
    if balance_scaled_delta and balance_exponent <= 0:
        raise ValueError("balance_exponent must be positive")

    lower_percentile = tail_fraction * 100
    upper_percentile = 100 - lower_percentile
    finite = frame.filter(
        pl.col(DELTA_NOX_COL).is_finite() & pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL).is_finite()
    )
    lower_bound, upper_bound = finite.select(
        pl.col(DELTA_NOX_COL).quantile(tail_fraction, interpolation="linear").alias("lower"),
        pl.col(DELTA_NOX_COL).quantile(1 - tail_fraction, interpolation="linear").alias("upper"),
    ).row(0)
    trimmed = finite.filter(pl.col(DELTA_NOX_COL).is_between(lower_bound, upper_bound, closed="both"))
    if trimmed.height < target_records:
        raise ValueError(
            f"[{split}] requested {target_records:,} records but only {trimmed.height:,} remain after tail trimming"
        )

    randomized = trimmed.with_columns(
        pl.struct(AOI_ID_COL, "emissions_hour_utc").hash(seed=seed).alias("_selection_tie_breaker")
    )
    if not balance_scaled_delta:
        selected = (
            randomized.sort("_selection_tie_breaker", AOI_ID_COL, "emissions_hour_utc")
            .head(target_records)
            .drop("_selection_tie_breaker")
            .sort(AOI_ID_COL, "emissions_hour_utc")
        )
        print(
            f"[{split}] raw delta P{lower_percentile:g}-P{upper_percentile:g} "
            f"[{lower_bound:.6g}, {upper_bound:.6g}] retained "
            f"{trimmed.height:,}/{finite.height:,}; randomly selected {selected.height:,} records"
        )
        return selected

    scaled_min, scaled_max = trimmed.select(
        pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL).min().alias("scaled_min"),
        pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL).max().alias("scaled_max"),
    ).row(0)
    if scaled_min == scaled_max:
        raise ValueError(f"[{split}] cannot balance a constant scaled effective-delta target")

    binned = randomized.with_columns(
        (
            (pl.col(DELTA_EFFECTIVE_NOX_SCALED_COL) - scaled_min)
            / (scaled_max - scaled_min)
            * balance_bin_count
        )
        .floor()
        .cast(pl.Int32)
        .clip(0, balance_bin_count - 1)
        .alias("_selection_bin"),
    )
    bin_counts = dict(binned.group_by("_selection_bin").len().iter_rows())
    quotas = _tempered_bin_quotas(bin_counts, target_records, balance_exponent)
    quota_frame = pl.DataFrame(
        {
            "_selection_bin": list(quotas),
            "_selection_quota": list(quotas.values()),
        },
        schema={"_selection_bin": pl.Int32, "_selection_quota": pl.UInt32},
    )
    selected = (
        binned.sort("_selection_bin", "_selection_tie_breaker", AOI_ID_COL, "emissions_hour_utc")
        .with_columns(pl.col("_selection_bin").cum_count().over("_selection_bin").alias("_selection_rank"))
        .join(quota_frame, on="_selection_bin", how="left")
        .filter(pl.col("_selection_rank") <= pl.col("_selection_quota"))
        .drop("_selection_bin", "_selection_tie_breaker", "_selection_rank", "_selection_quota")
        .sort(AOI_ID_COL, "emissions_hour_utc")
    )
    if selected.height != target_records:
        raise RuntimeError(f"[{split}] selected {selected.height:,} records instead of {target_records:,}")
    print(
        f"[{split}] raw delta P{lower_percentile:g}-P{upper_percentile:g} "
        f"[{lower_bound:.6g}, {upper_bound:.6g}] retained "
        f"{trimmed.height:,}/{finite.height:,}; selected {selected.height:,} records with tempered balancing "
        f"across {len(bin_counts):,} scaled-delta bins"
    )
    return selected


def classification_summary(source: pl.DataFrame, retained: pl.DataFrame) -> dict[str, object]:
    """Summarize binary-label retention overall and by AOI.

    Args:
        source: Records before the reported filtering or balancing stage.
        retained: Retained records carrying binary labels.

    Returns:
        JSON-safe overall and per-AOI counts.
    """

    def counts(frame: pl.DataFrame) -> dict[str, int | float | None]:
        # Count each class without assuming both are present
        negative = frame.filter(pl.col(LABEL_COL) == 0).height
        positive = frame.filter(pl.col(LABEL_COL) == 1).height
        total = negative + positive
        return {
            "retained_records": total,
            "negative_records": negative,
            "positive_records": positive,
            "positive_fraction": positive / total if total else None,
        }

    source_by_aoi = dict(source.group_by(AOI_ID_COL).len().iter_rows())
    retained_by_aoi = {int(group[AOI_ID_COL][0]): group for group in retained.partition_by(AOI_ID_COL)}
    by_aoi = []
    for aoi_id, source_records in sorted(source_by_aoi.items()):
        aoi_counts = counts(retained_by_aoi.get(int(aoi_id), retained.head(0)))
        by_aoi.append(
            {
                AOI_ID_COL: int(aoi_id),
                "source_records": int(source_records),
                "retained_fraction": aoi_counts["retained_records"] / source_records,
                **aoi_counts,
            }
        )
    overall = counts(retained)
    return {
        "overall": {
            "source_records": source.height,
            "retained_fraction": overall["retained_records"] / source.height if source.height else None,
            **overall,
        },
        "by_aoi": by_aoi,
    }


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


def add_tempo_sequences(
    frame: pl.DataFrame,
    observations: pl.DataFrame,
    timesteps: int = SEQUENCE_TIMESTEPS,
) -> pl.DataFrame:
    """Match complete causal TEMPO sequences to emissions clock hours.

    Args:
        frame: AOI-hour rows eligible for observation matching.
        observations: AOI scans with timestamps and source path lists.
        timesteps: Number of consecutive scans per record.

    Returns:
        Rows carrying oldest-to-newest scan timestamps, ages, and paths.
    """
    time_columns = [f"timestep_time_t{index}" for index in range(timesteps)]
    path_columns = [f"no2_paths_t{index}" for index in range(timesteps)]
    age_columns = [f"timestep_age_hours_t{index}" for index in range(timesteps)]
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
            pl.datetime_ranges(
                pl.col(time_columns[-2]).dt.truncate("1h"),
                (pl.col(time_columns[-1]) - pl.duration(microseconds=1)).dt.truncate("1h"),
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
                pl.min_horizontal(time_columns[-1], "_emissions_hour_end")
                - pl.max_horizontal(time_columns[-2], "_emissions_hour")
            ).alias("_overlap")
        )
        .filter(pl.col("_overlap") > pl.duration(microseconds=0))
        .with_columns(
            (pl.col("_overlap").dt.total_seconds() * 100 / SECONDS_PER_HOUR).alias("coverage_percent"),
            pl.col(interval_columns[-1]).alias("tempo_delta_minutes"),
            *[
                ((pl.col(time_columns[-1]) - pl.col(time_columns[index])).dt.total_seconds() / SECONDS_PER_HOUR).alias(
                    age_columns[index]
                )
                for index in range(timesteps)
            ],
        )
        .sort(
            [AOI_ID_COL, "date", "hour", "_overlap", time_columns[-2]],
            descending=[False, False, False, True, False],
        )
        .unique(subset=[AOI_ID_COL, "date", "hour"], keep="first", maintain_order=True)
        .select(
            AOI_ID_COL,
            "date",
            "hour",
            *time_columns,
            *age_columns,
            *path_columns,
            "tempo_delta_minutes",
            "coverage_percent",
        )
        .collect()
    )
    return frame.join(sequences, on=[AOI_ID_COL, "date", "hour"], how="left")


def _scan_ema(
    indexed: pl.DataFrame,
    hourly: pl.DataFrame,
    scan_time_column: str,
    output_prefix: str,
    timesteps: int,
    decay_timescale_hours: float,
) -> pl.DataFrame:
    # Integrate piecewise-constant hourly emissions over one causal scan window
    scan_time = pl.col(scan_time_column)
    contributions = (
        indexed.select(
            "_label_row",
            AOI_ID_COL,
            pl.col(scan_time_column).dt.truncate("1s").alias(scan_time_column),
        )
        .with_columns((scan_time - pl.duration(hours=timesteps)).alias("_window_start"))
        .with_columns(
            pl.datetime_ranges(
                pl.col("_window_start").dt.truncate("1h"),
                (scan_time - pl.duration(microseconds=1)).dt.truncate("1h"),
                interval="1h",
                time_zone="UTC",
            ).alias("_component_hour")
        )
        .explode("_component_hour", empty_as_null=True)
        .with_columns(
            pl.max_horizontal("_window_start", "_component_hour").alias("_overlap_start"),
            pl.min_horizontal(scan_time_column, pl.col("_component_hour") + pl.duration(hours=1)).alias("_overlap_end"),
        )
        .with_columns((pl.col("_overlap_end") - pl.col("_overlap_start")).dt.total_seconds().alias("_overlap_seconds"))
        .with_columns(
            ((scan_time - pl.col("_overlap_start")).dt.total_seconds() - pl.col("_overlap_seconds") / 2)
            .truediv(SECONDS_PER_HOUR)
            .alias("_age_hours")
        )
        .with_columns(
            (pl.col("_overlap_seconds") * (-pl.col("_age_hours") / decay_timescale_hours).exp()).alias("_raw_weight")
        )
        .join(
            hourly.select(
                AOI_ID_COL,
                pl.col("emissions_hour_utc").alias("_component_hour"),
                "nox_mass",
            ),
            on=[AOI_ID_COL, "_component_hour"],
            how="left",
        )
        .with_columns(
            (pl.col("_raw_weight") / pl.col("_raw_weight").sum().over("_label_row")).alias("_normalized_weight")
        )
        .sort("_label_row", "_component_hour")
    )
    aggregate = contributions.group_by("_label_row", maintain_order=True).agg(
        pl.col("_overlap_seconds").sum().alias("_total_overlap_seconds"),
        pl.col("_overlap_seconds").filter(pl.col("nox_mass").is_finite()).sum().alias("_valid_overlap_seconds"),
        (pl.col("nox_mass") * pl.col("_normalized_weight")).sum().alias(f"{output_prefix}_nox"),
        pl.col("_component_hour").alias(f"{output_prefix}_component_hours"),
        pl.col("nox_mass").alias(f"{output_prefix}_component_nox_mass"),
        pl.col("_overlap_seconds").alias(f"{output_prefix}_overlap_seconds"),
        pl.col("_age_hours").alias(f"{output_prefix}_age_hours"),
        pl.col("_normalized_weight").alias(f"{output_prefix}_normalized_weights"),
    )
    expected_overlap_seconds = timesteps * SECONDS_PER_HOUR
    return aggregate.with_columns(
        pl.when(
            (pl.col("_total_overlap_seconds") == expected_overlap_seconds)
            & (pl.col("_valid_overlap_seconds") == expected_overlap_seconds)
        )
        .then(pl.col(f"{output_prefix}_nox"))
        .otherwise(None)
        .alias(f"{output_prefix}_nox")
    ).drop("_total_overlap_seconds", "_valid_overlap_seconds")


def add_ema_targets(
    frame: pl.DataFrame,
    hourly: pl.DataFrame,
    timesteps: int = SEQUENCE_TIMESTEPS,
    decay_timescale_hours: float = EMA_DECAY_TIMESCALE_HOURS,
) -> pl.DataFrame:
    """Add exact-overlap current and previous scan EMA emissions targets.

    Args:
        frame: TEMPO-matched records carrying configured timestep timestamps.
        hourly: Aggregate AOI-hour emissions lookup.
        timesteps: Shared raster and EMA history length.
        decay_timescale_hours: Positive exponential e-folding time in hours.

    Returns:
        Records with continuous EMA targets and complete audit components.
    """
    if decay_timescale_hours <= 0:
        raise ValueError("decay_timescale_hours must be positive")
    indexed = frame.with_row_index("_label_row")
    current = _scan_ema(
        indexed,
        hourly,
        f"timestep_time_t{timesteps - 1}",
        "current_ema",
        timesteps,
        decay_timescale_hours,
    )
    previous = _scan_ema(
        indexed,
        hourly,
        f"timestep_time_t{timesteps - 2}",
        "previous_ema",
        timesteps,
        decay_timescale_hours,
    )
    return (
        indexed.join(current, on="_label_row", how="left")
        .join(previous, on="_label_row", how="left")
        .with_columns(
            pl.col("current_ema_nox").alias(EFFECTIVE_CURRENT_NOX_COL),
            pl.col("previous_ema_nox").alias(EFFECTIVE_PREVIOUS_NOX_COL),
        )
        .with_columns(
            (pl.col(EFFECTIVE_CURRENT_NOX_COL) - pl.col(EFFECTIVE_PREVIOUS_NOX_COL)).alias(EFFECTIVE_DELTA_NOX_COL)
        )
        .drop("_label_row", "current_ema_nox", "previous_ema_nox")
    )


def add_scaled_nox_targets(frame: pl.DataFrame) -> pl.DataFrame:
    """Add canonical raw and effective delta targets scaled by prior-quarter median NOx.

    Args:
        frame: AOI-hour records with raw and effective NOx changes.

    Returns:
        Records with unscaled aliases and asinh-scaled delta targets.
    """
    return frame.with_columns(
        pl.col("delta_nox_mass").alias(DELTA_NOX_COL),
    ).with_columns(
        (pl.col(DELTA_NOX_COL) / pl.col(PREV_QTR_MED_NOX_COL)).arcsinh().alias(DELTA_NOX_SCALED_COL),
        (pl.col(EFFECTIVE_DELTA_NOX_COL) / pl.col(PREV_QTR_MED_NOX_COL))
        .arcsinh()
        .alias(DELTA_EFFECTIVE_NOX_SCALED_COL),
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


def previous_quarter_power_priorities(
    records: pl.LazyFrame,
    membership: pl.DataFrame,
) -> pl.LazyFrame:
    """Build lagged AOI power summaries for record prioritization.

    Args:
        records: Unit-hour emissions records with fuel and gross-load fields.
        membership: Facility-to-AOI membership table.

    Returns:
        Total and coal power summaries shifted into the following quarter.
    """
    coal, _ = _fuel_flags()
    unit_quarter = (
        records.with_columns(
            pl.col("date").dt.year().alias("_priority_year"),
            pl.col("date").dt.quarter().alias("_priority_quarter"),
            coal.alias("_priority_is_coal"),
        )
        .filter(pl.col("grossLoad").is_finite())
        .group_by("facilityId", "unitId", "_priority_year", "_priority_quarter")
        .agg(
            pl.col("grossLoad").mean().alias("_unit_average_power"),
            pl.col("_priority_is_coal").any(),
        )
    )
    return (
        unit_quarter.join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "_priority_year", "_priority_quarter")
        .agg(
            pl.col("_unit_average_power").sum().alias(PREVIOUS_QUARTER_POWER_COL),
            pl.col("_unit_average_power")
            .filter(pl.col("_priority_is_coal"))
            .sum()
            .alias(PREVIOUS_QUARTER_COAL_POWER_COL),
        )
        .with_columns(
            pl.when(pl.col("_priority_quarter") == 4)
            .then(pl.col("_priority_year") + 1)
            .otherwise(pl.col("_priority_year"))
            .alias("_priority_year"),
            pl.when(pl.col("_priority_quarter") == 4)
            .then(1)
            .otherwise(pl.col("_priority_quarter") + 1)
            .alias("_priority_quarter"),
        )
    )


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


def add_previous_quarter_nox_median(hourly: pl.LazyFrame) -> pl.LazyFrame:
    """Add the AOI's median hourly NOx mass from the prior quarter.

    Args:
        hourly: AOI-hour rows containing date and aggregate NOx mass.

    Returns:
        Rows with a leakage-safe prior-quarter median when the immediately
        preceding quarter is available.
    """
    quarter_columns = hourly.with_columns(
        pl.col("date").dt.year().alias("_year"),
        pl.col("date").dt.quarter().alias("_quarter"),
    )
    previous_quarter = (
        quarter_columns.group_by(AOI_ID_COL, "_year", "_quarter")
        .agg(pl.col("nox_mass").median().alias(PREV_QTR_MED_NOX_COL))
        .with_columns(
            pl.when(pl.col("_quarter") == 4).then(pl.col("_year") + 1).otherwise(pl.col("_year")).alias("_year"),
            pl.when(pl.col("_quarter") == 4).then(1).otherwise(pl.col("_quarter") + 1).alias("_quarter"),
        )
    )
    return quarter_columns.join(previous_quarter, on=[AOI_ID_COL, "_year", "_quarter"], how="left").drop(
        "_year", "_quarter"
    )


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
) -> pl.DataFrame:
    """Aggregate unit observations and prediction-date attributes to AOI hours."""
    records_lazy = records.lazy() if isinstance(records, pl.DataFrame) else records
    coal, natural_gas = _fuel_flags()
    power_priorities = previous_quarter_power_priorities(records_lazy, membership)
    facility_units = (
        records_lazy.with_columns(coal.alias("is_coal"), natural_gas.alias("is_ng"))
        .group_by("facilityId", "unitId")
        .agg(pl.col("is_coal").any(), pl.col("is_ng").any())
    )
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
    unit_counts = (
        facility_units.join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL)
        .agg(
            pl.col("is_coal").sum().cast(pl.UInt32).alias("num_coal_units"),
            pl.col("is_ng").sum().cast(pl.UInt32).alias("num_ng_units"),
        )
    )
    facility_capacity = (
        records_lazy.select(
            "facilityId",
            "emissions_hour_utc",
            "facility_nameplate_capacity_mw",
        )
        .unique(subset=["facilityId", "emissions_hour_utc"])
        .join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(
            pl.col("facility_nameplate_capacity_mw").sum().alias("total_nameplate_capacity_mw"),
        )
    )
    hourly = (
        records_lazy.join(membership.lazy(), on="facilityId", how="inner")
        .group_by(AOI_ID_COL, "emissions_hour_utc")
        .agg(
            pl.col("noxMass").sum().alias("nox_mass"),
            pl.col("heatInput").mean().alias("_hourly_avg_heat_input"),
            pl.col("grossLoad").mean().alias("_hourly_avg_pwr_gen"),
        )
        .with_columns(
            pl.col("emissions_hour_utc").dt.date().alias("date"),
            pl.col("emissions_hour_utc").dt.hour().cast(pl.Int8).alias("hour"),
            pl.col("nox_mass").alias(NOX_COL),
        )
    )
    return (
        add_delta_nox_targets(add_previous_quarter_nox_median(add_previous_quarter_same_hour_averages(hourly)))
        .with_columns(
            pl.col("date").dt.year().alias("_priority_year"),
            pl.col("date").dt.quarter().alias("_priority_quarter"),
        )
        .join(power_priorities, on=[AOI_ID_COL, "_priority_year", "_priority_quarter"], how="left")
        .drop("_priority_year", "_priority_quarter")
        .join(facility_capacity, on=[AOI_ID_COL, "emissions_hour_utc"], how="left")
        .join(unit_counts, on=AOI_ID_COL, how="left")
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
