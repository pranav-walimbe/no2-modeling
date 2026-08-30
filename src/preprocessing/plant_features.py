"""Build spatial and historical power-plant features."""

import geopandas as gpd
import numpy as np
import pandas as pd

from config import EMISSIONS_RECORDS_CSV, IMG_RANGE, LABEL_COL, STRAT_INPUT_CSV


def project_to_meters(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Project latitude and longitude to NAD83 Conus Albers coordinates."""
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(
        "EPSG:5070"
    )
    return gdf.geometry.x.values, gdf.geometry.y.values


def aggregate_units(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete unit groups to one row per facility and hour."""
    units_per_facility = df.groupby("facilityId")["unitId"].nunique()
    n_units = df.groupby(["facilityId", "date", "hour"])["unitId"].transform("nunique")
    df = df[n_units == df["facilityId"].map(units_per_facility)].copy()
    df[LABEL_COL] = df.groupby(["facilityId", "date", "hour"])[LABEL_COL].transform("sum")

    unit_locs = df[["facilityId", "unitId", "lat", "lon"]].drop_duplicates(["facilityId", "unitId"]).copy()
    centroids = unit_locs.groupby("facilityId")[["lat", "lon"]].transform("mean")
    unit_locs["dist"] = np.sqrt(
        (unit_locs["lat"] - centroids["lat"]) ** 2 + (unit_locs["lon"] - centroids["lon"]) ** 2
    )
    rep_units = unit_locs.loc[unit_locs.groupby("facilityId")["dist"].idxmin(), ["facilityId", "unitId"]]

    df = df.merge(rep_units, on=["facilityId", "unitId"]).reset_index(drop=True)
    df["num_adj_units"] = df["facilityId"].map(units_per_facility - 1)
    return df


def compute_prev_qtr_mass(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate prior-quarter mean emissions at the same hour."""
    full = (
        pd.read_csv(
            EMISSIONS_RECORDS_CSV,
            usecols=["date", "hour", "facilityId", "opTime", LABEL_COL],
            parse_dates=["date"],
        )
        .query("opTime == 1.0")
        .dropna(subset=["facilityId", LABEL_COL])
    )
    facility_hours = full.groupby(["facilityId", "date", "hour"], as_index=False)[LABEL_COL].sum()
    facility_hours["year"] = facility_hours["date"].dt.year
    facility_hours["quarter"] = facility_hours["date"].dt.quarter
    lookup_df = (
        facility_hours.groupby(["facilityId", "year", "quarter", "hour"], as_index=False)[LABEL_COL]
        .mean()
        .rename(columns={LABEL_COL: "prev_qtr_mass"})
    )

    df = df.copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["prev_year"] = np.where(df["quarter"] == 1, df["year"] - 1, df["year"])
    df["prev_quarter"] = (df["quarter"] - 2) % 4 + 1
    df = df.merge(
        lookup_df,
        left_on=["facilityId", "prev_year", "prev_quarter", "hour"],
        right_on=["facilityId", "year", "quarter", "hour"],
        how="left",
    )
    return (
        df.drop(columns=["year_x", "quarter_x", "prev_year", "prev_quarter", "year_y", "quarter_y"])
        .dropna(subset=["prev_qtr_mass"])
        .reset_index(drop=True)
    )


def compute_adj_plants(df: pd.DataFrame) -> pd.DataFrame:
    """Count neighboring facilities within each image patch."""
    half_m = (IMG_RANGE / 2) * 1000
    query = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)
    all_plants = (
        pd.read_csv(STRAT_INPUT_CSV, usecols=["facilityId", "lat", "lon"])
        .drop_duplicates("facilityId")
        .reset_index(drop=True)
    )
    query_x, query_y = project_to_meters(query)
    all_x, all_y = project_to_meters(all_plants)
    in_patch = (np.abs(query_x[:, None] - all_x[None, :]) < half_m) & (
        np.abs(query_y[:, None] - all_y[None, :]) < half_m
    )
    query["num_adj_plants"] = (in_patch.sum(axis=1) - 1).clip(min=0)
    return query[["facilityId", "num_adj_plants"]]
