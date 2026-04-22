"""
Script to stratify power plant emission records into train/test/val splits

Output: train.csv, test.csv, val.csv located in output directory
"""

import sys
import os
from sklearn.model_selection import train_test_split
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from collections import defaultdict
import pandas as pd
from datetime import datetime
import numpy as np
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # access config
from config import *

# helper functions
def parse_tile(fname: str):
    """Return (dt, fname) if tile meets duration requirement"""
    try:
        fpath = os.path.join(TEMPO_DIR, fname)
        ds = xr.open_dataset(fpath, engine="netcdf4")
        start = pd.Timestamp(ds.attrs["time_coverage_start"])
        end = pd.Timestamp(ds.attrs["time_coverage_end"])
        ds.close()
        if (end - start).total_seconds() / 60 < MIN_TEMPO_DURATION:
            return None
        dt = pd.to_datetime(fname.split("_")[4], format="%Y%m%dT%H%M%SZ")
        return (dt, fname)
    except Exception as e:
        print(f"WARNING: skipping {fname}: {e}")
        return None

def build_tempo_mapping():
    """Return dict mapping dates to (timestamp, fname) for tiles meeting duration requirement"""
    # if we already computed this mapping then load it in
    if os.path.exists(TEMPO_MAPPING):
        with open(TEMPO_MAPPING, "rb") as f:
            return pickle.load(f)

    # otherwise run parallelized scan of tempo images
    fnames = [f for f in os.listdir(TEMPO_DIR) if f.endswith(".nc")]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(parse_tile, fnames))

    tempo_by_date = defaultdict(list)
    for result in results:
        if result is not None:
            dt, fname = result
            tempo_by_date[dt.date()].append((dt, fname))

    os.makedirs(os.path.dirname(TEMPO_MAPPING), exist_ok=True)
    with open(TEMPO_MAPPING, "wb") as f:
        pickle.dump(dict(tempo_by_date), f)

    return tempo_by_date

def map_to_tempo(row: pd.Series, tempo_by_date: dict):
    """Return (tempo, prev_hour_tempo) for a record"""
    target_dt = datetime(year=row['date'].year, month=row['date'].month, day=row['date'].day, hour=row['hour'])
    prev_dt = target_dt - pd.Timedelta(hours=1)

    tempo = None
    for curr_dt, fname in tempo_by_date.get(target_dt.date(), []):
        dt_diff = curr_dt - target_dt
        if pd.Timedelta(0) < dt_diff <= pd.Timedelta(minutes=MINS_FILTER):
            tempo = fname
            break

    prev_tempo = None
    for curr_dt, fname in tempo_by_date.get(prev_dt.date(), []):
        dt_diff = curr_dt - prev_dt
        if pd.Timedelta(0) < dt_diff <= pd.Timedelta(minutes=MINS_FILTER):
            prev_tempo = fname
            break

    return tempo, prev_tempo

def map_chunk(args):
    """Pickleable helper for parallelized record-tempo mapping"""
    chunk, tempo_by_date = args
    results = chunk.apply(lambda row: map_to_tempo(row, tempo_by_date), axis=1)
    chunk = chunk.copy()
    chunk["tempo"], chunk["prev_tempo"] = zip(*results)
    return chunk

def aggregate_units(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to complete groups and aggregate units to one row per facility per hour"""

    # keep only groups where all units are present
    units_per_facility = df.groupby("facilityId")["unitId"].nunique()
    df["_n_units"] = df.groupby(["facilityId", "date", "hour"])["unitId"].transform("nunique")
    df["_expected"] = df["facilityId"].map(units_per_facility)
    df = df[df["_n_units"] == df["_expected"]].drop(columns=["_n_units", "_expected"])

    # sum label across units per facility-hour
    df[LABEL_COL] = df.groupby(["facilityId", "date", "hour"])[LABEL_COL].transform("sum")

    # find representative unit per facility (closest to centroid) and filter to it
    unit_locs = df[["facilityId", "unitId", "lat", "lon"]].drop_duplicates(["facilityId", "unitId"])
    centroids = unit_locs.groupby("facilityId")[["lat", "lon"]].mean().rename(columns={"lat": "c_lat", "lon": "c_lon"})
    unit_locs = unit_locs.merge(centroids, on="facilityId")
    unit_locs["dist"] = np.sqrt((unit_locs["lat"] - unit_locs["c_lat"])**2 + (unit_locs["lon"] - unit_locs["c_lon"])**2)
    rep_units = unit_locs.loc[unit_locs.groupby("facilityId")["dist"].idxmin(), ["facilityId", "unitId"]]

    df = df.merge(rep_units, on=["facilityId", "unitId"]).reset_index(drop=True)
    df["num_adj_units"] = df["facilityId"].map(units_per_facility - 1)

    return df

def compute_prev_qtr_mass(df: pd.DataFrame) -> pd.DataFrame:
    """Attach prev_qtr_mass: prior-quarter total emissions at the same hour."""

    # build lookup: (facilityId, year, quarter, hour) -> total emissions
    full = (
        pd.read_csv(EMISSIONS_RECORDS_CSV, usecols=["date", "hour", "facilityId", "opTime", LABEL_COL], parse_dates=["date"])
        .query("opTime == 1.0")
        .dropna(subset=["facilityId", LABEL_COL])
    )
    full["year"] = full["date"].dt.year
    full["quarter"] = full["date"].dt.quarter
    lookup = full.groupby(["facilityId", "year", "quarter", "hour"])[LABEL_COL].sum().to_dict()

    # compute previous quarter for each row
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["prev_year"] = np.where(df["quarter"] == 1, df["year"] - 1, df["year"])
    df["prev_quarter"] = (df["quarter"] - 2) % 4 + 1

    df["prev_qtr_mass"] = df.apply(
        lambda row: lookup.get((row["facilityId"], row["prev_year"], row["prev_quarter"], row["hour"]), np.nan), axis=1
    )
    return (
        df.drop(columns=["year", "quarter", "prev_year", "prev_quarter"])
        .dropna(subset=["prev_qtr_mass"])
        .reset_index(drop=True)
    )

def compute_adj_plants(df: pd.DataFrame):
    """Count facilities from STRAT_INPUT_CSV within IMG_RANGE patch centered on each facility in df"""
    half_m = (IMG_RANGE / 2) * 1000

    query = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)
    all_plants = (
        pd.read_csv(STRAT_INPUT_CSV, usecols=["facilityId", "lat", "lon"])
        .drop_duplicates("facilityId")
        .reset_index(drop=True)
    )

    # project both sets to meters
    def to_meters(plants):
        gdf = gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants["lon"], plants["lat"]), crs="EPSG:4326").to_crs("EPSG:5070")
        return gdf.geometry.x.values, gdf.geometry.y.values

    qx, qy = to_meters(query)
    ax, ay = to_meters(all_plants)

    # count facilities in patch per query plant, subtract 1 to exclude self
    in_patch = (np.abs(qx[:, None] - ax[None, :]) < half_m) & (np.abs(qy[:, None] - ay[None, :]) < half_m)
    query["num_adj_plants"] = (in_patch.sum(axis=1) - 1).clip(min=0)

    return query[["facilityId", "num_adj_plants"]]

def cluster_plants(df: pd.DataFrame):
    """check for intersecting IMG_RANGE x IMG_RANGE bounding boxes before stratifying"""
    plants = df[["facilityId", "lat", "lon"]].drop_duplicates("facilityId").reset_index(drop=True)

    # convert from lat/lon (WGS84) format to meters format (NAD83 Conus Albers)
    gdf = gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants["lon"], plants["lat"]), crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:5070")
    x = np.array([geom.x for geom in gdf_proj.geometry])
    y = np.array([geom.y for geom in gdf_proj.geometry])

    # construct adjacency matrix to derive cluster values
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    adjacency = ((dx < IMG_RANGE * 1000) & (dy < IMG_RANGE * 1000)).astype(int)
    n_components, labels = connected_components(csr_matrix(adjacency))
    plants["cluster"] = labels
    return plants[["facilityId", "cluster"]]

def resample_uniform(df: pd.DataFrame, n: int, bins: int = 20):
    """Resample to uniform label distribution using oversampling"""
    np.random.seed(42)

    bin_labels = pd.cut(df[LABEL_COL], bins=bins, labels=False, include_lowest=True)
    active_bins = bin_labels.dropna().unique()
    n_per_bin = int(np.ceil(n / len(active_bins)))

    chosen = []
    for b in active_bins:
        idxs = np.where(bin_labels == b)[0]
        chosen.extend(np.random.choice(idxs, size=n_per_bin, replace=len(idxs) < n_per_bin).tolist())

    total, unique = len(chosen), len(set(chosen))
    print(f"resample_uniform: {total - unique}/{total} oversampled ({100*(total-unique)/total:.1f}%)")

    return df.iloc[chosen].reset_index(drop=True)

def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Visualize stratification in terms of geography and nox emissions"""

    us = gpd.read_file(COUNTRIES_URL)
    us = us[us.NAME == "United States of America"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    splits = [("Train", train), ("Val", val), ("Test", test)]

    # spatial visualization of stratification
    for ax, (label, split_df) in zip(axes[0], splits):
        us.plot(ax=ax, color="lightgray", edgecolor="black")
        ax.scatter(split_df["lon"], split_df["lat"], s=5, alpha=0.3)
        ax.set_title(f"{label} : n = {split_df.shape[0]}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-130, -65)
        ax.set_ylim(24, 50)

    # linear scale distribution
    for ax, (label, split_df) in zip(axes[1], splits):
        ax.hist(split_df[LABEL_COL], bins=50, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel(LABEL_COL)
        ax.set_ylabel("Count")

    plt.tight_layout()
    os.makedirs(os.path.dirname(STRAT_VIS_PNG), exist_ok=True)
    plt.savefig(STRAT_VIS_PNG, dpi=150)
    plt.close()

def main():
    df = pd.read_csv(STRAT_INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])

    # only include desired plant type
    df = df[df['primaryFuelInfo'] == PLANT_TYPE]

    # build TEMPO mapping and run parallelized record-tempo mapping
    tempo_by_date = build_tempo_mapping()
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), NUM_CORES) if len(idx) > 0]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(executor.map(map_chunk, [(c, tempo_by_date) for c in chunks]))
    df = pd.concat(results).reset_index(drop=True)
    df = df[df['tempo'].notna() & df['prev_tempo'].notna()].reset_index(drop=True)

    # filter to isolated plants before aggregation
    adj_map = compute_adj_plants(df)
    df = df.merge(adj_map, on="facilityId")
    df = df[df['num_adj_plants'] == 0].reset_index(drop=True)

    # aggregate multi-unit records per facility per hour and compute previous quarter mean emissions
    df = aggregate_units(df)
    df = compute_prev_qtr_mass(df)

    # remove outlier emissions values after facility-level aggreagation
    low = df[LABEL_COL].quantile(0.10)
    high = df[LABEL_COL].quantile(0.90)
    df = df[(df[LABEL_COL] >= low) & (df[LABEL_COL] <= high)]

    # reduce records down to desired sample size
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    # cluster plants and stratify into train/val/test
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")
    clusters = df[["cluster"]].drop_duplicates()

    train_c, temp_c = train_test_split(clusters, test_size=0.30, random_state=42)
    test_c, val_c = train_test_split(temp_c, test_size=0.50, random_state=42)
    train_samples = df[df["cluster"].isin(train_c["cluster"])].reset_index(drop=True)
    val_samples = df[df["cluster"].isin(val_c["cluster"])].reset_index(drop=True)
    test_samples = df[df["cluster"].isin(test_c["cluster"])].reset_index(drop=True)

    # resample splits to have uniform label distributions
    train = resample_uniform(train_samples, n=SPLIT_SIZES["train"]*3)
    val = resample_uniform(val_samples, n=SPLIT_SIZES["val"]*3)
    test = resample_uniform(test_samples, n=SPLIT_SIZES["test"]*3)

    # save splits and generate visualization
    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    plot_split_distributions(train, val, test)

if __name__ == "__main__":
    main()