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
import shapely
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
    """Return the first TEMPO filename captured within MINS_FILTER minutes before the record"""
    target_dt = datetime(year=row['date'].year, month=row['date'].month, day=row['date'].day, hour=row['hour'])
    for curr_dt, fname in tempo_by_date.get(row['date'].date(), []):
        dt_diff = curr_dt - target_dt
        if pd.Timedelta(0) < dt_diff <= pd.Timedelta(minutes=MINS_FILTER):
            return fname
    return None

def map_chunk(args):
    """pickleable helper function for parallelized record-tempo mapping"""
    chunk, tempo_by_date = args
    return chunk.assign(tempo=chunk.apply(lambda row: map_to_tempo(row, tempo_by_date), axis=1))

def aggregate_units(df: pd.DataFrame):
    """Aggregate units to facility level and attach prev_qtr_mass from full CSV"""

    # keep only groups where all units are present
    units_per_facility = df.groupby("facilityId")["unitId"].nunique()
    df["_n_units"] = df.groupby(["facilityId", "date", "hour"])["unitId"].transform("nunique")
    df["_expected"] = df["facilityId"].map(units_per_facility)
    df = df[df["_n_units"] == df["_expected"]].drop(columns=["_n_units", "_expected"])

    # sum label per group, keep unit closest to facility centroid
    df[LABEL_COL] = df.groupby(["facilityId", "date", "hour"])[LABEL_COL].transform("sum")
    centroids = (
        df[["facilityId", "unitId", "lat", "lon"]]
        .drop_duplicates(["facilityId", "unitId"])
        .groupby("facilityId")[["lat", "lon"]].mean()
        .rename(columns={"lat": "c_lat", "lon": "c_lon"})
    )
    df = df.merge(centroids, on="facilityId")
    df["dist_to_center"] = np.sqrt((df["lat"] - df["c_lat"])**2 + (df["lon"] - df["c_lon"])**2)
    df = (
        df.loc[df.groupby(["facilityId", "date", "hour"])["dist_to_center"].idxmin()]
        .drop(columns=["dist_to_center", "c_lat", "c_lon"])
        .reset_index(drop=True)
    )

    # attach prev_qtr_mass: prior-quarter mean of facility hourly totals at same hour, opTime == 1
    full = pd.read_csv(EMISSIONS_RECORDS_CSV, usecols=["date", "hour", "facilityId", "opTime", LABEL_COL],
                        parse_dates=["date"]).dropna(subset=["facilityId", LABEL_COL])
    full = full[full["opTime"] == 1.0]
    full["_date"] = full["date"].dt.date
    full["_hour"] = full["hour"].astype(np.int64)
    full["year"] = full["date"].dt.year
    full["quarter"] = full["date"].dt.quarter

    # sum across units per facility-hour
    facility_hourly = (
        full.groupby(["facilityId", "year", "quarter", "_date", "_hour"])[LABEL_COL]
        .sum().reset_index()
    )

    # quarterly mean per facility per hour
    qtr_mean = (
        facility_hourly.groupby(["facilityId", "year", "quarter", "_hour"])[LABEL_COL]
        .mean().reset_index()
        .rename(columns={LABEL_COL: "prev_qtr_mass"})
    )
    qtr_mean["target_year"] = np.where(qtr_mean["quarter"] == 1, qtr_mean["year"] + 1, qtr_mean["year"])
    qtr_mean["target_quarter"] = (qtr_mean["quarter"] % 4) + 1

    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df = df.merge(
        qtr_mean[["facilityId", "target_year", "target_quarter", "_hour", "prev_qtr_mass"]],
        left_on=["facilityId", "year", "quarter", "hour"],
        right_on=["facilityId", "target_year", "target_quarter", "_hour"],
        how="left",
    ).drop(columns=["year", "quarter", "target_year", "target_quarter", "_hour"])

    return df.dropna(subset=["prev_qtr_mass"]).reset_index(drop=True)

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

def compute_adj_plants(df: pd.DataFrame):
    """Count plants and units from STRAT_INPUT_CSV within an IMG_RANGE x IMG_RANGE box centred on each plant in df"""
    half_m = (IMG_RANGE / 2) * 1000

    plants = (
        df[["facilityId", "lat", "lon"]]
        .drop_duplicates("facilityId")
        .dropna(subset=["lat", "lon"])
        .reset_index(drop=True)
    )
    gdf_q = (
        gpd.GeoDataFrame(plants, geometry=gpd.points_from_xy(plants["lon"], plants["lat"]), crs="EPSG:4326")
        .to_crs("EPSG:5070")
    )
    x, y = gdf_q.geometry.x.values, gdf_q.geometry.y.values
    gdf_q["geometry"] = shapely.box(x - half_m, y - half_m, x + half_m, y + half_m)

    # load unique facilities and all facility-unit pairs for adj counts
    all_units = (
        pd.read_csv(STRAT_INPUT_CSV, usecols=["facilityId", "unitId", "lat", "lon"])
        .dropna(subset=["lat", "lon"])
        .drop_duplicates(["facilityId", "unitId"])
        .reset_index(drop=True)
    )
    all_plants = all_units.drop_duplicates("facilityId").reset_index(drop=True)

    # num_adj_plants: distinct facilities within box, excluding self
    gdf_plants = gpd.GeoDataFrame(all_plants, geometry=gpd.points_from_xy(all_plants["lon"], all_plants["lat"]), crs="EPSG:4326").to_crs("EPSG:5070")
    joined_plants = gpd.sjoin(gdf_plants, gdf_q[["facilityId", "geometry"]], how="inner", predicate="within")
    plant_counts = (joined_plants.groupby("facilityId_right").size() - 1).clip(lower=0).astype(int)
    plants["num_adj_plants"] = plants["facilityId"].map(plant_counts).fillna(0).astype(int)

    # num_adj_units: distinct units per facility, excluding self
    unit_counts = all_units.groupby("facilityId")["unitId"].nunique() - 1
    plants["num_adj_units"] = plants["facilityId"].map(unit_counts).fillna(0).astype(int)

    return plants[["facilityId", "num_adj_plants", "num_adj_units"]]

def resample_uniform(df: pd.DataFrame, n: int, bins: int = 10):
    """Resample to uniform label distribution using oversampling"""
    counts, bin_edges = np.histogram(df[LABEL_COL], bins=bins)
    bin_idx = np.clip(np.digitize(df[LABEL_COL], bin_edges[1:-1]), 0, len(counts) - 1)

    active_bins = np.where(counts > 0)[0]
    n_per_bin = int(np.ceil(n / len(active_bins)))

    rng = np.random.default_rng(42)
    chosen = []
    for b in active_bins:
        idxs = np.where(bin_idx == b)[0]
        chosen.extend(rng.choice(idxs, size=n_per_bin, replace=len(idxs) < n_per_bin).tolist())

    total = len(chosen)
    unique = len(set(chosen))
    print(f"resample_uniform: {total - unique}/{total} oversampled ({100*(total-unique)/total:.1f}%)")

    return df.iloc[chosen].reset_index(drop=True)

def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Visualize stratification in terms of geography and nox emissions"""

    us = gpd.read_file(COUNTRIES_URL)
    us = us[us.NAME == "United States of America"]
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
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

    # log scale distribution
    for ax, (label, split_df) in zip(axes[2], splits):
        ax.hist(np.log(split_df[LABEL_COL]), bins=50, alpha=0.7, color="orange")
        ax.set_title(label)
        ax.set_xlabel(f"log({LABEL_COL})")
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
    df = df[df['tempo'].notna()].reset_index(drop=True)

    # aggregate multi-unit records per facility per hour
    print(f"before agg units {df.shape[0]}")
    df = aggregate_units(df)
    print(f"after agg units {df.shape[0]}")

    # compute adjacent power plant counts and filter BEFORE sampling
    adj_map = compute_adj_plants(df)
    df = df.merge(adj_map, on="facilityId")
    print(f"before adj filter {df.shape[0]}")
    df = df[df['num_adj_plants'] == 0].reset_index(drop=True)
    print(f"after adj filter {df.shape[0]}")

    # remove outlier emissions values after facility-level aggreagation
    low = df[LABEL_COL].quantile(0.05)
    high = df[LABEL_COL].quantile(0.95)
    df = df[(df[LABEL_COL] >= low) & (df[LABEL_COL] <= high)]

    # reduce records down to desired sample size
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    # cluster plants and stratify into train/val/test
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")
    cluster_emissions = df.groupby("cluster")[LABEL_COL].mean()
    cluster_df = pd.DataFrame({"cluster": df["cluster"].unique()})
    cluster_df["emission_bin"] = pd.qcut(
        cluster_df["cluster"].map(cluster_emissions),
        q=3, labels=False, duplicates="drop"
    )

    train_c, temp_c = train_test_split(cluster_df, test_size=0.30, stratify=cluster_df["emission_bin"], random_state=42)
    val_c, test_c = train_test_split(temp_c, test_size=0.50, stratify=temp_c["emission_bin"], random_state=42)
    train_samples = df[df["cluster"].isin(train_c["cluster"])].reset_index(drop=True)
    val_samples = df[df["cluster"].isin(val_c["cluster"])].reset_index(drop=True)
    test_samples = df[df["cluster"].isin(test_c["cluster"])].reset_index(drop=True)

    # resample splits to have uniform label distributions
    train = resample_uniform(train_samples, n=SPLIT_SIZES["train"]*4)
    val = resample_uniform(val_samples, n=SPLIT_SIZES["val"]*4)
    test = resample_uniform(test_samples, n=SPLIT_SIZES["test"]*4)

    # save splits and generate visualization
    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    plot_split_distributions(train, val, test)

if __name__ == "__main__":
    main()