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
    fnames = [f for f in os.listdir(TEMPO_DIR) if f.endswith(".nc")]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:                                                                                                               
        results = list(executor.map(parse_tile, fnames))
    tempo_by_date = defaultdict(list)                                                                                                                                          
    for result in results:
        if result is not None:                                                                                                                                                 
            dt, fname = result
            tempo_by_date[dt.date()].append((dt, fname))                                                                                                                       
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
    """Sum LABEL_COL across units per (facilityId, date, hour) and keep unit closest to facility centroid"""                                                                   
                                                                                                                                                                                
    # total units per facility across entire df                                                                                                                                
    units_per_facility = df.groupby("facilityId")["unitId"].nunique()                                                                                                          
                                                                                                                                                                                
    # facility centroid as mean of all unit locations                                                                                                                          
    centroids = (                                                                                                                                                              
        df[["facilityId", "unitId", "lat", "lon"]]                                                                                                                             
        .drop_duplicates(["facilityId", "unitId"])                                                                                                                             
        .groupby("facilityId")[["lat", "lon"]]
        .mean()                                                                                                                                                                
        .rename(columns={"lat": "c_lat", "lon": "c_lon"})
    )                                                                                                                                                                          
    df = df.merge(centroids, on="facilityId")
    df["dist_to_center"] = np.sqrt((df["lat"] - df["c_lat"])**2 + (df["lon"] - df["c_lon"])**2)                                                                                
                                                                                                                                                                                
    # keep only groups where all units are present for that hour
    group_unit_counts = df.groupby(["facilityId", "date", "hour"])["unitId"].nunique()                                                                                         
    expected = group_unit_counts.index.get_level_values("facilityId").map(units_per_facility)                                                                                  
    complete_groups = group_unit_counts[group_unit_counts.values == expected.values].index                                                                                     
    df = df.set_index(["facilityId", "date", "hour"])                                                                                                                          
    df = df[df.index.isin(complete_groups)].reset_index()                                                                                                                                                                                                                                         
                                                                                                                                                                                
    # sum label per group                                                                                                                                                      
    label_sums = (                                                                                                                                                             
        df.groupby(["facilityId", "date", "hour"])[LABEL_COL]                                                                                                                  
        .sum()                                                                                                                                                                 
        .reset_index()                                                                                                                                                         
        .rename(columns={LABEL_COL: f"{LABEL_COL}_sum"})                                                                                                                       
    )                                   

    # keep unit closest to centroid per group, replace label with summed value                                                                                                 
    df = df.loc[df.groupby(["facilityId", "date", "hour"])["dist_to_center"].idxmin()].reset_index(drop=True)
    df = df.merge(label_sums, on=["facilityId", "date", "hour"])                                                                                                               
    df[LABEL_COL] = df[f"{LABEL_COL}_sum"]                                                                                                                                     
    df = df.drop(columns=["dist_to_center", "c_lat", "c_lon", f"{LABEL_COL}_sum"])
                                                                                                                                                                                
    return df.reset_index(drop=True)                                                                                                                                           
                                                                                                                                                                                
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
                                                                                                                                                                                
    # load all unique facilities and units using chunking for memory efficiency
    chunks = []                                                                                                                                                                
    for chunk in pd.read_csv(STRAT_INPUT_CSV, usecols=["facilityId", "unitId", "lat", "lon"], chunksize=1000000):                                                            
        chunks.append(chunk.dropna(subset=["lat", "lon"]).drop_duplicates(["facilityId", "unitId"]))                                                                           
    full_df = pd.concat(chunks).drop_duplicates(["facilityId", "unitId"]).reset_index(drop=True)                                                                            
    all_plants = full_df.drop_duplicates("facilityId").reset_index(drop=True)                                                                                                  
                                                                                                                                                                                
    # num_adj_plants: distinct facilities within box, excluding self                                                                                                           
    gdf_plants = gpd.GeoDataFrame(all_plants, geometry=gpd.points_from_xy(all_plants["lon"], all_plants["lat"]), crs="EPSG:4326").to_crs("EPSG:5070")                       
    joined_plants = gpd.sjoin(gdf_plants, gdf_q[["facilityId", "geometry"]], how="inner", predicate="within")                                                                  
    plant_counts = (joined_plants.groupby("facilityId_right").size() - 1).clip(lower=0).astype(int)                                                                           
    plants["num_adj_plants"] = plants["facilityId"].map(plant_counts).fillna(0).astype(int)                                                                                    
                                                                                                                                                                                
    # num_adj_units: distinct units per facility, excluding self                                                                                                               
    unit_counts = full_df.groupby("facilityId")["unitId"].nunique() - 1                                                                                                        
    plants["num_adj_units"] = plants["facilityId"].map(unit_counts).fillna(0).astype(int)                                                                                      
                                                                                                                                                                                
    return plants[["facilityId", "num_adj_plants", "num_adj_units"]]
                                                                                                                                                                                
def rebalance_splits(val: pd.DataFrame, test: pd.DataFrame, other_df: pd.DataFrame):                                                                                           
    """Rebalance val and test split to equal sizes using label-similarity weighted sampling"""
    target_size = max(len(val), len(test))                                                                                                                                     
                                                                                                                                                                                
    if len(other_df) < target_size:         
        sys.exit(f"ERROR: other_df (size={len(other_df)}) is smaller than target_size ({target_size}), cannot rebalance")                                                      
                
    def get_weights(target_df):                                                                                                                                                
        z_scores = ((other_df[LABEL_COL] - target_df[LABEL_COL].mean()) / (target_df[LABEL_COL].std() + 1e-10)).abs()
        weights = np.exp(-z_scores)                                                                                                                                            
        return weights / weights.sum()  
                                            
    if len(val) < target_size:                                                                                                                                                 
        w = get_weights(val).values
        chosen = np.random.choice(len(other_df), size=target_size - len(val), replace=False, p=w)                                                                              
        extra = other_df.iloc[chosen]
        other_df = other_df.drop(extra.index)                                                                                                                                  
        val = pd.concat([val, extra]).reset_index(drop=True)

    if len(test) < target_size:                                                                                                                                                
        w = get_weights(test).values
        chosen = np.random.choice(len(other_df), size=target_size - len(test), replace=False, p=w)                                                                             
        extra = other_df.iloc[chosen]   
        other_df = other_df.drop(extra.index)
        test = pd.concat([test, extra]).reset_index(drop=True)                                                                                                                 
                                        
    return val, test                                                                                                                                                           
                                                                                                                                                                                
def plot_split_distributions(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Visualize stratification in terms of geography and nox emissions"""                                                                                                     
                                            
    # get US outline for geographic visualization
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
                                                                                                                                                                                
    # histogram of noxMass distribution     
    for ax, (label, split_df) in zip(axes[1], splits):                                                                                                                         
        ax.hist(split_df["noxMass"], bins=50, alpha=0.7)
        ax.set_title(label)                                                                                                                                                    
        ax.set_xlabel("noxMass")
        ax.set_ylabel("Count")                                                                                                                                                 
                                        
    plt.tight_layout()
    os.makedirs(os.path.dirname(STRAT_VIS_PNG), exist_ok=True)                                                                                                                 
    plt.savefig(STRAT_VIS_PNG, dpi=150)
    plt.close()                                                                                                                                                                
                                        
def main():
    df = pd.read_csv(STRAT_INPUT_CSV)                                                                                                                                          
    df['date'] = pd.to_datetime(df['date'])
                                                                                                                                                                                
    # prefilter to only include coal + exclude outlier label values
    df = df[df['primaryFuelInfo'] == PLANT_TYPE]
    low, high = df[LABEL_COL].quantile(0.05), df[LABEL_COL].quantile(0.95)
    df = df[(df[LABEL_COL] >= low) & (df[LABEL_COL] <= high)].reset_index(drop=True)                                                                                           
                                                                                                                                                                                
    # build TEMPO mapping and run parallelized record-tempo mapping
    tempo_by_date = build_tempo_mapping()                                                                                                                                      
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), NUM_CORES) if len(idx) > 0]
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:                                                                                                               
        results = list(executor.map(map_chunk, [(c, tempo_by_date) for c in chunks]))
    df = pd.concat(results).reset_index(drop=True)                                                                                                                             
    df = df[df['tempo'].notna()].reset_index(drop=True)                                                                                                                        
                                        
    # aggregate multi-unit records per facility per hour                                                                                                                       
    if ADJ_UNITS:                                                                                                                                                              
        df = aggregate_units(df)                                                                                                                                               
                                                                                                                                                                                
    # apply strict lower and upper bound [TEMPORARY]                                                                                                                           
    df = df[(df[LABEL_COL] >= 100) & (df[LABEL_COL] <= 2000)].reset_index(drop=True)                                                                                             
                                                                                                                                                                                
    # compute adjacent power plant counts and filter BEFORE sampling
    adj_map = compute_adj_plants(df)    
    df = df.merge(adj_map, on="facilityId")                                                                                                                                    
    print(f"rows before adj filter: {len(df):,}")                                                                                                                              
    df = df[df['num_adj_plants'] == 0].reset_index(drop=True)                                                                                                                  
    print(f"rows after adj filter: {len(df):,}")                                                                                                                               
                                            
    # sample desired dataset size with inverse label frequency weights                                                                                                         
    counts, bin_edges = np.histogram(df[LABEL_COL], bins=50)
    bin_idx = np.clip(np.digitize(df[LABEL_COL], bin_edges[:-1]) - 1, 0, 49)                                                                                                   
    sample_weights = 1.0 / (counts[bin_idx] + 1e-10)
    sample_weights = sample_weights / sample_weights.sum()                                                                                                                     
                
    n = min(SAMPLE_SIZE, len(df))                                                                                                                                              
    sampled_idx = np.random.choice(len(df), size=n, replace=False, p=sample_weights)
    df = df.iloc[sampled_idx].reset_index(drop=True)    
                                                                                                                                                                                
    # cluster plants and compute per-cluster mean emission for stratification
    cluster_map = cluster_plants(df)
    df = df.merge(cluster_map, on="facilityId")                                                                                                                                
    cluster_emissions = df.groupby("cluster")[LABEL_COL].mean()
    cluster_df = pd.DataFrame({"cluster": df["cluster"].unique()})                                                                                                             
    cluster_df["emission_bin"] = pd.qcut(
        cluster_df["cluster"].map(cluster_emissions),
        q=2, labels=False, duplicates="drop"                                                                                                                                   
    )                                       
                                                                                                                                                                                
    # stratify clusters based on n=3 quantile binning on label
    strat_clusters, other_clusters = train_test_split(cluster_df, test_size=0.20, random_state=42)                                                                             
    train_c, temp_c = train_test_split(strat_clusters, test_size=0.40, stratify=strat_clusters["emission_bin"], random_state=42)
    val_c, test_c = train_test_split(temp_c, test_size=0.50, stratify=temp_c["emission_bin"], random_state=42)                                                               
                                            
    train = df[df["cluster"].isin(train_c["cluster"])]                                                                                                                      
    val = df[df["cluster"].isin(val_c["cluster"])]
    test = df[df["cluster"].isin(test_c["cluster"])]                                                                                                                       
    other_df = df[df["cluster"].isin(other_clusters["cluster"])]
                                                                                                                                                                                
    # rebalance val and test to equal sizes                                                                                                                                    
    val, test = rebalance_splits(val, test, other_df)
                                                                                                                                                                                
    # save splits and generate visualization                                                                                                                                   
    os.makedirs(STRAT_BASE_DIR, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)                                                                                                                                       
    val.to_csv(VAL_CSV, index=False)                                                                                                                                           
    test.to_csv(TEST_CSV, index=False)      
    plot_split_distributions(train, val, test)                                                                                                                                 
                
if __name__ == "__main__":                                                                                                                                                     
    main()