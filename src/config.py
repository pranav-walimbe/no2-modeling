"""Configuration values for collection, preprocessing, and modeling."""

import os
from datetime import date, datetime, timezone

from dotenv import load_dotenv

# ============================================================================
# API
# ============================================================================
load_dotenv()
CAMPD_API_KEY = os.getenv("CAMPD_API_KEY")
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

# ============================================================================
# Emissions scraping
# ============================================================================
EMISSIONS_START_DATE = date(2023, 8, 1)  # start of CAMPD hourly emissions pull
EMISSIONS_END_DATE = date.today()  # request through the latest date available from CAMPD
EMISSIONS_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/nox_emissions"  # HPC output directory for emissions
EMISSIONS_RECORDS_PARQUET = os.path.join(
    EMISSIONS_BASE_DIR, "nox_emissions_all.parquet"
)  # source hours in local standard time
FULL_DATA_PARQUET = os.path.join(
    EMISSIONS_BASE_DIR, "nox_emissions_full.parquet"
)  # enriched emissions with UTC date and hour

# ============================================================================
# TEMPO data scraping
# ============================================================================
TEMPO_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/TEMPO"
TEMPO_LEVEL = "L2"  # processing level used by collection and preprocessing
TEMPO_VERSION = "V04"  # supported values are V03 and V04
TEMPO_PRODUCT = f"TEMPO_NO2_{TEMPO_LEVEL}"
TEMPO_DIR = os.path.join(TEMPO_BASE_DIR, TEMPO_VERSION, TEMPO_LEVEL, "raw")
TEMPO_MAPPING_DIR = os.path.join(TEMPO_BASE_DIR, TEMPO_VERSION, TEMPO_LEVEL, "tempo_mapping")
TEMPO_GRANULE_MAPPING = os.path.join(TEMPO_MAPPING_DIR, "granules")
TEMPO_AOI_MAPPING = os.path.join(TEMPO_MAPPING_DIR, "aoi_observations")
TEMPO_START_DATE = "2023-08-02 00:00:00"  # beginning of the TEMPO science record
TEMPO_END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d 23:59:59")

TEMPO_CELL_OVERLAP_FLOOR_KM2 = 0.0  # retain every positive accepted footprint-cell overlap

# ============================================================================
# Stratification
# ============================================================================
TEMPO_MIN_DELTA_MINUTES = 50
TEMPO_MAX_DELTA_MINUTES = 70
IMG_RANGE = 72  # spatial extent of extracted image patch (km)
STRAT_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"  # stratified split output directory
TRAIN_RECORDS_CSV = os.path.join(STRAT_BASE_DIR, "train_records.csv")  # train split metadata
VAL_RECORDS_CSV = os.path.join(STRAT_BASE_DIR, "val_records.csv")  # validation split metadata
TEST_RECORDS_CSV = os.path.join(STRAT_BASE_DIR, "test_records.csv")  # test split metadata
VIS_DIR = "/global/home/users/pranavwalimbe/vis"  # output directory for visualizations

# ============================================================================
# Wind data scraping
# ============================================================================
ERA5_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/era5"  # output directory for ERA5 wind reanalysis
WIND_START_MONTH = 8  # ERA5 download start month
WIND_START_YEAR = 2023  # ERA5 download start year
WIND_END_MONTH = 12  # ERA5 download end month
WIND_END_YEAR = 2025  # ERA5 download end year

# ============================================================================
# HRRR data scraping
# ============================================================================
HRRR_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/HRRR"
HRRR_START_DATE = EMISSIONS_START_DATE
HRRR_END_DATE = EMISSIONS_END_DATE

# ============================================================================
# Dataset generation
# ============================================================================
DATASET_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/dataset"  # root output directory for final dataset
DATASET_RASTER_DIR = os.path.join(DATASET_DIR, "rasters")  # raster bundles for direct monolithic generation
DATASET_DF = os.path.join(DATASET_DIR, "dataframes")  # saved tabular features and labels
DATASET_TEMPO_CACHE_DIR = os.path.join(DATASET_DIR, "tempo-cache")  # persistent AOI-scan regridding cache
DATASET_WEATHER_CACHE_DIR = os.path.join(DATASET_DIR, "weather-cache")  # persistent aligned AOI-hour weather rasters
IMG_SIZE = 24  # image size in pixels (24x24)
MIN_PIXEL_CLOUD = 0.20  # TEMPO cloud fraction threshold per pixel
MIN_TIMESTEP_NO2_FINITE_FRACTION = 0.95  # inclusive coverage floor applied independently to every timestep
HOTSPOT_WINDOW_SIZE = 3  # odd source-centred square required to have complete NO2 support
MIN_HOTSPOT_NO2_FINITE_FRACTION = 1.0  # inclusive hotspot coverage floor applied to every timestep
LABEL_COL = "delta_nox_class"
EMA_DELTA_THRESHOLD = 100.0  # least absolute current-minus-previous effective NOx retained
TARGET_LABEL_MODE = "hard_hour"  # supported values: hard_hour, overlap_weighted
MIN_COVERAGE_PERCENT = 50.0  # least share of the emissions hour a delta window may cover
MIN_CITY_POPULATION = 500000  # metro population a populated place needs to count as a major city
MIN_MAJOR_CITY_DISTANCE_KM = 50.0  # minimum eligible plant distance from a major city in kilometers
# ============================================================================
# Modeling data contract
# ============================================================================
RUNS_DIR = "/global/home/users/pranavwalimbe/model_runs/"  # output directory for model checkpoints and results
SEQUENCE_TIMESTEPS = 5  # shared hourly raster and EMA history length
EMA_DECAY_TIMESCALE_HOURS = 2.0  # exponential e-folding time kept separate from the sequence length
MODEL_IMAGE_KEYS = ("current_no2", "delta_no2", "wind_u_80m_mps", "wind_v_80m_mps")
MODEL_MASK_KEYS = ("current_no2_mask", "delta_no2_mask")
MODEL_ROBUST_IMAGE_KEYS = ("current_no2", "delta_no2")
MODEL_IMAGE_CHANNELS = len(MODEL_IMAGE_KEYS)
MODEL_INPUT_CHANNELS = MODEL_IMAGE_CHANNELS + len(MODEL_MASK_KEYS)
MODEL_IMAGE_CLIP_ABS = 8.0  # bound rare raster extremes after train-only normalization
MODEL_RAW_FEATURES = (  # leakage-safe scalar inputs available to both tabular and fused models
    "num_coal_units",
    "num_ng_units",
    "total_nameplate_capacity_mw",
    "avg_heat_input",
    "avg_pwr_gen",
)
MODEL_CYCLIC_FEATURES = ("local_solar_hour", "day_of_year")  # each expands to sine and cosine

# ============================================================================
# Other
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))  # number of cores for parallelized jobs
