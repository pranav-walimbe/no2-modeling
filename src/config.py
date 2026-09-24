"""Shared configuration for collection, delta modeling, and pretraining."""

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
# Emissions data scraping
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
# Delta-model stratification
# ============================================================================
TEMPO_MIN_DELTA_MINUTES = 40
TEMPO_MAX_DELTA_MINUTES = 70
IMG_RANGE = 72  # spatial extent of extracted image patch (km)
TARGET_LABEL_MODE = "overlap_weighted"  # interpolate CAMPD hours over each TEMPO interval
MIN_CITY_POPULATION = 500000  # metro population a populated place needs to count as a major city
STRAT_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"  # stratified split output directory
STRATIFICATION_EMA_CHANGE_THRESHOLD = 100.0  # raw EMA NOx-change boundary used to balance metadata splits
STRATIFICATION_AOI_FRACTION = 0.50  # highest coal-NOx-ranked share of coal-containing AOIs
TRAIN_RECORDS_CSV = os.path.join(STRAT_BASE_DIR, "train_records.csv")  # train split metadata
VAL_RECORDS_CSV = os.path.join(STRAT_BASE_DIR, "val_records.csv")  # validation split metadata
TEST_RECORDS_CSV = os.path.join(STRAT_BASE_DIR, "test_records.csv")  # test split metadata
VIS_DIR = "/global/home/users/pranavwalimbe/vis"  # output directory for visualizations

# ============================================================================
# ERA5 wind data scraping
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
# Delta-model dataset generation
# ============================================================================
DATASET_DIR = os.getenv(  # root for shared generation outputs
    "NO2_DATASET_DIR",
    "/global/scratch/projects/fc_nitrates/ddp/nox/dataset",
)
DATASET_RASTER_DIR = os.path.join(DATASET_DIR, "rasters")  # raster bundles for direct monolithic generation
DATASET_DF = os.getenv(  # saved tabular features and targets
    "NO2_DATASET_DF",
    os.path.join(DATASET_DIR, "dataframes"),
)
DATASET_TEMPO_CACHE_DIR = os.path.join(DATASET_DIR, "tempo-cache")  # persistent AOI-scan regridding cache
DATASET_WEATHER_CACHE_DIR = os.path.join(DATASET_DIR, "weather-cache")  # persistent aligned AOI-hour weather rasters
DATASET_MAX_PARALLEL_SHARDS = 8  # maximum concurrently running Slurm shard tasks
DATASET_WORKERS_PER_SHARD = 8  # process workers and CPUs assigned to each shard task
IMG_SIZE = 24  # image size in pixels (24x24)
MIN_PIXEL_CLOUD = 0.20  # TEMPO cloud fraction threshold per pixel
MIN_TIMESTEP_NO2_FINITE_FRACTION = 0.90  # inclusive coverage floor applied independently to every timestep
HOTSPOT_WINDOW_SIZE = 3  # odd source-centred square required to have complete NO2 support
MIN_HOTSPOT_NO2_FINITE_FRACTION = 1.0  # inclusive hotspot coverage floor applied to every timestep

# ============================================================================
# Delta-model training contract
# ============================================================================
RUNS_DIR = "/global/home/users/pranavwalimbe/model_runs/"  # output directory for model checkpoints and results
PRETRAINED_ENCODER_WEIGHTS = os.getenv(  # masked-model checkpoint for delta transfer and gap filling
    "PRETRAINED_ENCODER_WEIGHTS",
    "/global/home/users/pranavwalimbe/masked_model_runs/masked_no2_20260920_211751/checkpoints/best_masked_no2.pt",
)
SEQUENCE_TIMESTEPS = 5  # four label rasters followed by one post-label raster
LABEL_TIMESTEP_INDEX = 3  # zero-based final raster timestep consumed by the label EMA
EMA_HISTORY_TIMESTEPS = 4  # interpolated timestep values consumed by the label EMA
EMA_DECAY_TIMESCALE_HOURS = 2.0  # exponential e-folding time kept separate from the sequence length
MODEL_IMAGE_KEYS = ("no2", "temperature_2m_k", "wind_u_80m_mps", "wind_v_80m_mps")
MODEL_MASK_KEYS = ("no2_mask",)
MODEL_ROBUST_IMAGE_KEYS = ("no2",)
MODEL_IMAGE_CHANNELS = len(MODEL_IMAGE_KEYS)
MODEL_INPUT_CHANNELS = MODEL_IMAGE_CHANNELS + len(MODEL_MASK_KEYS)
MODEL_IMAGE_CLIP_ABS = 8.0  # bound rare raster extremes after train-only normalization
MODEL_TARGET_COL = "delta_category"  # three-class decrease, steady, or increase target
MODEL_CLASS_NAMES = ("decrease", "steady", "increase")
MODEL_RAW_FEATURES = (  # leakage-safe scalar inputs available to both tabular and fused models
    "major_city_dist",
    "num_units",
    "total_nameplate_capacity_mw",
    "avg_heat_input",
    "avg_pwr_gen",
)
MODEL_CYCLIC_FEATURES = ("local_solar_hour", "day_of_year")  # each expands to sine and cosine

# ============================================================================
# Masked pretraining
# ============================================================================
MASKED_PRETRAINING_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/masked_pretraining"
MASKED_PRETRAINING_VALIDITY_INDEX = os.path.join(MASKED_PRETRAINING_BASE_DIR, "validity-index.parquet")
MASKED_PRETRAINING_VALIDITY_UPDATES_DIR = os.path.join(MASKED_PRETRAINING_BASE_DIR, "validity-updates")
MASKED_PRETRAINING_WORK_DIR = os.path.join(MASKED_PRETRAINING_BASE_DIR, "work")
MASKED_PRETRAINING_SHARD_DIR = os.path.join(MASKED_PRETRAINING_BASE_DIR, "shards")
MASKED_PRETRAINING_DF_DIR = os.path.join(MASKED_PRETRAINING_BASE_DIR, "dataframes")
MASKED_PRETRAINING_TRAIN_RECORDS = 500_000
MASKED_PRETRAINING_VAL_RECORDS = 50_000
MASKED_PRETRAINING_TEST_RECORDS = 50_000
MASKED_PRETRAINING_NUM_SHARDS = 8
MASKED_PRETRAINING_WORKERS_PER_SHARD = 8
MASKED_PRETRAINING_SPLIT_SEED = 42

# ============================================================================
# Shared runtime
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))  # number of cores for parallelized jobs
