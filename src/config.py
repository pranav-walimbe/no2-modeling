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
EMISSIONS_BASE_DIR = (
    "/global/scratch/projects/fc_nitrates/ddp/nox/nox_emissions"  # HPC output directory for emissions data
)
EMISSIONS_RECORDS_PARQUET = os.path.join(
    EMISSIONS_BASE_DIR, "nox_emissions_all.parquet"
)  # raw operating and non-operating hourly records
FULL_DATA_PARQUET = os.path.join(
    EMISSIONS_BASE_DIR,
    "nox_emissions_full.parquet",
)  # emissions merged with plant metadata

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
TEMPO_DOWNLOAD_BATCH_SIZE = 100  # bounds downloader memory and retry scope
TEMPO_START_DATE = "2023-08-02 00:00:00"  # beginning of the TEMPO science record
TEMPO_END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d 23:59:59")

# ============================================================================
# Stratification
# ============================================================================
TEMPO_MIN_DELTA_MINUTES = 45
TEMPO_MAX_DELTA_MINUTES = 75
TEMPO_GEOLOCATION_STRIDE = 4
IMG_RANGE = 72  # spatial extent of extracted image patch (km)
STRAT_BASE_DIR = (
    "/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"  # output directory for stratified splits
)
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
HRRR_MAX_WORKERS = 4

# ============================================================================
# Dataset generation
# ============================================================================
DATASET_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/dataset"  # root output directory for final dataset
IMAGES_DIR = os.path.join(DATASET_DIR, "images")  # saved image arrays (.npy)
DATASET_DF = os.path.join(DATASET_DIR, "dataframes")  # saved tabular features and labels
IMG_SIZE = 48  # image size in pixels (48x48)
PLUME_FILTER_PERCENTILE = 0.30  # drop samples with plume heuristic below this percentile
MAX_IMG_VAL = 1e17  # upper NO2 concentration clipping bound
MIN_IMG_VAL = -2e16  # lower NO2 concentration clipping bound
IMG_VAL_FILTER = 0.50  # max fraction of pixels at or above MAX_IMG_VAL
MIN_PIXEL_CLOUD = 0.20  # TEMPO cloud fraction threshold per pixel
IMG_CLOUD_FILTER = 0.50  # max fraction of pixels exceeding MIN_PIXEL_CLOUD
IMG_QA_FILTER = 0.80  # min fraction of pixels with QA flag == 0
NOX_MASS_COL = "nox_mass"
DELTA_NOX_MASS_COL = "delta_nox_mass"
DELTA_NOX_MED_COL = "delta_nox_med"
DELTA_NOX_SCALE_COL = "delta_nox_scale"
LABEL_COL = "delta_nox_norm"
MIN_DELTA_HISTORY = 168
MIN_DELTA_SCALE_LB = 0.1
MAD_NORMAL_SCALE = 1.4826
MIN_CITY_PROXIMITY = 100  # minimum distance to nearest major city (km)
SPLIT_SIZES = {"train": 18000, "val": 4000, "test": 4000}  # target sample count per split

# ============================================================================
# ML modeling
# ============================================================================
RUNS_DIR = "/global/home/users/pranavwalimbe/model_runs/"  # output directory for model checkpoints and results
BATCH_SIZE = 128  # training batch size
HEAD_DIM = 128  # hidden dimension of MLP regression head
LR = 1e-4  # Adam learning rate
NUM_EPOCHS = 300  # maximum training epochs
SCHEDULER_PATIENCE = 10  # epochs without val improvement before LR reduction
SCHEDULER_FACTOR = 0.50  # LR multiplier applied on plateau
EARLY_STOP_PATIENCE = 25  # epochs without val improvement before early stopping
KERNEL_SIZE = 3  # conv kernel size
STRIDE = 1  # conv stride
PADDING = 1  # conv padding (maintains spatial dimensions with kernel=3)
WEIGHT_DECAY = 1e-4  # regularization strength
DROPOUT = 0.30  # dropout rate in regression head

# ============================================================================
# Other
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))  # number of cores for parallelized jobs
COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"  # country polygons for US map background
CITIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places_simple.zip"  # major cities shapefile for proximity filtering
