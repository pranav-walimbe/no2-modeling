"""Configuration values for collection, preprocessing, and modeling."""

import os
from datetime import date

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
EMISSIONS_START_DATE = date(2023, 8, 1)  # start of CAMPD hourly NOx pull
EMISSIONS_END_DATE = date.today()  # request through the latest date available from CAMPD
EMISSIONS_BASE_DIR = (
    "/global/scratch/projects/fc_nitrates/ddp/nox/nox_emissions"  # HPC output directory for emissions data
)
EMISSIONS_RECORDS_CSV = os.path.join(
    EMISSIONS_BASE_DIR, "nox_emissions_all.csv"
)  # raw operating and non-operating hourly records
FULL_DATA_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full.csv")  # emissions merged with plant metadata

# ============================================================================
# Power price scraping
# ============================================================================
POWER_PRICE_START_DATE = EMISSIONS_START_DATE
POWER_PRICE_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/power_prices"
POWER_PRICE_HOURLY_DIR = os.path.join(POWER_PRICE_BASE_DIR, "hourly")
POWER_PRICE_DERIVED_DIR = os.path.join(POWER_PRICE_BASE_DIR, "derived", "hourly_spreads")
POWER_PRICE_METADATA_DIR = os.path.join(POWER_PRICE_BASE_DIR, "metadata")
POWER_PRICE_TEMP_DIR = os.path.join(POWER_PRICE_BASE_DIR, "temporary")

# ============================================================================
# Stratification
# ============================================================================
SAMPLE_SIZE = 500000  # number of rows to run stratification on
MINS_FILTER = 60  # max time delta (minutes) between emissions record and TEMPO image
IMG_RANGE = 72  # spatial extent of extracted image patch (km)
MIN_TEMPO_DURATION = 58  # minimum TEMPO scan duration to accept (minutes)
PLANT_TYPE = "Coal"  # power plant fuel type filter
TEMPO_MAPPING = (
    "/global/scratch/projects/fc_nitrates/ddp/nox/TEMPO/tempo_mapping/tempo.pkl"  # cached (image, timestamp) lookup map
)
TEMPO_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/TEMPO/V03/tmp"  # raw downloaded TEMPO files
STRAT_INPUT_CSV = os.path.join(
    EMISSIONS_BASE_DIR, "nox_emissions_full2.csv"
)  # emissions CSV with quality and proximity filters applied
STRAT_BASE_DIR = (
    "/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"  # output directory for stratified splits
)
TRAIN_CSV = os.path.join(STRAT_BASE_DIR, "train.csv")  # train split metadata
VAL_CSV = os.path.join(STRAT_BASE_DIR, "val.csv")  # val split metadata
TEST_CSV = os.path.join(STRAT_BASE_DIR, "test.csv")  # test split metadata
VIS_DIR = "/global/home/users/pranavwalimbe/vis"  # output directory for visualizations
STRAT_VIS_PNG = os.path.join(VIS_DIR, "strat_vis.png")  # stratification distribution plot

# ============================================================================
# Wind data scraping
# ============================================================================
ERA5_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/era5"  # output directory for ERA5 wind reanalysis
WIND_START_MONTH = 8  # ERA5 download start month
WIND_START_YEAR = 2023  # ERA5 download start year
WIND_END_MONTH = 12  # ERA5 download end month
WIND_END_YEAR = 2025  # ERA5 download end year

# ============================================================================
# TEMPO data scraping
# ============================================================================
TEMPO_START_DATE = "2023-08-12 00:00:00"  # TEMPO download start (instrument available from Aug 2023)
TEMPO_END_DATE = "2025-09-17 00:00:00"  # TEMPO download end
TEMPO_VERSION = "V03"  # TEMPO product version

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
LABEL_COL = "noxMass"  # target variable (hourly NOx mass, lb/hr)
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
