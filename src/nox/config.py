import os
import sys
from datetime import date
from dotenv import load_dotenv

# ============================================================================
# API
# ============================================================================
load_dotenv()
if not os.getenv("CAMPD_API_KEY"): # API key for CAMPD API (emissions data)
    sys.exit("ERROR: CAMPD_API_KEY not found in .env file.")
if not os.getenv("EARTHDATA_USERNAME") or not os.getenv("EARTHDATA_PASSWORD"): # API key for EarthData API (tempo data)
    sys.exit("ERROR: EARTHDATA_USERNAME or EARTHDATA_PASSWORD not found in .env file")
if not os.path.exists(os.path.expanduser("~/.cdsapirc")): # API key for CDS API (wind data)
    sys.exit("ERROR: ~/.cdsapirc not found. Configure CDS API credentials first")

# ============================================================================
# Emissions scraping
# ============================================================================
EMISSIONS_START_DATE = date(2023, 8, 1)
EMISSIONS_END_DATE = date(2025, 12, 31)
EMISSIONS_BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_emissions_1"
EMISSIONS_RECORDS_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_all.csv")
FULL_DATA_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full.csv")

# ============================================================================
# Stratification
# ============================================================================
SAMPLE_SIZE = 500000 # number of rows to run stratification on
MINS_FILTER = 60 # time window between emissions record and TEMPO image (minutes)
IMG_RANGE = 72  # km range captured by image
MIN_TEMPO_DURATION = 58 # min TEMPO image duration (minutes)
PLANT_TYPE = "Coal" # power plant type to filter for
TEMPO_MAPPING = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/tempo_mapping/tempo.pkl" # define cache for (image, timestamp) map
TEMPO_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V03/tmp"
STRAT_INPUT_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full2.csv")
STRAT_BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_powerplant_data"
TRAIN_CSV = os.path.join(STRAT_BASE_DIR, "train.csv")
VAL_CSV = os.path.join(STRAT_BASE_DIR, "val.csv")
TEST_CSV = os.path.join(STRAT_BASE_DIR, "test.csv")
VIS_DIR = "/global/home/users/pranavwalimbe/vis"
STRAT_VIS_PNG = os.path.join(VIS_DIR, "strat_vis.png")

# ============================================================================
# Wind data scraping
# ============================================================================
ERA5_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/era5"
WIND_START_MONTH = 8
WIND_START_YEAR = 2023
WIND_END_MONTH = 12
WIND_END_YEAR = 2025

# ============================================================================
# TEMPO data scraping
# ============================================================================
TEMPO_START_DATE = "2023-08-12 00:00:00"
TEMPO_END_DATE = "2025-09-17 00:00:00"
TEMPO_VERSION = "V03"

# ============================================================================
# Dataset generation
# ============================================================================
DATASET_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
DATASET_DF = os.path.join(DATASET_DIR, "dataframes")
IMG_SIZE = 48 # image size in pixels
PLUME_FILTER_PERCENTILE = 0.30 # drop samples with plume heuristic below this percentile
MAX_IMG_VAL = 1e17 # maximum concentration value in images
MIN_IMG_VAL = -2e16 # minimum concentration value in images
IMG_VAL_FILTER = 0.50 # max percent of image pixels >= MAX_IMG_VAL
MIN_PIXEL_CLOUD = 0.20 # tempo cloud fraction for pixel quality filtering
IMG_CLOUD_FILTER = 0.50 # percent of image pixels that meet MIN_PIXEL_CLOUD fraction
IMG_QA_FILTER = 0.80 # percent of images pixels with QA == 0
LABEL_COL = "noxMass" # emissions variable to predict
WIND_COLS = ["era5_u10", "era5_v10"] # wind vector data to include as model input
MIN_CITY_PROXIMITY = 100 # minimum distance to nearby major city (km)
SPLIT_SIZES = {"train": 18000, "val": 4000, "test": 4000} # size of each dataset split

# ============================================================================
# ML modeling
# ============================================================================
RUNS_DIR = "/global/home/users/pranavwalimbe/model_runs/"
BATCH_SIZE = 128
RESNET_HEAD_DIM = 128
LR = 1e-4
NUM_EPOCHS = 190
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.50
EARLY_STOP_PATIENCE = 20
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5

# ============================================================================
# Other
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK")) # number of cores for parallelized slurm jobs
COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip" # US map shapefile for visualizations