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
SAMPLE_SIZE = 50000 # subset count of rows to run stratification on
MINS_FILTER = 60 # time window between emissions record and TEMPO image (minutes)
IMG_RANGE = 72  # km range captured by image
MIN_TEMPO_DURATION = 58 # min TEMPO image duration (minutes)
COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip" # US map shapefile

TEMPO_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V04"
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
TEMPO_START_DATE = "2025-09-18 00:00:00"
TEMPO_END_DATE = "2025-12-31 23:59:59"
TEMPO_VERSION = "V04"

# ============================================================================
# Dataset generation
# ============================================================================
DATASET_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
DATASET_DF = os.path.join(DATASET_DIR, "dataframes")
IMG_SIZE = 56 # image size in pixels
LABEL_FILTER_PERCENTILE = 0.10 # upper and lower filter bound for label values
MAX_IMG_VAL = 3e16 # maximum concentration value in images
MIN_IMG_VAL = -1e16 # minimum concentration value in images
IMG_VAL_FILTER = 0.50 # max percent of image pixels >= MAX_IMG_VAL
MIN_PIXEL_CLOUD = 0.20 # tempo cloud fraction for pixel quality filtering 
IMG_CLOUD_FILTER = 0.75 # percent of image pixels that meet MIN_PIXEL_CLOUD fraction
LABEL_COL = "noxMass" # emissions variable to predict
WIND_COLS = ["era5_u10", "era5_v10"] # wind variables to include in dataset
MIN_CITY_PROXIMITY = 100 # minimum distance to nearby major city (km)
SPLIT_SIZES = {"train": 12000, "val": 4000, "test": 4000} # size of each dataset split

# ============================================================================
# ML modeling
# ============================================================================
RUNS_DIR = "/global/home/users/pranavwalimbe/model_runs/"
BATCH_SIZE = 64
RESNET_HEAD_DIM = 256
LR = 1e-4
NUM_EPOCHS = 100
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.75
EARLY_STOP_PATIENCE = 5
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
WEIGHT_DECAY = 1e-4
USE_PRETRAINED = True

# ============================================================================
# System
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK"))