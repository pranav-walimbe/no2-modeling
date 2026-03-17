import os                                                                                                                                       
import sys                                                                                                                                      
from datetime import date                                                                                                                       
from dotenv import load_dotenv                                                                                                               
                                                                                                                                                
# ============================================================================
# API
# ============================================================================                                                                                                                                   
load_dotenv()   
API_KEY = os.getenv("CAMPD_API_KEY") # API key for government CAMPD API                                                                                                    
if not API_KEY: 
    sys.exit("ERROR: CAMPD_API_KEY not found in .env file.")

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
DATASET_SIZE = 35000 # desired dataset size
MINS_FILTER = 60 # time window between emissions record and TEMPO image (minutes)
IMG_RANGE = 72  # km range captured by image
MIN_TEMPO_DURATION = 58 # min TEMPO image duration (minutes)

TEMPO_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V03/tmp"
STRAT_INPUT_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full2.csv")
STRAT_BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_powerplant_data"
TRAIN_CSV = os.path.join(STRAT_BASE_DIR, "train.csv")
VAL_CSV = os.path.join(STRAT_BASE_DIR, "val.csv")
TEST_CSV = os.path.join(STRAT_BASE_DIR, "test.csv")
STRAT_VIS_PNG = "/global/home/users/pranavwalimbe/vis/strat_vis.png"

# ============================================================================
# Wind data scraping
# ============================================================================
ERA5_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/era5"
WIND_START_MONTH = 8
WIND_START_YEAR = 2023
WIND_END_MONTH = 9
WIND_END_YEAR = 2025
if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
    sys.exit("ERROR: ~/.cdsapirc not found. Configure CDS API credentials first")

# ============================================================================
# TEMPO data scraping
# ============================================================================
TEMPO_START_DATE = "2025-01-24 00:00:00"
TEMPO_END_DATE = "2025-09-16 23:59:59"
EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")
if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
    sys.exit("ERROR: EARTHDATA_USERNAME or EARTHDATA_PASSWORD not found in .env file")

# ============================================================================
# Dataset generation
# ============================================================================
DATASET_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
WIND_DIR = os.path.join(DATASET_DIR, "wind")
IMG_SIZE = 64 # image size in pixels
MIN_PIXEL_CLOUD = 0.2 # tempo cloud threshold for pixel quality filtering 
MIN_IMG_CLOUD = 0.75 # percent of image pixels with valid cloud fraction
LABEL_COL = "noxMass" # emissions variable to predict
WIND_COLS = ["era5_u10", "era5_v10"] # wind variables to include in dataset
MIN_CITY_PROXIMITY = 100 # minimum distance to nearby major city
SPLIT_SIZES = {"train": 6000, "val": 1000, "test": 1000} # split size mapping (undersampling)

# ============================================================================
# System
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK"))