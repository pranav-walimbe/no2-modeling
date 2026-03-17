import os                                                                                                                                       
import sys                                                                                                                                      
from datetime import date                                                                                                                       
from dotenv import load_dotenv                                                                                                                  
                                                                                                                                                
# --- API ---                                                                                                                                   
load_dotenv()   
API_KEY       = os.getenv("CAMPD_API_KEY")                                                                                                      
if not API_KEY: 
    sys.exit("ERROR: CAMPD_API_KEY not found in .env file.")

# --- Emissions scraping ---
EMISSIONS_START_DATE  = date(2023, 8, 1)
EMISSIONS_END_DATE    = date(2025, 12, 31)
EMISSIONS_BASE_DIR    = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_emissions_1"
EMISSIONS_RECORDS_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_all.csv")
FULL_DATA_CSV         = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full.csv")

# --- Stratification ---
DATASET_SIZE       = 15000  # desired dataset size
MINS_FILTER        = 60     # time window between emissions record and TEMPO image (minutes)
IMG_SIZE           = 72     # image patch size (pixels)
MIN_TEMPO_DURATION = 58     # min TEMPO image duration (minutes)

TEMPO_DIR       = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V03/tmp"
STRAT_INPUT_CSV = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full2.csv")
STRAT_BASE_DIR  = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_powerplant_data"
TRAIN_CSV       = os.path.join(STRAT_BASE_DIR, "train.csv")
VAL_CSV         = os.path.join(STRAT_BASE_DIR, "val.csv")
TEST_CSV        = os.path.join(STRAT_BASE_DIR, "test.csv")
STRAT_VIS_PNG   = "/global/home/users/pranavwalimbe/vis/strat_vis.png"

# --- System ---
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK"))
# --- ERA5 extraction ---
ERA5_DIR     = "/global/scratch/projects/fc_nitrates/pranavwalimbe/era5"
ERA5_OUT_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_powerplant_data"

# --- ERA5 extraction ---
ERA5_DIR     = "/global/scratch/projects/fc_nitrates/pranavwalimbe/era5"
ERA5_OUT_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_powerplant_data"
