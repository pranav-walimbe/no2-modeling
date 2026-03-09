import os

# paths for stratification script
VIS_PNG = "/global/home/users/pranavwalimbe/vis/strat_vis.png"
TEMPO_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/TEMPO/V03/tmp"
BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe"
INPUT_CSV = os.path.join(BASE_DIR, "nox_emissions_1", "nox_emissions_full2.csv")
OUT_DIR  = os.path.join(BASE_DIR, "nox_powerplant_data")
TRAIN_CSV = os.path.join(OUT_DIR, "train.csv")
TEST_CSV = os.path.join(OUT_DIR, "test.csv")
VAL_CSV = os.path.join(OUT_DIR, "val.csv")

# variables for stratification script
NUM_SAMPLES = 10000 # desired dataset size
MINS_FILTER = 60 # time window between emissions record and tempo image
PATCH_SIZE = 72 # desired dataset image size
MIN_DURATION_MINS = 58 # min threshold for tempo image duration
MIN_PIXEL_QUALITY_AVG = 0.6 # min threshold for % of pixels with quality = 0

# core count for multiprocessing 
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK"))