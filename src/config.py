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
EMISSIONS_RECORDS_PARQUET = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_all.parquet")  # hourly records, any status
FULL_DATA_PARQUET = os.path.join(EMISSIONS_BASE_DIR, "nox_emissions_full.parquet")  # emissions with plant metadata

# ============================================================================
# TEMPO data scraping
# ============================================================================
TEMPO_BASE_DIR = "/global/scratch/projects/fc_nitrates/ddp/nox/TEMPO"
TEMPO_LEVEL = "L2"  # processing level used by collection and preprocessing
TEMPO_VERSION = "V04"  # supported values are V03 and V04
TEMPO_PRODUCT = f"TEMPO_NO2_{TEMPO_LEVEL}"
TEMPO_DIR = os.path.join(TEMPO_BASE_DIR, TEMPO_VERSION, TEMPO_LEVEL, "raw")
TEMPO_MAPPING_DIR = os.path.join(TEMPO_BASE_DIR, TEMPO_VERSION, TEMPO_LEVEL, "tempo_mapping")
TEMPO_L3_DIR = os.path.join(TEMPO_BASE_DIR, TEMPO_VERSION)  # NASA Level 3 scan files used as a regridding reference
TEMPO_GRANULE_MAPPING = os.path.join(TEMPO_MAPPING_DIR, "granules")
TEMPO_AOI_MAPPING = os.path.join(TEMPO_MAPPING_DIR, "aoi_observations")
TEMPO_START_DATE = "2023-08-02 00:00:00"  # beginning of the TEMPO science record
TEMPO_END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d 23:59:59")

TEMPO_GOOD_QUALITY_FLAG = 0  # V04 main_data_quality_flag value accepted into the NO2 mean
TEMPO_CELL_OVERLAP_FLOOR_KM2 = 0.25  # removes edge slivers with about 1 percentage point paired-coverage loss
TEMPO_EFFECTIVE_SAMPLE_FLOOR = 0.0  # ESS 1.25 cut paired-cell coverage from 58 percent to 21 percent

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
DATASET_RASTER_DIR = os.path.join(DATASET_DIR, "rasters")  # per-record compressed delta NO2 rasters
DATASET_DF = os.path.join(DATASET_DIR, "dataframes")  # saved tabular features and labels
IMG_SIZE = 48  # image size in pixels (48x48)
MIN_PIXEL_CLOUD = 0.20  # TEMPO cloud fraction threshold per pixel
MIN_PAIRED_FINITE_FRACTION = 0.50  # least share of cells finite in both scans
CENTRAL_COVERAGE_WINDOW_SIZE = 8  # centred 12 km window; even because the 48-cell grid centre is an intersection
MIN_CENTRAL_FINITE_FRACTION = 0.50  # least paired-finite share in the central window
NOX_MASS_COL = "nox_mass"
DELTA_NOX_MASS_COL = "delta_nox_mass"
DELTA_NOX_SCALE_COL = "delta_nox_scale"
LABEL_COL = "delta_nox_norm"
MIN_DELTA_HISTORY = 168
DELTA_SCALE_LEVEL_FRACTION = 0.03  # share of an AOI's median hourly NOx added to its scale
MIN_COVERAGE_PERCENT = 50.0  # least share of the emissions hour a delta window may cover
MIN_CITY_PROXIMITY = 50  # minimum distance to nearest major city (km)
MIN_CITY_POPULATION = 500000  # metro population a populated place needs to count as a major city
OUTLIER_LOWER_QUANTILE = 0.01  # learn continuous-variable lower bounds from the training split
OUTLIER_UPPER_QUANTILE = 0.99  # learn continuous-variable upper bounds from the training split
OUTLIER_FILTER_COLUMNS = (  # excludes coordinates, counts, time, and already bounded coverage
    "avg_heat_input",
    "avg_pwr_gen",
    NOX_MASS_COL,
    DELTA_NOX_MASS_COL,
    DELTA_NOX_SCALE_COL,
    LABEL_COL,
)

TRAIN_SIZE = 12_000
VAL_SIZE = 4_000
TEST_SIZE = 4_000
STRATIFY_CANDIDATE_MULTIPLIER = 3  # overdraw for paired TEMPO coverage rejection
TRAIN_RECORDS_SIZE = TRAIN_SIZE * STRATIFY_CANDIDATE_MULTIPLIER
VAL_RECORDS_SIZE = VAL_SIZE * STRATIFY_CANDIDATE_MULTIPLIER
TEST_RECORDS_SIZE = TEST_SIZE * STRATIFY_CANDIDATE_MULTIPLIER

# ============================================================================
# Modeling data contract
# ============================================================================
RUNS_DIR = "/global/home/users/pranavwalimbe/model_runs/"  # output directory for model checkpoints and results
MODEL_IMAGE_KEY = "delta_no2"  # array stored in each per-record NPZ bundle
MODEL_IMAGE_CHANNELS = 2  # standardized delta NO2 plus a binary finite-data mask
MODEL_IMAGE_CLIP_Z = 8.0  # bound rare raster extremes after train-only standardization
MODEL_RAW_FEATURES = (  # columns available before the prediction hour or from coincident meteorology
    "num_coal_units",
    "num_ng_units",
    "avg_heat_input",
    "avg_pwr_gen",
    DELTA_NOX_SCALE_COL,
    "temperature_2m_k",
    "wind_u_10m_mps",
    "wind_v_10m_mps",
    "boundary_layer_height_m",
)
MODEL_LOG1P_FEATURES = (  # stabilize strongly right-skewed, nonnegative quantities before z-scoring
    "avg_heat_input",
    "avg_pwr_gen",
    DELTA_NOX_SCALE_COL,
    "boundary_layer_height_m",
)
MODEL_CYCLIC_FEATURES = ("hour", "day_of_year")  # each expands to sine and cosine

# ============================================================================
# Other
# ============================================================================
NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))  # number of cores for parallelized jobs
COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"  # country polygons for US map background
CITIES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip"  # populated places shapefile for proximity filtering
