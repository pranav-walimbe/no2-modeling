import os                                   
import sys                              
from datetime import date
from dotenv import load_dotenv

# pull CAMPD API key from .env file
load_dotenv()
API_KEY = os.getenv("CAMPD_API_KEY")
if not API_KEY:
    sys.exit("ERROR: CAMPD_API_KEY not found in .env file.")

# emissions records scraping variables
STREAMING_URL = "https://api.epa.gov/easey/streaming-services/emissions/apportioned/hourly"
LOCATIONS_URL = "https://api.epa.gov/easey/facilities-mgmt/facilities/attributes"
START_DATE = date(2023, 8, 1)
END_DATE = date(2025, 12, 31)

# paths for storing data
BASE_DIR = "/global/scratch/projects/fc_nitrates/pranavwalimbe/nox_emissions_1"
EMISSIONS_RECORDS_CSV = os.path.join(BASE_DIR, "nox_emissions_all.csv")
FULL_DATA_CSV = os.path.join(BASE_DIR, "nox_emissions_full.csv")