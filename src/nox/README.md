# NOx Emissions from TEMPO Satellite Imagery

Deep learning framework to quantify hourly NOx emissions from U.S. coal power plants
using TEMPO geostationary satellite imagery. Combines NASA TEMPO NO2 retrievals,
EPA CAMPD emissions records, and ERA5 wind reanalysis to train a ResNet/DenseNet
CNN that predicts plant-level NOx mass emissions (lb/hr).

## Project Structure

```
nox/
├── concentrations/
│   ├── scrape_locations.py       # fetch plant coordinates
│   └── scrape_nox_emissions.py   # pull hourly NOx records from CAMPD API
├── external_data/
│   ├── scrape_era5.py            # download ERA5 wind reanalysis
│   └── scrape_tempo.py           # download TEMPO NO2 imagery from EarthData
├── powerplants/
│   ├── partition_plants.py       # train/val/test plant split
│   └── generate_dataset.py       # build final image dataset with labels
├── model/
│   ├── dataset.py                # PyTorch dataset class
│   ├── resnet_model.py           # custom ResNet CNN
│   ├── densenet_model.py         # custom DenseNet CNN
│   ├── train.py                  # training loop
│   └── utils.py                  # shared utilities
└── config.py                     # all pipeline parameters
```

## Setup

### 1. Set up the virtual environment

Shell scripts to create and configure the virtual environment are provided in `setup/`. Run these once before submitting any jobs.

### 2. Create a `.env` file

```
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

- `CAMPD_API_KEY` — EPA Clean Air Markets Program Data API key for hourly NOx emissions records
- `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` — NASA EarthData credentials for downloading TEMPO NO2 V03 imagery

### 3. Configure CDS API credentials

ERA5 wind data requires a `~/.cdsapirc` file with your Copernicus Climate Data Store credentials:

```
url: https://cds.climate.copernicus.eu/api/v2
key: your-uid:your-api-key
```

### 4. Configure parameters

All pipeline parameters are defined in `config.py` including data paths, filtering thresholds, and model hyperparameters.

### 5. Create job logs + visualization directories

```bash
mkdir -p /global/home/users/<USERNAME>/job_logs
mkdir -p /global/home/users/<USERNAME>/vis
```

## Running the Pipeline

**Step 1 — Download external data (`savio2`):**

`scrape_external.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=scrape_external
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio2_bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/global/home/users/<USERNAME>/job_logs/scrape_external_%j.out
#SBATCH --error=/global/home/users/<USERNAME>/job_logs/scrape_external_%j.err

source /global/home/users/<USERNAME>/venv/bin/activate

cd /global/home/users/<USERNAME>/conus_co2/src/nox/external_data/
python -u scrape_tempo.py
python -u scrape_era5.py
```

**Step 2 — Scrape emissions (run after Step 1 completes, `savio2`):**

`scrape_emissions.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=scrape_emissions
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio2_bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=/global/home/users/<USERNAME>/job_logs/scrape_emissions_%j.out
#SBATCH --error=/global/home/users/<USERNAME>/job_logs/scrape_emissions_%j.err

source /global/home/users/<USERNAME>/venv/bin/activate

cd /global/home/users/<USERNAME>/conus_co2/src/nox/concentrations/
python -u scrape_nox_emissions.py
python -u scrape_locations.py
```

**Step 3 — Partition plants and build dataset (run after Step 2 completes, `savio4_htc`):**

`build_dataset.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00
#SBATCH --output=/global/home/users/<USERNAME>/job_logs/build_dataset_%j.out
#SBATCH --error=/global/home/users/<USERNAME>/job_logs/build_dataset_%j.err

source /global/home/users/<USERNAME>/venv/bin/activate

cd /global/home/users/<USERNAME>/conus_co2/src/nox/powerplants/
python -u partition_plants.py
python -u generate_dataset.py
```

**Step 4 — Train (run after Step 3 completes, `savio3_gpu`):**

`train.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR EMAIL>
#SBATCH --output=/global/home/users/<USERNAME>/job_logs/train_%j.out
#SBATCH --error=/global/home/users/<USERNAME>/job_logs/train_%j.err

source /global/home/users/<USERNAME>/venv/bin/activate

cd /global/home/users/<USERNAME>/conus_co2/src/nox/model/
python -u train.py
```