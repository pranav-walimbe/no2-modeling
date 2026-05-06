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

### 1. Create a `.env` file

```
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

- `CAMPD_API_KEY` — EPA Clean Air Markets Program Data API key for hourly NOx emissions records
- `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` — NASA EarthData credentials for downloading TEMPO NO2 V03 imagery

### 2. Configure CDS API credentials

ERA5 wind data requires a `~/.cdsapirc` file with your Copernicus Climate Data Store credentials:

```
url: https://cds.climate.copernicus.eu/api/v2
key: your-uid:your-api-key
```

## Configuration

All parameters live in `config.py`.

### Emissions Scraping

| Parameter | Default | Description |
|---|---|---|
| `EMISSIONS_START_DATE` | 2023-08-01 | Start of emissions record pull |
| `EMISSIONS_END_DATE` | 2025-12-31 | End of emissions record pull |
| `EMISSIONS_BASE_DIR` | Savio path | Output directory for emissions CSVs |

### Stratification

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_SIZE` | 500,000 | Rows to run stratification on |
| `MINS_FILTER` | 60 | Max time delta (minutes) between emissions record and TEMPO image |
| `IMG_RANGE` | 72 | Spatial extent of extracted image patch (km) |
| `MIN_TEMPO_DURATION` | 58 | Minimum TEMPO scan duration to accept (minutes) |
| `PLANT_TYPE` | `"Coal"` | Power plant fuel type filter |
| `MIN_CITY_PROXIMITY` | 100 | Minimum distance to nearest major city (km) |

### Wind Data (ERA5)

| Parameter | Default | Description |
|---|---|---|
| `WIND_START_MONTH` / `WIND_START_YEAR` | 8 / 2023 | ERA5 download start |
| `WIND_END_MONTH` / `WIND_END_YEAR` | 12 / 2025 | ERA5 download end |

### TEMPO Data

| Parameter | Default | Description |
|---|---|---|
| `TEMPO_START_DATE` | 2023-08-12 | TEMPO download start |
| `TEMPO_END_DATE` | 2025-09-17 | TEMPO download end |
| `TEMPO_VERSION` | `V03` | TEMPO product version |

### Dataset Generation

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 48 | Image size in pixels (48x48) |
| `PLUME_FILTER_PERCENTILE` | 0.30 | Drop samples below this plume heuristic percentile |
| `MAX_IMG_VAL` / `MIN_IMG_VAL` | 1e17 / -2e16 | NO2 concentration clipping bounds |
| `IMG_VAL_FILTER` | 0.50 | Max fraction of pixels at or above `MAX_IMG_VAL` |
| `MIN_PIXEL_CLOUD` | 0.20 | TEMPO cloud fraction threshold per pixel |
| `IMG_CLOUD_FILTER` | 0.50 | Max fraction of pixels exceeding cloud threshold |
| `IMG_QA_FILTER` | 0.80 | Min fraction of pixels with QA flag == 0 |
| `SPLIT_SIZES` | train: 18K, val: 4K, test: 4K | Target samples per split |
| `LABEL_COL` | `noxMass` | Target variable (hourly NOx mass in lb/hr) |

### ML Modeling

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 128 | Training batch size |
| `HEAD_DIM` | 128 | Regression head hidden dimension |
| `LR` | 1e-4 | Adam learning rate |
| `NUM_EPOCHS` | 300 | Max training epochs |
| `SCHEDULER_PATIENCE` | 10 | LR scheduler plateau patience |
| `SCHEDULER_FACTOR` | 0.50 | LR reduction factor on plateau |
| `EARLY_STOP_PATIENCE` | 25 | Early stopping patience |
| `WEIGHT_DECAY` | 1e-4 | Weight Decay |
| `DROPOUT` | 0.30 | Dropout rate in regression head |
| `KERNEL_SIZE` / `STRIDE` / `PADDING` | 3 / 1 / 1 | Conv layer geometry |

### Other

| Parameter | Description |
|---|---|
| `NUM_CORES` | Set automatically from `SLURM_CPUS_PER_TASK` for Savio jobs |
| `COUNTRIES_URL` | Natural Earth shapefile used for U.S. map visualizations |