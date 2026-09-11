# Setup and pipeline guide

This project predicts hourly power-plant NOx emissions from TEMPO satellite
imagery, EPA CAMPD records, HRRR meteorology, and plant-level features. ERA5
stays available as a benchmark weather input.

## Prerequisites

- `uv`
- The Savio `python/3.11.6-gcc-11.4.0` module
- An EPA CAMPD API key
- A NASA Earthdata account with access to TEMPO products
- A Copernicus Climate Data Store account when collecting the ERA5 benchmark
- Access to the configured Savio project paths, or matching path changes in
  `src/config.py`

## Initial Python environment setup

Install `uv` once on a login node with its official installer. The default
location under `~/.local/bin` is visible from compute nodes, and jobs stop
needing `uv` once the environment exists:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, load Savio's Python 3.11 module and create the locked
environment:

```bash
module load python/3.11.6-gcc-11.4.0
make setup
```

`uv` creates `.venv` by default. When the environment exceeds your home quota,
place it in scratch and use the same path in Slurm jobs:

```bash
UV_CACHE_DIR=/global/scratch/users/$USER/uv-cache \
    make setup VENV=/global/scratch/users/$USER/no2-modeling-venv
```

Run this step only when `.venv` is missing or `pyproject.toml` or `uv.lock`
changes. Slurm jobs activate the existing environment and run neither
`make setup` nor `uv sync`.

Static and syntax checks:

```bash
make check
```

## Credentials

Create `.env` in the repository root, and keep it out of commits:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

These cover EPA CAMPD hourly NOx data and NASA Earthdata TEMPO downloads.

Optional ERA5 benchmark downloads use the CDS API. Create `~/.cdsapirc` from
your CDS account's API setup page:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-api-key
```

## Configuration

Review `src/config.py` before running the pipeline. It holds:

- Savio input and output paths
- collection date ranges and filtering thresholds
- train, validation, and test sample sizes
- image parameters
- model and training hyperparameters

The checked-in paths point at the `fc_nitrates` Savio project and one user's
home directory, so update user-specific entries such as `VIS_DIR` and
`RUNS_DIR`. Create the output directories before submitting jobs:

```bash
mkdir -p /global/home/users/<USERNAME>/no2-modeling/logs
mkdir -p /global/home/users/<USERNAME>/vis
```

## Pipeline order

Run commands from the repository root so `.env` resolves consistently. Load the
module and activate the environment once per shell or job:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
```

### 1. Select the TEMPO collection

In `src/config.py`, preprocessing uses `TEMPO_LEVEL = "L2"`, and
`TEMPO_VERSION` accepts `V03` or `V04`. V04 is the default. Files land under:

```text
TEMPO/<version>/<level>/raw/<year>/<month>/
```

### 2. Download TEMPO and HRRR data

```bash
python -u -m collection.scrape_tempo
python -u -m collection.scrape_hrrr
```

- The TEMPO scraper searches one month at a time, downloads in batches set by
  `DOWNLOAD_BATCH_SIZE` in `collection/scrape_tempo.py`, and skips files already
  present at their final path. Rerunning a range is idempotent for completed
  files.
- The HRRR scraper saves one atomic GRIB2 subset per UTC hour under
  `HRRR/raw/<year>/<month>/<day>`, each holding 80 m U/V wind, 2 m temperature,
  and boundary-layer height from the hourly `f00` analysis.
- The Slurm HRRR collector passes `--overwrite` for a full replacement. Wait
  for every array task to succeed before using the archive, then regenerate
  the dataset once with `--refresh-wind` to replace aligned 10 m cache entries.

### 3. Download EPA emissions and facility locations

```bash
python -u -m collection.scrape_emissions
python -u -m collection.scrape_locations
```

Facility attributes:

- Fetched in nationwide pages per prediction year, not one request per facility.
- Each hourly record takes the latest facility and unit record whose attribute
  year stays at or below the prediction year.
- The location stage stops without replacing its output when CAMPD requests fail
  or enrichment would drop any hourly rows.
- Generator nameplate capacities are parsed, and conflicting facility-generator
  values stay out of the sum.
Time handling:

- CAMPD source `date` and `hour` fields use local standard time.
- Location enrichment resolves each facility's IANA timezone from its
  coordinates, preserves the source fields as `local_standard_date` and
  `local_standard_hour`, and writes an explicit `emissions_hour_utc`.
- Downstream `date` and `hour` come from that UTC timestamp.
- Standard offsets apply year-round, since the EPA reporting clock skips
  daylight-saving time.

### 4. Build mappings, partition plants, generate datasets

```bash
scripts/slurm/submit_tempo_mapping.sh --overwrite
# Wait for the observation job array to finish successfully.
python -u -m preprocessing.stratify_plants
python -u -m preprocessing.generate_dataset --shard-size 20000
```

`preprocessing.tempo_mapping`:

- builds monthly granule-index Parquet files;
- assigns disjoint months to a dependent job array;
- writes daily AOI-observation shards;
- skips populated month directories without `--overwrite`;
- uses `NUM_CORES`, sourced from `SLURM_CPUS_PER_TASK`.

`preprocessing.stratify_plants`:

- reads the prebuilt TEMPO mapping;
- normalizes consecutive-hour AOI NOx changes with the previous completed
  quarter's median and MAD;
- assigns overlapping AOI clusters intact to 60/20/20 splits;
- fits historical-variable percentile bounds on training only;
- selects lagged coal-output AOIs first, then the general pool by lagged total
  power, using no target or current-quarter output;
- emits three times each configured final size as raster candidates.

`preprocessing.generate_dataset`:

- resolves current, previous, and prior-14-day same-time TEMPO scans through one
  deduplicated persistent image-cache plan;
- writes five numeric rasters and three independent NO2 masks per retained
  record;
- stores each unique AOI scan and aligned AOI-hour wind raster in persistent
  caches, grouping work so workers reuse each NetCDF or GRIB read;
- requires greater than 95 percent current coverage and greater than 80 percent
  paired coverage for both delta rasters, preserving gaps in separate masks;
- builds a per-pixel 14-day EMA from the available historical dates, requiring
  five finite dates per cell and using a seven-day half-life;
- selects the requested size through deterministic AOI and temporal round-robin,
  or the largest exactly balanced subset when either class is short;
- uses `NUM_CORES` workers, sourced from `SLURM_CPUS_PER_TASK` inside an
  allocation.

Every successful split-CSV row carries its relative `delta_no2_path`, plume
score, paired cloud, quality and retrieval-uncertainty means, and
centre-interpolated HRRR temperature and boundary-layer height.

Running the splits:

- Run `python -u -m preprocessing.generate_dataset --shard-size N` on a login
  node. The CLI assigns at most `N` consecutive source records to each array
  task across train, validation, and test, then submits a dependent finalizer.
- A launch is refused while dataset-generation or `train-no2` jobs are active.
  It deletes the existing shards and published metadata before submitting every
  planned shard, so the dataset is unavailable until finalization succeeds.
- Workers write rasters and candidate/failure CSVs directly under
  `shards/<split>/<shard>/`. Failed runs may leave partial shards; the next
  launch deletes the complete shard tree rather than resuming it.
- The finalizer runs only after every array task succeeds. It validates all
  source outcomes and referenced rasters, applies exact class balance and
  global AOI round-robin selection, then atomically publishes metadata whose
  raster paths point directly into the shards. It does not install or remove
  raster files, so selected and unselected successful rasters remain in place.
- `--refresh-cache`, `--refresh-tempo`, and `--refresh-wind` explicitly clear
  the selected persistent caches once before fan-out. Otherwise caches survive
  fresh dataset runs and concurrent shards reuse their atomic entries.
- Worker logs report elapsed time, peak memory, and TEMPO/wind cache hits. The
  finalizer log reports finalizer time, peak memory, and launch-to-publication
  wall time for warm-cache benchmark records.
- Direct `python -u -m preprocessing.generate_dataset` remains available for a
  fresh monolithic local run. Pass `--split` to limit that run to one split.

### 5. Train and evaluate

```bash
python -u -m modeling.train
```

The trainer reads current NO2, hourly delta NO2, EMA delta NO2, wind, and three
validity masks from each selected NPZ on demand. It fits memory-bounded robust
NO2 normalization statistics on the training split alone and records clipped
valid-pixel fractions by channel and split. It then predicts whether raw
delta-NOx falls below or above zero outside the fixed deadband and reports
classification metrics. See `docs/modeling.md` for the full contract.

### Regeneration

Dataset generation has no shard resume mode. Each launch discards the previous
shards and published metadata but reuses valid TEMPO-cache and wind-cache
entries. Do not regenerate or refresh caches while dataset generation or model
training is active. A worker failure leaves no published dataframe; launch the
complete run again after resolving the failure.

## Savio jobs

The original workflow used these resources:

| Stage | Savio partition | Typical time | CPU/GPU |
| --- | --- | ---: | --- |
| TEMPO and HRRR download | `savio4_htc` | Range-dependent | 4 CPUs |
| EPA emissions download | `savio2_bigmem` | 6 hours | 1 CPU |
| Partition and dataset build | `savio4_htc` | 4 hours | 56 CPUs |
| Model training | `savio3_gpu` | 2 hours | 8 CPUs, 1 A40 GPU |

Start from `scripts/slurm/example_job.sh` and tailor the command and resources
per stage. A representative job body:

```bash
#!/bin/bash
#SBATCH --job-name=no2_pipeline
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio2_bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<EMAIL>
#SBATCH --output=/global/home/users/<USERNAME>/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/<USERNAME>/no2-modeling/logs/%x-%j.err

cd /global/home/users/<USERNAME>/no2-modeling
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
srun python -u -m collection.scrape_tempo
srun python -u -m collection.scrape_hrrr
```

Notes:

- For a scratch environment, swap the activation line for the exact environment
  path.
- Create or update environments on a login node. Batch jobs activate and run
  them.
- For training, use the GPU partition and add the GPU and QoS directives:

  ```bash
  #SBATCH --partition=savio3_gpu
  #SBATCH --qos=a40_gpu3_normal
  #SBATCH --gres=gpu:A40:1
  ```

Cluster partitions, QoS names, and account policies change over time. Verify
them against current Savio documentation before submitting long-running jobs.
