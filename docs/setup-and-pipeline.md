# Setup and pipeline

This project predicts the direction of hourly power-plant NOx changes from
TEMPO imagery, EPA CAMPD records, HRRR weather, and plant attributes. ERA5 is
an optional weather benchmark.

## Quick path

| Step | Action |
|---|---|
| 1 | Load Python 3.11 and run `make setup` |
| 2 | Add CAMPD and Earthdata credentials to `.env` |
| 3 | Review paths and thresholds in `src/config.py` |
| 4 | Collect TEMPO, HRRR, emissions, and facility data |
| 5 | Build mappings, split AOIs, and generate rasters |
| 6 | Train with `python -m modeling.train` |

## Prerequisites

- `uv`
- The Savio `python/3.11.6-gcc-11.4.0` module
- An EPA CAMPD API key
- A NASA Earthdata account with access to TEMPO products
- A Copernicus Climate Data Store account when collecting the ERA5 benchmark
- Access to the configured Savio project paths, or matching path changes in
  `src/config.py`

## Initial Python environment setup

Install `uv` once on a login node:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the locked environment from the repository root:

```bash
module load python/3.11.6-gcc-11.4.0
make setup
```

`uv` creates `.venv`. To keep it in scratch, pass an explicit path and use that
path in Slurm jobs:

```bash
UV_CACHE_DIR=/global/scratch/users/$USER/uv-cache \
    make setup VENV=/global/scratch/users/$USER/no2-modeling-venv
```

Rerun setup when the environment is missing or the dependency files change.
Batch jobs activate the existing environment.

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

For ERA5 downloads, create `~/.cdsapirc` from the CDS account setup page:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-api-key
```

## Configuration

Review `src/config.py` before running the pipeline. It holds paths, date ranges,
filters, image parameters, and model settings.

The checked-in paths point at the `fc_nitrates` Savio project and one user's
home directory, so update user-specific entries such as `VIS_DIR` and
`RUNS_DIR`. Create the output directories before submitting jobs:

```bash
mkdir -p /global/home/users/<USERNAME>/no2-modeling/logs
mkdir -p /global/home/users/<USERNAME>/vis
```

## Run the pipeline

Run commands from the repository root. Load the module and environment once per
shell or job:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"
```

### 1. Choose the TEMPO collection

Set `TEMPO_VERSION` to `V03` or `V04` in `src/config.py`; `V04` is the default.
Preprocessing uses `TEMPO_LEVEL = "L2"`. Files land under:

```text
TEMPO/<version>/<level>/raw/<year>/<month>/
```

### 2. Download TEMPO and HRRR

```bash
python -u src/data-scraping/scrape_tempo.py
python -u src/data-scraping/scrape_hrrr.py
```

- The TEMPO scraper searches one month at a time, downloads in batches set by
  `DOWNLOAD_BATCH_SIZE` in `src/data-scraping/scrape_tempo.py`, and skips files already
  present at their final path. Rerunning a range is idempotent for completed
  files.
- The HRRR scraper saves one atomic GRIB2 subset per UTC hour under
  `HRRR/raw/<year>/<month>/<day>`, each holding 80 m U/V wind, 2 m temperature,
  and boundary-layer height from the hourly `f00` analysis.
- The Slurm HRRR collector passes `--overwrite` for a full replacement. Wait
  for every array task to succeed before using the archive, then regenerate
  the dataset once with `--refresh-weather` to replace aligned weather cache entries.

### 3. Download emissions and facility locations

```bash
python -u src/data-scraping/scrape_emissions.py
python -u src/data-scraping/scrape_locations.py
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
python -u -m preprocessing.tempo_mapping index --overwrite
python -u -m preprocessing.tempo_mapping observations \
    --task-id <TASK_ID> --task-count 32 --overwrite
python -u -m preprocessing.stratify_plants
python -u -m preprocessing.generate_dataset --shard-size 20000
```

Run the observation command once for each `TASK_ID` from 0 through 31. Start
those tasks after the index command succeeds. On Savio, use a dependent job
array such as `0-31%14`.

`preprocessing.tempo_mapping`:

- builds monthly granule-index Parquet files;
- assigns disjoint months to a dependent job array;
- writes daily AOI-observation shards;
- skips populated month directories without `--overwrite`;
- uses `NUM_CORES`, sourced from `SLURM_CPUS_PER_TASK`.

`preprocessing.stratify_plants`:

- reads the prebuilt TEMPO mapping;
- computes raw consecutive-hour AOI NOx changes;
- computes each AOI's absolute mean hourly NOx mass from the immediately
  preceding quarter as `prev_qtr_avg_nox`;
- stores raw and effective NOx changes scaled by the prior-quarter level;
- scores AOIs from full-history coal production share, NOx signal strength,
  event support, urban isolation, and complete-observation yield;
- retains the top `AOI_SELECTION_COUNT` AOIs, currently 100;
- assigns overlapping AOI clusters intact toward 70/15/15 record targets;
- removes records above each split's 95th percentile of aggregate AOI-hour NOx
  mass;
- deterministically random-samples up to 300,000 training and 75,000 validation
  and test records, retaining the full post-pruning pool when it is smaller;
- saves and emails a bar chart of each AOI's percentage of final records by
  split plus scaled hourly and effective NOx-delta histograms.

`preprocessing.generate_dataset`:

- resolves all configured TEMPO timesteps through one
  deduplicated persistent image-cache plan;
- writes NO2, NO2 validity, temperature, eastward wind, and northward wind as
  `T x 24 x 24` arrays per retained record;
- stores each unique AOI scan and aligned AOI-hour weather raster in persistent
  caches, grouping work so workers reuse each NetCDF or GRIB read;
- preserves each directly regridded NO2 scan and its independent mask;
- requires at least 95 percent finite NO2 coverage in every timestep;
- requires all nine cells around the highest-unit source cell to be valid in
  every timestep;
- keeps every successful record without label-based resampling;
- uses `NUM_CORES` workers, sourced from `SLURM_CPUS_PER_TASK` inside an
  allocation.

Every successful split-CSV row carries its relative `raster_bundle_path`,
selected hotspot cell, per-timestep full-raster and hotspot coverage, and
sequence-level cloud, quality, and retrieval-uncertainty summaries. Temperature
is stored only as a raster.

Running the splits:

- Run `python -u -m preprocessing.generate_dataset --shard-size N` on a login
  node. The CLI assigns at most `N` consecutive source records to each array
  task across train, validation, and test, then submits a dependent finalizer.
  By default, Slurm runs at most eight shard tasks concurrently and gives each
  shard eight CPUs and process workers. Use `--max-parallel-shards` and
  `--workers-per-shard` to override those limits.
- A launch is refused while dataset-generation or `train-no2` jobs are active.
  It deletes the existing shards and published metadata before submitting every
  planned shard, so the dataset is unavailable until finalization succeeds.
- Workers write rasters and candidate/failure CSVs directly under
  `shards/<split>/<shard>/`. Failed runs may leave partial shards; the next
  launch deletes the complete shard tree rather than resuming it.
- The finalizer runs only after every array task succeeds. It validates all
  source outcomes and referenced rasters, then atomically publishes every
  successful record with metadata whose raster paths point directly into the
  shards. It does not install or remove raster files.
- `--refresh-cache`, `--refresh-tempo`, and `--refresh-weather` explicitly clear
  the selected persistent caches once before fan-out. Otherwise caches survive
  fresh dataset runs and concurrent shards reuse their atomic entries.
- Worker logs report elapsed time, peak memory, and TEMPO/weather cache hits. The
  finalizer log reports finalizer time, peak memory, and launch-to-publication
  wall time for warm-cache benchmark records.
- Direct `python -u -m preprocessing.generate_dataset` remains available for a
  fresh monolithic local run. Pass `--split` to limit that run to one split.

### 5. Train and evaluate

```bash
python -u -m modeling.train
```

The trainer reads the consecutive NO2 scans and aligned validity-mask,
temperature, and geographic-wind sequences from each selected NPZ on demand.
It derives local mean solar hour from the stored UTC hour and AOI longitude. It
fits memory-bounded normalization statistics on the training split alone and
records clipped valid-pixel fractions by channel and split. The tabular MLP and
mask-aware raster ConvGRU train independently, with separate BCE losses,
optimizers, validation selection, and checkpoints. The ConvGRU receives no MLP
outputs or tabular features. The report compares both models on the same
records. See `docs/modeling.md` for the full contract.

### Regeneration

Dataset generation has no shard resume mode. Each launch discards the previous
shards and published metadata but reuses valid TEMPO-cache and wind-cache
entries. Do not regenerate or refresh caches while dataset generation or model
training is active. A worker failure leaves no published dataframe; launch the
complete run again after resolving the failure.

## Savio jobs

Each batch job should:

1. Use Bash with `set -euo pipefail`. Request the `fc_nitrates` account, one
   node, one task, and the stage resources listed below.
2. Write stdout and stderr to `logs/%x-%j.log` and `logs/%x-%j.err`; use `%A_%a`
   for arrays. Request `BEGIN`, `END`, and `FAIL` email notifications.
3. Change to the repository root, load `python/3.11.6-gcc-11.4.0`, activate
   `.venv`, and add both `src` and `src/delta-model` to `PYTHONPATH`.
4. Export `SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"`, then launch the stage
   command with `srun`.

Use these stage-specific allocations and commands:

| Stage | Savio request | Command and scheduler logic |
|---|---|---|
| TEMPO download | `savio4_htc`, `savio_normal`, 4 CPUs, 72 hours | Confirm `TEMPO_LEVEL="L2"` and `TEMPO_VERSION="V04"`, then run `python -u src/data-scraping/scrape_tempo.py` |
| HRRR download | `savio4_htc`, `savio_normal`, 4 CPUs per task, 48 hours | Split the date range across an array; pass each range to `src/data-scraping/scrape_hrrr.py` with `--workers "$SLURM_CPUS_PER_TASK" --overwrite` |
| Facility metadata | `savio4_htc`, `savio_normal`, 4 CPUs, 8 hours | Run `python -u src/data-scraping/scrape_locations.py` |
| TEMPO index | `savio4_htc`, `savio_normal`, 16 CPUs, 2 hours | Run `python -u -m preprocessing.tempo_mapping index`; require success before observation tasks start |
| TEMPO observations | `savio4_htc`, `savio_normal`, 4 CPUs per task, 8 hours | Use a `0-31%14` array and run `preprocessing.tempo_mapping observations --task-id "$SLURM_ARRAY_TASK_ID" --task-count 32` |
| Stratification | `savio4_htc`, `savio_normal`, `savio4_m512`, 16 CPUs, 2 hours | Run `python -u -m preprocessing.stratify_plants` |
| Dataset generation | `savio4_htc`, `savio_normal`, 8 concurrent tasks with 8 CPUs each, 12 hours | Submit a throttled shard array from the login node with `python -u -m preprocessing.generate_dataset --shard-size 20000`; Slurm starts another queued shard whenever a slot opens |
| Model training | `savio3_gpu`, `a40_gpu3_normal`, 8 CPUs, 1 A40, 2 hours | Run the training command below |

```bash
srun python -u -m modeling.train \
    --device cuda \
    --batch-size 128 \
    --epochs 300 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --prefetch-factor 2 \
    --seed 42 \
    --head-dim 128 \
    --dropout 0.30 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --gradient-clip-norm 5.0 \
    --scheduler-patience 10 \
    --scheduler-factor 0.50 \
    --early-stop-patience 25
```

Create or update environments on a login node. Batch jobs activate the existing
environment. Adjust the activation path when the environment lives in scratch.

Cluster partitions, QoS names, and account policies change over time. Verify
them against current Savio documentation before submitting long-running jobs.
