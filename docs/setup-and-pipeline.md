# Setup and pipeline guide

This project predicts hourly power-plant NOx emissions using TEMPO satellite
imagery, EPA CAMPD records, HRRR meteorology, and plant-level features. ERA5
remains available as a benchmark weather input.

## Prerequisites

- `uv`
- The Savio `python/3.11.6-gcc-11.4.0` module
- An EPA CAMPD API key
- A NASA Earthdata account with access to TEMPO products
- A Copernicus Climate Data Store account when collecting the ERA5 benchmark
- Access to the configured Savio project paths, or corresponding local path
  changes in `src/config.py`

## Initial Python environment setup

Install `uv` once on a login node using its official installer. The default
installation under `~/.local/bin` is visible from Savio compute nodes, although
jobs do not need `uv` after the environment has been created:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, load Savio's Python 3.11 module and create the locked
project environment:

```bash
module load python/3.11.6-gcc-11.4.0
make setup
```

`uv` creates `.venv` by default. If the environment is too large for your home
quota, place it in scratch instead and use the same path in Slurm jobs:

```bash
UV_CACHE_DIR=/global/scratch/users/$USER/uv-cache \
    make setup VENV=/global/scratch/users/$USER/no2-modeling-venv
```

This setup step is only needed when `.venv` does not exist or dependencies in
`pyproject.toml` or `uv.lock` change. Normal Slurm jobs activate the existing
environment directly and do not run `make setup` or `uv sync`.

Run static and syntax checks with:

```bash
make check
```

## Credentials

Create `.env` in the repository root:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

Do not commit this file. The variables are used for EPA CAMPD hourly NOx data
and NASA Earthdata TEMPO downloads.

Optional ERA5 benchmark downloads use the CDS API. Create `~/.cdsapirc` using
the credentials and format shown in your CDS account's API setup page. A
typical configuration is:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-api-key
```

## Configuration

Review `src/config.py` before running the pipeline. It contains:

- Savio input and output paths
- collection date ranges and filtering thresholds
- train, validation, and test sample sizes
- image parameters
- model and training hyperparameters

The checked-in paths point to the `fc_nitrates` Savio project and a specific
user's home directory. Update user-specific paths such as `VIS_DIR` and
`RUNS_DIR` for your account. Create the required output directories, including
job logs and visualizations, before submitting jobs:

```bash
mkdir -p /global/home/users/<USERNAME>/no2-modeling/logs
mkdir -p /global/home/users/<USERNAME>/vis
```

## Pipeline order

Run commands from the repository root so `.env` is discovered consistently.
Load the same Python module and activate the environment once per shell or job:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
```

1. Select the TEMPO collection in `src/config.py`. Preprocessing uses
   `TEMPO_LEVEL = "L2"`; `TEMPO_VERSION` accepts `V03` or `V04`. The default
   is V04. Files are stored under:

   ```text
   TEMPO/<version>/<level>/raw/<year>/<month>/
   ```

2. Download TEMPO and HRRR data:

   ```bash
   python -u -m collection.scrape_tempo
   python -u -m collection.scrape_hrrr
   ```

   The TEMPO scraper searches one month at a time, downloads in batches set by
   `DOWNLOAD_BATCH_SIZE` in `collection/scrape_tempo.py`, and skips files
   already present at their final path. Rerunning the same range is therefore
   idempotent for completed files.
   The HRRR scraper saves one atomic GRIB2 subset per UTC hour under
   `HRRR/raw/<year>/<month>/<day>`. Each file contains 10 m U/V wind, 2 m
   temperature, and boundary-layer height from the hourly `f00` analysis.

3. Download EPA emissions and facility locations:

   ```bash
   python -u -m collection.scrape_emissions
   python -u -m collection.scrape_locations
   ```

   Facility attributes are fetched in nationwide pages for each year rather
   than with one request per facility. The location stage stops without
   replacing its existing output if CAMPD requests fail or if enrichment would
   drop any hourly emissions rows.

4. Build the TEMPO mappings, partition plants, and generate model-ready datasets:

   ```bash
   scripts/slurm/submit_tempo_mapping.sh --overwrite
   # Wait for the observation job array to finish successfully.
   python -u -m preprocessing.stratify_plants
   python -u -m preprocessing.generate_dataset
   ```

   `preprocessing.tempo_mapping` owns TEMPO mapping construction. Its index job
   builds monthly granule-index Parquet files. A dependent job array owns
   disjoint months and builds daily AOI-observation shards containing stitched
   granule paths, mirror-step ranges, and observation times. Existing populated
   month directories are skipped. Pass `--overwrite` to rebuild the index and
   AOI mappings from scratch; the submit wrapper forwards it to both jobs. Each
   task uses `NUM_CORES`, which reads `SLURM_CPUS_PER_TASK`.

   `preprocessing.stratify_plants` only reads the prebuilt TEMPO mapping before
   matching observations and writing splits.
   The stratifier computes consecutive-hour AOI NOx mass changes and normalizes
   them with the previous completed quarter's median and MAD. It removes values
   below the 1st or above the 99th normalized-label percentile. Overlapping AOI
   clusters are assigned intact to the 60/20/20 train, validation, and test
   splits, which are capped at 100,000, 20,000, and 20,000 records.

   `preprocessing.generate_dataset` regrids the current and previous TEMPO
   scans onto the same AOI grid, requires finite NO2 in both scans, and writes
   one compressed delta-NO2 NPZ per retained record. It caches each unique AOI
   scan for the lifetime of the run. Every successful split-CSV row carries
   its relative `delta_no2_path`,
   plume score, paired cloud and quality means, and nearest-grid-point HRRR
   temperature, wind, and boundary-layer height. On Savio, the CLI defaults to
   `SLURM_CPUS_PER_TASK` workers through `NUM_CORES` and refuses to create more
   workers than that allocation.

   Train, validation, and test contain disjoint geographic clusters, so they
   have no useful cross-split AOI-scan cache sharing. A three-task Slurm array
   can therefore run them concurrently without redundant regridding. Array
   indices 0, 1, and 2 automatically select `train`, `val`, and `test`, so each
   task can invoke `python -u -m preprocessing.generate_dataset`. Outside an
   array, use `--split` for one split or omit it to process all splits.

5. Train and evaluate the model:

   ```bash
   python -u -m modeling.training
   ```

Each stage depends on the outputs of the preceding stage. The scripts currently
resume only where their individual implementation explicitly supports it; check
existing output files before rerunning a large collection or generation job.

## Savio jobs

The original workflow used these resources:

| Stage | Savio partition | Typical time | CPU/GPU |
| --- | --- | ---: | --- |
| TEMPO and HRRR download | `savio4_htc` | Range-dependent | 4 CPUs |
| EPA emissions download | `savio2_bigmem` | 6 hours | 1 CPU |
| Partition and dataset build | `savio4_htc` | 4 hours | 56 CPUs |
| Model training | `savio3_gpu` | 2 hours | 8 CPUs, 1 A40 GPU |

Start from `scripts/slurm/example_job.sh` and tailor the command and resources
for each stage. A representative job body is:

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

For a scratch environment, replace the activation line with the exact existing
environment path. Create or update environments on a login node only when
needed; the batch job only activates and runs them.

For training, use the GPU partition and add the appropriate GPU/QoS directives,
for example:

```bash
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
```

Cluster partitions, QoS names, and account policies can change; verify them
against current Savio documentation before submitting long-running jobs.
