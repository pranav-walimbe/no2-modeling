# Setup and pipeline guide

This project predicts hourly power-plant NOx emissions using TEMPO satellite
imagery, EPA CAMPD records, ERA5 wind reanalysis, and plant-level features.

## Prerequisites

- `uv`
- The Savio `python/3.11.6-gcc-11.4.0` module
- An EPA CAMPD API key
- A NASA Earthdata account with access to TEMPO products
- A Copernicus Climate Data Store account with ERA5 API access
- Access to the configured Savio project paths, or corresponding local path
  changes in `src/no2_modeling/config.py`

## Python environment

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

ERA5 downloads use the CDS API. Create `~/.cdsapirc` using the credentials and
format shown in your CDS account's API setup page. A typical configuration is:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-api-key
```

## Configuration

Review `src/no2_modeling/config.py` before running the pipeline. It contains:

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
mkdir -p /global/home/users/<USERNAME>/job_logs
mkdir -p /global/home/users/<USERNAME>/vis
```

## Pipeline order

Run commands from the repository root so `.env` is discovered consistently.
Load the same Python module and activate the environment once per shell or job:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
```

1. Download TEMPO and ERA5 data:

   ```bash
   python -u -m no2_modeling.collection.scrape_tempo
   python -u -m no2_modeling.collection.scrape_era5
   ```

2. Download EPA emissions and facility locations:

   ```bash
   python -u -m no2_modeling.collection.scrape_emissions
   python -u -m no2_modeling.collection.scrape_locations
   ```

3. Partition plants and generate model-ready datasets:

   ```bash
   python -u -m no2_modeling.preprocessing.partition_plants
   python -u -m no2_modeling.preprocessing.generate_dataset
   ```

4. Train and evaluate the model:

   ```bash
   python -u -m no2_modeling.modeling.training
   ```

Each stage depends on the outputs of the preceding stage. The scripts currently
resume only where their individual implementation explicitly supports it; check
existing output files before rerunning a large collection or generation job.

## Savio jobs

The original workflow used these resources:

| Stage | Savio partition | Typical time | CPU/GPU |
| --- | --- | ---: | --- |
| TEMPO and ERA5 download | `savio2_bigmem` | 8 hours | 1 CPU |
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
#SBATCH --output=/global/home/users/<USERNAME>/job_logs/%x_%j.out
#SBATCH --error=/global/home/users/<USERNAME>/job_logs/%x_%j.err

cd /global/home/users/<USERNAME>/no2-modeling
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
python -u -m no2_modeling.collection.scrape_tempo
python -u -m no2_modeling.collection.scrape_era5
```

For a scratch environment, replace the activation line with the exact path used
for `make setup VENV=...`. Create or update the environment on a login node;
the batch job only activates and runs it.

For training, use the GPU partition and add the appropriate GPU/QoS directives,
for example:

```bash
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
```

Cluster partitions, QoS names, and account policies can change; verify them
against current Savio documentation before submitting long-running jobs.
