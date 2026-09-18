# Delta-model setup and pipeline

The delta model combines TEMPO imagery, EPA CAMPD records, HRRR weather, and
plant attributes. ERA5 is an optional benchmark.

## Setup

Requirements:

- `uv` and Savio's `python/3.11.6-gcc-11.4.0` module;
- an EPA CAMPD API key;
- a NASA Earthdata account with TEMPO access;
- a Copernicus CDS account only for ERA5;
- access to the paths configured in `src/config.py`.

Create the locked environment:

```bash
module load python/3.11.6-gcc-11.4.0
make setup
```

To place it in scratch:

```bash
UV_CACHE_DIR=/global/scratch/users/$USER/uv-cache \
    make setup VENV=/global/scratch/users/$USER/no2-modeling-venv
```

Create `.env` in the repository root:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

ERA5 also requires `~/.cdsapirc`:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-api-key
```

Review `src/config.py`, especially Savio paths, dates, thresholds, `VIS_DIR`,
and `RUNS_DIR`. Run `make check` after setup.

## Shell environment

Run pipeline commands from the repository root:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"
```

Set `TEMPO_VERSION` to `V03` or `V04`; `V04` is the default. The pipeline uses
Level 2 data under `TEMPO/<version>/L2/raw/<year>/<month>/`.

## 1. Collect source data

```bash
python -u src/data-scraping/scrape_tempo.py
python -u src/data-scraping/scrape_hrrr.py
python -u src/data-scraping/scrape_emissions.py
python -u src/data-scraping/scrape_locations.py
```

The TEMPO collector skips completed files. HRRR output contains hourly `f00`
analyses for 80 m wind, 2 m temperature, and boundary-layer height. After
replacing HRRR files, regenerate the dataset with `--refresh-weather`.

Facility enrichment uses nationwide CAMPD pages and selects the latest
attribute year no later than each prediction year. It preserves CAMPD local
standard date and hour, resolves facility timezones, and writes
`emissions_hour_utc`. Standard offsets apply year-round because the EPA clock
does not use daylight-saving time. Failed or incomplete enrichment does not
replace the prior output.

## 2. Build mappings and splits

```bash
python -u -m preprocessing.tempo_mapping index --overwrite
python -u -m preprocessing.tempo_mapping observations \
    --task-id <TASK_ID> --task-count 32 --overwrite
python -u -m preprocessing.stratify_plants
```

Run observation tasks from 0 through 31 after indexing succeeds. The mapping
stage writes monthly granule indexes and daily AOI-observation shards.

Stratification computes consecutive-hour changes and prior-quarter baselines,
scores AOIs, keeps `AOI_SELECTION_COUNT`, assigns overlap clusters to
70/15/15 splits, removes each split's upper 5% NOx-mass tail, and samples up to
300,000/75,000/75,000 records. See [dataset_design.md](dataset_design.md).

## 3. Generate raster datasets

Launch the Slurm workflow from a login node:

```bash
python -u -m preprocessing.generate_dataset --shard-size 20000
```

The launcher submits a throttled shard array and dependent finalizer. Defaults
allow eight concurrent shards with eight CPUs and workers each. Override them
with `--max-parallel-shards` and `--workers-per-shard`.

Each launch clears old shards and published metadata but keeps persistent TEMPO
and weather caches. Use `--refresh-cache`, `--refresh-tempo`, or
`--refresh-weather` to clear selected caches before fan-out. Do not refresh or
regenerate while dataset generation or training is active.

Workers write five `T x 24 x 24` arrays per record: NO2, its mask, temperature,
and geographic wind U/V. They require 95% NO2 coverage per timestep and complete
3 by 3 source-hotspot coverage. The finalizer validates every source outcome
and raster before publishing relative paths. A failed worker prevents
publication; fix the cause and relaunch the complete workflow.

For a local monolithic run, call the module inside an allocation and use
`--split` when needed. See [regridding.md](regridding.md) for cache and raster
details.

## 4. Train and evaluate

```bash
python -u -m modeling.train
```

The trainer fits normalization on training pixels, loads NPZ files on demand,
and trains independent raster ConvGRU and tabular MLP models. See
[modeling.md](modeling.md) for inputs, leakage controls, architecture, and
evaluation.

## Savio allocations

Jobs should use `set -euo pipefail`, the `fc_nitrates` account, stage-specific
logs, `BEGIN,END,FAIL` email, the Python module and environment above, and
`srun`. Export `SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"`.

| Stage | Savio request | Command |
|---|---|---|
| TEMPO download | `savio4_htc`, `savio_normal`, 4 CPUs, 72 hours | `python -u src/data-scraping/scrape_tempo.py` |
| HRRR download | `savio4_htc`, `savio_normal`, 4 CPUs per task, 48 hours | Date-range array with `scrape_hrrr.py --workers "$SLURM_CPUS_PER_TASK" --overwrite` |
| Facility metadata | `savio4_htc`, `savio_normal`, 4 CPUs, 8 hours | `python -u src/data-scraping/scrape_locations.py` |
| TEMPO index | `savio4_htc`, `savio_normal`, 16 CPUs, 2 hours | `preprocessing.tempo_mapping index` |
| TEMPO observations | `savio4_htc`, `savio_normal`, 4 CPUs per task, 8 hours | `0-31%14` array with the observations command |
| Stratification | `savio4_htc`, `savio_normal`, `savio4_m512`, 16 CPUs, 2 hours | `python -u -m preprocessing.stratify_plants` |
| Dataset generation | `savio4_htc`, `savio_normal`, 8 concurrent tasks with 8 CPUs, 12 hours | Launch `preprocessing.generate_dataset` from the login node |
| Model training | `savio3_gpu`, `a40_gpu3_normal`, 8 CPUs, 1 A40, 2 hours | `python -u -m modeling.train --device cuda` |

Savio policies change. Verify partitions, QoS, and account limits before a long
run.

## Pretraining pipelines

Masked pretraining reuses delta-model TEMPO and weather caches but maintains its
own validity cache and AOI-disjoint splits. See
[`../masked-pretraining/setup-and-pipeline.md`](../masked-pretraining/setup-and-pipeline.md).
The next-raster pipeline is not implemented yet.
