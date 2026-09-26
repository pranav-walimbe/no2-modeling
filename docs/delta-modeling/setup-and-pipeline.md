# Delta-model setup and pipeline

The delta pipeline combines TEMPO NO2, EPA CAMPD emissions, HRRR weather, and
plant attributes to classify emissions changes as `decrease`, `steady`, or
`increase`.

## Environment

You need Savio access, `uv`, the `python/3.11.6-gcc-11.4.0` module, an EPA
CAMPD API key, and a NASA Earthdata account with TEMPO access. Put credentials
in `.env`:

```dotenv
CAMPD_API_KEY=your_campd_api_key
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password
```

Create the locked environment and verify the repository:

```bash
module load python/3.11.6-gcc-11.4.0
make setup
make check
```

Run Python commands from the repository root with:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"
```

Review the storage paths, date bounds, and thresholds in `src/config.py` before
launching a long job. `TEMPO_VERSION` defaults to `V04`.

## 1. Collect source data

The collection scripts write TEMPO, HRRR, CAMPD, and plant metadata to the
configured shared paths:

```bash
python -u src/data-scraping/scrape_tempo.py
python -u src/data-scraping/scrape_hrrr.py
python -u src/data-scraping/scrape_emissions.py
python -u src/data-scraping/scrape_locations.py
```

The Slurm launchers under `scripts/slurm/` supply the production resource
requests. Facility enrichment converts CAMPD local standard time to UTC and
writes `emissions_hour_utc`. CAMPD standard offsets apply throughout the year.

## 2. Build TEMPO mappings and metadata splits

Submit the mapping index, wait for it to finish, then submit the observation
array and stratification job:

```bash
sbatch scripts/slurm/build_tempo_mapping_index.sh --overwrite
sbatch scripts/slurm/build_tempo_mapping_observations.sh --overwrite
sbatch scripts/slurm/stratify_plants.sh
```

The observation launcher runs 32 tasks. Stratification ranks facility-centered
AOIs by activity-conditioned median NOx, retains the highest-scoring half,
builds causal EMA targets, requires 20 records in every class per AOI, assigns
overlapping AOIs to the same geographic split, and balances the three classes
within each split. See [dataset_design.md](dataset_design.md).

## 3. Generate raster records

Launch generation from a login node:

```bash
./scripts/launch_dataset_generation.sh
```

The launcher uses 16,000 source records per shard by default and submits up to
eight concurrent workers plus a dependent finalizer. The launcher sorts each
split by target hour and location, then each worker processes the requested
batch size. For each batch, the worker copies its required TEMPO, HRRR, and
cache files to node-local `/tmp`, publishes generated cache files and raster
bundles through one writer thread, and removes the local batch directory.

Pass the required batch size and optionally override the shard size or selected
split:

```bash
./scripts/launch_dataset_generation.sh --batch-size 500 --shard-size 12000 -- --split train
```

Each launch replaces disposable shards and published metadata while retaining
the TEMPO and weather caches. Workers check only the cache paths required by
their current batch. Pass `--refresh-tempo`, `--refresh-weather`, or
`--refresh-cache` after the final `--` when a raster contract or source file
changes. Do not regenerate while a model job reads the dataset.

Workers require at least 90% finite NO2 coverage at each timestep and full
coverage in the 3 by 3 source hotspot. The finalizer publishes split CSVs only
after it validates all shard outcomes. See [regridding.md](regridding.md).

## 4. Provide a masked-model checkpoint

Delta training uses a masked NO2 checkpoint to fill missing NO2 pixels before
classifier training. It does not copy the masked encoder into the classifier. Set
`PRETRAINED_MASKED_MODEL_WEIGHTS` to the checkpoint path or use the default in
`src/config.py`.

The masked-pretraining workflow lives in
[masked-pretraining/setup-and-pipeline.md](../masked-pretraining/setup-and-pipeline.md).

## 5. Train and evaluate

Submit the maintained launcher:

```bash
sbatch scripts/slurm/train_model.sh
```

The job stages split metadata in job-local `/tmp`, fills missing NO2 into
temporary memory-mapped arrays, and trains two classifiers:

- a four-input seasonal MLP;
- a random-initialized ConvGRU fused with the frozen seasonal logits.

The launcher requests one A5000 GPU, four CPUs, and four hours on
`savio4_gpu` with `a5k_gpu4_normal`. It emails split and class accuracy, both
training curves, and the test AOI and raster-quality comparison after a
successful run. See [modeling.md](modeling.md) for the model contract.

## Current Savio launchers

| Stage | Launcher | Main request |
|---|---|---|
| TEMPO index | `build_tempo_mapping_index.sh` | 16 CPUs, 2 hours |
| TEMPO observations | `build_tempo_mapping_observations.sh` | `0-31%14`, 4 CPUs, 8 hours |
| Stratification | `stratify_plants.sh` | 16 CPUs, 30 minutes, high-memory node |
| Dataset shards | `launch_dataset_generation.sh` | Up to 8 tasks, 8 CPUs each, 12 hours |
| Masked pretraining | `train_masked_pretraining.sh` | 1 A5000, 4 CPUs, 8 hours |
| Delta classification | `train_model.sh` | 1 A5000, 4 CPUs, 4 hours |

Savio policies and availability can change. Check the requested account,
partition, and QoS before submission.
