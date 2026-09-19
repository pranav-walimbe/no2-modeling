# Masked-pretraining dataset

This pipeline builds a single-timestep dataset from TEMPO NO2 and aligned HRRR
weather. Emissions values are neither labels nor inputs.

## Data contract

| Split | Target records |
|---|---:|
| Train | 500,000 |
| Validation | 50,000 |
| Test | 50,000 |

Only delta-model training AOIs are eligible, which keeps downstream validation
and test geography unseen. Overlapping 72 km AOIs remain in the same
pretraining split. Preparation assigns AOI groups and writes fixed candidate
manifests before shard work begins.

## Validity cache

The persistent cache records each AOI-scene outcome:

- `valid/<prefix>/<key>.npz` stores complete 24 by 24 NO2, mask, temperature,
  and geographic wind rasters;
- `invalid/<prefix>/<key>.json` records incomplete NO2 coverage without storing
  a raster.

Transient source or weather errors remain retryable. Uncached scenes regrid NO2
first and stop before weather processing when any NO2 cell is missing. Existing
delta-model TEMPO and weather caches supply reusable intermediate data.

`--clear-cache` clears positive and negative validity entries;
`--refresh-cache` is an alias. Without either flag, the cache persists. Every run
rebuilds candidate results, shards, and published dataframes.

## Shards and finalization

The launcher submits four dependent stages:

1. Clear disposable outputs and write candidate manifests.
2. Discover valid scenes until each split reaches its target or runs out.
3. Select records deterministically and materialize fresh shards.
4. Validate shards and publish split CSVs.

Discovery groups candidates by TEMPO scan to reuse granule reads. Materialized
shards contain at most 20,000 records by default, with `records.csv` and NPZ
bundles under `record-rasters/<split>/`. Each bundle copies the clean cache
arrays and adds masked NO2 plus an artificial mask. Each record draws a masking
fraction uniformly from 1% through 10%. The sampler favors outer pixels and
mildly boosts pixels near each previous selection, producing exact-size masks
with small clusters. Masked NO2 uses a finite zero fill; the artificial mask
uses one for observed pixels and zero for masked pixels.

The finalizer publishes dataset-root-relative paths in `train_df.csv`,
`val_df.csv`, and `test_df.csv`. It reports a shortfall when a candidate pool is
exhausted and never duplicates records.

## Launch

From the repository root on a Savio login node:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"

python -u src/pretraining/masked-model/preprocessing/dataset_generation.py
```

Use `--clear-cache` to rebuild validity entries. Use `--shard-size`,
`--max-parallel-shards`, and `--workers-per-shard` to change resource limits.
The jobs run through
`scripts/slurm/generate_masked_pretraining_dataset.sh`.
