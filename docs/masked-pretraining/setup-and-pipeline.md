# Masked pretraining dataset

The masked-pretraining pipeline builds a single-timestep, AOI-disjoint dataset
from TEMPO observations and aligned HRRR weather without using emissions values
as labels or inputs.

## Dataset targets

| Split | Requested records |
|---|---:|
| Train | 500,000 |
| Validation | 50,000 |
| Test | 50,000 |

Only AOIs from the delta model's training split are eligible, keeping its
validation and test geographies unseen during pretraining. Overlapping 72 km
AOIs stay in one pretraining split. The launcher assigns complete AOI groups
before any shard starts and writes immutable candidate manifests for the run.

## Validity cache

The persistent validity cache has two terminal entry types for each unique
AOI-scene key:

- `valid/<prefix>/<key>.npz` contains the original 24 by 24 NO2, NO2 validity,
  temperature, eastward-wind, and northward-wind rasters;
- `invalid/<prefix>/<key>.json` records that the scene failed complete coverage
  and stores no pretraining raster.

Source-read or transient weather failures are not negative-cached and can be
retried later. Existing delta-model TEMPO and weather cache entries are reused.
For an uncached scene, NO2 is regridded first. A scene below 100% finite NO2
coverage exits before HRRR alignment or validity-bundle storage.

Pass `--clear-cache` to clear both positive and negative validity entries. The
older `--refresh-cache` spelling remains an alias. Without that option, the
validity cache persists across runs. Candidate-discovery work, dataset shards,
and published dataframes are always cleared and rebuilt.

## Shards and finalization

A preparation job clears disposable outputs, performs the AOI split, and writes
candidate manifests on a compute node. A dependent discovery array checks the
validity cache and processes new candidates until each split reaches its target
or exhausts its candidates. Candidates sharing a TEMPO scan stay in one
discovery task so one granule read can serve multiple AOIs.

After discovery, a second array deterministically selects the requested records
and builds fresh dataset shards of at most 20,000 records by default. This
matches the delta-modeling pattern: each shard contains `records.csv` plus
per-record NPZ bundles beneath `record-rasters/<split>/`. Each disposable bundle
copies the clean original NO2, validity, temperature, and wind arrays from the
persistent validity cache and adds the masked NO2 and artificial mask. Change
the maximum with `--shard-size N`; `--max-parallel-shards` and
`--workers-per-shard` bound cluster and process concurrency.

The finalizer validates all fresh shards and publishes `train_df.csv`,
`val_df.csv`, and `test_df.csv` with dataset-root-relative raster-bundle paths.
If a split exhausts its candidate pool, it publishes the available records and
reports the shortfall without duplication. `mask_no2_raster` remains
intentionally unimplemented until EDA on delta-model validity masks defines the
masking distribution, so materialization currently fails loudly instead of
publishing an unchanged raster as a masked sample.

## Launch

From the repository root on a Savio login node:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"

python -u src/pretraining/masked-model/preprocessing/dataset_generation.py
```

To rebuild the validity cache:

```bash
python -u src/pretraining/masked-model/preprocessing/dataset_generation.py \
    --clear-cache
```

The launcher submits a preparation job, a dependent validity-discovery array,
a dependent shard-materialization array, and a final dependent finalizer through
`scripts/slurm/generate_masked_pretraining_dataset.sh`.
