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

Pass `--refresh-cache` to clear both positive and negative validity entries.
The work area and masked-raster output are refreshed on every launch regardless
of that option.

## Shards and finalization

A preparation job clears disposable outputs, performs the AOI split, and writes
the candidate manifests on a compute node. Candidates sharing a TEMPO scan are
assigned to the same shard so one granule read can serve multiple AOIs. Shards
publish progress atomically and stop after their split collectively reaches its
requested count or their candidate pools are exhausted. In-flight batches can
produce a small surplus; finalization deduplicates and deterministically selects
the exact requested count.

The finalizer will eventually write a disposable masked-NO2 bundle for every
selected original and publish `train_df.csv`, `val_df.csv`, and `test_df.csv`
with paths to both bundles. `mask_no2_raster` is intentionally unimplemented
until EDA on delta-model validity masks defines the masking distribution. The
finalizer currently fails loudly at that boundary instead of publishing an
unmasked raster under a misleading masked path.

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
    --refresh-cache
```

The launcher submits a preparation job, its dependent shard array, and a final
dependent finalizer through
`scripts/slurm/generate_masked_pretraining_dataset.sh`.
