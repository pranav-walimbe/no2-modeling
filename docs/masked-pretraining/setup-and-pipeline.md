# Masked-pretraining dataset

This pipeline builds a single-timestep dataset from TEMPO NO2 and aligned HRRR
weather. Emissions values are neither labels nor inputs.

See [`dataset_design.md`](dataset_design.md) for the sample, split, and masking
contracts. See [`modeling.md`](modeling.md) for the reconstruction model and
training pipeline.

## Data contract

| Split | Maximum records |
|---|---:|
| Train | 500,000 |
| Validation | 50,000 |
| Test | 50,000 |

Every AOI represented in the global emissions inventory and TEMPO mapping is
eligible. Overlapping 72 km AOIs remain in the same pretraining split.
Preparation assigns those AOI groups before ordering scenes by UTC hour and AOI
within each split and writing fixed candidate manifests.

## Validity index

The persistent Parquet index records each AOI-scene outcome without duplicating
raster arrays:

- valid rows store the TEMPO and weather cache paths needed to reconstruct a
  pretraining raster bundle;
- invalid rows store terminal coverage failures;
- retryable source and weather errors are not persisted.

The generator starts with an empty index when none exists and publishes newly
classified outcomes during finalization. Pass `--clear-cache` to discard the
existing validity index and rebuild it during the next generation run.

## Shards and finalization

The launcher submits three dependent stages:

1. Snapshot the validity index, build the AOI-disjoint splits, order each split
   by UTC hour and AOI, inventory the flat TEMPO and weather caches once, and
   divide each split into `N` contiguous candidate segments.
2. Let every discovery shard process its equal candidate segment. It reuses
   indexed cache paths and the cache-membership flags in its candidate manifest,
   then writes clean raster bundles directly to its output directory. Each shard
   atomically publishes its progress. Once their combined valid-record count
   reaches the split target, a shared completion marker tells every shard to
   stop at its next batch boundary. Targeted filesystem checks are limited to
   inventory misses, including files created after preparation.
3. Concatenate the small shard manifests, deterministically trim batch overshoot,
   and publish the split CSVs. If all candidate segments are exhausted below a
   target, publish every valid record found and report the shortfall.

The default is eight discovery shards. Every new run atomically moves stale
dataset shards into a cleanup area, creates an empty live shard directory, and
submits an independent cleanup job with at most eight deletion workers. Stale
work and dataframes are cleared during preparation. Completed shards remain in
place after successful finalization.

Shard files are disposable and never serve as restart checkpoints. Discovery
publishes compact validity-index updates periodically. A later run merges those
updates before clearing stale shards, so completed validity checks remain
reusable even when the earlier dataset run failed.

The finalizer publishes dataset-root-relative paths in `train_df.csv`,
`val_df.csv`, and `test_df.csv`. It records published counts and target
shortfalls, compacts validity updates into the main index, and does not issue
one filesystem lookup per raster. Synthetic masks are generated
deterministically by the model loader rather than stored during dataset
generation.

## Launch

From the repository root on a Savio login node:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"

python -u src/pretraining/masked-model/preprocessing/dataset_generation.py
```

Use `--clear-cache` to rebuild validity entries. Use `--num-shards`,
`--workers-per-shard`, and `--batch-size` to change resource limits. The jobs
run through `scripts/slurm/generate_masked_pretraining_dataset.sh`.
