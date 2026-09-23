# Masked-pretraining dataset

This pipeline builds a single-timestep dataset from TEMPO NO2 and aligned HRRR
weather. Emissions values are neither labels nor inputs.

See [`dataset_design.md`](dataset_design.md) for the sample, split, and masking
contracts. See [`modeling.md`](modeling.md) for the reconstruction model and
training pipeline.

## Data contract

| Split | Target records |
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
  masked dataset bundle;
- invalid rows store terminal coverage failures;
- retryable source and weather errors are not persisted.

Run the standalone migration once before using the redesigned generator:

```bash
sbatch scripts/slurm/migrate_masked_pretraining_validity_cache.sh
```

After inspecting the published `validity-index.parquet`, rerun the migration
with `--overwrite --delete-legacy-files` to remove the redundant legacy NPZ and
JSON files. Deletion occurs only after every valid entry resolves to existing
TEMPO and weather cache files and the new index is atomically published.
The generation launcher refuses to submit jobs until this index exists.

## Shards and finalization

The launcher submits three dependent stages:

1. Snapshot the validity index, build the AOI-disjoint splits, order each split
   by UTC hour and AOI, and divide it into `N` contiguous candidate segments.
2. Let each discovery shard fill its deterministic split quota. It reuses
   indexed cache paths, performs targeted TEMPO and weather cache lookups for
   misses, and writes freshly masked bundles directly to its output directory.
3. Concatenate the small shard manifests and publish the split CSVs.

The default is eight discovery shards. Per-shard quotas sum exactly to the split
targets, so workers do not coordinate through a shared counter. Every new run
deletes stale work, dataframes, and dataset shards before creating fresh ones.
Completed shards remain in place after successful finalization.

Shard files are disposable and never serve as restart checkpoints. Discovery
publishes compact validity-index updates periodically. A later run merges those
updates before clearing stale shards, so completed validity checks remain
reusable even when the earlier dataset run failed.

The finalizer publishes dataset-root-relative paths in `train_df.csv`,
`val_df.csv`, and `test_df.csv`. It performs count validation, compacts validity
updates into the main index, and does not issue one filesystem lookup per
raster. Synthetic masks remain generated and stored during dataset generation.

## Launch

From the repository root on a Savio login node:

```bash
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/src/delta-model"

python -u src/pretraining/masked-model/preprocessing/dataset_generation.py
```

Use `--num-shards`, `--workers-per-shard`, and `--batch-size` to change resource
limits. The jobs run through
`scripts/slurm/generate_masked_pretraining_dataset.sh`.
