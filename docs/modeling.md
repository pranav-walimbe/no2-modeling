# Modeling

This document defines the binary baseline for classifying hourly power-plant
NOx changes from paired TEMPO observations.

## Prediction target

- Fit a symmetric cutoff at the training 20th percentile of
  `abs(delta_nox_mass)`.
- Freeze the cutoff for validation, test, and inference.
- Remove records inside the closed deadband.
- Label negative changes as 0 and positive changes as 1.
- Balance each split to equal label counts after raster QC.
- Preserve raw `delta_nox_mass` for reporting, never as an input.

## Inputs and leakage policy

Each sample has five aligned 48 by 48 raster channels and scalar context. The
scalar inputs are:

- coal and natural-gas unit counts;
- total generator nameplate capacity;
- previous-quarter average heat input and power generation;
- the historical NOx-change variability from the prior completed quarter;
- coincident HRRR 2 m temperature and boundary-layer height;
- sine/cosine encodings of UTC hour and day of year.

### Feature transformations

The pipeline transforms each scalar according to its distribution, then
standardizes every scalar with the training-split mean and standard deviation.
Validation, test, and inference reuse those statistics.

| Transform before standardization | Features | Reason |
|---|---|---|
| `log1p` | nameplate capacity, heat input, power generation, NOx-change scale, boundary-layer height | These nonnegative features have long right tails; compression limits the influence of extreme values and preserves zero. |
| None | coal and gas unit counts, temperature | Counts retain their discrete spacing, and temperature has a moderate range. |
| Sine and cosine | UTC hour, day of year | Circular encoding keeps adjacent boundary values close, such as hours 23 and 0. |

NO2 rasters use robust linear scaling. Wind rasters use ordinary
standardization. All image statistics come from training pixels only.

Coordinates, AOI IDs, current emissions, plume score, and raster-quality scores
are excluded. This prevents geographic memorization, direct target leakage,
and conditioning predictions on a diagnostic extracted from the response
image. Coverage remains available for sliced evaluation; the image mask
already gives the network spatial coverage information.

## Image representation and normalization

The model receives five channels:

1. paired-valid current NO2;
2. paired-valid delta NO2;
3. geographic eastward wind aligned from the native HRRR grid;
4. geographic northward wind aligned from the native HRRR grid; and
5. a binary mask whose value is one where both scans supplied accepted NO2.

For each NO2 channel:

- center at the median of valid training pixels;
- scale by `IQR / 1.349`; and
- clip to `[-8, 8]`.

Wind channels use their training-pixel mean and population standard deviation,
then apply the same clipping range. Validation, test, and inference reuse all
frozen training statistics. The mask stays binary and unstandardized.

The transform is

```text
normalized[channel] =
    (raster[channel] - train_center[channel]) / train_scale[channel]
```

Only finite pixels under the stored NO2 validity mask fit NO2 statistics.
Missing NO2 becomes normalized zero, while the mask distinguishes missing
support from a measured value at the training median.

The implementation uses temporary node-local arrays for exact quartiles rather
than building an in-memory pixel archive. `normalization_stats.json` stores the
frozen center, scale, and valid-pixel count for each channel. `run_config.json`
records the fraction of valid pixels clipped for every channel and split.

Robust linear scaling limits outlier influence without compressing the whole
NO2 distribution. Per-image normalization remains unsuitable because absolute
enhancement magnitude is part of the emissions signal.

## Network

The default network is a compact residual CNN plus an MLP scalar branch.
Residual stages reduce 48 by 48 images to a 6 by 6 feature map. A 3 by 3
adaptive average pool retains coarse plume location, while a global maximum
pool preserves localized enhancements that an average can dilute. Their fused
embedding is joined with the scalar embedding for one classification logit.

GroupNorm replaces BatchNorm throughout the image encoder. GroupNorm does not
depend on batch-level statistics and is stable if memory pressure forces small
batches; this is the central result of the original
[Group Normalization paper](https://arxiv.org/abs/1803.08494). LayerNorm is used
in the MLP projections. The older DenseNet alternative was removed because it
duplicated an obsolete input signature and was not selected by training.

We do not rotate or flip rasters in the baseline. HRRR grid-relative wind is
rotated to geographic east and north during alignment. Any later spatial
augmentation must transform the wind vector values along with the raster
coordinates and mask.

## Optimization and I/O

Training uses:

- AdamW;
- unweighted binary cross-entropy with logits;
- gradient clipping and mixed precision on CUDA;
- validation-loss scheduling; and
- early stopping.

`config.py` owns only the shared modeling data contract: paths, raster key and
channels, image clipping, and input-feature definitions. Training defaults live
in `modeling/train.py`, while architecture defaults live in `modeling/resnet.py`;
both are exposed through training CLI flags. This keeps preprocessing and
collection code independent of a particular training run while ensuring each
run records its resolved settings.

CUDA runs use automatic mixed precision for convolutions and linear layers,
with gradient scaling. PyTorch documents AMP as selecting lower precision for
eligible high-throughput operations while retaining float32 where its range is
needed; see the [PyTorch AMP documentation](https://docs.pytorch.org/docs/2.3/amp.html).

The map-style dataset lazily decompresses the selected per-record NPZ files.
DataLoader workers overlap that I/O with GPU computation, remain persistent
between epochs, and prefetch a bounded two batches per worker. Pinned memory is
enabled only for CUDA. These controls and their memory implications are
described in the [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/2.3/data.html).
The configured worker count is capped by the allocated CPUs; increasing it can
hurt shared-filesystem performance and multiply parent-process memory.

## Evaluation philosophy

Report accuracy, balanced accuracy, precision, recall, specificity, F1, ROC
AUC, and the full confusion matrix. Also record:

- class counts and natural pre-balancing prevalence;
- equal-count test slices by absolute raw delta-NOx magnitude;
- per-AOI metrics;
- row-level logits, probabilities, and predictions; and
- probability distributions and held-out AOI accuracy maps.

Model selection uses validation loss only. Test outputs describe the final
chosen system and must not drive normalization, architecture, thresholds, or
hyperparameters. Because the split is geographic, validation and test measure
transfer to non-overlapping plant regions rather than memorization of known
AOIs.

## Required comparisons

Before treating the CNN as scientifically useful, compare it with:

1. constant and natural-prevalence classifiers;
2. a tabular-only MLP with the same scalar features;
3. an image-only model;
4. the full image-plus-tabular model;
5. a delta-plus-mask versus current-plus-delta-plus-mask ablation; and
6. a mask ablation, while keeping the same eligible records.

Report all comparisons on the same frozen validation and test records. The
full model is justified only if image information improves held-out-AOI error
and the improvement is not confined to unusually clear or high-plume scenes.
The same trainer exposes these controlled ablations through `--inputs tabular`,
`--inputs image`, and the default `--inputs full`.

## Run artifacts

Each UTC-stamped directory under `RUNS_DIR` contains:

- `normalization_stats.json` with train-only preprocessing and the deadband;
- `run_config.json` with features, settings, clipped-pixel fractions, and
  parameter count;
- `checkpoints/best_model.pt` selected by validation loss;
- `results.json` with metrics and pre-balancing prevalence;
- one prediction CSV per split;
- loss, probability-distribution, and spatial-accuracy plots.

Run training on a compute node with

```bash
python -u -m modeling.train
```

Use `--workers`, `--batch-size`, and `--epochs` for allocation-specific
overrides, and `--inputs` for the controlled branch ablations. `--stats` may
reuse a compatible statistics JSON, but only when the training dataset and
configured feature order are unchanged.
