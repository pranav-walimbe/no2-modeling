# Modeling

This document defines the first production modeling baseline for inferring an
hourly change in power-plant NOx emissions from paired TEMPO observations. It
replaces the legacy raw-NOx model and its dense two-image array contract.

## Prediction target

The network predicts the dataset's signed, robust target

```text
delta_nox_norm = asinh(delta_nox_mass / delta_nox_scale)
```

where `delta_nox_scale` is estimated only from the AOI's previous completed
quarter. The inverse used for physical-unit evaluation is

```text
predicted_delta_nox_mass = delta_nox_scale * sinh(predicted_delta_nox_norm)
```

The asinh transform is approximately linear near zero, preserves the direction
of a change, and compresses large positive and negative changes. The training
target is additionally standardized with the training-split mean and standard
deviation for optimization. Predictions are returned to `delta_nox_norm`
before metrics or plots are produced.

This is a change model, not an absolute-emissions model. Current `nox_mass`,
`delta_nox_mass`, and `delta_nox_norm` must never be input features.

## Inputs and leakage policy

Each sample has three aligned 48 by 48 raster channels and scalar context. The
scalar inputs are:

- coal and natural-gas unit counts;
- total generator nameplate capacity and its unit coverage rate;
- previous-quarter average heat input and power generation;
- the historical NOx-change scale used by the target definition;
- coincident HRRR 2 m temperature, 10 m U/V wind, and boundary-layer height;
- sine/cosine encodings of UTC hour and day of year.

### Feature transformations

The pipeline transforms each scalar according to its distribution, then
standardizes every scalar with the training-split mean and standard deviation.
Validation, test, and inference reuse those statistics.

| Transform before standardization | Features | Reason |
|---|---|---|
| `log1p` | nameplate capacity, heat input, power generation, NOx-change scale, boundary-layer height | These nonnegative features have long right tails; compression limits the influence of extreme values and preserves zero. |
| None | coal and gas unit counts, capacity coverage, temperature, U/V wind | Counts and coverage retain their bounded spacing, temperature has a moderate range, and wind components can be negative. |
| Sine and cosine | UTC hour, day of year | Circular encoding keeps adjacent boundary values close, such as hours 23 and 0. |

The signed target and both numeric NO2 rasters use `asinh`. The transform stays
close to linear near zero and compresses positive and negative tails.

Coordinates, AOI IDs, current emissions, plume score, and raster-quality scores
are excluded. This prevents geographic memorization, direct target leakage,
and conditioning predictions on a diagnostic extracted from the response
image. Coverage remains available for sliced evaluation; the image mask
already gives the network spatial coverage information.

## Image representation and normalization

The model receives three channels:

1. paired-valid current NO2 transformed with a robust signed asinh;
2. paired-valid delta NO2 transformed with a separate robust signed asinh; and
3. a binary mask whose value is one where both scans supplied accepted NO2.

Each numeric channel uses its own training-pixel mean, standard deviation, and
robust scale, then clips to plus or minus 8 standard deviations. The mask stays
binary and unstandardized.

The transform is

```text
transformed[channel] = asinh(raster[channel] / image_scale[channel])
normalized[channel] =
    (transformed[channel] - train_mean[channel]) / train_std[channel]
```

Each `image_scale` is the median of that channel's per-record median absolute
finite value. This record-balanced definition prevents high-coverage rasters
from dominating either scale. Asinh preserves sign, is approximately linear
for weak values, and becomes logarithmic in both tails.

After standardization, missing values in both numeric channels are filled with
zero. Zero is the transformed training mean, not a claim that physical NO2 was
zero, and the mask channel makes the distinction explicit. This
follows the general missing-image principle that the validity mask is
information rather than an implementation detail; specialized mask-updating
partial convolutions remain an experiment rather than part of this baseline.
See the original [partial-convolution paper](https://arxiv.org/abs/1804.07723).

Normalization uses two sequential training-bundle passes. The first derives
both robust scales; the second accumulates transformed finite-pixel means and
variance with a numerically stable combined-Welford update. Only one compressed
NPZ is open at a time, and the implementation never concatenates the roughly
28 million training pixels or builds another dense image archive. The JSON
statistics file stores the transform name, scale, mean, and standard deviation
and is used unchanged for validation, test, and later inference.

Asinh plus global standardization was selected over these options:

- Per-image normalization was rejected because absolute enhancement magnitude
  is part of the emissions signal.
- Treating NaN as an ordinary zero without a mask was rejected because scan
  coverage would be indistinguishable from measured zero change.
- Raw z-scoring is cheaper by one scan but lets extreme retrieval differences
  exert more influence on its mean, variance, and gradients.
- Signed `log1p` also compresses both tails but has a less direct smooth signed
  formulation than asinh around zero.
- Percentile min-max scaling depends strongly on chosen endpoints and can hide
  distribution shift by saturating all values outside the training range.

The 8-sigma bound is intentionally conservative. Tune it only on training and
validation data and record the retained-pixel distribution before changing it.

## Network

The default network is a compact residual CNN plus an MLP scalar branch.
Residual stages reduce 48 by 48 images to a 6 by 6 feature map. A 3 by 3
adaptive average pool retains coarse plume location, while a global maximum
pool preserves localized enhancements that an average can dilute. Their fused
embedding is joined with the scalar embedding for one regression output.

GroupNorm replaces BatchNorm throughout the image encoder. GroupNorm does not
depend on batch-level statistics and is stable if memory pressure forces small
batches; this is the central result of the original
[Group Normalization paper](https://arxiv.org/abs/1803.08494). LayerNorm is used
in the MLP projections. The older DenseNet alternative was removed because it
duplicated an obsolete input signature and was not selected by training.

We do not rotate or flip rasters in the baseline. Grid direction is physical,
wind U/V uses that direction, and arbitrary transforms would require exactly
consistent wind and mask transformations. Spatial augmentation is a valid
future experiment only with those transformations implemented together.

## Optimization and I/O

Training uses AdamW, weighted Huber loss in standardized-target units, gradient
clipping, validation-loss scheduling, and early stopping. Twenty train-derived
signed histogram bins keep negative and positive labels separate. Each bin gets
an inverse-frequency weight capped at 5. Validation and test loss remain
unweighted. Every run saves the bin edges, counts, and weights in
`loss_weights.json`.

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

The primary validation and test metrics are MAE, RMSE, bias, R-squared, and
Pearson correlation. They are reported in both normalized-target units and
physical NOx-mass-change units. MAPE is excluded because the target is signed
and can legitimately equal or cross zero.

The run also records:

- zero-change and training-mean baselines;
- equal-count test slices by absolute target magnitude;
- per-AOI physical MAE and bias;
- row-level predictions for later coverage, season, fuel, and geography slices;
- normalized and physical prediction plots, signed residuals, and held-out AOI
  spatial error.

Model selection uses validation loss only. Test outputs describe the final
chosen system and must not drive normalization, architecture, thresholds, or
hyperparameters. Because the split is geographic, validation and test measure
transfer to non-overlapping plant regions rather than memorization of known
AOIs.

## Required comparisons

Before treating the CNN as scientifically useful, compare it with:

1. zero-change and training-mean constants already emitted by the trainer;
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

- `normalization_stats.json` with the exact train-only preprocessing state;
- `run_config.json` with features, seed, optimization settings, and parameter
  count;
- `checkpoints/best_model.pt` selected by validation loss;
- aggregate `results.json` and one prediction CSV per split;
- loss, prediction, residual, and spatial-error plots.

Run training on a compute node with

```bash
python -u -m modeling.train
```

Use `--workers`, `--batch-size`, and `--epochs` for allocation-specific
overrides, and `--inputs` for the controlled branch ablations. `--stats` may
reuse a compatible statistics JSON, but only when the training dataset and
configured feature order are unchanged.
