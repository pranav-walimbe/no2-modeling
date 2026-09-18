# Modeling

The delta-model baseline classifies hourly power-plant NOx changes from causal
TEMPO sequences and aligned HRRR weather.

## Baseline

| Component | Choice |
|---|---|
| Target | Sign of effective hourly NOx change outside a 100 lb deadband |
| Raster input | `T` NO2 scans, validity masks, temperature, and geographic wind U/V |
| Scalar input | Plant attributes, prior-quarter activity, and time |
| Split | Geographic AOI clusters, approximately 70/15/15 |
| Raster model | Shared mask-aware encoder and ConvGRU |
| Comparison | Independent tabular MLP |
| Selection | Validation log loss |

Apply the same symmetric 100 lb cutoff to every split and inference. Remove
records inside the closed deadband, label decreases as 0 and increases as 1,
then balance each split. Retain the continuous change for reporting only.

## Inputs and leakage controls

Each sample contains `T` aligned 24 by 24 raster timesteps. Scalar inputs are
coal and gas unit counts, total nameplate capacity, previous-quarter heat input
and power generation, local solar hour, and day of year.

| Excluded input | Reason |
|---|---|
| Coordinates and AOI IDs | Prevent geographic memorization; longitude only derives solar hour |
| Current emissions | Direct target leakage |
| Previous-quarter average NOx | Defines relative-change filtering and identifies operating regimes |
| `prev_qtr_rel_delta` | Contains target magnitude and serves stratification only |

Coverage supports sliced evaluation but does not enter the model.

All scalar features are standardized with training means and standard
deviations. Local solar hour is `(UTC hour + longitude / 15) mod 24`; hour and
day of year use sine and cosine encodings. Validation, test, and inference reuse
the training statistics.

No feature uses `log1p`. On the first training split, it increased skew for all
three tested features and changed logistic-probe validation AUC by less than
0.005:

| Feature | Raw skew | `log1p` skew | Raw tail share | `log1p` tail share |
|---|---:|---:|---:|---:|
| `avg_heat_input` | 0.378 | -0.673 | 0.032 | 0.041 |
| `total_nameplate_capacity_mw` | 0.789 | -0.643 | 0.038 | 0.044 |
| `avg_pwr_gen` | 0.062 | -1.009 | 0.027 | 0.046 |

Tail share is the fraction of absolute deviation from the median held by the top
1% of records.

## Raster normalization

Each timestep has NO2, 2 m temperature, eastward wind, northward wind, and a
binary NO2 support mask. Fit statistics on finite training pixels only:

| Channels | Center | Scale |
|---|---|---|
| NO2 | Median | `IQR / 1.349` |
| Temperature and wind | Mean | Population standard deviation |

Clip numeric channels to `[-8, 8]`. Replace invalid normalized values with zero
only at model loading, while keeping the binary mask unscaled. Reuse the frozen
statistics for validation, test, and inference. Per-image normalization is not
used because absolute enhancement magnitude carries signal.

`normalization_stats.json` stores centers, scales, and valid counts.
`run_config.json` stores clipped-pixel fractions. Exact quartiles use temporary
node-local arrays instead of an in-memory pixel archive.

## Network

The raster branch applies one spatial encoder to every timestep. A two-layer
partial-convolution stem handles the NO2 value and mask. A dense stem handles
weather. Residual blocks fuse the stems and reduce each 24 by 24 input to 6 by
6. A 96-channel ConvGRU combines the ordered sequence, and global average and
maximum pooling produce a 128-value embedding.

The raster embedding feeds its own Bernoulli classifier. It receives no scalar
features or MLP output. The tabular baseline uses a 32-value hidden layer, a
16-value embedding, and a separate classifier. Each model has its own BCE loss,
optimizer, validation selection, and checkpoint.

GroupNorm handles the raster stems and encoder; LayerNorm handles MLP
projections. See the [Group Normalization paper](https://arxiv.org/abs/1803.08494).
The baseline applies no rotations or flips. Any future spatial transform must
also rotate wind vectors.

## Training and data loading

Training uses AdamW, unweighted binary cross-entropy with logits, gradient
clipping, CUDA mixed precision, validation-loss scheduling, and early stopping.

| File | Owns |
|---|---|
| `src/config.py` | Shared paths and delta-model input contract |
| `src/delta-model/modeling/train.py` | Training defaults and CLI |
| `src/delta-model/modeling/convgru.py` | Raster model |
| `src/delta-model/modeling/mlp.py` | Tabular model |

The map-style dataset decompresses record NPZ files on demand. DataLoader
workers overlap reads with GPU work and prefetch two batches. CUDA runs use
pinned memory. Keep worker counts within the CPU allocation because excessive
workers can hurt shared-filesystem throughput.

Run on a compute node:

```bash
python -u -m modeling.train
```

Use `--workers`, `--batch-size`, `--epochs`, and `--tabular-epochs` for run-level
overrides. Every run recomputes normalization statistics from its training split.

## Evaluation

Select models on validation loss alone. Do not use test outputs for thresholds,
normalization, architecture, or hyperparameters. Report accuracy, balanced
accuracy, precision, recall, specificity, F1, ROC AUC, log loss, and confusion
matrices. Also retain class prevalence, magnitude slices, per-AOI metrics,
row-level predictions, probability plots, and held-out AOI maps.

Run all comparisons on the same frozen records:

| Comparison | Question |
|---|---|
| Constant and prevalence classifiers | Does either learned model beat trivial predictions? |
| Tabular MLP | Does raster data add value? |
| Raster ConvGRU | What can the raster sequence predict without scalar context? |
| With and without masks | Does explicit support improve results? |

## Run artifacts

Each UTC-stamped directory under `RUNS_DIR` contains:

| Artifact | Contents |
|---|---|
| `normalization_stats.json` | Train-only preprocessing and cutoff |
| `run_config.json` | Features, settings, clipping rates, and parameter count |
| `checkpoints/best_raster_convgru.pt` | Selected raster checkpoint |
| `checkpoints/best_tabular_mlp.pt` | Selected tabular checkpoint |
| `results.json` | Metrics, model differences, and prevalence |
| `*_predictions.csv` | Row-level predictions by model and split |
| Plots | Model comparison, loss, probabilities, and spatial accuracy |
