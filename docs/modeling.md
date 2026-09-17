# Modeling

The binary baseline classifies power-plant NOx changes from causal consecutive
TEMPO scans and aligned hourly HRRR fields.

## Baseline at a glance

| Component | Choice |
|---|---|
| Target | Sign of hourly NOx change outside a 100 lb deadband |
| Raster input | `T` consecutive NO2 scans, validity masks, and aligned 2 m temperature and wind U/V fields |
| Context input | Plant attributes, prior-quarter activity, and time |
| Split | Geographic AOI clusters, approximately 70/15/15 |
| Raster encoder | Shared mask-aware spatial encoder followed by a ConvGRU |
| Models | Independent raster-only ConvGRU and tabular-only MLP |
| Selection metric | Validation log loss |
| Final metrics | ROC AUC and log loss across seeds, with subgroup results |

## Prediction target

- Apply the fixed symmetric 100 lb `EMA_DELTA_THRESHOLD` cutoff on the absolute
  current-minus-previous effective EMA emissions difference.
- Use the same cutoff for train, validation, test, and inference.
- Remove records inside the closed deadband.
- Label negative changes as 0 and positive changes as 1.
- Balance each split to equal label counts after raster QC.
- Preserve the continuous EMA difference for reporting, never as an input.

## Inputs and leakage policy

Each sample carries `T` aligned 24 by 24 raster timesteps and scalar context.

Scalar inputs:

- coal and natural-gas unit counts;
- total generator nameplate capacity;
- previous-quarter average heat input and power generation;
- sine/cosine encodings of local mean solar hour and day of year.

Leakage controls:

| Excluded | Reason |
|---|---|
| Raw coordinates, AOI IDs | Prevent geographic memorization; longitude only converts UTC to local solar hour |
| Current emissions | Direct target leakage |
| Previous-quarter average NOx | Defines the relative-change filter and can identify plant operating regimes |
| `prev_qtr_rel_delta` | Contains target magnitude and is used only for stratification |

Coverage stays available for sliced evaluation but is not a model input.

### Feature transformations

Every scalar enters raw, then gets standardized with the training-split mean and
standard deviation. Validation, test, and inference reuse those statistics.

| Transform | Features |
|---|---|
| None | all scalar inputs |
| Sine and cosine | Local mean solar hour, day of year |

Local mean solar hour is `(UTC hour + longitude / 15) mod 24`. This keeps solar
time continuous across civil-time boundaries and daylight-saving changes. Sine
and cosine keep hour 23 adjacent to hour 0.

No feature carries a `log1p` transform. Measurements on the first generated
training split rejected it, split by feature:

| Feature | Raw skew | `log1p` skew | Tail share raw | Tail share `log1p` |
|---|---|---|---|---|
| `avg_heat_input` | 0.378 | -0.673 | 0.032 | 0.041 |
| `total_nameplate_capacity_mw` | 0.789 | -0.643 | 0.038 | 0.044 |
| `avg_pwr_gen` | 0.062 | -1.009 | 0.027 | 0.046 |

Tail share is the fraction of total absolute deviation from the median held by
the top 1 percent of records. A logistic probe on the retained tabular features
moved validation AUC by less than 0.005 across every combination tested. Nothing
justified the added distortion, so the transform is gone.

## Raster representation and normalization

Four numeric channels and one mask reach the model at every timestep:

1. directly regridded NO2 on finite native support;
2. HRRR 2 m temperature sampled at AOI cell centers;
3. geographic eastward wind sampled at AOI cell centers;
4. geographic northward wind sampled at AOI cell centers;
5. an independent binary NO2 validity mask.

Every statistic comes from training pixels alone:

| Channels | Center | Scale |
|---|---|---|
| NO2 | median of finite pixels | `IQR / 1.349` |
| temperature, wind u, wind v | mean of finite pixels | population standard deviation |

```text
normalized[channel] =
    (raster[channel] - train_center[channel]) / train_scale[channel]
```

- Clip all four numeric channels to `[-8, 8]` and represent invalid normalized
  values with zero only when loading the model input.
- Fit every channel on its finite training pixels.
- Reuse the frozen training statistics for validation, test, and inference.

The mask remains binary and unscaled. A two-layer partial-convolution stem
consumes each scan's NO2 value-mask pair. The resulting features join the dense
weather stem before the shared residual encoder.

Two design notes:

- Robust linear scaling limits outlier influence without compressing the whole
  NO2 distribution.
- Per-image normalization stays unsuitable because absolute enhancement
  magnitude carries part of the emissions signal.

Run outputs store centers, scales, and valid counts in
`normalization_stats.json`; `run_config.json` stores clipped-pixel fractions.

Exact quartiles come from temporary node-local arrays, so the fit never builds
an in-memory pixel archive.

## Network

The raster branch applies the same spatial encoder to every hour. A partial-
convolution NO2 stem uses the validity mask to renormalize local support rather
than treating missing cells as physical zeros. A conventional weather stem
encodes temperature and wind. Their fused features pass through residual blocks
that reduce each 24 by 24 timestep to 6 by 6, then a 96-channel ConvGRU fuses
the ordered sequence. Global average and maximum pooling produce a 128-value
raster embedding.

The raster embedding passes through its own multilayer classifier to produce a
single Bernoulli logit. The raster model receives no tabular features or MLP
outputs. The tabular model is a separate 32-value hidden layer followed by a
16-value embedding and its own Bernoulli classifier. Each model has its own BCE
loss, optimizer, validation selection, and checkpoint. Their sigmoid outputs
are compared on the same records; they are not fused during training or
inference.

Normalization choices:

- GroupNorm in the NO2 stem, weather stem, and shared spatial encoder avoids
  batch-level statistics when memory pressure forces small batches. See the
  [Group Normalization paper](https://arxiv.org/abs/1803.08494).
- LayerNorm in the MLP projections.

The DenseNet alternative is gone. It duplicated an obsolete input signature and
training never selected it.

The baseline applies no rotation or flip. Alignment rotates HRRR grid-relative
wind to geographic east and north, so any later spatial augmentation must
transform the wind vector values along with the raster coordinates.

## Optimization and I/O

Training uses:

- AdamW;
- unweighted binary cross-entropy with logits;
- gradient clipping and mixed precision on CUDA;
- validation-loss scheduling;
- early stopping.

Where settings live:

| File | Owns |
|---|---|
| `config.py` | Shared data contract: paths, raster keys and channels, image clipping, input-feature definitions |
| `modeling/train.py` | Training defaults |
| `modeling/convgru.py` | Mask-aware spatial and temporal raster model |
| `modeling/mlp.py` | Compact tabular model and embedding dimensions |

Training CLI flags expose the last two, which keeps preprocessing and collection
code independent of any single run while each run still records its resolved
settings.

CUDA runs use automatic mixed precision for convolutions and linear layers, with
gradient scaling. AMP selects lower precision for eligible high-throughput
operations and keeps float32 where the range matters; see the
[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.3/amp.html).

Data loading:

- The map-style dataset decompresses the selected per-record NPZ files on
  demand.
- DataLoader workers overlap that I/O with GPU computation, persist between
  epochs, and prefetch two batches each.
- Pinned memory stays on for CUDA alone.
- The allocated CPUs cap the worker count. Raising it can hurt
  shared-filesystem performance and multiply parent-process memory.

The [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/2.3/data.html)
covers these controls and their memory implications.

## Evaluation philosophy

Report accuracy, balanced accuracy, precision, recall, specificity, F1, ROC AUC,
and the full confusion matrix. Also record:

- class counts and natural pre-balancing prevalence;
- equal-count test slices by absolute raw delta-NOx magnitude;
- per-AOI metrics;
- row-level logits, probabilities, and predictions;
- probability distributions and held-out AOI accuracy maps.

Rules:

- Select models on validation loss alone.
- Keep test outputs out of decisions about normalization, architecture,
  thresholds, and hyperparameters.

The split is geographic, so validation and test measure transfer to
non-overlapping plant regions rather than memorization of known AOIs.

## Required comparisons

| Comparison | Question |
|---|---|
| Constant and prevalence classifiers | Does the model beat trivial predictions? |
| Tabular-only MLP | Does image data add value to the neural model? |
| Raster-only ConvGRU | How much can the raster sequence predict without tabular features? |
| With and without masks | Does explicit support information add value? |

Report every comparison on the same frozen validation and test records. Each run
reports the independently trained raster ConvGRU against the independently
trained MLP.

## Run artifacts

Each UTC-stamped directory under `RUNS_DIR` contains:

| Artifact | Contents |
|---|---|
| `normalization_stats.json` | Train-only preprocessing and deadband cutoff |
| `run_config.json` | Features, settings, clipping rates, and parameter count |
| `checkpoints/best_raster_convgru.pt` | Validation-selected raster-only ConvGRU checkpoint |
| `checkpoints/best_tabular_mlp.pt` | Validation-selected tabular-only MLP checkpoint |
| `results.json` | Metrics, raster ConvGRU minus MLP differences, and prevalence |
| `*_predictions.csv` | Row-level predictions for each model and split |
| `model_comparison.png` | Side-by-side split metrics |
| Other plots | Loss, probability distributions, and spatial accuracy |

Run training on a compute node:

```bash
python -u -m modeling.train
```

Flags:

- `--workers`, `--batch-size`, and `--epochs` for allocation-specific overrides;
- `--tabular-epochs` for the independent MLP training phase.

Every run recomputes normalization statistics from the training split and writes
them to its own run directory. No flag reuses a saved file, so a stale statistics
JSON can never normalize a run against the wrong feature order.
