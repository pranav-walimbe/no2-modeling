# Modeling

The binary baseline classifies hourly power-plant NOx changes from paired TEMPO
observations.

## Prediction target

- Apply the fixed symmetric 75 lb `DELTA_THRESHOLD` cutoff on
  `abs(delta_nox_mass)`.
- Use the same cutoff for train, validation, test, and inference.
- Remove records inside the closed deadband.
- Label negative changes as 0 and positive changes as 1.
- Balance each split to equal label counts after raster QC.
- Preserve raw `delta_nox_mass` for reporting, never as an input.

## Inputs and leakage policy

Each sample carries four aligned 48 by 48 raster channels and scalar context.

Scalar inputs:

- coal and natural-gas unit counts;
- total generator nameplate capacity;
- previous-quarter average AOI hourly NOx mass;
- previous-quarter average heat input and power generation;
- coincident HRRR 2 m temperature and boundary-layer height;
- sine/cosine encodings of UTC hour and day of year.

Excluded inputs and the reason for each:

| Excluded | Reason |
|---|---|
| Coordinates, AOI IDs | Prevent geographic memorization |
| Current emissions | Direct target leakage |
| Plume score, raster-quality scores | Diagnostics extracted from the response image |
| Prior-quarter NOx-change scale | Encodes how far a plant usually swings, which tracks crossing a fixed magnitude cutoff |

Coverage stays available for sliced evaluation but is not a model input.

### Feature transformations

Every scalar enters raw, then gets standardized with the training-split mean and
standard deviation. Validation, test, and inference reuse those statistics.

| Transform | Features |
|---|---|
| None | all scalar inputs |
| Sine and cosine | UTC hour, day of year |

Sine and cosine keep hour 23 adjacent to hour 0.

No feature carries a `log1p` transform. Measurements on the first generated
training split rejected it, split by feature:

| Feature | Raw skew | `log1p` skew | Tail share raw | Tail share `log1p` |
|---|---|---|---|---|
| `avg_heat_input` | 0.378 | -0.673 | 0.032 | 0.041 |
| `total_nameplate_capacity_mw` | 0.789 | -0.643 | 0.038 | 0.044 |
| `avg_pwr_gen` | 0.062 | -1.009 | 0.027 | 0.046 |
| `boundary_layer_height_m` | 1.401 | -0.875 | 0.058 | 0.051 |

Tail share is the fraction of total absolute deviation from the median held by
the top 1 percent of records. Only `boundary_layer_height_m` improves on both
measures, and a logistic probe on the tabular features moved validation AUC by
less than 0.005 across every combination tested. Nothing justified the added
distortion, so the transform is gone.

## Image representation and normalization

Four numeric channels and two masks reach the model:

1. smoothed current NO2 on finite native support;
2. smoothed current-minus-smoothed previous NO2 on paired support;
3. geographic eastward wind aligned from the native HRRR grid;
4. geographic northward wind aligned from the native HRRR grid;
5. independent binary validity masks for current and hourly-delta NO2.

Every statistic comes from training pixels alone:

| Channels | Center | Scale |
|---|---|---|
| current NO2, hourly delta NO2 | median of finite pixels | `IQR / 1.349` |
| wind u, wind v | mean of finite pixels | population standard deviation |

```text
normalized[channel] =
    (raster[channel] - train_center[channel]) / train_scale[channel]
```

- Clip all four numeric channels to `[-8, 8]` and replace invalid normalized
  values with zero only when loading the model input.
- Fit every channel on its finite training pixels.
- Reuse the frozen training statistics for validation, test, and inference.

The two masks remain binary and unscaled. A separate two-layer partial-
convolution stem consumes each NO2 value-mask pair. The resulting features join
the dense wind stem before the shared residual encoder.

Two design notes:

- Robust linear scaling limits outlier influence without compressing the whole
  NO2 distribution.
- Per-image normalization stays unsuitable because absolute enhancement
  magnitude carries part of the emissions signal.

Where the numbers live:

- `normalization_stats.json` holds the center, scale, and valid-pixel count per
  channel.
- `run_config.json` holds the clipped valid-pixel fraction per channel and split.

Exact quartiles come from temporary node-local arrays, so the fit never builds
an in-memory pixel archive.

## Network

A compact residual CNN plus an MLP scalar branch:

- Residual stages reduce 48 by 48 images to a 6 by 6 feature map.
- A 3 by 3 adaptive average pool retains coarse plume location.
- A global maximum pool preserves localized enhancements that an average
  dilutes.
- The fused image embedding joins the scalar embedding for one classification
  logit.

Normalization choices:

- GroupNorm throughout the image encoder, since it avoids batch-level
  statistics and holds up when memory pressure forces small batches. See the
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
| `modeling/resnet.py` | Architecture defaults |

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

Before treating the CNN as scientifically useful, compare it with:

1. constant and natural-prevalence classifiers;
2. an XGBoost tabular baseline trained after every deep-learning run;
3. a tabular-only MLP with the same scalar features;
4. an image-only model;
5. the full image-plus-tabular model;
6. a delta-plus-mask versus current-plus-delta-plus-mask ablation;
7. a mask ablation over the same eligible records.

Report every comparison on the same frozen validation and test records. The full
model earns its place only when image information improves held-out-AOI error
and the gain extends past unusually clear or high-plume scenes. The trainer
exposes these ablations through `--inputs tabular`, `--inputs image`, and the
default `--inputs full`.

## Run artifacts

Each UTC-stamped directory under `RUNS_DIR` holds:

- `normalization_stats.json` with train-only preprocessing and the deadband
  cutoff;
- `run_config.json` with features, settings, clipped-pixel fractions, and
  parameter count;
- `checkpoints/best_model.pt` selected by validation loss;
- `results.json` with deep-learning and XGBoost metrics, direct metric
  differences, and pre-balancing prevalence. Existing top-level metric fields
  continue to describe the deep-learning model;
- one deep-learning prediction CSV per split plus matching
  `xgboost_*_predictions.csv` files;
- `checkpoints/xgboost_model.json` selected by validation log loss;
- `model_comparison.png` with side-by-side split metrics;
- loss, probability-distribution, and spatial-accuracy plots.

Run training on a compute node:

```bash
python -u -m modeling.train
```

Flags:

- `--workers`, `--batch-size`, `--epochs` for allocation-specific overrides;
- `--inputs` for the controlled branch ablations.

Every run recomputes normalization statistics from the training split and writes
them to its own run directory. No flag reuses a saved file, so a stale statistics
JSON can never normalize a run against the wrong feature order.
