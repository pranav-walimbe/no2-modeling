# Modeling

The delta-model baseline regresses effective power-plant NOx changes from five
causal TEMPO scans and aligned hourly HRRR fields.

## Contract

| Component | Choice |
|---|---|
| Target | `delta_effective_nox_scaled` |
| Raster input | Five NO2, validity-mask, temperature, and wind U/V rasters |
| Raster shape | `5 x 5 x 24 x 24` |
| Tabular input | Ten standardized plant, activity, and cyclic-time features |
| Split | Geographic AOI clusters, about 80k/15k/15k after raster QC |
| Raster model | Mask-aware spatial encoder followed by a ConvGRU |
| Baseline model | Independent tabular MLP |
| Training loss | LDS-weighted Huber |
| Selection | Lowest weighted validation Huber loss |

The target comes from dataset generation:

```text
delta_effective_nox_scaled =
    asinh(effective_delta_nox / prior_quarter_median_nox)
```

Five scans remain available to the model. The target is aligned to the fourth
scan: label coverage uses the `t2` to `t3` interval, and the effective change
compares four-hour EMAs ending at `t3` and `t2`. The `t4` raster remains stored
as the fifth model input.

Model loading copies this column without another transform, normalization, or
clip. Both networks use an unrestricted one-value output head.

## Inputs and leakage controls

The raster model receives five ordered timesteps. Each timestep contains:

1. NO2 on finite native support;
2. 2 m temperature;
3. geographic eastward wind;
4. geographic northward wind;
5. the binary NO2 validity mask.

The tabular MLP receives coal and natural-gas unit counts, total nameplate
capacity, prior-quarter average heat input and generation, local solar hour,
and day of year. Sine and cosine encode both time features.

Coordinates, AOI identity, current emissions, prior-quarter NOx, and target
derivatives stay out of model inputs. Longitude only converts UTC to local
solar hour.

## Input normalization

Training pixels define all raster statistics:

| Channels | Center | Scale |
|---|---|---|
| NO2 | Median | `IQR / 1.349` |
| Temperature and wind | Mean | Population standard deviation |

The loader clips normalized numeric raster values to `[-8, 8]`. It fills an
invalid numeric value with zero and supplies the independent validity mask.
Validation and test data reuse the training statistics.

The loader standardizes each tabular feature using its training mean and
standard deviation. It does not standardize or transform the regression
target.

## Models

The raster model has about 700,000 trainable parameters. A shared spatial
encoder reduces each 24 by 24 frame to a 6 by 6 feature map. A 96-channel
ConvGRU consumes the five maps in time order. Global average and maximum pools
produce a 128-value embedding for the regression head.

The NO2 stem uses partial convolutions, which exclude missing cells and adjust
for available kernel support. A conventional stem handles the complete weather
rasters. GroupNorm avoids dependence on batch statistics.

The tabular MLP has about 1,000 parameters. It uses a 32-value hidden layer, a
16-value embedding, and an independent scalar regression head. The MLP provides
a low-capacity baseline on the same records and target. Its output never enters
the raster model.

The model sizes fit the expected 80,000-record training split. Weight sharing
limits raster-encoder capacity, dropout regularizes the raster head, and early
stopping uses the geographically held-out validation split.

## Skew-aware loss weights

Most target values lie near zero. Training fits loss weights from the training
labels only:

1. Assign targets to 101 equal-width bins across the training range.
2. Smooth bin counts with a Gaussian kernel whose sigma is two bins.
3. Give each target inverse-square-root smoothed-density weight.
4. Scale the uncapped weights to mean one over training records.
5. Cap individual weights at five.

The loss for one batch is:

```text
sum(weight * huber(prediction, target, delta=0.1)) / sum(weight)
```

The same training-fitted bins and weights define validation loss. Test labels
do not affect weights or checkpoint selection. Run metadata records bin count,
kernel width, cap, Huber delta, target range, and realized weight statistics.

`--loss-weighting none` runs the required unweighted Huber baseline. It changes
only sample weights and leaves targets untouched.

The weighting follows label distribution smoothing from
[Yang et al. (ICML 2021)](https://proceedings.mlr.press/v139/yang21m/yang21m.pdf).
The square-root inverse scheme matches the authors'
[reference implementation](https://github.com/YyzHarry/imbalanced-regression/blob/main/agedb-dir/datasets.py).

## Optimization

Both models use AdamW, gradient clipping, mixed precision on CUDA, validation
loss scheduling, and early stopping. The defaults allow 100 raster epochs and
75 MLP epochs with 12 epochs of early-stop patience. These are ceilings rather
than expected run lengths.

The map-style dataset decompresses raster bundles on demand. Persistent loader
workers overlap I/O with GPU work, and pinned memory applies on CUDA.

Run on a compute node:

```bash
python -u -m modeling.train
```

## Evaluation

Each split reports:

- mean squared error, mean absolute error, and root mean squared error;
- R-squared, Pearson correlation, and Spearman correlation;
- mean prediction bias.

The test set also reports these metrics in equal-count low, middle, and high
absolute-target thirds. This separates performance on the dense near-zero
region from performance on larger changes. Row-level output includes the stored
target, prediction, signed residual, and absolute error.

The raster and MLP models use the same splits. Model selection uses validation
loss only. Test metrics remain reporting outputs.

The emailed prediction artifact shows test-set predicted-versus-observed
scatterplots for both models. Both panels use shared limits from the pooled
0.5th and 99.5th percentiles, and each panel reports its model's test MSE.
