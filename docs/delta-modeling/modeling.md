# Modeling

The delta-model baseline classifies power-plant NOx changes from five causal
TEMPO scans and aligned hourly HRRR fields.

## Contract

| Component | Choice |
|---|---|
| Target | `delta_category` |
| Classes | `decrease`, `steady`, `increase` |
| Raster input | Five NO2, validity-mask, temperature, and wind U/V rasters |
| Raster shape | `5 x 5 x 24 x 24` |
| Tabular input | Ten standardized plant, activity, and cyclic-time features |
| Expected split sizes | About 100k train, 20k validation, and 20k test |
| Raster model | Partial-convolution spatial encoder followed by a ConvGRU |
| Baseline model | Independent tabular MLP classifier |
| Training loss | Three-class cross-entropy |
| Selection | Lowest validation cross-entropy |

Dataset generation preserves `delta_category` from the geographically
stratified source records. Model loading maps the ordered class names to the
indices 0, 1, and 2. It does not recreate classes from a continuous target.

Five scans remain available to the raster model. The target is aligned to the
fourth scan: label coverage uses the `t2` to `t3` interval, and the fifth scan
remains stored as model input.

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
Validation and test data reuse the training statistics. The loader standardizes
each tabular feature using its training mean and standard deviation.

## Model sizing

The raster classifier has about 701,000 trainable parameters. A shared spatial
encoder reduces each 24 by 24 frame to a 6 by 6 feature map. A 96-channel
ConvGRU consumes the five maps in time order. Global average and maximum pools
produce a 128-value embedding, followed by a 128-value regularized head and
three output logits.

The NO2 stem uses partial convolutions, which exclude missing cells and adjust
for available kernel support. A conventional stem handles the complete weather
rasters. GroupNorm avoids dependence on batch statistics.

The tabular MLP has about 1,000 parameters. It uses a 32-value hidden layer, a
16-value embedding, and three output logits. It remains deliberately compact
so it measures the information in scalar features rather than matching the
raster model through excess capacity.

At roughly 100,000 training examples, the raster model has about seven trainable
parameters per record. This is moderate for a convolutional sequence model
because spatial and temporal weights are shared. A 30% head dropout, weight
decay, and validation early stopping further constrain capacity. Increasing the
encoder size is not the first response to underfitting; first compare train and
validation learning curves and per-class recall.

## Optimization

Both models use unweighted cross-entropy because stratification balances the
three target classes before raster quality filtering. Saved run metadata records
the final class counts so any filtering-induced imbalance remains visible.

Both models use AdamW, gradient clipping, mixed precision on CUDA, validation
loss scheduling, and early stopping. The defaults allow 100 raster epochs and
75 MLP epochs with 12 epochs of early-stop patience. These are ceilings rather
than expected run lengths. A batch size of 128 provides about 780 optimizer
steps per raster epoch for 100,000 training records.

Run on a compute node:

```bash
python -u -m modeling.train
```

## Evaluation

Each split reports:

- accuracy and balanced accuracy;
- macro F1 and one-vs-rest macro ROC AUC;
- class counts and per-class recall;
- the three-class confusion matrix.

The raster and MLP classifiers use the same splits. Model selection uses
validation cross-entropy only. Test metrics remain reporting outputs. Saved
prediction CSVs contain each class logit and probability, and the generated
figures compare both models and show row-normalized test confusion matrices.
