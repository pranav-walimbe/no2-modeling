# Modeling

The masked model reconstructs artificially hidden NO2 pixels from visible NO2,
temperature, and wind. It predicts NO2 alone. The training loss excludes each
visible pixel.

## Input and target

The loader normalizes four physical channels in this order:

1. NO2
2. 2 m temperature
3. eastward wind at 80 m
4. northward wind at 80 m

It appends a fifth channel containing the visible-pixel mask. The loader sets
hidden NO2 pixels to zero after normalization, which represents the train-split
NO2 center rather than a physical zero. The independent mask tells partial
convolutions which values carry observations.

The target contains the untouched normalized NO2 raster. The loss mask selects
pixels that were valid in the source raster and hidden by the artificial mask.

## Normalization

Training records define all normalization statistics:

| Channel | Center | Scale |
|---|---|---|
| NO2 | Median | `IQR / 1.349` |
| Temperature and wind | Mean | Population standard deviation |

The loader clips normalized values to `[-8, 8]`. Validation, test, checkpoint
inference, and downstream transfer reuse the saved training statistics.

## Architecture

The encoder matches the delta model's single-frame spatial stack:

| Stage | Output |
|---|---|
| Partial-convolution NO2 stem | 16 by 24 by 24 |
| Weather stem | 16 by 24 by 24 |
| 1 by 1 fusion | 32 by 24 by 24 |
| Residual spatial encoder | 64 by 6 by 6 |

The NO2 stem applies 5 by 5 and 3 by 3 partial convolutions. Each layer excludes
hidden inputs, rescales for available kernel support, and propagates an updated
support mask. The weather stem uses a conventional 5 by 5 convolution. GroupNorm
and SiLU follow the convolutional stages.

The decoder contains a 64-channel residual bottleneck, bilinear upsampling to
12 by 12 with a 48-channel residual block, bilinear upsampling to 24 by 24 with
a 32-channel residual block, and a final 1 by 1 NO2 projection. It has no encoder
skip connections. The current implementation contains 394,209 trainable
parameters, including 243,680 in the encoder.

## Objective and inference

Training minimizes mean per-pixel L1 error in normalized NO2 over the artificial
gaps:

```text
sum(abs(prediction - target) * loss_mask) / sum(loss_mask)
```

The model predicts a complete NO2 raster, but evaluation scores only hidden
observed pixels. Gap filling preserves visible NO2 values and inserts model
predictions where the visible-pixel mask equals zero.

## Optimization

The training command uses AdamW, gradient clipping, mixed precision on CUDA,
and a `ReduceLROnPlateau` scheduler. It selects the epoch with the lowest
validation masked L1 and stops after the configured patience without an
improvement.

Default settings include:

| Setting | Value |
|---|---:|
| Batch size | 128 |
| Maximum epochs | 300 |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Gradient norm limit | 5.0 |
| Scheduler patience | 10 epochs |
| Early-stop patience | 25 epochs |
| Seed | 42 |

## Evaluation procedure

The evaluation pass runs on validation and test loaders without updating model
state. It compares the autoencoder with separable horizontal and vertical linear
interpolation. Both methods receive the same normalized input and visible mask.

For each method, the evaluator reports normalized and physical-unit MAE, RMSE,
and signed bias over artificial gaps. It also reports the autoencoder's absolute
and relative normalized-L1 improvement over interpolation. This document records
the evaluation contract and does not record run results.

## Checkpoints and run artifacts

The best checkpoint stores:

- Full model and encoder state dictionaries.
- Architecture name and input-channel contract.
- Ordered image keys and normalization state.
- Selected epoch and validation loss.
- SHA-256 hash of the three published split manifests.

Each run also writes `normalization_stats.json`, `run_config.json`,
`results.json`, a loss curve, and evaluation plots. The run configuration records
the model and encoder parameter counts plus optimization settings.

## Delta-model transfer

The delta training pipeline loads the masked checkpoint, uses the full model to
fill missing NO2 cells, and maps partial-convolution kernel weights into the
delta model's conventional single-frame encoder. It initializes only the shared
spatial encoder. The delta model trains its temporal ConvGRU and classifier for
the downstream objective.
