# Modeling results

The September 20 masked-pretraining run produced the NO2 imputer and encoder
checkpoint used by the delta classifier. The learned reconstruction cut
held-out masked-pixel MAE by 55.36% relative to bilinear interpolation.

## Run summary

| Item | Value |
|---|---|
| Run directory | `masked_no2_20260920_211751` |
| Training job | `39070356` |
| Model | 394,209-parameter masked NO2 convolutional autoencoder |
| Data | 500,000 train, 50,000 validation, and 50,000 test scenes |
| Best checkpoint | Epoch 62 |
| Best validation normalized L1 | 0.12416 |
| Evaluation job | `39079776` |
| Evaluated test pixels | 1,585,861 artificially hidden valid pixels |

The training job reached the eight-hour limit during epoch 63. The trainer had
already written the epoch-62 checkpoint, and epoch 63 did not improve validation
loss (`0.12462`). Training did not run through early stopping, so later epochs
might improve the checkpoint.

Validation L1 fell from `0.29647` at epoch 1 to `0.12416` at epoch 62. Compared
with the earlier epoch-20 checkpoint's `0.15526`, the newer checkpoint reduced
validation L1 by 20.03%.

## Test comparison with bilinear interpolation

The evaluation applies each method to the same deterministic test split and the
same synthetic masks. Both methods retain measured pixels and fill only the
artificial gaps.

| Test metric on hidden pixels | Epoch-62 model | Bilinear interpolation | Relative reduction |
|---|---:|---:|---:|
| Normalized MAE / L1 | 0.13197 | 0.29563 | 55.36% |
| Normalized RMSE | 0.22106 | 0.40502 | 45.42% |
| Normalized bias | -0.00029 | 0.00170 | n/a |

The model removes more than half of bilinear interpolation's average
hidden-pixel error and also reduces larger misses. Both methods have small
aggregate bias, so better pixel estimates account for the gain rather than a
global offset correction.

The evaluation wrote its detailed metrics and reconstruction panel to:

- `/global/home/users/pranavwalimbe/vis/masked-model-test-reconstruction-latest-39079776.json`
- `/global/home/users/pranavwalimbe/vis/masked-model-test-reconstruction-latest-39079776.png`

## Scope

These numbers measure artificial 1% to 10% gaps in scenes that originally had
complete NO2 coverage. They do not establish performance on large cloud gaps or
scenes selected by the downstream 90% coverage rule. The delta-model comparison
must determine whether the pretrained encoder improves emissions-change
classification; reconstruction accuracy alone cannot answer that question.
