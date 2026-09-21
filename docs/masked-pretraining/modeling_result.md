# Modeling results

## Run summary

| Item | Value |
|---|---|
| Run directory | `masked_no2_20260920_211751` |
| Model | 394,209-parameter masked NO2 convolutional autoencoder |
| Data | 500,000 train, 50,000 validation, and 50,000 test scenes |
| Best checkpoint | Epoch 62 |
| Best validation normalized L1 | 0.12416 |
| Evaluation job | `39079776` |
| Evaluated test pixels | 1,585,861 artificially hidden valid pixels |
| Comparison basis | Same 50,000-record test split and synthetic masks |

## Test comparison with bilinear interpolation

| Test metric on hidden pixels | Epoch-62 model | Bilinear interpolation | Relative reduction |
|---|---:|---:|---:|
| Normalized MAE / L1 | 0.13197 | 0.29563 | 55.36% |
| Normalized RMSE | 0.22106 | 0.40502 | 45.42% |
