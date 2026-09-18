# Previous modeling results

Run: `delta_nox_classification_20260917_184122`

## Run summary

| Item | Value |
|---|---|
| Seed | 42 |
| Train / validation / test rows | 83,764 / 21,244 / 22,746 |
| Unique train / validation / test records | 58,860 / 15,940 / 15,562 |
| Train / validation / test AOIs | 556 / 107 / 134 |
| Target | Sign of causal effective NOx change outside a 100 lb deadband |
| Raster sequence | Five 24 × 24 hourly frames |
| Raster inputs | NO2, validity mask, 2 m temperature, and 80 m winds |
| Tabular inputs | Plant attributes, prior-quarter activity, solar hour, and day of year |
| Raster model | Independent ConvGRU |
| Tabular model | Independent MLP |

## Balanced split metrics

| Split | Model | Accuracy | Balanced accuracy | Precision | TPR | TNR | F1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | Raster ConvGRU | 0.8578 | 0.8578 | 0.8355 | 0.8911 | 0.8246 | 0.8624 | 0.9095 |
| Train | Tabular MLP | 0.8478 | 0.8478 | 0.8357 | 0.8659 | 0.8298 | 0.8505 | 0.9180 |
| Validation | Raster ConvGRU | 0.7915 | 0.7915 | 0.7821 | 0.8080 | 0.7749 | 0.7949 | 0.8441 |
| Validation | Tabular MLP | 0.7643 | 0.7643 | 0.7888 | 0.7217 | 0.8068 | 0.7538 | 0.8418 |
| Test | Raster ConvGRU | 0.7700 | 0.7700 | 0.7343 | 0.8462 | 0.6938 | 0.7863 | 0.8357 |
| Test | Tabular MLP | 0.7525 | 0.7525 | 0.7191 | 0.8289 | 0.6762 | 0.7701 | 0.8107 |

## Balanced test metrics by absolute emissions-change tertile

| Absolute change | Range (lb) | Rows | Model | Accuracy | AUROC |
|---|---:|---:|---|---:|---:|
| Low | 100.0–132.4 | 7,582 | Raster ConvGRU | 0.7609 | 0.8396 |
| Low | 100.0–132.4 | 7,582 | Tabular MLP | 0.7576 | 0.8168 |
| Middle | 132.4–201.0 | 7,582 | Raster ConvGRU | 0.7824 | 0.8410 |
| Middle | 132.4–201.0 | 7,582 | Tabular MLP | 0.7580 | 0.8200 |
| High | 201.0–4,300.3 | 7,582 | Raster ConvGRU | 0.7668 | 0.8258 |
| High | 201.0–4,300.3 | 7,582 | Tabular MLP | 0.7420 | 0.7967 |

## Balanced test metrics by AOI stratum

| Stratum | Rows | AOIs | Model | Accuracy | AUROC |
|---|---:|---:|---|---:|---:|
| Prior-quarter coal generation share ≥ 50% | 11,184 | 48 | Raster ConvGRU | 0.7410 | 0.7973 |
| Prior-quarter coal generation share ≥ 50% | 11,184 | 48 | Tabular MLP | 0.7283 | 0.7875 |
| Major-city distance ≥ 50 km | 10,735 | 84 | Raster ConvGRU | 0.7590 | 0.8241 |
| Major-city distance ≥ 50 km | 10,735 | 84 | Tabular MLP | 0.7372 | 0.8038 |
| Both strata | 6,081 | 35 | Raster ConvGRU | 0.7597 | 0.8073 |
| Both strata | 6,081 | 35 | Tabular MLP | 0.7264 | 0.7857 |

## Unique-record metrics

| Split or stratum | Records | AOIs | Model | Accuracy | Balanced accuracy | AUROC |
|---|---:|---:|---|---:|---:|---:|
| Train | 58,860 | 556 | Raster ConvGRU | 0.8742 | 0.8619 | 0.9126 |
| Train | 58,860 | 556 | Tabular MLP | 0.8596 | 0.8550 | 0.9231 |
| Validation | 15,940 | 107 | Raster ConvGRU | 0.7971 | 0.7916 | 0.8442 |
| Validation | 15,940 | 107 | Tabular MLP | 0.7502 | 0.7644 | 0.8420 |
| Test | 15,562 | 134 | Raster ConvGRU | 0.8090 | 0.7770 | 0.8409 |
| Test | 15,562 | 134 | Tabular MLP | 0.7910 | 0.7586 | 0.8160 |
| Test: prior-quarter coal generation share ≥ 50% | 7,544 | 48 | Raster ConvGRU | 0.7549 | 0.7507 | 0.8052 |
| Test: prior-quarter coal generation share ≥ 50% | 7,544 | 48 | Tabular MLP | 0.7735 | 0.7414 | 0.7928 |
| Test: major-city distance ≥ 50 km | 6,685 | 84 | Raster ConvGRU | 0.7928 | 0.7774 | 0.8303 |
| Test: major-city distance ≥ 50 km | 6,685 | 84 | Tabular MLP | 0.7687 | 0.7520 | 0.8054 |
| Test: both strata | 3,834 | 35 | Raster ConvGRU | 0.7770 | 0.7741 | 0.8186 |
| Test: both strata | 3,834 | 35 | Tabular MLP | 0.7679 | 0.7486 | 0.7896 |
