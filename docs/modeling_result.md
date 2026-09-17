# Previous modeling results

Run: `delta_nox_classification_20260917_115919`

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
| Models | Frozen MLP with additive ConvGRU correction; tabular MLP |

## Saved balanced artifacts

| Split | Model | Accuracy | AUROC | Log loss | Brier score |
|---|---|---:|---:|---:|---:|
| Train | ConvGRU + MLP | 0.8578 | 0.9265 | 0.3469 | 0.1056 |
| Train | MLP | 0.8478 | 0.9180 | 0.3636 | 0.1119 |
| Validation | ConvGRU + MLP | 0.7701 | 0.8491 | 0.5363 | 0.1690 |
| Validation | MLP | 0.7643 | 0.8418 | 0.5186 | 0.1685 |
| Test | ConvGRU + MLP | 0.7577 | 0.8267 | 0.5524 | 0.1778 |
| Test | MLP | 0.7525 | 0.8107 | 0.5612 | 0.1824 |

## Unique-record results

| Split | Model | Accuracy | Balanced accuracy | AUROC | Log loss |
|---|---|---:|---:|---:|---:|
| Train | ConvGRU + MLP | 0.8635 | 0.8636 | 0.9308 | 0.3512 |
| Train | MLP | 0.8596 | 0.8550 | 0.9231 | 0.3498 |
| Validation | ConvGRU + MLP | 0.7527 | 0.7703 | 0.8493 | 0.5930 |
| Validation | MLP | 0.7502 | 0.7644 | 0.8420 | 0.5550 |
| Test | ConvGRU + MLP | 0.7843 | 0.7630 | 0.8322 | 0.5189 |
| Test | MLP | 0.7910 | 0.7586 | 0.8160 | 0.4986 |

## Unique test records by absolute emissions-change tertile

| Absolute change | Range (lb) | N | ConvGRU + MLP accuracy | MLP accuracy | ConvGRU + MLP AUROC | MLP AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Low | 100.0–133.3 | 5,188 | 0.7787 | 0.7866 | 0.8316 | 0.8178 |
| Middle | 133.3–201.5 | 5,187 | 0.7872 | 0.7941 | 0.8461 | 0.8282 |
| High | 201.5–4,300.3 | 5,187 | 0.7870 | 0.7924 | 0.8204 | 0.8037 |

## Unique test records by AOI capacity tertile

| Capacity range (MW) | AOIs | Records | ConvGRU + MLP accuracy | MLP accuracy | ConvGRU + MLP AUROC | MLP AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 133–1,269 | 46 | 2,462 | 0.8034 | 0.8034 | 0.8339 | 0.8250 |
| 1,305–2,354 | 43 | 5,041 | 0.7419 | 0.7457 | 0.7667 | 0.7411 |
| 2,367–9,539 | 45 | 8,059 | 0.8049 | 0.8156 | 0.8790 | 0.8716 |

## Per-AOI summary on unique test records

| Scope | AOIs | Records | Median accuracy | Median AUROC | AOIs with accuracy gain | AOIs with AUROC gain |
|---|---:|---:|---:|---:|---:|---:|
| AOIs with at least 100 records and both classes | 44 | 13,030 | 0.8120 | 0.8453 | 15 | 30 |
