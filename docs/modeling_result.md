# Modeling results and open questions

Last updated: 2026-09-14

This file tracks the latest model result, the questions it raises, and the next experiments. Treat subgroup findings as exploratory because we inspected the current test set while developing these questions. Confirm decisions on a fresh geographic holdout and across several training seeds.

## Current reference result

Reference run: `delta_nox_classification_20260914_215026`, Slurm job `38853856`. It contains 40,630 training, 8,212 validation, and 9,842 balanced test records.

| Test metric | CNN | XGBoost | CNN minus XGBoost |
|---|---:|---:|---:|
| Accuracy | 0.6577 | 0.6515 | +0.0062 |
| ROC AUC | 0.7107 | 0.6840 | +0.0267 |
| Recall for increases | 0.6889 | 0.6537 | +0.0352 |
| Specificity for decreases | 0.6265 | 0.6493 | -0.0228 |

The CNN improves ranking more than 0.5-threshold decisions. It predicts the increase class more often than XGBoost, which creates the recall gain and specificity loss. Validation-selected thresholds raise the CNN versus XGBoost balanced-accuracy gap to 1.18 percentage points, but threshold tuning does not fix the underlying decrease weakness.

## Answers to current questions

### Is plume scoring and filtering sensible?

The regional score makes sense as a post-hoc plume-presence diagnostic. The current hard filter does not make sense as a training-record or headline-test eligibility rule.

The score takes the larger positive-plume score from the current and previous rasters. It therefore conditions the sample on observed NO2 structure, which the CNN receives as input. The 40th-percentile training cutoff retains 60.0% of training candidates, 52.4% of validation candidates, and 57.7% of test candidates. On test it retains 61.0% of decreases and 55.4% of increases, so simple class attrition does not explain weak decrease specificity. The filter still changes class-conditional raster distributions. Among records that already pass the gate, tighter plume cutoffs do not improve validation accuracy.

Recommendation: remove plume score from dataset selection. Keep coverage, retrieval quality, corridor coverage, coherent wind, and raster-extent checks. Report plume-score strata after inference. If deployment needs abstention, select a gate on validation data and publish its class-specific risk-coverage curve.

### Did the dual spatial and magnitude encoder improve performance?

The single-seed ablation does not support the magnitude branch.

| Model | Validation accuracy | Validation AUC | Test accuracy | Test AUC |
|---|---:|---:|---:|---:|
| Reference dual encoder | 0.6443 | 0.6784 | 0.6577 | 0.7107 |
| Magnitude encoder disabled | 0.6416 | 0.6762 | **0.6652** | **0.7164** |

Removing the magnitude encoder improves test accuracy by 0.75 points, test AUC by 0.57 points, and test log loss by 0.0045. AOI-cluster bootstrap intervals for the test improvement exclude zero, but they do not include training-seed variance. Validation moves slightly in the other direction. Keep the simpler spatial encoder as the working model and run at least five seeds before making the architecture decision permanent.

### Would persistent-change framing increase accuracy?

It should increase headline accuracy by removing ambiguous one-hour changes, while changing the estimand to sustained plant behavior.

About 74% of current test transitions retain at least half of the initial raw change for the next two hours. On this post-hoc subset, CNN accuracy rises from 65.77% to 69.28%; XGBoost rises from 65.15% to 68.77%. Nonpersistent cases score 55.81% and 54.87%. Selection explains part of that increase, and neither model received retraining. The CNN's relative gain stays near 0.5 points on persistent records, so persistence should improve absolute accuracy more than CNN advantage.

Recommendation: adopt the sustained-regime rule because it matches the scientific question. Use future emissions for eligibility alone. Require complete CAMPD measurements and retain every audit field in `AGENTS.md` before treating the estimate as final.

### Which XGBoost features matter now?

XGBoost excludes `delta_flux_norm`. Tree total gain ranks the remaining features as follows:

| Feature group | Share of total gain |
|---|---:|
| 2 m temperature | 59.7% |
| Day-of-year sine and cosine | 17.3% |
| Local-solar-hour sine and cosine | 15.8% |
| Plant activity and capacity | 6.3% |
| Boundary-layer height | 0.9% |

XGBoost relies on meteorological and calendar shortcuts more than plant descriptors. Tree gain favors continuous features and does not measure causality. Run grouped permutation ablations for weather, calendar, and plant blocks on validation data before removing features. The present ranking raises a useful concern: season and temperature may proxy for label-selection or operating-pattern differences across held-out AOIs.

### Does the CNN need the flux feature?

No measurable accuracy benefit appears in the controlled ablation.

| Model | Validation accuracy | Validation AUC | Test accuracy | Test AUC |
|---|---:|---:|---:|---:|
| Reference with `delta_flux_norm` | 0.6443 | 0.6784 | 0.6577 | 0.7107 |
| Flux scalar disabled | **0.6470** | **0.6795** | 0.6577 | **0.7133** |

Removing the scalar leaves test accuracy unchanged and improves test AUC by 0.26 points and log loss by 0.0020. The AOI-cluster AUC interval is +0.14 to +0.36 points, conditional on this seed. Remove `delta_flux_norm` from the next model. Preserve flux outputs as diagnostics for physical agreement and observability.

### Would scan-time interpolation help?

Exact overlap weighting alone should have little effect on this dataset. A post-hoc reconstruction weights the current and next hourly changes by their overlap with each paired L2 scan interval. Of 9,842 test records, 8,656 remain outside the 100 lb deadband. Only 0.058% of those records change class. Relabeling without retraining raises both models' accuracy by 0.058 points.

Implement overlap weighting for correctness and continuous-target fidelity. Do not expect it to drive a material gain by itself. The three-hour exponentially weighted `effective_delta_nox` target in `AGENTS.md` changes the label more substantially and better represents transport and chemical persistence.

## Ideas from weather-modeling research

1. **Use a physics-guided transport residual.** Advect or warp the previous NO2 field with the observed wind, then ask the network to encode the current-minus-advected residual. [NowcastNet](https://www.nature.com/articles/s41586-023-06184-4) combines a physical evolution scheme with learned correction. [NeuralGCM](https://www.nature.com/articles/s41586-024-07744-y) uses the same hybrid principle at global scale. This is the highest-priority architecture idea for plume transport.

2. **Condition on time and use nested spatial context.** Supply scan interval, emissions-bin ages, transport distance, and boundary-layer state as explicit conditioning variables. Pair a source-centered crop with a longer downwind corridor. [MetNet-3](https://arxiv.org/abs/2306.06079) shows how a model can combine dense fields with sparse observations at high spatial and temporal resolution.

3. **Predict a distribution over the continuous target.** Train on `effective_delta_nox` with Huber loss or a heteroscedastic likelihood, then derive deadband classes from the predicted distribution. This preserves magnitude and expresses uncertainty near zero. Ensembles can measure seed and observation uncertainty; [FourCastNet](https://arxiv.org/abs/2202.11214) emphasizes the value of cheap neural ensembles.

4. **Reserve graph or spectral models for a larger dataset.** [GraphCast](https://www.science.org/doi/10.1126/science.adi2336) motivates message passing when irregular sources and multi-scale transport interactions matter. A graph network or Fourier operator is premature for 48 by 48 patches and roughly 40,000 training examples.

## Recommended next model

1. Rebuild the label and eligibility logic around three-hour `effective_delta_nox`, sustained regimes, complete emissions, coherent winds, corridor coverage, and raster extent.
2. Remove plume-score filtering and the CNN flux scalar. Start with the spatial encoder without the magnitude branch.
3. Predict continuous effective delta and binary direction together. Select loss weights and probability calibration on validation AOIs.
4. Compare XGBoost, tabular MLP, image-only CNN, and fused CNN on identical records across at least five seeds. Report AUC, log loss, calibration, accuracy, recall, specificity, and AOI-cluster intervals.
5. Add wind-advection residuals and nested downwind context only after the revised target establishes a clean baseline.

The current 25-epoch early-stopping patience adds about 15 unproductive epochs after validation peaks near epoch 3. Reducing patience will save GPU time without changing checkpoint selection.

## Experiment ledger

| Date | Run or artifact | Question | Status |
|---|---|---|---|
| 2026-09-14 | `delta_nox_classification_20260914_215026` | Reference dual encoder | Complete |
| 2026-09-14 | Slurm `38857420`, run suffix `no-magnitude` | Magnitude-encoder ablation | Complete |
| 2026-09-14 | Slurm `38857421`, run suffix `no-flux` | CNN flux-scalar ablation | Complete |
| 2026-09-14 | `reports/model_strata_eda_20260914/` | Test strata and persistence proxy | Complete |
| 2026-09-14 | `reports/modeling_result_20260914/` | Plume retention, scan overlap, XGBoost gain, ablation intervals | Complete |

## Open questions

- Does the three-hour target reduce the increase/decrease calibration asymmetry on new AOIs?
- Does the spatial encoder retain its advantage after removing response-based plume selection?
- Does a wind-advected residual improve decreases, where the previous plume should decay or leave the corridor?
- Do temperature and calendar features remain useful after sustained-regime selection?
- Does the magnitude branch help under the revised continuous target across training seeds?
