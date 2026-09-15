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

## Main implications

Two problems are entangled in the current result.

1. **The target is instantaneous, but the observed column is a state with memory.** A TEMPO NO2 column contains material emitted during several earlier hours, transported into and out of the crop and removed by chemistry. Pairing two scans does not make either scan representative of one clock-hour change. The physically aligned target is therefore a *causal, transport-weighted emission state* evaluated at each scan, or, in a more ambitious sequence model, the current emission conditioned on an explicit latent plume state.
2. **The fused model has no incentive to learn a weaker raster signal after weather and calendar variables solve much of the task.** This is a known multimodal optimization failure, not proof that the rasters contain no information. The next evaluation should make incremental raster skill the estimand, and the next fusion model should learn a correction to a frozen tabular baseline rather than allow unconstrained feature concatenation.

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

Removing the magnitude encoder improves test accuracy by 0.75 points, test AUC by 0.57 points, and test log loss by 0.0045. AOI-cluster bootstrap intervals for the test improvement exclude zero, but they do not include training-seed variance. Validation moves slightly in the other direction. The magnitude and restitution paths are now removed from the working architecture rather than exposed as runtime options. A future architecture comparison should still use several seeds.

### What label respects plume memory?

The present sign of one hourly CAMPD difference is not well aligned with what the raster pair can observe. A column observation at scan time `i` is better represented schematically as

```text
Omega_i(x) = background_i(x)
             + sum_k H_i,k(x; meteorology) * emission_i-k
             + retrieval_error_i(x)
```

`H_i,k` is the sensitivity of the observed column to emissions `k` hours earlier. Atmospheric inverse models call this a source-receptor footprint. It contains advection, dilution, residence in the crop, and chemical loss; consequently it changes with wind, boundary-layer structure, sunlight, and plume chemistry. This source-history formulation is standard in receptor-oriented inverse modeling,[^2] while satellite NO2 studies explicitly fit transport and exponential chemical decay and find lifetimes of several hours rather than zero memory.[^3] Recent power-plant work also shows that plume-scale NOx lifetime and NO2-to-NOx conversion require explicit correction.[^4]

The first replacement target should be a continuous, causal effective-emission state:

```text
effective_emission_i = sum_k w_i,k * interval_emission_i-k
delta_effective_i    = effective_emission_i - effective_emission_i-1
```

Use only emissions available at or before each scan. Normalize nonnegative weights `w_i,k` to one and truncate the history only after its remaining weight is negligible. The strongest version derives `w_i,k` from an LPDM/adjoint footprint. A practical first version uses a fixed exponential residence kernel with scan-interval overlap handled exactly. Evaluate `tau` values of 1, 2, 4, and 6 h on training/validation AOIs before trying a meteorology-dependent lifetime. This sensitivity range is deliberately broad: published lifetime estimates depend on source, season, wind, and the spatial scale of the retrieval.[^3][^4] Because a meteorology-dependent kernel writes weather directly into the target, it should advance only if it improves raster-only and residual skill rather than merely strengthening the tabular baseline.

This target represents the emission history visible to the satellite, not instantaneous stack emissions. If instantaneous emissions remain the scientific estimand, smoothing the label is the wrong final solution. Instead train on sequences with a latent plume state:

```text
state_i       = transport(state_i-1, meteorology_i) + source(emission_i)
observed_NO2_i = observation_operator(state_i) + error_i
```

Then supervise the current scan-interval emission or its change while carrying the state forward. This is harder but preserves the desired instantaneous estimand.

The current persistence result remains useful, but it answers a different question. It selects sustained operating events using future stack measurements; it does not model material from previous hours that is present in the current plume.

About 74% of current test transitions retain at least half of the initial raw change for the next two hours. On this post-hoc subset, CNN accuracy rises from 65.77% to 69.28%; XGBoost rises from 65.15% to 68.77%. Nonpersistent cases score 55.81% and 54.87%. Selection explains part of that increase, and neither model received retraining. The CNN's relative gain stays near 0.5 points on persistent records, so persistence should improve absolute accuracy more than CNN advantage.

Recommendation: use persistent-change eligibility only for a separately named offline estimand such as “sustained regime change.” Do not use future emissions in the primary real-time label. For the primary experiment, compare the hard-hour target with causal effective-emission targets on frozen splits and preserve continuous values before deriving deadband classes. Re-estimate a defensible deadband for each continuous target; the current 100 lb threshold does not automatically retain the same meaning after temporal filtering.

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

Removing the scalar leaves test accuracy unchanged and improves test AUC by 0.26 points and log loss by 0.0020. The AOI-cluster AUC interval is +0.14 to +0.36 points, conditional on this seed. `delta_flux_norm` is absent from dataset generation and model inputs, and the standalone estimator has been removed.

### Would scan-time interpolation help?

Exact overlap weighting alone should have little effect on this dataset. A post-hoc reconstruction weights the current and next hourly changes by their overlap with each paired L2 scan interval. Of 9,842 test records, 8,656 remain outside the 100 lb deadband. Only 0.058% of those records change class. Relabeling without retraining raises both models' accuracy by 0.058 points.

Implement overlap weighting for correctness and continuous-target fidelity. Do not expect it to drive a material gain by itself: overlap weighting fixes clock alignment, whereas a causal footprint or residence kernel fixes atmospheric memory. These are complementary operations.

## How to make the raster earn its place

### Diagnose conditional information before changing the network

The relevant question is not whether the CNN beats XGBoost. It is whether NO2 adds held-out information *conditional on* the weather, calendar, and plant baseline. Use identical records and seeds for the following ladder:

| Model or intervention | What it establishes |
|---|---|
| Prevalence/constant | Dataset floor |
| Tabular-only MLP and XGBoost | Shortcut ceiling available without NO2 |
| Raster-only model | Standalone learnability of the image signal |
| Frozen-tabular-plus-raster residual model | Incremental NO2 skill |
| Fused model with NO2 values removed but masks retained | Benefit due only to coverage/missingness |
| Fused model with NO2 rasters permuted within AOI, season, solar-hour, and weather bins | Whether aligned plume structure matters beyond matched context |
| Wind rotated or spatial quadrants occluded | Whether the model uses a source-relative downwind pattern |

Make improvement in validation log loss over the frozen tabular model the primary selection statistic. Also report AUC, Brier score, calibration, and accuracy, with paired AOI-cluster bootstrap intervals. A fusion model should advance only if it beats the tabular baseline across seeds and loses that advantage when the NO2 values are conditionally permuted. Global feature importance or a visually plausible saliency map does not establish incremental information.

### Train the image as a residual expert

Simple late fusion allows temperature and time to reduce the loss quickly while the noisier image encoder receives little useful gradient. Multimodal studies describe this as modality competition or modality laziness and show that independently useful encoders can be under-trained during joint optimization.[^8][^9]

Use a two-stage additive model:

```text
baseline_logit = g(plant, temperature, PBL, solar_hour, day_of_year)
final_logit    = stop_gradient(baseline_logit) + r(NO2_sequence, wind, physics_context)
```

1. Fit `g` first and freeze it. Generate out-of-fold training logits so that the raster branch never learns against an overfit nuisance prediction.
2. Train `r` to improve likelihood beyond that frozen logit. Give the image encoder its own auxiliary raster-only head so it must retain label-relevant information.
3. Only after this works, optionally unfreeze both branches with a much smaller learning rate for `g`. Compare against separate unimodal training followed by a calibrated weighted average and against tabular-modality dropout.

This construction does not manufacture image information: `r` will converge toward zero if the raster has no conditional signal. It makes that failure observable. Weather needed for plume transport may enter `r`; freezing `g` prevents the same variables from erasing the optimization pressure on the raster. Do not adversarially remove temperature or season from the image representation, because chemistry, mixing, and retrieval sensitivity genuinely depend on them.

## Physics-guided raster representations

1. **Turn the scan pair into a continuity-equation innovation.** The vertically integrated tracer balance contains a tendency term, horizontal transport/divergence, emissions, and chemical loss. A 2025 study applied this directly to hourly TEMPO observations at 14 US power plants; adding the column-tendency term improved correlation with CEMS at 10 plants and overall.[^1] Supply the network with `dOmega/dt`, `u dot grad(Omega)` or `div(u * Omega)`, and an advected residual `Omega_i - Advect(Omega_i-1)` rather than expecting a generic CNN to discover these derivatives from two channels. Calculate derivatives on native L2 geometry before coarse regridding when feasible, because numerical differentiation amplifies smoothing and retrieval noise.[^1]

2. **Use a short sequence, not independent pairs.** Feed at least the last 3-6 scans, their masks, exact observation times, and meteorology at each step. Encode each scan spatially, then use a small ConvGRU/temporal attention block or an explicitly advected recurrent state. Autoregressive weather models treat the atmosphere as an evolving state, while NowcastNet explicitly advects the previous field and learns an intensity residual.[^6] NeuralGCM similarly retains a dynamical core and learns unresolved physical tendencies.[^7]

3. **Use source-receptor geometry.** Rotate or resample crops into along-wind/cross-wind coordinates; include distance and travel time from the stack; and compare surface wind with wind near effective plume height. Add nested source-centered and longer downwind crops. The network should see where an emission from each prior hour can be at the current scan, not just four colocated channels.

4. **Give the model an inexpensive transport prior.** A Gaussian-puff or differentiable semi-Lagrangian layer can advect, diffuse, and exponentially decay prior emissions or the previous column. Let the CNN learn the correction. FootNet predicts atmospheric source-receptor footprints from HRRR variables and found that a Gaussian-plume first guess materially improved its transport emulator; its history sensitivity saturated after 6 h in that application, although the authors caution that the window is scale-dependent.[^5] GATES likewise learns time-integrated transport footprints from several meteorological snapshots rather than treating the measurement time independently.[^10]

5. **Add forward and auxiliary tasks.** Pretrain the raster encoder on all eligible consecutive scans to predict the next observed NO2 field or its advected residual, using masks in the loss. During supervised training, jointly predict continuous `delta_effective`, its direction, and physically interpretable summaries such as downwind excess mass or continuity-based emission proxy. A forward state-prediction loss gives many more training examples and makes ignoring the raster impossible, while the final held-out incremental-skill test still decides whether that representation helps emissions inference.

6. **Model uncertainty instead of forcing ambiguous labels.** Predict a distribution over the continuous target with Huber/Student-t or heteroscedastic likelihood, then calculate `P(delta_effective > deadband)` and `P(delta_effective < -deadband)`. Abstain when neither probability is high. Retrieval error, missing support, uncertain winds, background subtraction, lifetime, and NOx/NO2 conversion should propagate into evaluation rather than be hidden by a hard class.

7. **Use simulation pretraining only after the simple operator baseline.** If observed pairs remain too noisy, pretrain the transport/image branch on Gaussian-puff, LPDM, LES, or CTM scenes with randomized lifetime, winds, backgrounds, missingness, and retrieval noise, then fine-tune on TEMPO/CEMS. Learned transport emulators demonstrate that this direction is feasible,[^5][^10] but synthetic-to-real bias makes it a second-stage experiment, not the first fix.

Full PINN training over concentration, velocity, diffusion, chemistry, and source terms is not the first recommendation. The inverse problem is weakly identified from sparse, noisy columns, and an unconstrained PINN would add many latent degrees of freedom. A fixed differentiable transport operator plus a learned residual is easier to audit and follows successful hybrid weather-modeling practice.[^6][^7]

## Recommended next model

Run the work in this order so that each experiment answers one question.

1. **Label sweep:** retain the exact overlap calculation and build causal exponential targets for `tau = 1, 2, 4, 6 h`. Predict the continuous effective-emission difference and derive direction classes afterward. Keep the hard-hour and future-persistent targets as explicitly separate comparators.
2. **Information audit:** rerun tabular-only, raster-only, naive fusion, mask-only, and conditionally permuted-raster controls on identical records across at least five seeds. This determines whether a predictive edge exists before more architecture work.
3. **Residual fusion:** freeze an out-of-fold tabular baseline and train the image/physics correction with an auxiliary raster-only loss. Select by improvement in validation log loss over the frozen baseline.
4. **Physics channels:** add tendency, advection/divergence, and previous-field advection residuals. Test winds at plausible plume heights and a source-aligned crop. Do not reintroduce the old learned scalar flux feature; expose the spatial physics fields themselves.
5. **Temporal state:** if the correction has reproducible skill, replace scan pairs with a 3-6-scan recurrent sequence and compare the causal effective-emission label with instantaneous-emission supervision under an explicit latent plume state.
6. **Only then expand:** try footprint emulation or synthetic CTM/LPDM pretraining. Graph and spectral architectures are premature for 48 by 48 crops and about 40,000 training examples.

Keep plume score diagnostic-only, require complete emissions histories for every kernel window, and retain coherent-wind, corridor-coverage, and raster-extent checks. Report the target kernel, all scan/emissions timestamps, every contributing hourly emission and weight, and the retained kernel mass so the label can be audited.

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

- Which causal kernel window and lifetime maximize validation likelihood without erasing real operating changes?
- Does a footprint-weighted target improve raster-only skill, or merely make the calendar baseline easier?
- Does the spatial encoder retain its advantage after removing response-based plume selection?
- Does a frozen-baseline raster residual improve log loss across seeds and lose that gain under conditional raster permutation?
- Does a continuity/advection representation improve decreases, where the previous plume should decay or leave the corridor?
- How much additional skill comes from wind at plume height, solar radiation/photochemical proxies, and time-varying lifetime?
- Can an explicit latent-state sequence model recover instantaneous emissions well enough to avoid changing the estimand?

## Sources

[^1]: K. Sun et al., “[Hourly Nitrogen Oxides Emissions Estimated From TEMPO and Comparison With Facility-Level Monitoring Data](https://doi.org/10.1029/2025JD044565),” *Journal of Geophysical Research: Atmospheres*, 2025.
[^2]: C. Gerbig et al., “[Toward constraining regional-scale fluxes of CO2 with atmospheric observations over a continent: 2. Analysis of COBRA data using a receptor-oriented framework](https://doi.org/10.1029/2003JD003770),” *Journal of Geophysical Research: Atmospheres*, 2003.
[^3]: S. Beirle et al., “[Megacity emissions and lifetimes of nitrogen oxides probed from space](https://doi.org/10.1126/science.1207824),” *Science*, 2011.
[^4]: G. Kuhlmann et al., “[Temporal variability of NOx emissions from power plants: a comparison of satellite- and inventory-based estimates](https://doi.org/10.5194/acp-26-4405-2026),” *Atmospheric Chemistry and Physics*, 2026.
[^5]: T.-L. He et al., “[FootNet v1.0: development of a machine learning emulator of atmospheric transport](https://doi.org/10.5194/gmd-18-1661-2025),” *Geoscientific Model Development*, 2025.
[^6]: Y. Zhang et al., “[Skilful nowcasting of extreme precipitation with NowcastNet](https://doi.org/10.1038/s41586-023-06184-4),” *Nature*, 2023.
[^7]: D. Kochkov et al., “[Neural general circulation models for weather and climate](https://doi.org/10.1038/s41586-024-07744-y),” *Nature*, 2024.
[^8]: W. Wang, D. Tran, and M. Feiszli, “[What Makes Training Multi-Modal Classification Networks Hard?](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html),” *CVPR*, 2020.
[^9]: C. Du et al., “[On Uni-Modal Feature Learning in Supervised Multi-Modal Learning](https://proceedings.mlr.press/v202/du23e.html),” *Proceedings of Machine Learning Research*, 2023.
[^10]: E. Fillola et al., “[Enabling fast greenhouse gas emissions inference from satellites with GATES: a Graph-Neural-Network Atmospheric Transport Emulation System](https://doi.org/10.5194/gmd-19-1893-2026),” *Geoscientific Model Development*, 2026.
