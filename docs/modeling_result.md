# Modeling results and next experiments

Last updated: 2026-09-17

## Executive readout

- The latest completed run, `delta_nox_classification_20260917_115919`, reaches
  0.8267 test AUROC and 75.77% accuracy on the saved balanced test table.
- The frozen tabular MLP already reaches 0.8107 AUROC and 75.25% accuracy. The
  raster branch adds 0.0160 AUROC and 0.51 percentage points of accuracy.
- Day of year supplies most of the MLP's ranking signal. A season-only probe
  reaches about 0.787 test AUROC, while disrupting season within AOI reduces the
  trained MLP from 0.816 to about 0.553 AUROC.
- Evaluation oversampling duplicates 31.6% of test rows. On the 15,562 unique
  test records, fusion raises AUROC from 0.8160 to 0.8322 but lowers accuracy
  from 79.10% to 78.43% and worsens log loss from 0.4986 to 0.5189.
- Raster gains appear across emissions-change levels and most adequately sized
  AOIs, but the model remains close to random ranking at several held-out AOIs.
- The raster branch contains NO2, its validity mask, temperature, and wind.
  These results show conditional value from the branch as a whole; they do not
  isolate information from NO2 columns.
- The fused model selected epoch 1, overfit after that, and produced NaN losses
  at epoch 16. Architecture work should start with evaluation and optimization
  fixes, followed by controlled raster ablations and physics-guided pretraining.

## Run contract

| Item | Value |
|---|---|
| Run | `delta_nox_classification_20260917_115919` |
| Seed | 42 |
| Train / validation / test rows | 83,764 / 21,244 / 22,746 |
| Unique raster records | 58,860 / 15,940 / 15,562 |
| Train / validation / test AOIs | 556 / 107 / 134 |
| Target | Sign of causal effective NOx change outside a 100 lb deadband |
| Raster sequence | Five 24 x 24 hourly frames, oldest to newest |
| Raster inputs | NO2, validity mask, 2 m temperature, and 80 m winds |
| Tabular inputs | Plant attributes, prior-quarter same-hour activity, solar hour, and season |
| Baseline | Frozen 993-parameter tabular MLP |
| Fused model | 700,545 trainable parameters, spatial encoder and ConvGRU |
| Fusion | Additive correction to the frozen MLP logit |

The split is geographic, so validation and test measure transfer to held-out
AOIs. Exact class balancing oversampled the minority class inside every split.
That choice created duplicate validation and test rows and changed their class
prevalence from 66.6% and 73.1% positive to 50%. The two evaluation views below
serve different purposes:

- **Artifact view:** reproduces `results.json` and weights duplicated rows.
- **Unique-record view:** removes repeated `raster_bundle_path` values and
  retains the raster-qualified sample's observed prevalence.

Future reports should use unique records as the primary evaluation set.
Balancing belongs in the training sampler or loss, not validation or test data.

## Performance across splits

### Artifact-reported balanced view

| Split | Model | Accuracy | AUROC | Log loss | Brier score |
|---|---|---:|---:|---:|---:|
| Train | ConvGRU + MLP | 0.8578 | 0.9265 | 0.3469 | 0.1056 |
| Train | MLP | 0.8478 | 0.9180 | 0.3636 | 0.1119 |
| Validation | ConvGRU + MLP | 0.7701 | 0.8491 | 0.5363 | 0.1690 |
| Validation | MLP | 0.7643 | 0.8418 | **0.5186** | **0.1685** |
| Test | ConvGRU + MLP | 0.7577 | 0.8267 | **0.5524** | **0.1778** |
| Test | MLP | 0.7525 | 0.8107 | 0.5612 | 0.1824 |

Fusion improves accuracy and AUROC on all three splits. The AUROC gain is
smaller on validation (+0.0073) than test (+0.0160), and the train-to-test gap
is 0.100 AUROC. Both patterns point to limited transfer across AOIs. Validation
log loss selects the MLP, despite the fused model's higher validation AUROC.

### Unique-record view

| Split | Model | Accuracy | Balanced accuracy | AUROC | Log loss |
|---|---|---:|---:|---:|---:|
| Train | ConvGRU + MLP | 0.8635 | 0.8636 | 0.9308 | 0.3512 |
| Train | MLP | 0.8596 | 0.8550 | 0.9231 | **0.3498** |
| Validation | ConvGRU + MLP | 0.7527 | 0.7703 | 0.8493 | 0.5930 |
| Validation | MLP | 0.7502 | 0.7644 | 0.8420 | **0.5550** |
| Test | ConvGRU + MLP | 0.7843 | 0.7630 | 0.8322 | 0.5189 |
| Test | MLP | **0.7910** | 0.7586 | 0.8160 | **0.4986** |

The natural test view is 73.1% positive. Fusion shifts test logits downward by
0.146 on average. This improves specificity from 0.6882 to 0.7169 but lowers
recall from 0.8289 to 0.8091, so accuracy falls. The raster correction changes
the predicted class correctly for 231 records and incorrectly for 336 records.
It reduces per-record log loss for 63.1% of records, yet a smaller set of large
errors raises mean log loss.

An AOI-cluster bootstrap with 1,000 resamples gives these 95% intervals for
ConvGRU plus MLP minus MLP on unique test records:

| Difference | 95% interval |
|---|---:|
| Accuracy | -0.0105 to -0.0022 |
| AUROC | +0.0112 to +0.0210 |
| Log loss | +0.0096 to +0.0309 |
| Brier score | +0.0013 to +0.0070 |

The intervals cover geographic sampling for this trained seed. They omit
training-seed variance and choices made after reviewing earlier test runs.

## Test performance by emissions-change magnitude

The following tertiles use unique records and absolute effective NOx change.

| Absolute change | Range (lb) | N | Accuracy | AUROC | Accuracy gain vs MLP | AUROC gain vs MLP |
|---|---:|---:|---:|---:|---:|---:|
| Low | 100.0 to 133.3 | 5,188 | 0.7787 | 0.8316 | -0.0079 | +0.0138 |
| Middle | 133.3 to 201.5 | 5,187 | 0.7872 | **0.8461** | -0.0069 | **+0.0179** |
| High | 201.5 to 4,300.3 | 5,187 | 0.7870 | 0.8204 | -0.0054 | +0.0167 |

Large changes are not easier. The high tertile has the weakest AUROC, although
its labels should sit farthest from the deadband. Large facilities, shutdowns,
startup periods, nonlinear chemistry, and plume motion outside the local crop
could all contribute. The current outputs cannot separate those explanations.

The raster branch adds a similar AUROC increment in all three tertiles. Its
accuracy penalty also appears in all three because the downward logit shift
trades positive recall for negative specificity under positive-heavy natural
prevalence.

## Test performance across AOIs

Per-AOI metrics become unstable for small sites or sites with one observed
class. Restricting the summary to 44 AOIs with at least 100 unique records and
both classes leaves 13,030 of 15,562 test records.

| Per-AOI statistic | Accuracy | AUROC | Accuracy gain vs MLP | AUROC gain vs MLP |
|---|---:|---:|---:|---:|
| Median | 0.8120 | 0.8453 | -0.0056 | +0.0085 |
| 25th percentile | 0.7705 | 0.7889 | | |
| 75th percentile | 0.8563 | 0.8929 | | |
| AOIs with a positive gain | | | 15 of 44 | 30 of 44 |

Performance varies more than the pooled score suggests. AOIs 55463 and 2951
have AUROCs of 0.5215 and 0.5421 across 251 and 256 records. Their accuracies
remain near 84% because about 90% of their labels are positive. At the other
end, AOIs 634, 8049, 7238, and 54466 exceed 0.977 AUROC with 437 to 509 records.
The model can rank changes at many plants, but it does not learn a transport or
operations relationship that transfers to every held-out setting.

Capacity strata show where the middle of the fleet falls short:

| AOI capacity tertile | AOIs | Records | Accuracy | AUROC | Accuracy gain vs MLP | AUROC gain vs MLP |
|---|---:|---:|---:|---:|---:|---:|
| Low, 133 to 1,269 MW | 46 | 2,462 | 0.8034 | 0.8339 | 0.0000 | +0.0089 |
| Middle, 1,305 to 2,354 MW | 43 | 5,041 | 0.7419 | 0.7667 | -0.0038 | **+0.0256** |
| High, 2,367 to 9,539 MW | 45 | 8,059 | 0.8049 | **0.8790** | -0.0107 | +0.0074 |

The middle-capacity group has the lowest absolute performance and the largest
raster AUROC gain. Fuel grouping changes little: gas-only, mixed, and coal-only
AOIs reach 0.819, 0.846, and 0.819 AUROC. Raster AUROC gains range from +0.011
to +0.014 across those groups.

## How much do the rasters inform the model?

The tabular MLP explains most of the observed signal, and season explains most
of the MLP. It sees plant capacity and unit counts, prior-quarter same-hour
average heat input and power generation, solar hour, and day of year. It does
not receive AOI ID, raw coordinates, date, year, current emissions, or current
power generation.

The label distribution contains a strong annual pattern. Among unique test
records, the positive share is 12.3% in January, reaches 87.8% to 90.0% from
June through September, and falls to 27.5% in December. Training records follow
the same shape. A smoothed day-of-year lookup fitted on unique training records
reaches 0.787 test AUROC without plant or operations features.

Post-hoc checks on the selected MLP give the same result:

| Check on unique test records | AUROC |
|---|---:|
| Full MLP | 0.8160 |
| Keep only day-of-year inputs; set other normalized inputs to zero | 0.7831 |
| Keep all time inputs; set other normalized inputs to zero | 0.7899 |
| Remove day-of-year inputs by setting them to zero | 0.5293 |
| Remove prior-quarter activity inputs | 0.8086 |

Conditional permutation within each AOI lowers AUROC by about 0.263 for day of
year, 0.034 for solar hour, and 0.005 for prior-quarter activity. These effects
are not additive because the MLP learns interactions, and zeroing inputs creates
out-of-distribution combinations. The gap is large enough to identify a smooth
seasonal shortcut as the main source of tabular skill. The geographic split
prevents direct AOI memorization, but it does not prevent a calendar pattern
shared across train and test AOIs.

The current analysis does not establish why the label is so seasonal. The
causal target compares two overlapping five-hour emissions averages, TEMPO
restricts examples to daytime observation windows, and the 100 lb deadband
keeps only large changes. Seasonal generation ramps, observation timing, and
selection effects can all create the observed prevalence curve. The next data
audit should report label prevalence by day of year and local solar hour before
and after the deadband and raster-quality filters.

The fused model can only add a correction to this strong frozen prediction.

The raster branch adds a repeatable-looking 0.016 test AUROC for this seed, and
the gain spans magnitude and fuel groups. Its mean absolute logit correction is
0.407 on unique test records. That is enough to change ranking, but it does not
improve probability quality or natural-prevalence accuracy.

No current ablation identifies the source of the gain. The branch can use:

- spatial and temporal NO2 structure;
- weather fields without NO2;
- mask patterns tied to clouds, season, or scan geometry;
- AOI-specific backgrounds and artifacts.

Calling the measured increment an NO2 benefit would overstate the evidence.
NO2-only, weather-only, mask-only, and conditional-permutation controls should
precede claims about satellite information.

## Training failure modes

The frozen MLP reached its best validation loss at epoch 2. The fused model
selected epoch 1 at 0.5363, already worse than the MLP's 0.5186. Training loss
then fell from 0.3521 to 0.0209 while validation loss rose to 1.424 by epoch 15.
Both losses became NaN at epoch 16 and remained NaN until early stopping at
epoch 26. Checkpoint restoration preserved usable predictions, but the run
spent most of its budget fitting noise and then operating in an invalid state.

Likely contributors include a 700K-parameter branch learning a small residual,
the frozen baseline's in-sample training logits, repeated training records, and
one optimization schedule for spatial and temporal components. Multimodal work
has found that streams can overfit at different rates and that joint models can
underlearn one modality.[^gradient-blending] The current trajectory matches
that risk.

## Next modeling phase

### 1. Repair evaluation and establish attribution

1. Oversample or reweight training only. Keep validation and test rows unique,
   report natural prevalence, and add a separately reweighted balanced view.
2. Repeat MLP, fused, raster-only, NO2-only, weather-only, and mask-only models
   across at least five seeds. Use paired AOI-cluster intervals.
3. Permute NO2 within AOI, season, solar-hour, and weather bins. Also shuffle
   temporal order and rotate wind. These controls preserve easy nuisance cues
   while breaking the relationships the raster encoder should learn.
4. Choose the operating threshold and any calibration map on validation AOIs.
   Keep AUROC, log loss, Brier score, recall, and specificity beside accuracy.

### 2. Stabilize fusion before adding capacity

- Stop on non-finite loss and save diagnostics from the first bad batch. Lower
  the raster learning rate, shorten patience, and test stronger weight decay.
- Generate out-of-fold MLP logits for training the correction branch. This
  prevents the raster encoder from fitting residuals against in-sample baseline
  predictions that are cleaner than validation predictions.
- Add a raster-only auxiliary direction head and track gradient cosine
  similarity on the shared encoder. Decay the auxiliary weight during training.
- Compare the additive correction with a small gated fusion layer. Keep the
  parameter budget fixed and require validation log-loss improvement.

### 3. Make next-raster prediction the first physics experiment

The next phase should follow the experiment order in `AGENTS.md`. Pretrain the
spatial encoder and ConvGRU on four causal frames to predict the fifth:

```text
baseline = advect(previous_no2, wind_u, wind_v)
predicted_next_no2 = baseline + bounded_correction
```

Use masked Huber loss on observed target pixels. Compare unconstrained
next-raster prediction, advection plus correction, and random initialization
under identical downstream splits and budgets. Retain a fifth-raster auxiliary
head during classification and decay its loss weight.

Hourly TEMPO research found that adding the observed NO2 column tendency
improved facility-level emission estimates, which supports learning temporal
evolution rather than treating frames as unordered texture.[^tempo-tendency]
NowcastNet offers a useful architectural precedent for differentiable advection
plus a learned intensity residual, though precipitation and NO2 chemistry obey
different source and loss processes.[^nowcastnet]

Test one constrained positive global lifetime only after advection alone. Add
diffusion, conditioned decay, a continuity residual, or bounded wind correction
one at a time. Avoid per-pixel lifetimes and free motion fields that could absorb
the emissions-change target.

### 4. Add source-relative geometry

Provide along-wind distance, cross-wind distance, travel time, source masks,
and a Gaussian-plume footprint without observed NO2 magnitude. FootNet found
surface wind and a Gaussian-plume first guess to be its most useful transport
emulator inputs.[^footnet] These fields may help the model transfer plume
geometry between AOIs and give it a useful prior when TEMPO retrievals are
noisy.

Use a larger or downwind-elongated context only if advection diagnostics show
that plume support often leaves the 72 km crop. Otherwise it adds compute and
background sources without addressing the present attribution problem.

### 5. Preserve target magnitude

Add a continuous effective-NOx-change head with a robust loss and keep direction
as an auxiliary target. The high-magnitude tertile's weak AUROC suggests that a
binary sign discards useful structure. Direct emissions supervision should
precede coupling the scalar head to a transport decoder.

Masked-patch reconstruction remains a cheaper pretraining baseline. MAE and
SatMAE support hidden-patch reconstruction and independent temporal masking for
satellite sequences.[^mae][^satmae] For this dataset, calculate loss only on
artificially hidden pixels that TEMPO observed, and judge pretraining by held-out
classification and calibration rather than reconstruction error.

## Experiment order and gates

| Priority | Experiment | Pass condition |
|---:|---|---|
| 0 | Unique validation/test records and stable training | No duplicated evaluation rows or non-finite losses |
| 1 | Five-seed modality and permutation suite | Raster gain survives nuisance-preserving controls |
| 2 | Unconstrained next-raster pretraining | Better validation log loss than random initialization |
| 3 | Advection plus bounded correction | Beats capacity-matched unconstrained pretraining |
| 4 | Global lifetime, then source-relative plume prior | Incremental validation gain across seeds |
| 5 | Continuous change plus direction heads | Better ranking, calibration, and magnitude error |

Promote an architecture only after it improves validation log loss across seeds
and raises raster-only skill. Test AUROC should confirm the locked decision, not
select it.

[^tempo-tendency]: K. Sun et al., “[Hourly Nitrogen Oxides Emissions Estimated From TEMPO and Comparison With Facility-Level Monitoring Data](https://doi.org/10.1029/2025JD044565),” *Journal of Geophysical Research: Atmospheres*, 2025.
[^footnet]: T.-L. He et al., “[FootNet v1.0: development of a machine learning emulator of atmospheric transport](https://doi.org/10.5194/gmd-18-1661-2025),” *Geoscientific Model Development*, 2025.
[^nowcastnet]: Y. Zhang et al., “[Skilful nowcasting of extreme precipitation with NowcastNet](https://doi.org/10.1038/s41586-023-06184-4),” *Nature*, 2023.
[^gradient-blending]: W. Wang, D. Tran, and M. Feiszli, “[What Makes Training Multi-Modal Classification Networks Hard?](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html),” *CVPR*, 2020.
[^mae]: K. He et al., “[Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html),” *CVPR*, 2022.
[^satmae]: Y. Cong et al., “[SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html),” *NeurIPS*, 2022.
