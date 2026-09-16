# Performance analysis and architecture change report

Last updated: 2026-09-16

## Scope and decision status

This report evaluates the latest completed run,
`delta_nox_classification_20260916_073403` from Slurm job `38969986`.
The run is a single-seed result on the first five-scan causal-EMA dataset. It is
useful evidence for architecture planning, but it is not the new reference model.

The queued follow-up, Slurm job `38976295`, is still pending. It will evaluate
the current fusion change with the regenerated, larger sample. The latest
stratification produced 307,491 training, 65,420 validation, and 68,344 test
candidates before raster generation and class handling. Final raster-qualified
counts are not known yet. Decisions about fusion, sample-size effects, or a new
headline score must wait for that run and repeated seeds.

## Completed-run contract

| Item | Value |
|---|---|
| Train / validation / test records | 15,708 / 4,256 / 3,326 |
| Train / validation / test AOIs | 330 / 69 / 66 |
| Class handling | Exactly balanced within each final split |
| Target | Sign of causal effective NOx change outside a 100 lb deadband |
| Effective-emissions kernel | Five hourly steps, exponential decay with 2 h timescale |
| Raster sequence | Five 24 x 24 hourly frames, oldest to newest |
| Per-frame inputs | NO2, 2 m temperature, 80 m eastward wind, 80 m northward wind, NO2 validity mask |
| Tabular inputs | Unit counts, capacity, prior-quarter activity, solar hour, and day of year |
| Baseline | 961-parameter tabular MLP, selected first and frozen during fusion |
| Raster model | 702,994 trainable parameters; mask-aware spatial encoder, ConvGRU, advection-decay residual |
| Fusion used by artifact | Additive correction to the frozen MLP logit, initialized at zero |
| Seed | 42 |

The transport layer back-advects the previous NO2 raster with the current wind,
applies one bounded global exponential lifetime, and supplies the innovation to
the raster encoder. The learned lifetime ended at 3.94 h from a 4 h
initialization within a 0.5 to 12 h range. That small movement is not evidence
for a physical lifetime estimate; classification loss may provide too little
identifying information.

## Performance

### Validation and test metrics

| Split and metric | ConvGRU residual + MLP | Frozen MLP | Difference |
|---|---:|---:|---:|
| Validation accuracy | 0.7131 | **0.7199** | -0.0068 |
| Validation ROC AUC | **0.8028** | 0.7913 | +0.0115 |
| Validation log loss | 0.5786 | **0.5754** | +0.0032 |
| Validation Brier score | **0.19380** | 0.19386 | -0.00006 |
| Validation ECE, 10 bins | 0.1003 | **0.0855** | +0.0148 |
| Test accuracy | **0.7402** | 0.7162 | +0.0241 |
| Test ROC AUC | **0.8124** | 0.7756 | +0.0367 |
| Test log loss | **0.5487** | 0.5893 | -0.0406 |
| Test Brier score | **0.1823** | 0.1989 | -0.0165 |
| Test ECE, 10 bins | 0.0771 | 0.0773 | -0.0002 |

The raster correction helps on the test AOIs. An AOI-cluster bootstrap with
1,000 resamples gives the following 95% intervals for ConvGRU minus MLP:

| Metric difference | 95% interval |
|---|---:|
| Accuracy | +0.0158 to +0.0316 |
| ROC AUC | +0.0282 to +0.0453 |
| Log loss | -0.0586 to -0.0238 |
| Brier score | -0.0229 to -0.0102 |

These intervals describe geographic sampling uncertainty for this trained seed.
They do not include training-seed variance, dataset-regeneration variance, or
the architecture choices made after inspecting earlier test results.

### The validation result does not yet approve the architecture

The tabular MLP reached its best validation loss at epoch 9. The fused model
reached its best value at epoch 4, but that value was 0.0032 worse than the
frozen MLP. It raised validation AUC while lowering validation accuracy and
worsening calibration error. Because the experiment selected checkpoints by
validation loss, the test improvement cannot override that result.

Training after epoch 4 added no validation value. Fused training loss fell from
0.427 at epoch 4 to 0.007 at epoch 29 while validation loss rose from 0.579 to
1.541. Early stopping restored the epoch-4 checkpoint, but the trajectory shows
that a 703K-parameter raster correction can memorize 15,708 records quickly.
Reduce patience for future sweeps and spend the saved compute on seeds and
ablations.

## Exploratory error analysis

All findings in this section use the frozen validation or test predictions from
the completed run. They diagnose the model; they do not establish new selection
rules.

### Does the raster correction add information?

The correction changed the MLP logit by a mean absolute 0.44 on test. It reduced
per-record log loss for 67.8% of test records and 64.6% of validation records.
The test accuracy gain was similar for both classes:

| Test class | ConvGRU + MLP | MLP | Difference |
|---|---:|---:|---:|
| Emissions decrease: specificity | 0.6837 | 0.6590 | +0.0247 |
| Emissions increase: recall | 0.7968 | 0.7733 | +0.0235 |

The gain also appears across absolute effective-change tertiles. Accuracy gains
were +3.70, +1.17, and +2.35 percentage points from the lowest to highest
tertile. The raster branch is therefore not helping only on the largest target
changes.

Among the 42 test AOIs with at least 20 records and both classes present, the
fused model improved accuracy at 26, lost accuracy at 5, and tied at 11. The
median AOI accuracy gain was 1.96 points, but individual changes ranged from
-5.45 to +7.53 points. Model selection still needs several seeds and paired
AOI-level intervals.

Seasonal AUC gains were positive in all four exploratory groups, from +0.036 in
spring to +0.100 in winter. The groups differ in geography and class mix, so
this pattern does not prove a seasonal mechanism.

### Coverage still controls learnability

The fused-minus-MLP AUC gain was +0.0439 for test sequences with complete NO2
coverage and +0.0175 for sequences whose minimum timestep coverage was between
0.90 and 0.99. The weaker group still showed a positive gain, which supports
keeping masks and testing more permissive coverage rather than treating every
incomplete sequence as unusable.

Coverage gates and balancing remove much of the available supervision. The
completed dataset began with 88,167 stratified training candidates, generated
26,266 qualifying raster sequences, and retained 15,708 after balancing. Among
the generated training sequences, 18,412 were increases and 7,854 were
decreases, so exact balancing discarded 10,558 otherwise eligible increases.
Earlier coverage EDA found that only 25.8% of a 1,000-sequence sample passed the
combination of at least 95% coverage at every timestep and complete 3 x 3 source
coverage.

## Dataset changes that could improve performance

1. **Train on every raster-qualified record.** Use class-weighted loss or a
   balanced batch sampler instead of deleting the majority class. Keep a frozen
   balanced test view for comparison, but also report log loss, precision-recall
   AUC, calibration, and decision costs under natural prevalence.

2. **Separate scientific eligibility from observation quality.** Keep hard
   requirements for valid targets, geographic independence, and source support.
   Represent partial scan coverage with masks, coverage features, and a missing
   timestep indicator. Evaluate 0.90 and 0.95 coverage floors on validation
   AOIs. Do not choose the floor from test performance.

3. **Prevent sequence duplication from inflating sample size.** Consecutive
   five-hour windows share four frames. Split by geographic cluster first,
   report unique AOI-date sequences and unique scans, and sample windows by
   AOI-date during training. A nominal 200K examples can contain far fewer than
   200K independent atmospheric states.

4. **Preserve the continuous target.** Predict continuous effective NOx change
   with a robust or Student-t likelihood and retain direction as an auxiliary
   head. Calculate class probabilities relative to the deadband after fitting.
   This uses magnitude information and exposes uncertainty near the threshold.

5. **Audit target timescales.** Compare causal EMA timescales of 1, 2, 4, and 6
   h on fixed geographic splits. Scan-time overlap and the causal emissions
   window must remain explicit in every record. A TEMPO power-plant study found
   that adding the NO2 column tendency term improved agreement with CEMS at 10
   of 14 plants, which supports modeling atmospheric memory rather than pairing
   each scan with one isolated emissions hour.[^tempo]

6. **Keep selection independent of observed plume strength.** Use plume scores,
   cloud, uncertainty, and coverage for subgroup reporting or validated
   abstention curves. A raster-derived plume score should not decide whether a
   sample enters the headline test set.

## Architecture changes after the queued run

### Establish the fusion result first

The queued fusion-plus-larger-sample run should answer the next question before
more components are added. Compare the fused model with its exact frozen MLP on
the same records across at least five seeds. Use validation log-loss improvement
as the primary gate, then report AUC, Brier score, calibration, accuracy,
recall, and specificity with paired AOI-cluster intervals.

Include raster-only, mask-only, and conditionally permuted-NO2 controls. Permute
NO2 within AOI, season, solar-hour, and weather bins while leaving masks and
weather aligned. A real spatial contribution should disappear under that
control. Multimodal research documents that joint models can undertrain one
modality when streams overfit at different rates, so a strong tabular baseline
can hide a useful but noisy image signal.[^gradient-blending][^modality-laziness]

If the residual fusion passes, make two changes:

- Build out-of-fold MLP logits for the training records. A frozen baseline fit
  on the same rows can give the correction branch overconfident nuisance
  residuals even when its validation checkpoint is sound.
- Add a raster-only auxiliary head during supervised training. Tune its weight
  on validation AOIs and remove the head at inference. This forces the encoder
  to retain image information while the additive final logit still measures
  conditional improvement over tabular context.

### Keep the physics module auditable

Run capacity-matched ablations for raw ConvGRU, advection only, advection plus
decay, and advection-decay plus learned correction. Report the learned lifetime
across seeds. The current 3.94 h value is too close to initialization to support
interpretation.

Add source-relative distance, along-wind and cross-wind coordinates, travel
time, and a cheap Gaussian-plume footprint as input fields before increasing
network depth. FootNet found surface winds and a Gaussian-plume first guess to
be its most informative transport-emulator inputs.[^footnet] Hybrid models such
as NowcastNet also pair an explicit advection evolution with a learned intensity
residual, which matches the structure of this problem.[^nowcastnet]

Do not add diffusion, a free learned motion field, or per-pixel chemical
lifetimes until advection and one global lifetime show repeatable validation
value. Those additions can absorb the emissions-change signal.

## Masked-raster pretraining on 200K+ sequences

Separate self-supervised pretraining is worth testing. It directly addresses
the present regime: millions of valid raster pixels and many unlabeled temporal
windows, but only 15,708 balanced labeled training examples for a 703K-parameter
model. Masked autoencoders learn by reconstructing deliberately hidden patches
with an encoder and lightweight decoder; satellite-specific work extends the
method with independent temporal masking and time embeddings.[^mae][^satmae]

The experiment should pretrain the same spatial encoder and ConvGRU that the
classifier will use:

1. Draw at least 200K five-scan windows from training AOIs. Include windows
   without deadband-eligible emissions labels. Deduplicate exact scans and cap
   sampling per AOI-date so large plants do not dominate.
2. Preserve the native validity mask. Add a second artificial mask over pixels
   that were originally observed. Hide 50% to 75% of 3 x 3 or 4 x 4 patches,
   mixing independent spatial masks with temporal tube masks.
3. Feed visible NO2, weather, elapsed time, coordinates, and both mask types to
   the encoder. Compute reconstruction loss only on deliberately hidden,
   originally valid NO2 pixels. Never ask the model to reconstruct pixels with
   no retrieval target.
4. Use a small decoder to reconstruct robust-normalized NO2 and the
   advection-decay innovation. Add next-scan prediction as a secondary objective
   only after masked reconstruction works. Weather should condition the task;
   reconstructing smooth weather fields can become an easy shortcut.
5. Discard the decoder. Compare a frozen linear probe, full encoder fine-tuning,
   and random initialization on identical downstream records and seeds. Plot
   learning curves at 10K, 25K, 50K, and the full labeled set.

MAE used a high mask ratio to prevent local pixel interpolation from making the
task trivial, and SatMAE found value in treating temporal satellite observations
explicitly.[^mae][^satmae] Our 24 x 24 rasters are much smaller than their input
images, so patch size and mask ratio need validation rather than direct copying.
A convolutional masked autoencoder is preferable for the first test because it
can reuse the deployed encoder; a larger transformer would confound the value
of pretraining with a capacity change.

Use these safeguards:

- Exclude validation and test AOIs from pretraining for the strict geographic
  generalization result. If a transductive experiment uses their unlabeled
  rasters, label it separately.
- Keep train-only normalization and never use CAMPD outcomes, deadband labels,
  or future emissions during pretraining.
- Compare random masks, block masks, and time-tube masks. Report reconstruction
  error by coverage, season, AOI, and distance from the source.
- Run a mask-only downstream control. Satellite missingness correlates with
  clouds, season, and scan geometry, so a representation can appear useful
  while learning acquisition patterns instead of NO2 structure.
- Test temporal-order shuffling and wind rotation. Useful pretraining should
  lose downstream skill when transport relationships are broken.

Pretraining earns a place if it improves validation log loss across seeds,
reduces the labeled-data requirement, and strengthens raster-only skill. Low
reconstruction error alone is insufficient. Spatial redundancy, overlapping
windows, and the gap between reconstruction and emissions inference could
otherwise produce a good autoencoder with no downstream gain.

## Recommended experiment order

| Priority | Experiment | Decision it supports |
|---:|---|---|
| 0 | Finish queued fusion run on the larger generated sample | Whether the current architecture and data scale clear the validation gate |
| 1 | Five-seed MLP, raster-only, fused, mask-only, and conditional-permutation comparison | Whether NO2 adds repeatable conditional information |
| 2 | 200K+ masked-raster pretraining with random-init and linear-probe controls | Whether unlabeled sequences improve representation and label efficiency |
| 3 | Continuous effective-change head plus auxiliary direction head | Whether hard labels discard useful target structure |
| 4 | Raw ConvGRU versus advection versus decay ablations | Which physics component contributes skill |
| 5 | Source-relative geometry and Gaussian-plume prior | Whether explicit source-receptor structure improves transfer |

The immediate success criterion is modest: repeatable validation-log-loss
improvement over the same frozen tabular baseline, without relying on masks or
test-set tuning. The completed run shows enough spatial signal to justify the
next experiments, but the queued larger-sample result must determine whether
that signal survives the current data and fusion changes.

[^tempo]: K. Sun et al., “[Hourly Nitrogen Oxides Emissions Estimated From TEMPO and Comparison With Facility-Level Monitoring Data](https://doi.org/10.1029/2025JD044565),” *Journal of Geophysical Research: Atmospheres*, 2025.
[^gradient-blending]: W. Wang, D. Tran, and M. Feiszli, “[What Makes Training Multi-Modal Classification Networks Hard?](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html),” *CVPR*, 2020.
[^modality-laziness]: C. Du et al., “[On Uni-Modal Feature Learning in Supervised Multi-Modal Learning](https://proceedings.mlr.press/v202/du23e.html),” *ICML*, 2023.
[^footnet]: T.-L. He et al., “[FootNet v1.0: development of a machine learning emulator of atmospheric transport](https://doi.org/10.5194/gmd-18-1661-2025),” *Geoscientific Model Development*, 2025.
[^nowcastnet]: Y. Zhang et al., “[Skilful nowcasting of extreme precipitation with NowcastNet](https://doi.org/10.1038/s41586-023-06184-4),” *Nature*, 2023.
[^mae]: K. He et al., “[Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html),” *CVPR*, 2022.
[^satmae]: Y. Cong et al., “[SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery](https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html),” *NeurIPS*, 2022.
