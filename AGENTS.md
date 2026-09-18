# Active pretraining plan

This file is the living design record for pretraining work. Keep confirmed
decisions, proposed defaults, and open questions distinct. Update it when an
experiment resolves an open choice. Do not move or regenerate existing datasets
as part of source-code reorganization.

## Goal and order of work

1. Build masked NO2 reconstruction first. Its encoder should learn spatial plume
   structure and its full model should support missing-pixel imputation.
2. Build next-raster prediction second. It should teach temporal transport and
   plume evolution from an ordered raster sequence.
3. Compare both pretrained encoders with the delta model trained from scratch.
   Downstream validation performance, not reconstruction appearance alone,
   decides whether pretraining is useful.

Only the first item is currently in design. Do not implement next-raster
pretraining until the masked baseline and its evaluation are working.

## Masked-pretraining data contract

Proposed initial sample:

- an ordered sequence of aligned 24 by 24 TEMPO NO2, temperature, eastward-wind,
  and northward-wind rasters;
- the original NO2 validity mask;
- a newly sampled artificial mask; and
- the untouched NO2 raster as the reconstruction target.

Begin with rasters having 100% valid NO2 coverage so every artificially hidden
pixel has an observed target. Require finite weather context too. Measure and
report how this complete-coverage cohort differs by AOI, season, local time,
wind, and NO2 level from the downstream population; complete-case filtering can
create a clear-sky distribution shift. A later experiment may admit incomplete
rasters while drawing artificial targets only from originally valid pixels.

Use unique AOI-scan rasters when reporting dataset size. Overlapping five-step
windows reuse many of the same rasters and must not be presented as independent
pretraining examples. Reuse the existing raster archives in place; do not copy
or move them into `src/pretraining`.

Fit normalization on the pretraining training split only. Preserve the existing
NO2 validity mask separately from the artificial mask. A naturally missing
pixel never contributes to reconstruction loss.

For an honest transfer experiment, keep downstream validation and test
geographies out of pretraining. If a later model is pretrained on all available
unlabelled TEMPO data, label that experiment transductive and do not compare it
directly with the train-geography-only result.

## Proposed first objective

Use temperature and wind as visible conditioning channels, but reconstruct NO2
only in the first baseline. Reconstructing smooth weather fields may dominate
the objective without improving plume representations.

The baseline loss is the mean per-pixel Huber or L1 error in normalized NO2,
evaluated only where a pixel is both originally valid and artificially masked.
Compare it with masked MSE. Do not score visible pixels. At inference, retain
observed pixels and insert predictions only at missing pixels rather than asking
the decoder to replace the entire observed field.

Treat physical and structural terms as later ablations:

- gradient or edge loss for plume boundaries;
- integrated-column or plume-mass error;
- weak wind-aligned temporal consistency; and
- a physics residual only after its required source, chemistry/lifetime,
  diffusivity, vertical-column, and boundary assumptions are defensible.

Do not put a hard advection-diffusion residual in version one. TEMPO observes a
retrieved vertical column, while the current weather context does not identify
all source, chemical-loss, mixing, and vertical-transport terms. An incorrect
constraint can make a reconstruction look physical while biasing real NO2.

## Masking experiments

Random single-pixel deletion is useful for interpolation testing but is likely
too easy to be the only pretext task on a 24 by 24 grid. Generate masks during
training and compare:

- scattered pixels;
- contiguous 2 by 2 and 3 by 3 blocks;
- mixtures of scattered and block masks; and
- held-out masks shaped like real TEMPO missingness patterns.

Start with masked-area ratios of 15%, 30%, and 50%. Select the ratio using both
masked-pixel validation error and frozen/fine-tuned downstream encoder transfer.
Do not copy the 75% MAE image default without an ablation: natural-image MAE
uses much larger images and patch-token redundancy than these small scientific
rasters.

For temporal sequences, begin with independently sampled masks at each time so
the model can use adjacent observations. Compare against masks held consistent
through time, which force spatial inference instead of allowing direct temporal
lookup. Record the mask seed and strategy in every run artifact.

## Architecture constraints

The first encoder should be compatible with the delta model's convolutional
spatial encoder so its weights transfer without a loosely defined adapter. Use
a deliberately lightweight decoder and avoid full-resolution U-Net skip paths
that let reconstruction bypass the encoder. Keep the checkpoint contract
explicit: architecture version, input channels, normalization state, encoder
state, and training-data manifest hash.

An MAE-style vision transformer is a valid later comparison, especially for
patch masking, but adopting it would also require a deliberate delta-model
encoder change. Do not call a decoder-only imputer successful pretraining until
the transferred encoder improves the downstream task.

## Dataset sizing and sampling

There is no literature-derived universal sample count for this domain. Use all
available unique qualifying rasters that fit the compute budget, prioritize
diversity over repeated overlapping windows, and make the choice empirical:

1. deduplicate by AOI and TEMPO scan identity;
2. stratify coverage across AOI, month/season, local observation hour, wind
   regime, and NO2 distribution without using downstream labels;
3. train otherwise identical models on 10%, 25%, 50%, and 100% of the unique
   training pool; and
4. plot reconstruction and downstream-transfer metrics against unique samples
   and GPU-hours.

Stop scaling when additional unique data no longer improves held-out
reconstruction or downstream transfer at fixed model capacity. Report both the
number of unique rasters and the number of sampled sequences.

## Evaluation requirements

Compare against zero/mean fill, nearest-neighbor or local interpolation,
spatial interpolation, temporal interpolation, and a simple wind-advection
baseline when available. Evaluate only artificially hidden observed pixels and
report:

- normalized and physical-unit MAE/RMSE;
- bias in high-NO2 pixels and integrated NO2 column;
- plume-gradient or edge error;
- performance by mask type and ratio;
- performance by AOI, season, wind speed/direction, and NO2 quantile; and
- downstream delta-model validation metrics across multiple fixed seeds.

Tune with validation data only. Keep the downstream test split frozen until the
masked design, checkpoint-selection rule, and transfer procedure are fixed.

## Open questions

- Whether the first model consumes one timestamp or the full five-step sequence.
- Whether independent temporal masks, consistent temporal masks, or a mixture
  transfers best.
- Whether a convolutional masked autoencoder or a small patch transformer gives
  the better compute/transfer tradeoff on 24 by 24 rasters.
- Whether Huber/L1 or MSE best preserves plume peaks without over-weighting
  retrieval outliers.
- Whether incomplete real rasters should join pretraining after the clean-cohort
  baseline.
- Whether physics-aware losses add value once compared with a data-only model.

## Research basis

- [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)
  introduced the asymmetric encoder/light-decoder design, masked-only pixel
  loss, and a 75% patch-mask default for large natural images.
- [SimMIM](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.html)
  found a simple masked-pixel objective competitive and used random block masks,
  a 60% ratio, and L1 reconstruction, reinforcing that mask unit and loss need
  empirical comparison.
- [SatMAE](https://sustainlab-group.github.io/SatMAE/) adds temporal embeddings
  for satellite imagery and reports stronger representations from masks sampled
  independently across time than masks fixed across all timestamps.
- [Prithvi](https://research.ibm.com/publications/prithvi-v10-generalist-geospatial-foundation-model-on-global-hls-data)
  demonstrates masked temporal/multispectral Earth-observation pretraining at
  large scale; its millions of HLS samples are a scaling reference, not a
  minimum requirement for this specialized task.
- [On Data Scaling in Masked Image Modeling](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_On_Data_Scaling_in_Masked_Image_Modeling_CVPR_2023_paper.html)
  shows that more data is not automatically useful when model capacity and
  overfitting are already controlled, motivating explicit learning curves.
- [Physics-Informed Neural Network Super Resolution for Advection-Diffusion Models](https://ml4physicalsciences.github.io/2020/files/NeurIPS_ML4PS_2020_117.pdf)
  reports benefits from an advection-diffusion residual on simulated plume data
  with known governing terms and missing pixels. That evidence motivates a
  later controlled ablation, not immediate use on retrieved TEMPO columns.
- [STILT-NOx](https://gmd.copernicus.org/articles/16/6161/2023/) documents why
  satellite NO2 plume evolution depends on nonlinear chemistry, turbulent
  mixing, vertical-column treatment, emissions, and wind, cautioning against an
  underspecified hard physics loss.
- The [NASA TEMPO Level 2/3 user guide](https://asdc.larc.nasa.gov/documents/tempo/guide/TEMPO_Level-2-3_trace_gas_clouds_user_guide_V2.0.pdf)
  is authoritative for product maturity, quality filtering, retrieval caveats,
  and scan timing.
