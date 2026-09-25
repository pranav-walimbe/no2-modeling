# AOI plume-quality score

## Purpose

The AOI score ranks locations by two properties: whether TEMPO contains a
detectable source-localized plume and whether that plume responds in the
expected direction when CAMPD NOx changes. The score selects AOIs for modeling;
it is not a model input.

## Observation and label contract

Each sample contains four causal TEMPO observations, `t0` through `t3`. The
first three observations establish the prior emissions state. The innovation
at the fourth observation supplies the label.

CAMPD hourly NOx is interpolated to each observation time. An irregular EMA
with a two-hour decay timescale is updated across the observations:

```text
r_i = exp(-(time_i - time_{i-1}) / 2 hours)
E_i = r_i E_{i-1} + (1 - r_i) NOx_i
innovation = (E_3 - E_2) / (1 - r_3)
```

The innovation is classified as decrease, steady, or increase with the larger
of the configured absolute floor and an AOI-relative floor. Scoring requires at
least eight finite samples in every class. Candidate preparation retains at
most 64 deterministic samples per AOI and class, for 24 to 192 samples per
eligible AOI.

## Raster heuristic

Valid NO2 pixels use the fixed robust normalization recorded in `AGENTS.md`:

```text
z = clip((NO2 - 1.868138303979520e15) / 1.1997222249899362e15, -8, 8)
```

For each observation, the heuristic subtracts a mask-normalized Gaussian
background and searches directions within 45 degrees of the current and prior
local 80 m wind. It compares a source-anchored downwind core with crosswind
flanks, then penalizes broad or source-disconnected positive structure.

```text
noise = max(1.4826 * background_MAD, 0.10)
raw_SNR = max((core_response - flank_response) / noise, 0)
morphology = sqrt(localization * anchored_fraction) * exp(-3 * broad_fraction)
plume_SNR = raw_SNR * morphology
signed_amplitude = ((core_response - flank_response) / noise) * morphology
```

The record SNR is the median of the finite timestep SNRs and requires at least
two valid observations. Absolute contrast is the median of
`abs(signed_amplitude) * noise` across the four observations. Record
detectability is:

```text
detectability = tanh(record_SNR) * tanh(absolute_contrast / 0.2)
```

The same irregular EMA is applied to the four signed plume amplitudes. Its
final change is divided by a robust scale, `max(IQR / 1.349, 0.1)`, without
clipping or a hyperbolic tangent.

## AOI aggregation

For each AOI and label class, calculate mean detectability and mean scaled
plume response. AOIs must retain all three classes and at least eight finite
records per class.

```text
class_balanced_detectability = mean(class mean detectability)
directional_separation = mean_response_increase - mean_response_decrease

AOI_score = 0.8 * percentile(class_balanced_detectability)
          + 0.2 * percentile(directional_separation)
```

The percentile ranks are calculated across eligible AOIs in the full run. The
score rewards visible, localized plumes in all emissions regimes while keeping
a smaller term for the expected signed response.

## Validation

Savio job `39217486` tested the accepted score on 96 development AOIs. It
scored 14,937 cache-complete histories and produced 78 AOIs with at least eight
finite records in all three classes. Peak resident memory was 5.9 GB. Relative
to the accepted iteration result, the production implementation reproduces all
78 scores exactly.

The full-data candidate and scoring path uses bounded batches. This avoids the
147 GB allocation that caused the earlier full-frame Polars run to exceed its
node memory limit.

Nearest-hour HRRR remains the weather contract. A comparison against temporal
interpolation found median wind-vector and temperature differences of 0.27 m/s
and 0.18 K, with plume-SNR rank correlation of 0.990 across 322 histories. This
did not justify the added interpolation work for the current heuristic.

## Production workflow

`src/delta-model/preprocessing/aoi_heuristic.py` owns the complete workflow:

1. stream label preparation in bounded Polars batches;
2. resolve exact cache keys and partition missing TEMPO and HRRR work into a
   Slurm array;
3. run bounded worker pools within each array task;
4. score the completed candidates and atomically replace `AOI_SCORE_JSON`;
5. write the run tables and a 20-history montage, then email the PNG.

The default submission uses eight array tasks with eight workers each. Both
values are command-line options.

## Artifacts

- Accepted validation: `/global/home/users/pranavwalimbe/vis/aoi-score-cache-scan-39217486/`
- Earlier continuous baseline: `/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-39206139/`
- Earlier cache gap-fill: `/global/home/users/pranavwalimbe/vis/aoi-score-cache-scan-39206363/`
