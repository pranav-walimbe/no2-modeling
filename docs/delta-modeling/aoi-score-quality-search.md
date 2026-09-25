# AOI plume-quality score

## Purpose

The score ranks AOIs for dataset selection. It combines:

- TEMPO evidence for a detectable source-localized plume;
- plume response when CAMPD NOx increases or decreases.

The model does not receive the score as an input.

## Observation and label contract

Each sample follows this label contract:

- Four causal TEMPO observations span `t0` through `t3`.
- The first three observations establish the prior emissions state.
- CAMPD hourly NOx is interpolated to each observation time.
- An irregular EMA uses a two-hour decay timescale.
- The innovation at `t3` supplies the label.

```text
r_i = exp(-(time_i - time_{i-1}) / 2 hours)
E_i = r_i E_{i-1} + (1 - r_i) NOx_i
innovation = (E_3 - E_2) / (1 - r_3)
```

Candidate selection then:

- classifies the innovation as decrease, steady, or increase using the larger
  of the configured absolute floor and an AOI-relative floor;
- requires at least eight finite samples in each class;
- retains at most 64 deterministic samples per AOI and class, for 24 to 192
  samples per eligible AOI.

## Raster heuristic

Valid NO2 pixels use the fixed robust normalization recorded in `AGENTS.md`:

```text
z = clip((NO2 - 1.868138303979520e15) / 1.1997222249899362e15, -8, 8)
```

For each observation, the heuristic:

1. subtracts a mask-normalized Gaussian background;
2. searches directions within 45 degrees of the current and prior local 80 m
   wind;
3. compares a source-anchored downwind core with crosswind flanks;
4. penalizes broad or source-disconnected positive structure.

```text
noise = max(1.4826 * background_MAD, 0.10)
raw_SNR = max((core_response - flank_response) / noise, 0)
morphology = sqrt(localization * anchored_fraction) * exp(-3 * broad_fraction)
plume_SNR = raw_SNR * morphology
signed_amplitude = ((core_response - flank_response) / noise) * morphology
```

The history-level calculation:

- requires at least two valid observations;
- sets record SNR to the median finite timestep SNR;
- sets absolute contrast to the median of
  `abs(signed_amplitude) * noise` across the four observations;
- combines SNR and contrast into record detectability.

```text
detectability = tanh(record_SNR) * tanh(absolute_contrast / 0.2)
```

The heuristic also applies the label EMA timing to the four signed plume
amplitudes. It divides the final change by the robust scale
`max(IQR / 1.349, 0.1)` without clipping or a hyperbolic tangent.

## AOI aggregation

For each AOI and label class, the scorer calculates mean detectability and mean
scaled plume response. An AOI remains eligible when it retains all three
classes and at least eight finite records per class.

```text
class_balanced_detectability = mean(class mean detectability)
directional_separation = mean_response_increase - mean_response_decrease

AOI_score = 0.8 * percentile(class_balanced_detectability)
          + 0.2 * percentile(directional_separation)
```

The scorer calculates percentile ranks across eligible AOIs in the full run.
Detectability contributes 80% of the final score. Directional separation
contributes 20%.

## What the score incentivizes

| Level | Rewarded | Penalized |
|---|---|---|
| Timestep | source-anchored enhancement; wind-aligned downwind structure; stronger core than crosswind flanks; adequate observed support | broad regional enhancement; source-disconnected structure; crosswind response; weak support |
| History | plume SNR across at least two observations; absolute plume contrast; finite temporal plume response | blank or noisy rasters; weak contrast; incomplete plume or wind support |
| AOI | detectable plumes in decrease, steady, and increase classes; larger plume response for increases than decreases | detectability confined to one emissions regime; reversed or indistinguishable increase/decrease response; fewer than eight finite histories in any class |
| Source geometry, as an indirect effect | one dominant hotspot; compact or co-varying sources; limited competing plume structure | dispersed or independently operating sources; competing nearby plumes; diffuse background structure |

## What the score does not directly incentivize

The formula omits:

- coal or gas status;
- fuel-specific unit counts or generation;
- heat input, capacity, or facility count;
- city proximity or land-use class;
- record count above the eligibility floor;
- low score variance or standard error;
- a stable plume response in the steady class;
- magnitude agreement between CAMPD change and plume change within a class.

These attributes may correlate with the score through plume detectability or
source attribution. For example, a compact coal facility can score well because
its plume is localized, not because the facility burns coal. A gas AOI with the
same plume evidence receives the same treatment.

## Production workflow

`src/delta-model/preprocessing/aoi_heuristic.py` owns the complete workflow:

1. stream label preparation in bounded Polars batches;
2. resolve exact cache keys and partition missing TEMPO and HRRR work into a
   Slurm array;
3. run bounded worker pools within each array task;
4. score the completed candidates and atomically replace `AOI_SCORE_JSON`;
5. write the run tables and a 20-history montage, then email the PNG.

The workflow uses bounded batches for candidate preparation and scoring. The
default submission uses eight exclusive array nodes with eight workers each.
Both values are command-line options.
