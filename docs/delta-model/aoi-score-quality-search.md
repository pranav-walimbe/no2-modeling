# AOI directional plume-quality score

## Purpose

We designed an AOI score to identify locations where source-localized NO2
plumes change in the same direction as the CAMPD emissions label. The score
uses the label, so it belongs in dataset selection and diagnostics. It must not
enter the emissions-change model as an input.

## Data contract

Savio job `39195305` evaluated 28,937 raster bundles from the prior delta-model
dataset, with at most 32 histories per AOI. The analysis rebuilt each label
from the current stratification code:

- CAMPD hourly NOx totals receive weights based on their overlap with each
  TEMPO observation interval.
- A two-hour irregular-time EMA updates from `t0` through `t3`.
- `EMA(t3) - EMA(t2)` defines the label. Changes below -100 lb are decreases,
  and changes above +100 lb are increases.
- The raster score uses `t0` through `t3` and excludes the post-label `t4`
  raster.

The scorer uses the fixed robust raster normalization recorded in `AGENTS.md`:

```text
center = 1.868138303979520e15 molecules/cm2
scale  = 1.1997222249899362e15 molecules/cm2
```

## Raster signal

For each timestep, the scorer:

1. subtracts a mask-normalized Gaussian background;
2. searches directions within 45 degrees of the current or previous local
   80 m wind;
3. compares a source-anchored downwind core with crosswind flanks; and
4. downweights broad fields and positive regions disconnected from the source.

The matched-filter response uses background pixels outside the source
neighborhood to estimate noise:

```text
noise = max(1.4826 * background_MAD, 0.10)
raw_SNR = max((core_response - flank_response) / noise, 0)
```

The strongest wind-constrained response supplies a nonnegative SNR and a
signed plume amplitude. The scorer applies the emissions-label EMA timing to
the four plume amplitudes and standardizes the resulting change.

## Final AOI score

The parameter search selected the maximum timestep SNR, mean record quality,
seven neutral pseudo-records per class, and a standard-error penalty:

```text
record_quality = tanh(max_timestep_SNR / 0.50)
                 * tanh(label_sign * plume_delta_z / 0.75)

class_center_c = mean(record_quality_c) * n_c / (n_c + 7)

AOI_score = mean(class_center_increase, class_center_decrease)
            - 0.25 * mean(SE_increase, SE_decrease)
```

The two `tanh` terms bound outliers and encode diminishing returns. Their
product requires a detectable plume whose change matches the label. The
class-specific centers prevent the more common class from controlling the AOI
score. An AOI needs at least three increase and three decrease histories.

The initial search used median SNR across `t0` through `t3`. Median SNR took
the median of at least two finite timestep SNRs. Maximum SNR produced better
held-out AOI rankings, so the final score uses the strongest of the four
timesteps.

## Validation

The search assigned records to three folds using a seeded raster-path hash.
Each fold measured quality on records excluded from its AOI-score fit. The
seed was `20260923`.

| Metric | Result |
|---|---:|
| Mean eligible AOIs per fold | 149.3 |
| Mean Spearman correlation | 0.308 |
| Mean top-quartile quality lift | 0.066 |
| Top-quartile lift SD | 0.0047 |
| Minimum fold top-quartile lift | 0.061 |
| Mean top-minus-bottom-quartile separation | 0.155 |

Mean held-out quality rose from -0.193 in the lowest score decile to 0.048 in
the highest. Decile seven fell below decile six, so the relationship was not
monotonic at every boundary.

## Why high-scoring AOIs score well

The score decomposition compared the 104 AOIs in each score extreme. High
scorers had weaker raw plume detection and stronger label agreement.

| Score quartile | Mean max SNR | Detectability | Direction correct | Record quality |
|---|---:|---:|---:|---:|
| Highest | 0.533 | 0.668 | 0.707 | 0.271 |
| Lowest | 0.574 | 0.721 | 0.417 | -0.094 |

Direction accuracy reached 0.815 for increase histories and 0.652 for decrease
histories in the top score quartile. The bottom quartile reached 0.500 and
0.360. The score measures agreement between a visible plume and the AOI
emissions label more than raw raster SNR.

### Plant characteristics

The analysis joined the final scores for 413 eligible AOIs to 43 CAMPD and
source-geometry features. Rank correlations used Benjamini-Hochberg correction
across all 43 tests.

| Feature | Spearman | Adjusted q |
|---|---:|---:|
| Largest-facility capacity share | 0.177 | 0.013 |
| Facility count | -0.159 | 0.026 |
| Median source distance from AOI center | -0.146 | 0.039 |
| Mean source distance from AOI center | -0.143 | 0.039 |
| Mean active operating units | -0.134 | 0.041 |
| Combined-cycle unit fraction | -0.131 | 0.041 |
| Mean active gross load | -0.131 | 0.041 |

Top-quartile AOIs contained more coal NOx, fewer sources, and a more dominant
facility than bottom-quartile AOIs:

| Characteristic | Highest quartile median | Lowest quartile median |
|---|---:|---:|
| Active coal NOx share | 0.820 | 0.250 |
| Active gas NOx share | 0.178 | 0.689 |
| Unit count | 12.0 | 16.5 |
| Active operating units | 5.91 | 7.82 |
| Largest-facility capacity share | 0.534 | 0.411 |
| Mean source distance from AOI center | 15.4 km | 17.3 km |
| Mean active gross load | 1,377 | 1,780 |
| Median active NOx | 881 lb | 671 lb |

Coal share had a weak unadjusted association with score (`rho = 0.096`,
`p = 0.051`, adjusted `q = 0.116`). Controlling for NOx scale, heat input,
load, facility dominance, source count, source distance, and operating
variability reduced the partial rank correlation to `0.041` (`p = 0.411`).
Coal share acts as a proxy for plant structure in this sample.

A shallow random forest using all 43 features reached a five-fold mean
Spearman correlation of 0.133 and a mean top-quartile lift of 0.014. Two folds
had negative lift. The available plant descriptors explain little of the AOI
score on held-out AOIs.

### Interpretation

Source attribution provides the best explanation for the observed pattern.
An AOI with one dominant facility, fewer operating units, and compact source
geometry produces an emissions label that represents the same plume measured
near the hotspot. In a complex AOI, total emissions can change because of a
distant or independent unit while the hotspot plume moves in another
direction. Coal-heavy AOIs often have the simpler configuration, but coal fuel
does not retain an independent association after adjustment.

The geometry fields measure facility distance from the AOI center. They do not
measure distance to a city. The largest-source field describes facility
capacity share rather than unit capacity share. The feature set contains
operating and emitting fractions but no direct operating-time volatility
metric.

## Use and limitations

Use the continuous score to rank AOIs and inspect retention thresholds. Do not
apply a coal-share cutoff. A pre-generation filter should focus on facility
dominance, source count, source geometry, and enough NOx to produce a visible
plume. Validate any filter on new AOIs before changing dataset generation.

The parameter search and reported three-fold metrics use the same sweep, so
they do not form a nested cross-validation estimate. The target is an
engineered plume proxy rather than downstream model accuracy. The source data
cover AOIs from the prior raster dataset. Weather, terrain, retrieval quality,
and plume timing remain unmeasured drivers.

## Artifacts

- Raster scan and feature sweep, job `39195305`:
  `/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-39195305/`
- Final score validation, job `39195783`:
  `/global/home/users/pranavwalimbe/vis/aoi-score-refinement-39195783/`
- Final 413-AOI ranking: `final_aoi_scores.csv` in the validation directory.
- Driver analysis:
  `/global/home/users/pranavwalimbe/vis/aoi-score-driver-analysis-20260924/`
- Good-versus-poor raster montage, job `39197345`:
  `/global/home/users/pranavwalimbe/vis/aoi-final-score-montage-39197345.png`
