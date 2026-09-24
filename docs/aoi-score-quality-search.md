# AOI directional plume-quality search

## Question

This analysis asks whether a deterministic AOI score can identify locations
whose NO2 rasters contain source-localized plume changes that agree with the
emissions label. It also tests whether attributes available before raster
generation can filter for those AOIs.

The analysis is exploratory. It uses the label to measure and rank AOI quality,
so the resulting score is appropriate for dataset design and diagnostics, not
as an input to an emissions-change model.

## Data and label contract

Slurm job `39195305` evaluated 28,937 raster bundles sampled deterministically
from the previous delta-model dataset, with at most 32 histories per AOI. It
recomputed labels from the current stratification implementation rather than
using the old dataset labels:

- CAMPD hourly NOx totals are weighted by their exact overlap with each TEMPO
  observation interval.
- The irregular-time EMA starts at `t0` and updates through `t3` using the
  actual time between observations and the configured two-hour decay
  timescale.
- The target is `EMA(t3) - EMA(t2)`. Values below -100 lb are decreases, values
  above +100 lb are increases, and the rest are steady.
- Raster scoring uses `t0` through `t3`. It does not inspect the post-label
  `t4` raster.

The raster normalization was frozen before tuning. A deterministic sample of
10,000 bundles, or 50,000 raster timesteps, provided 28,668,797 finite pixels.
The robust center is `1.868138303979520e15` molecules/cm² and the scale is
`IQR / 1.349 = 1.1997222249899362e15` molecules/cm². Every search configuration
uses these same values.

## Raster quality signal

For each timestep, the scorer:

1. subtracts a mask-normalized Gaussian background;
2. searches directions within 45 degrees of the local current or previous
   80 m wind;
3. compares a narrow, source-anchored downwind core with crosswind flanks;
4. divides by a robust background-noise estimate; and
5. discounts broad positive fields and responses not connected to the source.

The strongest constrained response supplies both a nonnegative plume SNR and a
signed plume amplitude. The four amplitudes pass through the same irregular
EMA timing used for the emissions label. For a record, directional strength is

```text
margin = label_sign * standardized_plume_EMA_change
quality = tanh(plume_SNR / SNR_scale) * tanh(margin / direction_scale)
```

This target rewards strong, correctly directed plume changes. Strong changes
in the wrong direction receive a negative value, while uncertain or weak
changes remain near zero. The robust plume-change scale reached its configured
floor of 0.1.

## Deterministic evaluation

Records are assigned to three folds by a seeded hash of raster path. For each
fold, candidate AOI scores use the other two folds and quality is measured only
on the held-out records. Both increase and decrease histories must be present.
The fixed seed is `20260923`.
The search evaluates:

- SNR scales of 0.25, 0.5, and 1.0;
- directional scales of 0.5, 1.0, and 2.0;
- mean, median, and upper-quartile within-class aggregation;
- class-imbalance penalties of 0, 0.5, and 1.0; and
- uncertainty penalties of 0, 0.5, and 1.0.

This gives 243 configurations. The selection objective is mean top-quartile
quality lift plus 0.25 times Spearman correlation, minus 0.10 times the
cross-fold standard deviation of top-quartile lift. All folds, hashes, and
tie-breaking rules are deterministic.

## Results

The best configuration uses an SNR scale of 0.25, directional scale of 1.0,
the within-class mean, no explicit increase/decrease imbalance penalty, and a
0.5 standard-error penalty. Across the three held-out record folds it produced:

| Metric | Result |
|---|---:|
| Mean eligible AOIs per fold | 149.3 |
| Mean Spearman correlation | 0.269 |
| Mean top-quartile quality lift | 0.052 |
| Standard deviation of top-quartile lift | 0.005 |
| Mean top-minus-bottom-quartile separation | 0.141 |

The score-decile curve is not perfectly monotonic, but its main separation is
clear: the lowest deciles are strongly negative, while the upper deciles are
near zero or positive. Mean aggregation consistently outranked median and
upper-quartile aggregation near the top of the sweep. This suggests that AOI
quality is better represented by repeatable directional evidence than by a few
exceptional scenes.

## Emissions-feature sweep

The feature pass used Polars streaming operations and the same PyCanopy spatial
membership procedure as stratification. It built 43 predictors from the full
emissions parquet, including unit and facility counts, fuel and unit-type mix,
NOx controls, capacity, heat input, load, emissions levels and variability,
operating frequency, and source geometry. Features calculated over active
hours use each AOI's median operating-time filter.

Simple high/low threshold rules were evaluated in five deterministic AOI folds.
Each cutoff was fitted outside its evaluation fold. The strongest individual
rules were:

| Rule | Retained | Mean quality lift | Fold SD |
|---|---:|---:|---:|
| History emitting fraction <= 0.917 | 20% | 0.033 | 0.026 |
| History operating fraction <= 0.917 | 20% | 0.033 | 0.026 |
| Active gas NOx share <= 0.042 | 20% | 0.025 | 0.010 |
| Largest-facility capacity share >= 0.718 | 20% | 0.027 | 0.025 |
| Active median total NOx >= 1,408 lb | 20% | 0.028 | 0.030 |
| Mean active operating units <= 3.05 | 20% | 0.022 | 0.008 |
| Maximum source distance <= 20.14 km | 20% | 0.025 | 0.021 |

The first two rules select the same AOIs in this sample. Their apparent result
fits a useful hypothesis: plants with meaningful on/off variation provide more
observable temporal contrast than plants that emit almost continuously.
Low gas share, higher coal NOx, fewer simultaneous sources, one dominant
facility, and sources nearer the AOI center also agree with a source-isolation
interpretation.

A shallow random forest did not improve the case for an emissions-only score.
Its five-fold mean Spearman correlation was 0.133 and its mean top-quartile
lift was 0.014. Two folds had negative top-quartile lift. Gas-unit count,
total-unit count, active operating-unit count, dominant-facility capacity
share, and total nameplate capacity had the largest impurity importances, but
these importances were variable across folds.

## Recommendation

Use the best raster heuristic as the primary EDA ranking. It directly measures
the desired property and has consistent held-out top-quartile lift. Keep the
continuous AOI score instead of immediately imposing a hard cutoff so dataset
size and geographic coverage can be inspected at several retention levels.

For pre-generation filtering, test a conservative composite centered on source
isolation and temporal contrast: lower operating/emitting fraction, low gas
NOx share, fewer active units, a dominant facility, and moderate source
distance. Do not yet hard-code the exact univariate cutoffs. They were selected
from this exploratory sweep and several have substantial fold variation.

The next validation should rerun the frozen raster score and a small number of
predeclared feature filters on newly generated AOIs. AOI or overlap-cluster
folds should remain grouped when estimating downstream model performance.

## Reproduction and artifacts

Run the analysis with:

```bash
sbatch scripts/slurm/aoi_score_quality_search.sh
```

Job `39195305` wrote its complete tables and diagnostic figure to
`/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-39195305/`.
Important files are `summary.json`, `heuristic_sweep.csv`,
`feature_threshold_sweep.csv`, `random_forest_metrics.csv`, and
`aoi_score_quality_search.png`.

The search chooses parameters from the same three-fold sweep used to summarize
them, so its reported heuristic metrics are selection-aware but not a nested-CV
estimate. The held-out target is also an engineered directional-plume proxy,
not human annotation or downstream model performance. The historical rasters
cover only AOIs represented in the prior dataset. These constraints make the
new-AOI validation above necessary before the score becomes a
dataset-generation default.
