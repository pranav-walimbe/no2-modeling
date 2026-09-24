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

This gives 243 initial configurations. Refinement uses a stronger stability
penalty: mean top-quartile quality lift plus 0.25 times Spearman correlation,
minus 0.25 times the cross-fold standard deviation of top-quartile lift. All
folds, hashes, and tie-breaking rules are deterministic.

## Results

The initial best configuration used the median SNR across `t0` through `t3`.
Three subsequent searches changed one part of the score at a time:

1. Temporal aggregation compared the median, mean, upper-two mean, maximum,
   label-pair summaries, and current-timestep SNR. The maximum performed best,
   but the upper-two mean provided the most uniform first-pass improvement.
2. Reliability adjustment tested neutral pseudo-count shrinkage, class balance,
   standard-error penalties, and lower-quartile downside penalties. Neutral
   shrinkage improved both ranking and fold stability. Class and downside
   penalties did not help.
3. A local search interpolated between the upper-two mean and maximum and
   tightened the scale, shrinkage, and uncertainty grids. It improved the
   selection objective by only 0.0018 and slightly reduced top-quartile lift.
   This was the stopping point.

| Iteration | Spearman | Top-quartile lift | Lift SD | Top-bottom separation | Objective |
|---|---:|---:|---:|---:|---:|
| Initial median SNR | 0.269 | 0.052 | 0.0050 | 0.141 | 0.119 |
| Temporal maximum | 0.286 | 0.067 | 0.0194 | 0.152 | 0.134 |
| Reliability refinement | 0.299 | 0.067 | 0.0067 | 0.158 | 0.140 |
| Local refinement | 0.308 | 0.066 | 0.0047 | 0.155 | 0.142 |

The table recalculates every objective with the final 0.25 stability penalty.
The temporal-only artifact used the initial 0.10 penalty and therefore stores
0.137 for that row.

The selected final score uses:

- the maximum matched-filter SNR across `t0` through `t3`;
- SNR scale 0.50 and directional-strength scale 0.75;
- the mean record quality within increase and decrease classes;
- neutral shrinkage equivalent to seven pseudo-records for each class;
- a 0.25 standard-error penalty; and
- no explicit class-imbalance or bad-scene penalty.

For class `c` with `n_c` histories, the score is:

```text
record_quality = tanh(max_timestep_SNR / 0.50)
                 * tanh(label_sign * plume_delta_z / 0.75)
class_center_c = mean(record_quality_c) * n_c / (n_c + 7)
AOI_score = mean(class_center_increase, class_center_decrease)
            - 0.25 * mean(SE_increase, SE_decrease)
```

Across the three held-out record folds, the final score produced:

| Metric | Result |
|---|---:|
| Mean eligible AOIs per fold | 149.3 |
| Mean Spearman correlation | 0.308 |
| Mean top-quartile quality lift | 0.066 |
| Standard deviation of top-quartile lift | 0.0047 |
| Minimum fold top-quartile lift | 0.061 |
| Mean top-minus-bottom-quartile separation | 0.155 |

The final cross-fitted deciles are not perfectly monotonic because decile seven
dips below decile six. The broad ordering is clear: mean held-out quality rises
from -0.193 in the lowest decile to 0.048 in the highest. Mean record
aggregation consistently outranked median and upper-quartile aggregation. The
neutral pseudo-count result shows that repeatable evidence across several
histories is preferable to a large score from a small sample.

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

Use the final maximum-SNR, shrinkage-adjusted heuristic as the primary EDA
ranking. It directly measures the desired property and its top-quartile lift
was positive in every held-out fold. Keep the continuous AOI score instead of
immediately imposing a hard cutoff so dataset size and geographic coverage can
be inspected at several retention levels. Require at least three increase and
three decrease histories before assigning a score.

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
sbatch scripts/slurm/aoi_score_refinement.sh --stage temporal
sbatch scripts/slurm/aoi_score_refinement.sh --stage reliability
sbatch scripts/slurm/aoi_score_refinement.sh --stage local
```

Job `39195305` wrote its complete tables and diagnostic figure to
`/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-39195305/`.
Important files are `summary.json`, `heuristic_sweep.csv`,
`feature_threshold_sweep.csv`, `random_forest_metrics.csv`, and
`aoi_score_quality_search.png`.

Refinement jobs `39195670`, `39195736`, and `39195756` wrote their tables and
summaries under `/global/home/users/pranavwalimbe/vis/aoi-score-refinement-*`.
Validation job `39195783` reproduced the local optimum and wrote the final 413
eligible AOI rankings to `final_aoi_scores.csv`.

The search chooses parameters from the same three-fold sweep used to summarize
them, so its reported heuristic metrics are selection-aware but not a nested-CV
estimate. The held-out target is also an engineered directional-plume proxy,
not human annotation or downstream model performance. The historical rasters
cover only AOIs represented in the prior dataset. These constraints make the
new-AOI validation above necessary before the score becomes a
dataset-generation default.
