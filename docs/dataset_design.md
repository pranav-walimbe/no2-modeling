# Dataset design

Turning the full AOI-hour population into a fixed-size modeling dataset rests on
one principle: keep scientific eligibility, raster quality, and diversity
sampling separate. A record must clear the first two. Quality alone must never
decide which population the third step represents.

## Output-size contract

The controls in `config.py`:

```python
TRAIN_SIZE = 12_000
VAL_SIZE = 4_000
TEST_SIZE = 4_000
```

- Dataset generation produces exactly those sizes or fails with the count of
  eligible records available.
- An undersized split never passes silently.

Stratification writes three times each requested size: 36,000 train, 12,000
validation, 12,000 test candidates. The first production pilot set that
multiplier. Paired-raster generation plus the whole-raster and central-coverage
gates retained 43.5 percent of train, 55.5 percent of validation, and 52.0
percent of test candidates, so a 3x overdraw leaves 30 to 67 percent headroom
without regridding the much larger eligible metadata population.

On a shortfall from a future archive or a stricter filter, raise
`STRATIFY_CANDIDATE_MULTIPLIER`. Never weaken a quality threshold to compensate.

## Split independence

- Overlapping 72 km AOIs form geographic clusters.
- Each cluster belongs to exactly one of train, validation, or test.
- No plant region leaks across splits, so evaluation measures generalization to
  unseen geographic regions instead of interpolation at known plants.
- The split precedes quantile fitting, so validation and test data never reach
  preprocessing statistics.

## Metadata eligibility and outliers

Before any image processing, a candidate needs:

- usable CAMPD measurements and enough prior-quarter history for its label;
- current and previous TEMPO observations separated by 50 to 70 minutes;
- at least 50 percent temporal overlap with the assigned emissions hour;
- a mapped HRRR analysis path, with file existence checked during generation;
- finite prior-quarter power generation and distance to a city of 500,000 or
  more people for priority sampling.

Outlier bounds:

- Fit 1st/99th percentile bounds on training data alone, covering prior-quarter
  heat input, prior-quarter power generation, and the robust NOx-change scale.
- Freeze and reuse those bounds for validation and test.
- Leave coordinates, time fields, unit counts, city distance, target values, and
  coverage percentages untrimmed.
- Expect the independent filters to remove more than two percent of rows
  overall.

## Label and tabular features

One UTC clock governs everything:

- Facility-location enrichment converts emission hours from CAMPD local standard
  time to UTC.
- AOI aggregation, TEMPO pairing, HRRR lookup, and the emitted `date` and `hour`
  all share that clock.
- The enriched archive keeps the source local-standard fields, each facility's
  timezone, and its standard offset for auditability.

The binary target uses raw `delta_nox_mass`:

- Read one fixed cutoff from the `DELTA_THRESHOLD` configuration constant.
- Remove records with absolute change at or below that cutoff in every split.
- Assign class 0 to negative changes and class 1 to positive changes.
- Select equal class counts in every candidate and final split.
- Record the cutoff in each stratification and generation summary.

Stratification and final generation write JSON summaries carrying overall and
per-AOI retention, natural pre-balancing prevalence, and selected class counts.

Each sample stores five arrays on one fixed grid:

| Array | Notes |
|---|---|
| current regridded NO2 | finite only where both scans have accepted support |
| current minus previous NO2 | same paired-valid support |
| eastward wind, northward wind | bilinearly aligned from the native HRRR grid, populated outside the NO2 mask |
| paired-valid mask | binary support indicator |

HRRR temperature and boundary-layer height come from interpolation at the AOI
centre. Prior-quarter heat input and power generation keep contemporaneous
operational leakage out.

Nameplate capacity:

- Collection parses each CAMPD generator-capacity pair and deduplicates
  generators within a facility and attribute year.
- AOI aggregation sums each member facility once.
- Conflicting values for one facility-generator pair contribute nothing.
- Each prediction uses the latest attribute year at or before its own year.
- Total capacity in MW enters the model as a numeric feature.

`TARGET_LABEL_MODE` selects the target construction:

| Mode | Behavior |
|---|---|
| `hard_hour` | Keeps the change for the clock hour with the best scan overlap |
| `overlap_weighted` | Averages every hourly change the scan interval touches, weighted by overlap seconds, and requires complete label coverage across that interval |

Keep `hard_hour` as the default until both modes compete on frozen splits.

## Raster-quality gates

The native regridder accepts an NO2 contributor only when:

- its quality flag is zero;
- cloud fraction is at most 0.20;
- value and geometry are valid;
- at least 0.25 km2 of accepted support reaches an output cell.

After pairing scans, a candidate must satisfy both:

- `paired_finite_fraction >= 0.50` across all 48 by 48 cells;
- `central_finite_fraction >= 0.50` in the central 8 by 8 cells.

The central window spans 12 by 12 km. Its even size follows from the centre of a
48-cell raster landing at the intersection of the middle four cells. The gate
stops a scan-edge fragment far from the AOI centre from passing as usable.

Eligible records first receive a bounded harmonic-mean coverage score:

```text
coverage_quality = 2 * paired * central / (paired + central)
```

Then uncertainty enters the ranking:

- Average retrieval uncertainty across both scans on paired-valid cells.
- Reject candidates without a finite uncertainty summary whenever the configured
  weight is positive.
- Give the rest an inverse uncertainty percentile within their split.

```text
raster_quality_score =
    (1 - RASTER_UNCERTAINTY_WEIGHT) * coverage_quality
    + RASTER_UNCERTAINTY_WEIGHT * uncertainty_quality
```

The configured weight is 0.25, which shifts final ranking and leaves the NO2
tessellation itself unweighted.

`coverage_percent` and `paired_finite_fraction` measure different things. The
first measures temporal overlap with a CAMPD clock hour. The second measures
spatial support shared by two rasterized scans.

## Quantities that do not select records

| Quantity | Role | Why not a filter |
|---|---|---|
| `plume_score` | Diagnostic | Selecting visible plumes conditions the dataset on an easily observed response and biases evaluation toward easy cases. Its percentile-ratio definition also destabilizes as the lower spread approaches zero. |
| Mean cloud and quality fractions | Diagnostics | Native cloud and quality filtering already decides whether NO2 is accepted. |

## Candidate selection

Before raster generation, apply these rules to every split:

- Require each AOI to sit at least 50 km from a major city.
- Average each unit's previous-quarter output, then sum the unit averages by AOI.
- Select records from AOIs with positive coal-unit output first, ranked by coal
  output.
- Fill remaining slots from the general pool, ranked by total AOI power.
- Apply the priority ordering and AOI round-robin within each label
  independently.
- Keep current-quarter output and target magnitude out of the ordering.

## Final raster selection

Quality-gated candidates are selected deterministically:

1. Form strata by AOI, year, quarter, and four-hour UTC bin.
2. Rank records within each stratum by raster quality.
3. Interleave strata within each AOI, deferring repeated records from one narrow
   time period.
4. Round-robin globally across AOIs so every available AOI receives one record
   before any receives a second.
5. Break competition within each round on coverage-plus-uncertainty quality, and
   stop at the exact configured split size.

Final selection takes equal counts from both labels after the raster-quality
gates. Read balanced metrics against the saved pre-balancing prevalence.

## Performance and persistence

- Metadata operations use Polars and project only the required columns.
- Generation bounds the number of pending worker futures and caches each unique
  AOI scan for one run.
- Candidate delta rasters live in the run's temporary directory.
- Only final selected rasters move into the persistent split directory.
- Replacing a split directory clears stale, unreferenced files from earlier runs.

For large archives, run stratification and raster generation through Slurm.
Never regrid the entire metadata population just to rank it. Raise the candidate
multiplier only when measured post-QC yield shows the requested final size is out
of reach.

## Evaluation checklist

For every generated dataset, record:

- candidate, processing-success, coverage-eligible, and final counts;
- AOIs and geographic clusters per split;
- records per AOI, year, quarter, and observation hour;
- distributions of label, fuel mix, plant size, paired coverage, and weather;
- metrics overall and by AOI, label magnitude, coverage, season, and fuel;
- a trivial tabular-only baseline against image-plus-tabular models.

Freeze the test set once these checks pass, then choose filter thresholds and
feature definitions from training and validation alone.
