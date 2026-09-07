# Dataset design

This document defines how the project turns the full AOI-hour population into
a fixed-size modeling dataset. The central principle is to separate scientific
eligibility, raster quality, and diversity sampling. A record must pass the
first two; quality alone does not determine the population represented by the
third.

## Output-size contract

The user-facing controls in `config.py` are:

```python
TRAIN_SIZE = 12_000
VAL_SIZE = 4_000
TEST_SIZE = 4_000
```

Dataset generation must either produce exactly those sizes or fail with the
number of eligible records available. It must not silently return an undersized
split.

Stratification writes three times each requested size, currently 36,000 train,
12,000 validation, and 12,000 test candidates. The multiplier is derived from
the first production pilot. Basic paired-raster generation plus the selected
whole-raster and central-coverage gates retained an estimated 43.5 percent of
train, 55.5 percent of validation, and 52.0 percent of test candidates. A 3x
overdraw therefore provides approximately 30 to 67 percent headroom without
regridding the much larger eligible metadata population.

If a future archive or stricter filter causes a shortfall, increase
`STRATIFY_CANDIDATE_MULTIPLIER`; do not weaken a quality threshold implicitly.

## Split independence

Overlapping 72 km AOIs form geographic clusters. Every cluster belongs to one
of train, validation, or test, preventing the same or overlapping plant region
from leaking across splits. This evaluates generalization to unseen geographic
plant regions rather than interpolation at already-seen plants.

The split happens before quantile bounds are learned. Validation and test data
therefore cannot influence preprocessing statistics.

## Metadata eligibility and outliers

Before expensive image processing, a candidate must have:

- usable CAMPD measurements and enough prior-quarter history for its label;
- current and previous TEMPO observations separated by 50 to 70 minutes;
- at least 50 percent temporal overlap with the assigned emissions hour;
- a mapped HRRR analysis path (file existence is checked during generation);
  and
- finite prior-quarter power generation and distance to a city of 500,000 or
  more people for priority sampling.

- Fit 1st/99th percentile bounds on training data only for prior-quarter heat
  input, prior-quarter power generation, and the robust NOx-change scale.
- Freeze and reuse those bounds for validation and test.
- Do not trim coordinates, time fields, unit counts, city distance, target
  values, or coverage percentages.
- Independent filters can remove more than two percent of rows overall.

## Label and tabular features

All emission hours are converted from CAMPD local standard time to UTC during
facility-location enrichment. AOI aggregation, TEMPO pairing, HRRR lookup, and
the emitted modeling `date` and `hour` therefore share one UTC clock. The
enriched emissions archive retains the source local-standard fields and each
facility's resolved timezone and standard offset for auditability.

The binary target uses raw `delta_nox_mass`:

- Read one fixed cutoff from the `DELTA_THRESHOLD` configuration constant.
- Remove records with absolute change at or below that cutoff in every split.
- Assign class 0 to negative changes and class 1 to positive changes.
- Select equal class counts in every candidate and final split.
- Record the cutoff in each stratification and generation summary.

Stratification and final generation write JSON summaries with overall and
per-AOI retention, natural pre-balancing prevalence, and selected class counts.

Each sample stores current regridded NO2, current minus previous NO2, geographic
eastward and northward wind, and a paired-valid mask on the same fixed grid.
Both NO2 rasters are finite only where both scans have accepted support. Wind
is bilinearly aligned from the native HRRR grid and remains populated outside
the NO2 mask. HRRR temperature and boundary-layer height are interpolated at
the AOI centre. Prior-quarter heat input and power generation avoid
contemporaneous operational leakage.

Each AOI-hour also carries total generator nameplate capacity in MW. Collection
parses each CAMPD generator-capacity pair and deduplicates generators within a
facility and attribute year. AOI aggregation then sums each member facility
once. Conflicting values for one facility-generator pair do not contribute to
the sum. Total capacity enters the model as a numeric feature. Each prediction
uses the latest attribute year that does not exceed its year.

`TARGET_LABEL_MODE` selects the target construction. `hard_hour` retains the
change for the clock hour with the best scan overlap. `overlap_weighted`
averages every hourly change touched by the scan interval using overlap seconds
and requires complete label coverage across that interval. Keep `hard_hour` as
the default until both modes have been compared on frozen splits.

## Raster-quality gates

The native regridder accepts an NO2 contributor only when its quality flag is
zero, cloud fraction is at most 0.20, value and geometry are valid, and at
least 0.25 km2 of accepted support reaches an output cell.

After pairing scans, a candidate must satisfy both:

- `paired_finite_fraction >= 0.50` across all 48 by 48 cells; and
- `central_finite_fraction >= 0.50` in the central 8 by 8 cells.

The central window is 12 by 12 km. It is even-sized because the centre of an
even 48-cell raster lies at the intersection of its middle four cells. This
gate prevents a scan-edge fragment far from the modeled AOI centre from making
a record appear usable.

Eligible records first receive the bounded harmonic-mean coverage score

```text
coverage_quality = 2 * paired * central / (paired + central)
```

Retrieval uncertainty is averaged across both scans on paired-valid cells.
When the configured weight is positive, candidates without a finite uncertainty
summary are rejected. Remaining candidates receive an inverse uncertainty
percentile within their split, then the final ranking is

```text
raster_quality_score =
    (1 - RASTER_UNCERTAINTY_WEIGHT) * coverage_quality
    + RASTER_UNCERTAINTY_WEIGHT * uncertainty_quality
```

The configured uncertainty weight is 0.25. This changes final ranking without
uncertainty-weighting the NO2 tessellation itself.

`coverage_percent` and `paired_finite_fraction` are not interchangeable. The
first measures temporal overlap with a CAMPD clock hour; the second measures
spatial support shared by two rasterized scans.

## Quantities that do not select records

`plume_score` is retained as a diagnostic, not a filter or ranking term.
Selecting visible plumes would condition the dataset on an easily observed
satellite response and bias evaluation toward easy cases. Its percentile-ratio
definition also becomes unstable when the lower spread approaches zero.

Mean cloud and quality fractions remain diagnostics rather than additional
ranking terms. Native cloud and quality filtering already determines whether
NO2 is accepted.

## Candidate selection

Before raster generation, apply these rules to every split:

- Require each AOI to be at least 50 km from a major city.
- Average each unit's previous-quarter output, then sum the unit averages by
  AOI.
- Select records from AOIs with positive coal-unit output first, ranked by coal
  output.
- Fill any remaining slots from the general pool, ranked by total AOI power.
- Apply the priority and AOI round-robin independently within each label.
- Do not use current-quarter output or target magnitude for ordering.

## Final raster selection

Quality-gated candidates are selected deterministically:

1. Form strata by AOI, year, quarter, and four-hour UTC bin.
2. Rank records within each stratum by raster quality.
3. Interleave strata within each AOI so repeated records from one narrow time
   period are deferred.
4. Round-robin globally across AOIs so each available AOI receives one record
   before any receives its next.
5. Use coverage-plus-uncertainty quality to break competition within each round
   and stop at the exact configured split size.

Final selection takes equal counts from both labels after raster-quality gates.
Use the saved pre-balancing prevalence when interpreting balanced metrics.

## Performance and persistence

Metadata operations use Polars and project only required columns. Dataset
generation bounds the number of pending worker futures and caches each unique
AOI scan for one run. Candidate delta rasters live in the run's temporary
directory; only final selected rasters are moved into the persistent split
directory. Replacing a split directory also prevents stale, unreferenced files
from earlier runs.

For large archives, run stratification and raster generation through Slurm.
Do not regrid the entire metadata population merely to rank it. Increase the
candidate multiplier only when measured post-QC yield shows that the requested
final size cannot be reached reliably.

## Evaluation checklist

For every generated dataset, record:

- candidate, processing-success, coverage-eligible, and final counts;
- AOIs and geographic clusters per split;
- records per AOI, year, quarter, and observation hour;
- distributions of label, fuel mix, plant size, paired coverage, and weather;
- metrics overall and by AOI, label magnitude, coverage, season, and fuel; and
- a trivial tabular-only baseline versus image-plus-tabular models.

The test set should be frozen once these checks pass. Filter thresholds and
feature definitions should then be chosen using training and validation only.
