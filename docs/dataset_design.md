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
- a mapped HRRR analysis path (file existence is checked during generation); and
- at least 50 km distance from a city of 500,000 or more people.

The 1st and 99th percentiles are learned only from the training split for these
continuous variables:

- prior-quarter average heat input;
- prior-quarter average power generation;
- AOI NOx mass;
- hourly change in AOI NOx mass;
- the robust NOx-change scale; and
- the normalized target.

The same numerical bounds are then applied unchanged to every split. Applying
six independent bounds can remove more than two percent of rows overall; that
is expected. Coordinates, dates, hours, unit counts, city distance, and
coverage percentages are not percentile-trimmed because their tails describe
real subpopulations or already have meaningful hard bounds.

## Label and tabular features

The current target is

```text
delta_nox_norm = asinh(delta_nox_mass / delta_nox_scale)
```

where `delta_nox_scale` is based on the AOI's previous completed quarter. The
transform preserves sign, compresses extreme changes, and makes changes at
different-sized plants more comparable. Exact zero-change records are valid
and remain in the population.

The image is current regridded NO2 minus the previous regridded NO2 on the same
fixed grid. A cell is finite only if both scans have accepted support there.
HRRR temperature, 10 m wind components, and boundary-layer height use the
native grid point nearest the AOI centre. Prior-quarter heat input and power
generation avoid contemporaneous operational leakage.

The current hard clock-hour label is a baseline. A future experiment should
compare it with an overlap-weighted combination of adjacent hourly emission
changes when the TEMPO interval straddles a clock boundary. That experiment
must be resolved before interpreting `coverage_percent` as anything stronger
than a temporal-alignment diagnostic.

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

Eligible records receive the bounded harmonic-mean score

```text
raster_quality_score = 2 * paired * central / (paired + central)
```

which penalizes weak coverage in either region.

`coverage_percent` and `paired_finite_fraction` are not interchangeable. The
first measures temporal overlap with a CAMPD clock hour; the second measures
spatial support shared by two rasterized scans.

## Quantities that do not select records

`plume_score` is retained as a diagnostic, not a filter or ranking term.
Selecting visible plumes would condition the dataset on an easily observed
satellite response and bias evaluation toward easy cases. Its percentile-ratio
definition also becomes unstable when the lower spread approaches zero.

Mean cloud and quality fractions are also diagnostics rather than additional
ranking terms. Native cloud and quality filtering already determines whether
NO2 is accepted, while paired-finite coverage captures whether enough usable
image remains. Selecting only the clearest scenes would change the deployment
population.

## Final selection

Quality-gated candidates are selected deterministically:

1. Form strata by AOI, year, quarter, and four-hour UTC bin.
2. Rank records within each stratum by raster quality.
3. Interleave strata within each AOI so repeated records from one narrow time
   period are deferred.
4. Round-robin globally across AOIs so each available AOI receives one record
   before any receives its next.
5. Use raster quality to break competition within each round and stop at the
   exact configured split size.

This prefers strong rasters while retaining plant and temporal diversity. It
does not balance on the target label, so validation and test remain suitable
for estimating performance on the quality-eligible population. If rare target
ranges need more training emphasis, use training-time sample weights or a
sampler and continue reporting unweighted validation and test metrics.

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

For every generated version, record:

- candidate, processing-success, coverage-eligible, and final counts;
- AOIs and geographic clusters per split;
- records per AOI, year, quarter, and observation hour;
- distributions of label, fuel mix, plant size, paired coverage, and weather;
- metrics overall and by AOI, label magnitude, coverage, season, and fuel; and
- a trivial tabular-only baseline versus image-plus-tabular models.

The test set should be frozen once these checks pass. Filter thresholds and
feature definitions should then be chosen using training and validation only.
