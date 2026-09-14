# Dataset design

Retain each scientifically eligible AOI-hour until raster generation determines
whether it has enough coverage.

## Design summary

| Decision | Rule |
|---|---|
| Split unit | Geographic clusters of overlapping 72 km AOIs |
| Split target | Approximately 70% train, 15% validation, 15% test |
| Label | Sign of raw hourly NOx change outside a 100 lb deadband |
| Metadata filters | Required source data, NOx percentile bounds, relative-change floor, and coal dominance |
| Raster gates | More than 95% current coverage and 80% paired coverage |
| Final selection | Largest balanced subset after deterministic AOI and temporal round-robin |
| Model selection | Validation data only; freeze test data for final comparison |

## Output contract

Stratification assigns intact geographic clusters toward a 70/15/15 split and
keeps each eligible record. After raster failures, finalization keeps the
complete smaller class and a deterministic equal-size sample from the larger
class. Reports record eligible counts, final size, and discarded imbalance.

## Split independence

- Overlapping 72 km AOIs form geographic clusters.
- Each cluster belongs to exactly one of train, validation, or test.
- A deterministic largest-cluster-first assignment minimizes deviations from
  70/15/15 targets for total, negative, and positive eligible record counts.
- No plant region leaks across splits, so evaluation measures generalization to
  unseen geographic regions instead of interpolation at known plants.
- The split precedes train-only raster and tabular normalization. The aggregate
  NOx eligibility percentiles are a global metadata gate applied before the
  split and are recorded in the stratification summary.

## Metadata eligibility and outliers

Before any image processing, a candidate needs:

- usable CAMPD measurements and a finite previous-quarter NOx average;
- current and previous TEMPO observations separated by 50 to 70 minutes;
- at least 50 percent temporal overlap with the assigned emissions hour;
- a mapped HRRR analysis path, with file existence checked during generation;
- finite prior-quarter power generation and distance to a city of 500,000 or
  more people for priority sampling.

After metadata eligibility, stratification calculates the global 1st and 99th
percentiles of finite aggregate AOI-hour `nox_mass` and retains records inside
the inclusive bounds. Coal share is not an eligibility constraint, so gas and
mixed-fuel AOIs remain candidates. `NOX_LOWER_PERCENTILE` and
`NOX_UPPER_PERCENTILE` configure the cutoffs. The fitted values and retention
rule are written to the stratification summary.

After the fixed absolute deadband, stratification calculates
`prev_qtr_rel_delta` as `abs(delta_nox_mass) / abs(prev_qtr_avg_nox)`. It
requires a value of at least the configured `MIN_PREV_QTR_REL_DELTA`, currently
0.10. The dataframe retains the metric for diagnostics, but the model does not
receive it or `prev_qtr_avg_nox` as an input.

## Label and tabular features

All joins use UTC:

- Facility-location enrichment converts emission hours from CAMPD local standard
  time to UTC.
- AOI aggregation, TEMPO pairing, HRRR lookup, and the emitted `date` and `hour`
  all share that clock.
- The model converts UTC hour and AOI longitude to local mean solar hour at load
  time. UTC remains the stored and joined clock.
- The enriched archive keeps the source local-standard fields, each facility's
  timezone, and its standard offset for auditability.

The target uses raw `delta_nox_mass`:

- Read the fixed 100 lb cutoff from the `DELTA_THRESHOLD` configuration
  constant.
- Remove records with absolute change at or below that cutoff in every split.
- Assign class 0 to negative changes and class 1 to positive changes.
- Select equal class counts only in the final generated splits.
- Record the cutoff in each stratification and generation summary.

Stratification reports natural class prevalence. Final generation reports
overall and per-AOI retention, natural pre-balancing prevalence, and selected
class counts.

Each sample stores four numeric arrays and two masks on a fixed grid:

| Array | Notes |
|---|---|
| smoothed and upwind-normalized current regridded NO2 | finite where native QA-passing support exists; record coverage must exceed 95% |
| current minus previous smoothed and upwind-normalized NO2 | finite on the current/previous mask intersection; coverage must exceed 80% |
| eastward wind, northward wind | bilinearly aligned from the native HRRR grid and finite across the image |
| two NO2 validity masks | separate binary support for current and hourly delta |

HRRR temperature and boundary-layer height come from interpolation at the AOI
centre. Prior-quarter heat input and power generation keep contemporaneous
operational leakage out. `prev_qtr_avg_nox` is the mean level of the AOI's
hourly `nox_mass` totals over the immediately preceding calendar quarter (not a
delta). Stratification uses it to calculate `prev_qtr_rel_delta`, but the model
does not receive either field.

The source-aware aggregate flux estimator remains available as a standalone
analysis module. Dataset generation does not run it, store its outputs, or use
a flux-derived model feature. This keeps the generated data contract independent
of the experimental flux formulation.

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

## Raster eligibility and masks

The native regridder accepts an NO2 contributor only when:

- its quality flag is zero;
- cloud fraction is at most 0.20;
- value and geometry are valid;
- at least 0.25 km2 of accepted support reaches an output cell.

Missing cells are never interpolated. Current coverage must be greater than
95%. The current/previous intersection must cover more than 80% of the raster.
After those gates pass, each scan is smoothed independently with its retrieval
uncertainty and observation-time wind. Each scan then receives a source-relative
upwind background subtraction using its own wind. The paired operation becomes
eligible only when both scans have at least 12 paired-valid background pixels.
Smoothing and normalization preserve the original masks.

After pairing scans, generated records use `paired_finite_fraction` as the only
ranking signal after the fixed gates. Central coverage and retrieval uncertainty
do not filter or rank records. Generation summaries report retained counts,
full-coverage rates, and represented AOIs overall and by class. Paired coverage
remains a dataset diagnostic and is not supplied to the model.

## Quantities that do not select records

| Quantity | Role | Why not a filter |
|---|---|---|
| `plume_score` | Diagnostic | Selecting visible plumes conditions the dataset on an easily observed response and biases evaluation toward easy cases. Its percentile-ratio definition also destabilizes as the lower spread approaches zero. |
| Mean cloud and quality fractions | Diagnostics | Native cloud and quality filtering already decides whether NO2 is accepted. |

`plume_score` remains in each output row for post-hoc stratification. It is not
required to be finite and never affects eligibility, ranking, or class balance.

## Candidate selection

Before raster generation, apply these rules to every split:

- Require each AOI to sit at least 50 km from a major city.
- Average each unit's previous-quarter output, then sum the unit averages by AOI.
- Retain only AOIs where coal units supply more than 50 percent of that total.
- Retain records within the configured aggregate AOI-hour NOx percentile bounds.
- Require deadband-eligible records to meet the configured
  `MIN_PREV_QTR_REL_DELTA` floor.
- Keep every record that passes the eligibility rules.

## Final raster selection

Successfully generated candidates are selected deterministically:

1. Form strata by AOI, year, quarter, and four-hour UTC bin.
2. Rank records within each stratum by paired raster coverage.
3. Interleave temporal strata within each AOI.
4. Round-robin globally across AOIs.
5. Retain the largest balanced subset, limited only by the smaller class.

## Performance and persistence

- Metadata operations use Polars and project only the required columns.
- Generation bounds the number of pending worker futures and caches each unique
  AOI scan for one run.
- Candidate delta rasters and outcome CSVs are written directly into disposable
  shards. Every launch first removes the previous shard tree and published
  metadata while retaining the TEMPO and wind caches.
- Final dataframes reference selected rasters by paths relative to the dataset
  root, such as `shards/train/000003/record-rasters/train/000012.npz`.
- Finalization performs no per-raster link, copy, move, or deletion. Selected
  and unselected successful rasters remain in the one current shard tree.
- A failed worker prevents finalization and may leave partial shards. The next
  launch starts with an empty shard tree rather than resuming them.

For large archives, run stratification and raster generation through Slurm.

## Evaluation checklist

For every generated dataset, record:

- candidate, processing-success, and final counts;
- AOIs and geographic clusters per split;
- records per AOI, year, quarter, and observation hour;
- distributions of label, fuel mix, plant size, and weather;
- metrics overall and by AOI, label magnitude, season, and fuel;
- a trivial tabular-only baseline against image-plus-tabular models.

Freeze the test set once these checks pass, then choose filter thresholds and
feature definitions from training and validation alone.
