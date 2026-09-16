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
| Final selection | Exact class balance by deterministic minority-row duplication |
| Model selection | Validation data only; freeze test data for final comparison |

## Output contract

Stratification assigns intact geographic clusters toward a 70/15/15 split and
keeps each eligible record. After raster failures, finalization keeps the
complete majority class and duplicates deterministically ranked minority rows
until both classes have equal size. Reports record eligible counts, final size,
and the number of duplicated rows.

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
- current and previous TEMPO observations separated by 40 to 70 minutes;
- at least 50 percent temporal overlap with the assigned emissions hour;
- a mapped HRRR analysis path, with file existence checked during generation;
- finite prior-quarter power generation and distance to a city of 500,000 or
  more people for priority sampling.

Coal share is not an eligibility constraint, so gas and mixed-fuel AOIs remain
candidates.

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

The target uses the current-minus-previous effective EMA emissions difference:

- Read the fixed 100 lb cutoff from the `EMA_DELTA_THRESHOLD` configuration
  constant.
- Remove records with absolute change at or below that cutoff in every split.
- Assign class 0 to negative changes and class 1 to positive changes.
- Select equal class counts only in the final generated splits.
- Record the cutoff in each stratification and generation summary.

Stratification reports natural class prevalence. Final generation reports
overall and per-AOI retention, natural pre-balancing prevalence, and selected
class counts.

Each sample stores five time-major arrays on a fixed 24 by 24 grid:

| Array | Notes |
|---|---|
| directly regridded NO2 | one direct field per scan; finite where native QA-passing support exists |
| NO2 validity mask | independent binary support for each scan's NO2 field |
| 2 m temperature | sampled from HRRR at every AOI cell center and scan-aligned hour |
| eastward wind, northward wind | sampled from HRRR at every AOI cell center and scan-aligned hour |

Prior-quarter heat input and power generation keep contemporaneous operational
leakage out. `prev_qtr_avg_nox` is the mean level of the AOI's
hourly `nox_mass` totals over the immediately preceding calendar quarter (not a
delta). Stratification uses it to calculate `prev_qtr_rel_delta`, but the model
does not receive either field. The model does receive `major_city_dist`, which
is normalized from the training split with the other scalar inputs.

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
- a positive area of accepted support reaches an output cell.

Missing cells are never interpolated. Every configured timestep must have at
least 95% finite NO2 coverage. It must also have complete coverage in a 3 by 3
window around the raster cell containing the largest cluster of modeled units.
Facilities in the same raster cell contribute their combined unit count. Equal
counts are resolved by unit-weighted distance to the AOI centre. After both
gates pass, each scan retains its directly regridded values and validity mask.

Generated records use `min_no2_finite_fraction` across the sequence as the only
ranking signal after the fixed gates. Retrieval uncertainty does not filter or
rank records. Generation summaries report retained counts,
full-sequence coverage rates, and represented AOIs overall and by class. Coverage
remains a dataset diagnostic and is not supplied to the model.

## Quantities that do not select records

| Quantity | Role | Why not a filter |
|---|---|---|
| Mean cloud and quality fractions | Diagnostics | Native cloud and quality filtering already decides whether NO2 is accepted. |
| Distance to the nearest major city | Tabular feature | Urban context may be predictive, but centroid distance is not a reliable contamination boundary. |

## Candidate selection

Before raster generation, apply these rules to every split:

- Preserve every finite major-city distance without imposing a minimum distance.
- Average each unit's previous-quarter output, then sum the unit averages by AOI.
- Retain only AOIs where coal units supply more than 50 percent of that total.
- Keep every record that passes the eligibility rules.

## Final raster selection

Successfully generated candidates are selected deterministically:

1. Form strata by AOI, year, quarter, and four-hour UTC bin.
2. Rank records within each stratum by paired raster coverage.
3. Interleave temporal strata within each AOI.
4. Round-robin globally across AOIs.
5. Keep every generated record and repeat ranked minority rows until the class
   counts match.

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
