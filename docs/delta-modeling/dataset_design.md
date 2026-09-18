# Dataset design

The delta-model dataset keeps each eligible AOI-hour until raster generation
applies its coverage checks.

## Design summary

| Decision | Rule |
|---|---|
| Split unit | Geographic clusters of overlapping 72 km AOIs |
| Split target | Approximately 70% train, 15% validation, 15% test |
| Targets | Raw and scaled hourly and effective NOx changes |
| Metadata filters | Required source data and each split's upper 5% NOx-mass cutoff |
| Raster gates | At least 95% coverage per timestep and complete 3 by 3 hotspot coverage |
| Final selection | Every record that passes raster checks |
| Model selection | Validation only; test remains frozen |

Stratification scores eligible AOIs, keeps the top `AOI_SELECTION_COUNT`, assigns
intact overlap clusters to splits, applies each split's NOx-mass cutoff, and
samples at most 300,000/75,000/75,000 records. Dataset generation keeps all
sampled records that pass raster checks. It never resamples by label.

## Split independence

- Each cluster of overlapping 72 km AOIs belongs to one split.
- A deterministic largest-cluster-first assignment targets the 70/15/15 ratio.
- AOI selection precedes splitting. Splitting precedes outlier pruning, record
  sampling, and train-only normalization.
- Each split computes its own 95th-percentile NOx-mass cutoff.

This order prevents a plant region from crossing split boundaries and tests
transfer to unseen regions.

## Metadata eligibility and AOI selection

A candidate requires:

- usable CAMPD measurements and a finite previous-quarter NOx average;
- consecutive TEMPO observations 40 to 70 minutes apart;
- at least 50% overlap with the assigned emissions hour;
- a mapped HRRR analysis path, checked during generation;
- finite prior-quarter power generation and major-city distance for scoring.

Coal share does not determine eligibility. The AOI score is a weighted sum on a
0 to 100 scale:

| Component | Weight | Definition |
|---|---:|---|
| Coal production share | 25% | Positive coal generation divided by all positive generation in the AOI archive |
| Signal strength | 25% | Percentile rank of log-transformed hourly NOx P75 |
| Event support | 20% | Rank of meaningful-change counts, with a reward for both directions |
| Urban isolation | 15% | Linear score from 25 km to 150 km from the nearest major city |
| Observation yield | 15% | 75% complete-record count rank and 25% complete-record-rate rank |

A meaningful event exceeds 100 lb or 25% of the AOI's previous-quarter median
NOx, whichever is larger. AOI ID breaks score ties. The split chart reports each
AOI's share of final sampled records.

## Targets and features

All joins use UTC. Facility enrichment converts CAMPD local standard time to
UTC and retains the source fields, facility timezone, and standard offset for
auditing. The model derives local mean solar hour from UTC and longitude.

Stratification preserves raw hourly NOx change, effective EMA change, and their
prior-quarter-scaled forms. It does not create classes or apply a deadband.
`TARGET_LABEL_MODE` controls label alignment:

| Mode | Behavior |
|---|---|
| `hard_hour` | Uses the clock hour with the greatest scan overlap |
| `overlap_weighted` | Averages touched hourly changes by overlap seconds and requires complete label coverage |

`hard_hour` remains the default until both modes are compared on frozen splits.

Each record stores five time-major 24 by 24 arrays:

| Array | Contents |
|---|---|
| `no2` | Directly regridded NO2 on native QA-passing support |
| `no2_mask` | Independent binary support mask |
| `temperature_2m_k` | HRRR temperature at AOI cell centers |
| `wind_u_80m_mps`, `wind_v_80m_mps` | Geographic eastward and northward HRRR wind |

Scalar model features use prior-quarter heat input and power generation to
avoid contemporaneous leakage. `prev_qtr_avg_nox` is the preceding calendar
quarter's mean AOI-hour NOx level. Stratification uses it for
`prev_qtr_rel_delta`, but neither field enters the model. Major-city distance
does enter the model.

Generator capacity is deduplicated by facility, generator, and attribute year.
Conflicting values contribute nothing. Each prediction uses the latest
attribute year no later than its prediction year, then sums each facility once.

## Raster eligibility

The regridder accepts an NO2 contributor when its quality flag is zero, cloud
fraction is at most 0.20, its value and geometry are valid, and it has positive
overlap with an output cell. Missing cells remain missing.

Every timestep needs at least 95% finite NO2 coverage and complete coverage in
a 3 by 3 window around the cell containing the largest modeled-unit cluster.
Facilities in one cell contribute their combined unit count. Unit-weighted
distance to the AOI center resolves ties.

Retrieval uncertainty, cloud, quality, and coverage summaries remain
diagnostics. They do not filter, rank, or enter the model.

## Generation and evaluation

Generation uses bounded worker queues and persistent TEMPO and weather caches.
Each launch clears disposable shards and published metadata, then runs a
throttled Slurm array. The finalizer validates every shard and publishes paths
relative to the dataset root without copying raster files. A failed worker
prevents publication; the next launch rebuilds all shards.

For each dataset, record:

- candidate, successful, and final counts;
- AOIs and geographic clusters per split;
- records by AOI, time, target magnitude, fuel mix, plant size, and weather;
- overall and subgroup metrics;
- a tabular-only baseline against raster models.

Freeze the test set after these checks. Choose thresholds, features, and models
from training and validation only.
