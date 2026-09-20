# Dataset design

The delta-model dataset keeps each eligible AOI-hour until raster generation
applies its coverage checks.

## Design summary

| Decision | Rule |
|---|---|
| Split unit | Geographic clusters of overlapping 72 km AOIs |
| Split target | Approximately 70% train, 15% validation, 15% test |
| Targets | Raw and scaled hourly and effective NOx changes |
| Metadata filters | Required source data and agreement between raw +/-100 and normalized +/-0.05 classes |
| Raster gates | At least 95% coverage per timestep and complete 3 by 3 hotspot coverage |
| Final selection | Equal decrease, steady, and increase samples based on the smallest bucket in each split |
| Model selection | Validation only; test remains frozen |

Stratification uses every eligible AOI and assigns raw effective EMA changes
below -100, from -100 through 100, or above 100. It assigns the normalized
effective EMA change with the same three classes at a 0.05 boundary and keeps
records only when the two assignments agree. Intact overlap clusters are then
assigned to approximately 70/15/15 splits by targeting each class separately.
Each split downsamples all three buckets to that split's smallest bucket.

## Split independence

- Each cluster of overlapping 72 km AOIs belongs to one split.
- A deterministic largest-cluster-first assignment targets 70/15/15 within
  each filtered label class.
- Dual-deadband filtering precedes geographic splitting. Splitting precedes
  within-split bucket balancing and train-only normalization.

This order prevents a plant region from crossing split boundaries and tests
transfer to unseen regions.

## Metadata eligibility and balancing

A candidate requires:

- usable CAMPD measurements and a finite previous-quarter NOx average;
- consecutive TEMPO observations 40 to 70 minutes apart;
- at least 50% overlap with the assigned emissions hour;
- a mapped HRRR analysis path, checked during generation;
- finite prior-quarter power generation and major-city distance.

AOI ranking is not applied. A decrease has a raw effective EMA change below
-100, an increase is above 100, and the inclusive interval from -100 through
100 is steady. The normalized boundaries are -0.05 and 0.05. A record is
eligible only when its raw and normalized classes agree. Exact boundaries
belong to the steady bucket.

## Targets and features

All joins use UTC. Facility enrichment converts CAMPD local standard time to
UTC and retains the source fields, facility timezone, and standard offset for
auditing. The model derives local mean solar hour from UTC and longitude.

Stratification preserves raw hourly NOx change, effective EMA change, and their
prior-quarter-scaled forms. It writes the sampling bucket to `delta_category`
as `decrease`, `steady`, or `increase`.
`TARGET_LABEL_MODE` controls label alignment:

| Mode | Behavior |
|---|---|
| `hard_hour` | Uses the clock hour with the greatest scan overlap |
| `overlap_weighted` | Averages touched hourly changes by overlap seconds and requires complete label coverage |

`hard_hour` remains the default until both modes are compared on frozen splits.

Each record retains five scans from `t0` through `t4`. Label matching and
coverage use the interval ending at `t3`, the fourth scan. The current and
previous effective NOx values are four-hour exponential moving averages ending
at `t3` and `t2`. The fifth scan remains stored as model input.

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
