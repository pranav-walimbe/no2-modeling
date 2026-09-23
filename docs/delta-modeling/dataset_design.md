# Dataset design

Each record joins four TEMPO scans, hourly HRRR fields, plant metadata, and one
three-class emissions-change label for a 72 km area of interest (AOI).

## Contract

| Component | Rule |
|---|---|
| Target column | `delta_category` |
| Classes | `decrease`, `steady`, `increase` |
| Split unit | Cluster of overlapping 72 km AOIs |
| Split target | About 70% train, 15% validation, 15% test within each class |
| Metadata sampling | Equal class counts within each split before raster quality control |
| Raster quality | At least 90% finite NO2 per timestep and full 3 by 3 source-hotspot coverage |
| Model selection | Validation split only; test remains frozen |

## Record selection

The pipeline applies these steps in order:

1. Average unit operating time within each AOI-hour and calculate its median
   for each AOI. Retain AOI-hours at or above that median, then average their
   hourly coal-unit NOx sums. Remove AOIs without coal units and retain the
   highest-ranked half.
2. Aggregate usable CAMPD measurements for the selected AOIs by UTC hour. Add
   unit counts, major-city distance, and full-history heat-input and generation
   averages calculated over the same higher-activity AOI-hours.
3. Match four consecutive TEMPO scans whose adjacent timestamps are 40 to 70
   minutes apart. The `t1` to `t2` interval must cover at least 50% of its
   assigned emissions hour. Store the scan times as `t0_timestamp` through
   `t3_timestamp` and the `t1` to `t2` duration as `label_delta_mins`.
4. Calculate four-hour, exponentially weighted NOx averages ending at `t1` and
   `t2`. The history reaches before `t0` and includes the preceding emissions
   hour. Both windows require complete CAMPD coverage.
5. Assign classes from the raw effective NOx change.
6. Assign each overlap cluster to one split with a deterministic procedure that
   targets the 70/15/15 ratio for each class.
7. Downsample each class to the smallest class count within its split.
8. Generate rasters and reject records that fail coverage or source-file checks.

Raster quality control can change class counts after balancing. The finalizer
reports those counts and does not rebalance the retained records.

## Label construction

The audit target is:

```text
effective_delta_nox = current_ema_nox - previous_ema_nox
```

`current_ema_nox` covers the four hours ending at `t2`; `previous_ema_nox`
covers the four hours ending at `t1`. Both use a two-hour exponential decay
timescale.

The class uses `effective_delta_nox` with boundaries at -100 and +100. Values
on a boundary belong to `steady`. The pipeline writes the class to
`delta_category`, which the classifier consumes without recreating it from a
continuous value.

Metadata records this target construction as `label_mode=causal_ema`. The label
ends at `t2`, but the raster classifier receives the full `t0` through `t3`
sequence. Its prediction therefore uses one observation after the labeled
interval.

## Stored inputs

Each raster bundle stores five arrays with shape `4 x 24 x 24`:

| Array | Contents |
|---|---|
| `no2` | Area-weighted NO2 on accepted TEMPO support |
| `no2_mask` | Binary NO2 support mask |
| `temperature_2m_k` | HRRR temperature at AOI cell centers |
| `wind_u_80m_mps` | Geographic eastward HRRR wind |
| `wind_v_80m_mps` | Geographic northward HRRR wind |

Stratification metadata stores coal, natural-gas, and total unit counts. It also
stores major-city distance and activity-conditioned averages for heat input,
generation, and coal NOx. It does not store nameplate capacity or normalized
NOx-change targets.

## Raster checks and publication

The regridder accepts a TEMPO value when its quality flag is zero, cloud
fraction is at most 0.20, its value and geometry are valid, and its footprint
overlaps an output cell. Missing cells remain `NaN` in the stored raster and
zero in `no2_mask`.

Each timestep needs at least 90% finite NO2 cells. A 3 by 3 window around the
cell with the largest modeled-unit count needs full support. Unit-weighted
distance to the AOI center breaks source-cell ties.

The generator records cloud, quality, uncertainty, and coverage diagnostics in
the companion metadata. These diagnostics do not enter the classifiers.

Workers write disposable shards against persistent TEMPO and HRRR caches. The
finalizer validates all source outcomes and publishes split CSVs with raster
paths relative to the dataset root. One failed shard blocks publication.
