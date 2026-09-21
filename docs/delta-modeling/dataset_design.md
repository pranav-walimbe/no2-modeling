# Dataset design

Each record joins five TEMPO scans, hourly HRRR fields, plant metadata, and one
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

1. Aggregate usable CAMPD measurements by AOI and UTC hour. Add prior-quarter
   same-hour heat input and generation, prior-quarter median hourly NOx, plant
   attributes, and major-city distance.
2. Match five consecutive TEMPO scans whose adjacent timestamps are 40 to 70
   minutes apart. The `t2` to `t3` interval must cover at least 50% of its
   assigned emissions hour.
3. Calculate four-hour, exponentially weighted NOx averages ending at `t2` and
   `t3`. Both windows require complete CAMPD coverage.
4. Assign raw and scaled effective-change classes. Keep a record only when both
   classes agree.
5. Assign each overlap cluster to one split with a deterministic procedure that
   targets the 70/15/15 ratio for each class.
6. Downsample each class to the smallest class count within its split.
7. Generate rasters and reject records that fail coverage or source-file checks.

Raster quality control can change class counts after balancing. The finalizer
reports those counts and does not rebalance the retained records.

## Label construction

The audit target is:

```text
delta_effective_nox_scaled =
    asinh((current_ema_nox - previous_ema_nox) / prev_qtr_med_nox)
```

`current_ema_nox` covers the four hours ending at `t3`; `previous_ema_nox`
covers the four hours ending at `t2`. Both use a two-hour exponential decay
timescale.

The raw class uses `effective_delta_nox` with boundaries at -100 and +100. The
scaled class uses `delta_effective_nox_scaled` with boundaries at -0.05 and
+0.05. Values on a boundary belong to `steady`. The pipeline writes the agreed
class to `delta_category`, which the classifier consumes without recreating it
from a continuous value.

Metadata records this target construction as `label_mode=causal_ema`. The label
ends at `t3`, but the current raster classifier receives the full `t0` through
`t4` sequence. Its prediction therefore uses the observation after the labeled
interval.

## Stored inputs

Each raster bundle stores five arrays with shape `5 x 24 x 24`:

| Array | Contents |
|---|---|
| `no2` | Area-weighted NO2 on accepted TEMPO support |
| `no2_mask` | Binary NO2 support mask |
| `temperature_2m_k` | HRRR temperature at AOI cell centers |
| `wind_u_80m_mps` | Geographic eastward HRRR wind |
| `wind_v_80m_mps` | Geographic northward HRRR wind |

The tabular classifier derives total unit count from the stored coal and
natural-gas counts. It combines that feature with major-city distance,
nameplate capacity, prior-quarter same-hour heat input and generation, local
solar hour, and day of year. Sine and cosine encode both time features, which
yields nine scalar inputs.

AOI identity, coordinates, current emissions, and prior-quarter NOx stay out of
the model inputs. Longitude contributes only to local solar hour.

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
