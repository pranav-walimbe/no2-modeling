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
| Raster quality | At least 90% finite NO2 per timestep |
| Model selection | Validation split only; test remains frozen |

## Record selection

The pipeline applies these steps in order:

1. Build the complete facility-centered AOI set. For each AOI-hour, calculate
   mean unit operating time and total usable NOx emissions. Retain hours at or
   above the AOI's median operating time, then score the AOI by its median NOx
   over those higher-activity hours. Rank AOIs by this emissions-only score
   with AOI ID as the deterministic tie-breaker and retain the highest-scoring
   half. No fuel-type or plant-characteristic value enters the score.
   Stratification saves a complete AOI selection audit and characteristic plot.
2. Aggregate usable CAMPD measurements for the selected AOIs by UTC hour. Add
   unit counts, major-city distance, and full-history heat-input and generation
   averages calculated over the same higher-activity AOI-hours.
3. Match four consecutive TEMPO scans whose adjacent timestamps are 40 to 70
   minutes apart. Store raster scans as `t0_timestamp` through `t3_timestamp`.
4. At each raster timestamp, linearly interpolate the CAMPD NOx rate between
   the surrounding UTC-hour values. Store these four interpolated rates as
   `t0_nox` through `t3_nox`.
5. Apply a continuous-time EMA to `t0_nox` through `t3_nox`. Each update uses
   the actual time between scans, so a 70-minute interval admits more of the
   new value than a 40-minute interval.
6. Undo the final EMA update attenuation to recover the innovation relative to
   the preceding EMA. Assign classes using the larger of a 200 lb/hr absolute
   floor or 25% of the AOI's median positive interpolated timestep NOx.
7. Retain only AOIs providing at least 20 candidate records in every class.
8. Assign each overlap cluster to one split with a deterministic procedure that
   targets the 70/15/15 ratio for each class.
9. Downsample each class to the smallest class count within its split. Use
   deterministic weighted sampling for steady records, with weight increasing
   as absolute EMA innovation approaches zero.
10. Generate rasters and reject records that fail coverage or source-file checks.

Raster quality control can change class counts after balancing. The finalizer
reports those counts and does not rebalance the retained records.

## Label construction

The audit target is:

```text
effective_delta_nox = EMA(t3) - EMA(t2)
```

The EMA starts from the point-interpolated `t0_nox` value and updates through
`t3_nox`. Each update retains `exp(-elapsed_hours / 2)` of the preceding EMA.

For the final update, define:

```text
alpha = 1 - exp(-(t3 - t2) / 2 hours)
ema_innovation_nox = effective_delta_nox / alpha
aoi_active_median_nox = median(positive t0_nox ... t3_nox values for the AOI)
hybrid_innovation_threshold = max(200, 0.25 * aoi_active_median_nox)
```

Innovations at or below the negative threshold are decreases. Innovations at
or above the positive threshold are increases, and values between them are
steady. The pipeline writes the derived scale, update weight, innovation,
threshold, and class to the split metadata. The classifier consumes
`delta_category` without recreating it from a continuous value.

Metadata records this target construction as
`label_mode=linear_interpolated_timestep_ema`. The label and the causal raster
sequence both end at `t3`; no post-label observation is included.

## Stored inputs

Each raster bundle stores five arrays with shape `4 x 24 x 24`:

| Array | Contents |
|---|---|
| `no2` | Area-weighted NO2 on accepted TEMPO support |
| `no2_mask` | Binary NO2 support mask |
| `temperature_2m_k` | HRRR temperature at AOI cell centers |
| `wind_u_80m_mps` | Geographic eastward HRRR wind |
| `wind_v_80m_mps` | Geographic northward HRRR wind |

Stratification metadata stores the activity-conditioned median-NOx score and percentile,
coal, natural-gas, and total unit counts. It also stores major-city distance
and activity-conditioned averages for heat input, generation, and coal NOx.
The AOI score is selection metadata and does not enter the model. The metadata
does not store nameplate capacity or normalized NOx-change targets.

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
