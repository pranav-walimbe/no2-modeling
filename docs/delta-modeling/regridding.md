# Regridding TEMPO Level 2 pixels

The regridder converts irregular TEMPO footprints into a fixed 24 by 24 grid
covering 72 km around each AOI. It projects footprints and cells to EPSG:5070,
intersects indexed polygon candidates, and averages accepted NO2 by overlap area.

## Native-pixel filters

An NO2 value contributes when:

- `main_data_quality_flag == 0`;
- cloud fraction is at most `MIN_PIXEL_CLOUD`, currently 0.20;
- its value and geometry are valid.

Filtering occurs before averaging. Cloud and quality diagnostics still cover
all valid overlapping footprints, including those rejected from NO2.

## AOI-scan cache

Each AOI scan produces one compressed NPZ with aligned 24 by 24 `float32`
rasters:

| Raster | Definition | No support |
|---|---|---|
| `no2` | Area-weighted tropospheric NO2 from accepted footprints | `NaN` |
| `weighted_cloud_fraction` | Area-weighted cloud fraction from all valid footprints | `NaN` |
| `good_quality_fraction` | Share of overlapping area with quality flag 0 | `NaN` |
| `retrieval_uncertainty` | Area-weighted uncertainty from accepted footprints | `NaN` |
| `sum_weight` | Accepted overlap area in km2 | `0.0` |

Ancillary rasters may remain finite where NO2 is missing. Downstream code keeps
these gaps and creates an explicit NO2 mask.

`src/delta-model/preprocessing/generate_dataset.py` deduplicates AOI-scan work
and manages two persistent caches:

| Cache | Key | Read grouping | Refresh flag |
|---|---|---|---|
| TEMPO | AOI and source granules | Shared granule set | `--refresh-tempo` |
| Weather | AOI and hour | Shared HRRR file | `--refresh-weather` |

`--refresh-cache` clears both caches. A refresh runs before array fan-out because
all shards share the cache directories. Cache files are written atomically.

## Model records

Each successful record stores oldest-to-newest arrays with shape
`T x 24 x 24`:

- directly regridded `no2` and its independent `no2_mask`;
- `temperature_2m_k`;
- geographic `wind_u_80m_mps` and `wind_v_80m_mps`.

The companion CSV stores `raster_bundle_path`, per-timestep finite fractions,
minimum sequence coverage, and cloud, quality, and uncertainty summaries.

Weather alignment projects AOI cell centers onto the HRRR Lambert grid,
bilinearly samples the 3 km fields, and rotates grid-relative wind into
geographic east and north. See the
[NOAA HRRR overview](https://rapidrefresh.noaa.gov/).

Every timestep needs at least 95% finite NO2 coverage and complete coverage in
the 3 by 3 source hotspot. The regridder also calculates squared weight,
effective sample size, overlap area, contributor count, and worst quality for
validation. These values do not enter model bundles.

## EDA decisions

Job 38571783 tested 12 fixed scans and 40 sampled scan pairs. Production keeps:

- cloud fraction at most 0.20;
- area-only NO2 weights;
- every positive accepted overlap;
- no effective-sample cutoff.

| Choice | Result |
|---|---|
| No overlap floor | Preserved NASA-style area-weighted gridding |
| Effective-sample floor of 1.25 | Reduced survival to 21.1%; rejected |
| Area-only weighting | Beat linear and squared inverse-uncertainty weights on median normalized RMS disagreement with Level 3 |

The vectorized implementation matched an independent overlap reference to
floating-point precision. Warm tessellation took 0.13 to 0.16 seconds per AOI.
NASA Level 3 is a geometry and unit check, not ground truth, because its grid and
input filters differ.
