# Regridding TEMPO Level 2 pixels

The regridder converts irregular TEMPO footprints into a `24 x 24` grid that
covers 72 km around each AOI. It projects footprints and cells to EPSG:5070,
finds intersecting polygons, and averages accepted NO2 by overlap area.

## Native-pixel filters

A TEMPO NO2 value contributes when:

- `main_data_quality_flag == 0`;
- cloud fraction is at most 0.20;
- its value and footprint geometry are valid;
- its footprint has positive overlap with an output cell.

The production calculation uses area-only weights and no effective-sample or
minimum-overlap cutoff. Cloud and quality diagnostics include valid overlapping
footprints that fail the NO2 filter.

## Persistent caches

Each AOI scan produces a compressed cache file with aligned `float32` rasters:

| Raster | Definition | Missing support |
|---|---|---|
| `no2` | Area-weighted tropospheric NO2 from accepted footprints | `NaN` |
| `weighted_cloud_fraction` | Area-weighted cloud fraction | `NaN` |
| `good_quality_fraction` | Area share with quality flag 0 | `NaN` |
| `retrieval_uncertainty` | Area-weighted uncertainty from accepted footprints | `NaN` |
| `sum_weight` | Accepted overlap area in km2 | `0.0` |

Generation deduplicates AOI-scan and AOI-hour work across shards:

| Cache | Key | Refresh flag |
|---|---|---|
| TEMPO | AOI and source-granule set | `--refresh-tempo` |
| Weather | AOI and UTC hour | `--refresh-weather` |

`--refresh-cache` clears both. When requested, the launcher clears caches before
array fan-out. Workers write cache files atomically.

## Published raster bundles

Each retained record stores oldest-to-newest arrays with shape `5 x 24 x 24`:

- `no2` and its independent `no2_mask`;
- `temperature_2m_k`;
- `wind_u_80m_mps` and `wind_v_80m_mps`.

Weather alignment projects AOI cell centers onto the HRRR Lambert grid,
bilinearly samples the 3 km fields, and rotates grid-relative winds into
geographic east and north.

Each timestep must contain at least 90% finite NO2 cells and complete NO2
support in the 3 by 3 source hotspot. The companion CSV includes per-timestep
coverage plus sequence-level cloud, quality, and uncertainty summaries.

## Training-time completion

Published NO2 rasters retain their gaps. Delta training normalizes the physical
channels with the masked-model checkpoint statistics, then passes NO2, weather,
and `no2_mask` through that reconstruction model. The imputer preserves observed
NO2 and fills mask gaps.

Training writes the completed four-channel sequences to job-local memory-mapped
arrays under `/tmp`. Both raster classifiers read those arrays and receive
`NO2`, temperature, wind U, and wind V. They do not receive `no2_mask`.
