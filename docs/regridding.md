# Regridding TEMPO Level 2 pixels onto AOI rasters

TEMPO Level 2 reports one NO2 value per irregular ground footprint. The model
needs a fixed 24 by 24 raster covering 72 km around each AOI, with the same
3 km cells in both scans of every delta.

The regridder:

- Projects each footprint and output cell to the equal-area EPSG:5070 system.
- Intersects their polygons and averages accepted NO2 by overlap area in square
  kilometres.
- Uses a spatial index to find only the cells a footprint can touch, so it never
  builds a dense `native pixels x output cells` matrix.

## Filter native pixels

A native value enters the NO2 mean when:

- `main_data_quality_flag == 0`;
- effective cloud fraction is at most `MIN_PIXEL_CLOUD`, currently 0.20;
- the NO2 value and footprint geometry are valid.

Filtering runs before averaging. Cloud and quality diagnostics still cover all
geometrically valid overlapping footprints, which keeps rejected inputs visible.

## Save each AOI scan

Each AOI scan is saved as one compressed `.npz` holding five aligned 24 by 24
`float32` rasters:

| Raster | What one output pixel displays | Missing value |
|---|---|---|
| `no2` | Overlap-area-weighted tropospheric NO2 from quality-0 footprints with cloud fraction at most 0.20, in molecules/cm2 | `NaN` when no accepted footprint overlaps the cell |
| `weighted_cloud_fraction` | Overlap-area-weighted cloud fraction from all valid native footprints, including footprints rejected from NO2 | `NaN` when no footprint with valid cloud information overlaps the cell |
| `good_quality_fraction` | Share of total overlapping footprint area carrying quality flag 0, from 0 to 1 | `NaN` when no valid native footprint overlaps the cell |
| `retrieval_uncertainty` | Arithmetic overlap-area-weighted mean of NO2 retrieval uncertainty from accepted footprints, in molecules/cm2 | `NaN` when no accepted footprint has finite uncertainty |
| `sum_weight` | Total area in km2 where accepted native footprints overlap the cell | `0.0` when no accepted footprint overlaps; zero marks a real absence of support, not a missing value |

The ancillary rasters stay populated where `no2` is `NaN`, which preserves
information about cloudy or low-overlap cells. Downstream code preserves these
gaps and forms explicit masks instead of interpolating them.

## Build model records

`preprocessing.generate_dataset` deduplicates AOI-scan work and writes the
five-raster bundles to the persistent TEMPO cache under `DATASET_DIR`.

Caching behavior:

| Cache | Key | Batching | Refresh flag |
|---|---|---|---|
| TEMPO scans | AOI and source granules | Scans sharing a granule set run together, so a worker opens each large NetCDF granule once for several AOIs | `--refresh-tempo` |
| Aligned weather | AOI and hour | HRRR files are grouped so each full grid is read once for several AOIs | `--refresh-weather` |

`--refresh-cache` fully deletes both configured cache directories before
rebuilding entries referenced by the selected split. `--refresh-tempo` and
`--refresh-weather` fully delete only their respective cache. Refresh must run as a
single non-array process because all split jobs share these directories. The
TEMPO rebuild includes all configured sequence scans. Individual
cache files are still written atomically.

Pass the matching flag after changing TEMPO processing or wind alignment.

Each successful model record persists one compressed NPZ holding five
oldest-to-newest arrays with shape `T x 24 x 24`:

- `no2`, directly regridded NO2 on each timestep's native QA-passing support;
- `no2_mask`, an independent `uint8` validity mask for every NO2 timestep;
- `temperature_2m_k`, bilinearly sampled temperature at AOI cell centers;
- `wind_u_80m_mps`, geographic eastward wind;
- `wind_v_80m_mps`, geographic northward wind.

Every row in the companion split CSV carries its NPZ path in `raster_bundle_path`
plus these derived features:

- `no2_finite_fraction_t0` through `no2_finite_fraction_t{T-1}`;
- `min_no2_finite_fraction`, used to rank otherwise eligible records;
- cloud, native-pixel quality, and retrieval-uncertainty summaries over each
  timestep's valid NO2 support.

Wind alignment, on the 3 km HRRR grid NOAA describes:

1. Project the 24 by 24 TEMPO cell centres onto the native Lambert grid.
2. Bilinearly interpolate the wind components.
3. Rotate the grid-relative values to geographic east and north before caching.

Every scan retains its directly regridded values. Every timestep must have at
least 90 percent finite NO2 coverage or the complete record is rejected.

See the [NOAA Global Systems Laboratory HRRR overview](https://rapidrefresh.noaa.gov/).

The regridder also calculates squared weight, effective sample size, total
overlap area, contributor counts, and worst quality internally. These support
masking and validation and stay out of the modeling bundle.

## EDA decisions

Job 38571783 validated the implementation on 12 fixed scans and 40 sampled scan
pairs. Production uses:

- cloud fraction at most 0.20;
- overlap area alone for the NO2 weights;
- no positive accepted-overlap floor;
- no additional effective-sample floor.

Dataset generation requires at least 90% finite NO2 coverage independently at
every configured timestep. It retains missing cells and selects eligible records
through temporal and AOI round-robin with minimum sequence coverage as the sole
quality rank. Cloud, uncertainty, and quality
summaries remain diagnostics and rank nothing.

Measured tradeoffs:

| Choice | Effect |
|---|---|
| No overlap floor | Matches NASA-style area-weighted gridding by retaining every positive overlap. |
| 1.25 effective-sample floor | Survival fell to 21.1 percent. Rejected. |
| Area-only weighting | Lower median normalized RMS disagreement with Level 3 than linear or squared inverse-uncertainty weighting. Kept. |

The vectorized implementation matched the independent overlap reference to
floating-point precision. Warm tessellation took 0.13 to 0.16 seconds per AOI.

NASA Level 3 is not ground truth here, since it uses a geographic grid and
different input filtering. Its similarity catches geometry, unit, and diagnostic
mistakes. It cannot select the scientifically best custom filter or weight.
