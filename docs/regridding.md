# Regridding TEMPO Level 2 pixels onto AOI rasters

TEMPO Level 2 reports one NO2 value per irregular ground footprint. The model
needs a fixed 48 by 48 raster covering 72 km around each AOI, with the same
1.5 km cells in both scans of every delta.

How the regridder bridges that gap:

- Projects each footprint and output cell to the equal-area EPSG:5070 system.
- Intersects their polygons and averages accepted NO2 by overlap area in square
  kilometres.
- Uses a spatial index to find only the cells a footprint can touch, so it never
  builds a dense `native pixels x output cells` matrix.

## Filtering and diagnostics

A native value enters the NO2 mean when:

- `main_data_quality_flag == 0`;
- effective cloud fraction is at most `MIN_PIXEL_CLOUD`, currently 0.20;
- the NO2 value and footprint geometry are valid.

Filtering runs before averaging. Cloud and quality diagnostics still cover all
geometrically valid overlapping footprints, which keeps rejected inputs visible.

## Saved raster bundle

Each AOI scan is saved as one compressed `.npz` holding five aligned 48 by 48
`float32` rasters:

| Raster | What one output pixel displays | Missing value |
|---|---|---|
| `no2` | Overlap-area-weighted tropospheric NO2 from quality-0 footprints with cloud fraction at most 0.20, in molecules/cm2 | `NaN` when no accepted value reaches the cell or accepted overlap is below 0.25 km2 |
| `weighted_cloud_fraction` | Overlap-area-weighted cloud fraction from all valid native footprints, including footprints rejected from NO2 | `NaN` when no footprint with valid cloud information overlaps the cell |
| `good_quality_fraction` | Share of total overlapping footprint area carrying quality flag 0, from 0 to 1 | `NaN` when no valid native footprint overlaps the cell |
| `retrieval_uncertainty` | Overlap-area-weighted NO2 retrieval uncertainty from accepted footprints, in molecules/cm2 | `NaN` when no accepted footprint has finite uncertainty |
| `sum_weight` | Total area in km2 where accepted native footprints overlap the cell | `0.0` when no accepted footprint overlaps; zero marks a real absence of support, not a missing value |

The ancillary rasters stay populated where `no2` is `NaN`, which preserves
information about cloudy or low-overlap cells. Downstream code treats finite
`no2` values as the validity mask and requires both scans to be finite before
forming a delta.

## Dataset-generation output

`preprocessing.generate_dataset` deduplicates AOI-scan work and writes the
five-raster bundles to the persistent TEMPO cache under `DATASET_DIR`.

Caching behavior:

| Cache | Key | Batching | Refresh flag |
|---|---|---|---|
| TEMPO scans | AOI and source granules | Scans sharing a granule set run together, so a worker opens each large NetCDF granule once for several AOIs | `--refresh-tempo` |
| Aligned wind | AOI and hour | HRRR files are grouped so each full grid is read once for several AOIs | `--refresh-wind` |

`--refresh-cache` fully deletes both configured cache directories before
rebuilding entries referenced by the selected split. `--refresh-tempo` and
`--refresh-wind` fully delete only their respective cache. Refresh must run as a
single non-array process because all split jobs share these directories. The
TEMPO rebuild includes current, previous-hour, and EMA-history scans. Individual
cache files are still written atomically.

Pass the matching flag after changing TEMPO processing or wind alignment.

Each successful model record persists one compressed NPZ holding five aligned
`float32` arrays:

- `current_no2`, current-scan NO2 restricted to paired-valid support;
- `delta_no2`, current minus previous NO2 on the same support;
- `ema_delta_no2`, current minus a causal same-time 14-day NO2 EMA;
- `wind_u_10m_mps`, geographic eastward wind;
- `wind_v_10m_mps`, geographic northward wind.

The EMA uses one closest scan per preceding calendar day within 60 minutes of
the current scan time, a 5-day half-life, at least seven daily scans per record,
and at least five observations per output pixel. Historical scans use the same
persistent TEMPO image cache and are normalized only after the EMA delta is
formed.

Every row in the companion split CSV carries its NPZ path in `delta_no2_path`
plus these derived features:

- `plume_score`, from finite delta pixels as `(p99 - p50) / (p50 - p10)`;
- `paired_finite_fraction`, the share of the 48 by 48 grid finite in both scans;
- `central_finite_fraction`, the paired-finite share of the central 8 by 8 cells;
- `raster_quality_score`, equal to paired coverage for final record ranking;
- `mean_weighted_cloud_fraction` and `mean_good_quality_fraction`, each averaged
  over both scans at paired-valid delta cells;
- `mean_retrieval_uncertainty`, averaged over both scans at paired-valid cells;
- `temperature_2m_k` and `boundary_layer_height_m`, bilinearly interpolated at
  the AOI centroid.

Wind alignment, on the 3 km HRRR grid NOAA describes:

1. Project the 48 by 48 TEMPO cell centres onto the native Lambert grid.
2. Bilinearly interpolate the wind components.
3. Rotate the grid-relative values to geographic east and north before caching.

See the [NOAA Global Systems Laboratory HRRR overview](https://rapidrefresh.noaa.gov/).

The regridder also calculates squared weight, effective sample size, total
overlap area, contributor counts, and worst quality internally. These support
masking and validation and stay out of the modeling bundle.

## EDA decisions

Job 38571783 validated the implementation on 12 fixed scans and 40 sampled scan
pairs. Production uses:

- cloud fraction at most 0.20;
- overlap area alone for the NO2 weights;
- an accepted-overlap floor of 0.25 km2;
- no additional effective-sample floor.

Dataset generation then requires paired coverage of at least 0.50 and ranks
eligible records by paired coverage alone. Central coverage and retrieval
uncertainty remain diagnostics. None of these change per-scan tessellation or
its 0.25 km2 cell-support floor. Plume, cloud, and quality summaries also remain
diagnostics and rank nothing.

What the measurements showed:

| Choice | Effect |
|---|---|
| 0.25 km2 overlap floor | Paired-cell survival fell from 58.0 to 57.0 percent while removing very small edge overlaps. Kept. |
| 1.25 effective-sample floor | Survival fell to 21.1 percent. Rejected. |
| Area-only weighting | Lower median normalized RMS disagreement with Level 3 than linear or squared inverse-uncertainty weighting. Kept. |

The vectorized implementation matched the independent overlap reference to
floating-point precision. Warm tessellation took 0.13 to 0.16 seconds per AOI.

NASA Level 3 is not ground truth here, since it uses a geographic grid and
different input filtering. Its similarity catches geometry, unit, and diagnostic
mistakes. It cannot select the scientifically best custom filter or weight.
