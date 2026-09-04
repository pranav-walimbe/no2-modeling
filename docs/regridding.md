# Regridding TEMPO Level 2 pixels onto AOI rasters

TEMPO Level 2 reports one NO2 value per irregular ground footprint. The model
needs a fixed 48 by 48 raster covering 72 km around each AOI, with the same
1.5 km cells in both scans of every delta.

The regridder projects each footprint and output cell to the equal-area
EPSG:5070 coordinate system, intersects their polygons, and averages accepted
NO2 values using overlap area in square kilometres. A spatial index finds only
the cells each footprint can touch, so it never constructs a dense
`native pixels x output cells` matrix.

## Filtering and diagnostics

A native value enters the NO2 mean when:

- `main_data_quality_flag == 0`;
- effective cloud fraction is at most `MIN_PIXEL_CLOUD`, currently 0.20; and
- the NO2 value and footprint geometry are valid.

Filtering occurs before averaging. Cloud and quality diagnostics still use all
geometrically valid overlapping footprints so rejected inputs remain visible.

## Saved raster bundle

Each AOI scan is saved as one compressed `.npz` containing five aligned 48 by
48 `float32` rasters:

| Raster | What one output pixel displays | Missing value |
|---|---|---|
| `no2` | Overlap-area-weighted tropospheric NO2 from quality-0 footprints with cloud fraction at most 0.20, in molecules/cm2 | `NaN` when no accepted value reaches the cell or accepted overlap is below 0.25 km2 |
| `weighted_cloud_fraction` | Overlap-area-weighted cloud fraction from all valid native footprints, including footprints rejected from NO2 | `NaN` when no footprint with valid cloud information overlaps the cell |
| `good_quality_fraction` | Share of total overlapping footprint area carrying quality flag 0, from 0 to 1 | `NaN` when no valid native footprint overlaps the cell |
| `retrieval_uncertainty` | Overlap-area-weighted NO2 retrieval uncertainty from accepted footprints, in molecules/cm2 | `NaN` when no accepted footprint has finite uncertainty |
| `sum_weight` | Total area in km2 where accepted native footprints overlap the cell | `0.0` when no accepted footprint overlaps; zero is a real absence of support rather than a missing value |

The ancillary rasters can remain populated where `no2` is `NaN`. This preserves
information about cloudy or low-overlap cells. Downstream code uses finite
`no2` values as the validity mask and requires both scans to be finite before
forming a delta.

The regridder also calculates squared weight, effective sample size, total
overlap area, contributor counts, and worst quality internally. These values
support masking and validation but are not persisted in the modeling bundle.

## EDA decisions

Job 38571783 validated the implementation on 12 fixed scans and 40 sampled
scan pairs. Production uses:

- cloud fraction at most 0.20;
- overlap area alone for the NO2 weights;
- an accepted-overlap floor of 0.25 km2; and
- no additional effective-sample floor.

The 0.25 km2 floor reduced paired-cell survival from 58.0 percent to 57.0
percent while removing very small edge overlaps. An effective-sample floor of
1.25 reduced survival to 21.1 percent and was rejected. Area-only weighting had
lower median normalized RMS disagreement with Level 3 than linear or squared
inverse-uncertainty weighting.

The vectorized implementation matched the independent overlap reference to
floating-point precision. Warm tessellation took 0.13 to 0.16 seconds per AOI.

NASA Level 3 is not treated as ground truth because it uses a geographic grid
and different input filtering. Its similarity is useful for catching geometry,
unit, and diagnostic mistakes; it cannot by itself select the scientifically
best custom filter or weight.
