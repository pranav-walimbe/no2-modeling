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

Filtering occurs before averaging. Diagnostics are nevertheless calculated
from all geometrically valid overlapping footprints so rejected inputs remain
visible. Each output contains:

| Raster | Meaning |
|---|---|
| `no2` | Accepted overlap-area-weighted tropospheric NO2 |
| `sum_weight` | Total accepted overlap area in km2 |
| `sum_weight_squared` | Sum of squared accepted overlap areas in km4 |
| `effective_sample_size` | `(sum_weight)^2 / sum_weight_squared` |
| `total_overlap_area` | Overlap area in km2 from all native contributors |
| `native_pixel_count` | Number of all native contributors |
| `accepted_pixel_count` | Number of contributors accepted into NO2 |
| `weighted_cloud_fraction` | Overlap-weighted cloud fraction from all inputs |
| `good_quality_fraction` | Fraction of all overlap area carrying quality flag 0 |
| `main_data_quality_flag` | Worst overlapping native flag, like NASA's conservative L3 flag |
| `retrieval_uncertainty` | Overlap-weighted uncertainty of accepted contributors |

Only `no2` is changed by cell masking. The raw diagnostics remain available so
the support rule can be revised without rerunning tessellation.

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
