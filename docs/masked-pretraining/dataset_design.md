# Dataset design

Masked pretraining uses one TEMPO scene at a time. Each sample pairs NO2 with
aligned HRRR temperature and wind, hides part of the NO2 field, and keeps the
original field as the reconstruction target. CAMPD emissions values serve no
role as labels or inputs. Facility coordinates define the candidate AOIs.

## Design philosophy

The first dataset favors clean supervision. A sample qualifies when all 576 NO2
cells and all weather cells contain finite values. Each hidden NO2 pixel has a
measured target, with no interpolation or neighboring timestamp acting as ground
truth.

Complete scenes form a selected subset of the downstream population. Cloud,
season, geography, wind, and NO2 level may affect membership. Dataset audits
must compare this cohort with the downstream data rather than treating a large
sample count as protection against complete-case bias.

The generator assigns overlapping 72 km AOI groups to one split. This keeps
nearby plants and shared raster footprints from crossing geographic boundaries.
The configured targets are 500,000 train scenes and 50,000 scenes in each
evaluation split. The generator deduplicates records by AOI and TEMPO scan.

## Sample contract

Each 24 by 24 bundle contains:

| Field | Role |
|---|---|
| `no2` | Untouched reconstruction target |
| `no2_mask` | Original TEMPO validity mask |
| `temperature_2m_k` | Visible weather context |
| `wind_u_80m_mps`, `wind_v_80m_mps` | Visible wind context |
| `artificial_mask` | One for visible NO2 and zero for synthetic gaps |
| `masked_no2` | NO2 with synthetic gaps filled by zero |

The loader keeps the original validity mask and synthetic mask separate. A
source-missing pixel must not contribute to reconstruction loss if a later
experiment admits incomplete scenes.

## Synthetic masks

The mask should hide enough context to require spatial inference while leaving
enough NO2 to identify plume structure. The current sampler draws a mask ratio
from 1% through 10%. It gives boundary pixels more weight and raises the chance
of selecting pixels near prior selections. The result mixes isolated gaps with
small clusters instead of scattering all missing pixels with equal probability.

This mask family offers a controlled baseline. Small gaps resemble the missing
regions that the downstream imputer must fill, and the low ratios limit the gap
between pretraining and downstream coverage. The design does not claim to model
the full cloud and retrieval process. Mask-ratio and shape ablations should
follow evidence from observed validity masks.

The split seed, shard task, and record offset determine each mask. Repeating a
run with the same shard settings reproduces it. The bundle stores the resulting
mask, though it does not store the seed or sampled ratio as separate fields.
