# Wind- and uncertainty-aware NO2 denoising

## Status

This document records the denoising kernel selected in the September 2026 EDA.
The dataset generator applies it to each current and previous TEMPO scan before
forming the hourly delta. The model loader consumes the resulting rasters.

The kernel aims to reduce pixel-scale retrieval noise while retaining plume
shape and magnitude. It processes a single TEMPO scan at a time and does not use
emissions, labels, facility coordinates, or unit activity.

## Inputs

Each call receives four aligned 48 by 48 rasters:

| Input | Role |
|---|---|
| Tropospheric NO2 | Observed image to denoise |
| Retrieval uncertainty | Sets pixel confidence and edge significance |
| Eastward wind | Sets the local smoothing direction |
| Northward wind | Sets the local smoothing direction |

The EDA uses geographic 80 m HRRR winds matched to the TEMPO observation time.
For an hourly delta, process the current and previous scans with their own
time-matched winds before subtracting them.

The kernel uses the full wind raster. It does not rotate the input image or
replace the field with one median wind vector. EDA montages may rotate finished
rasters around the AOI-centre wind to make downwind point right.

## Algorithm

### 1. Build the valid-pixel graph

Treat each finite NO2 cell as a graph node. Connect horizontal, vertical, and
diagonal neighbours when both cells contain valid NO2. Missing NO2 remains
missing and does not enter the solve.

If a valid NO2 cell lacks a positive uncertainty, substitute the scan median
uncertainty. This preserves the NO2 support mask.

### 2. Set data fidelity from uncertainty

Each node receives this observation precision:

```text
precision = clip((median uncertainty / pixel uncertainty)^2, 0.25, 4.0)
```

Low-uncertainty pixels stay closer to their observations. The clipping bounds
stop uncertainty from either freezing a pixel or removing its influence. This
matters because retrieval uncertainty can increase with absolute NO2 amount.

### 3. Set wind-directed edge weights

Average the two endpoint wind vectors for each graph edge. Let `alignment` be
the absolute cosine between the graph edge and that local wind. The directional
factor is:

```text
directional factor = 0.15 + 0.85 * alignment^2
```

The graph gives along-wind neighbours the largest coupling and uses 15 percent
of that coupling across the wind. Calm or invalid wind produces
isotropic coupling.

The kernel also protects observed boundaries. It measures each neighbouring
NO2 difference relative to the two pixels' combined uncertainty, then reduces
coupling when that standardized difference exceeds the 2.5 threshold. Diagonal
edges receive an additional inverse-distance-squared adjustment.

### 4. Solve the regularized image

The sparse linear solve balances observation fidelity against weighted
neighbour agreement. Its selected regularization strength is 1.1. In compact
form, it minimizes:

```text
precision-weighted change from observed NO2
    + 1.1 * wind- and edge-weighted neighbour differences
```

This step suppresses isolated variation and favors features that continue in
the local wind direction.

### 5. Restore coherent plume detail

Smoothing can weaken narrow plumes. A structure-tensor pass identifies image
ridges using three signals:

- local directional coherence;
- gradient strength relative to retrieval uncertainty;
- ridge-tangent agreement with local wind.

The kernel restores 70 percent of the removed residual at full confidence and
less elsewhere:

```text
final = smoothed + 0.70 * confidence * (observed - smoothed)
```

This restoration favors uncertainty-significant wind-aligned ridges. It leaves
incoherent speckle close to the smoothed solution.

## Selected settings

| Setting | Value |
|---|---:|
| Regularization strength | 1.1 |
| Crosswind smoothing ratio | 0.15 |
| Edge-significance threshold | 2.5 |
| Minimum and maximum data precision | 0.25 and 4.0 |
| Coherent-detail restoration | 0.70 |
| Calm-wind threshold | 0.5 m/s |

## Evaluation

The final evaluation used 1,000 records: 500 negative and 500 positive hourly
NOx changes across 182 AOIs. AOI-clustered bootstrap intervals used 2,000
resamples.

The 95 percent cluster-bootstrap intervals excluded zero for lower background
MAD, higher edge retention, and higher pixel correlation. The flux-preservation
slope interval included zero. Label AUC stayed near chance for both kernels, so
it serves as a leakage and degradation check rather than the optimization
target.

The sweep also tested Huber edge penalties and iterative edge reweighting.
Those variants reduced background variation but removed more plume-edge signal.
The selected kernel combines a stronger quadratic solve with targeted detail
restoration.

Relevant Savio jobs:

- `38700086`: selected kernel on 1,000 records;
- `38700279`: matched previous-kernel baseline;
- `38700536`: 20-scene original-versus-selected montage.

## Limits and integration requirements

- Ten-metre wind may differ from transport winds at plume height. A future
  comparison should test boundary-layer or pressure-level winds.
- The optimization metrics estimate image quality without plume ground truth.
  They do not establish that denoising improves the downstream classifier.
- The kernel handles curved and multi-source scenes through local wind without
  fixed source corridors. Overlapping plumes can still be inseparable.
- Current and previous scans are denoised separately with winds matched to each
  observation's nearest HRRR analysis hour.
- Later normalization is fit on training data alone. Existing masks, coverage
  gates, and split rules remain unchanged.

## Method context

The kernel is a project-specific estimator rather than a reproduction of one
published algorithm. Wind rotation and plume-coordinate analysis follow the
general treatment used in satellite NO2 plume studies such as
[Griffin et al. (2021)](https://amt.copernicus.org/articles/14/7929/2021/amt-14-7929-2021.html).
The coherent-detail gate draws on directional structure-tensor regularization,
including the approach described by
[Demircan-Tureyen and Kamasak (2021)](https://www.sciencedirect.com/science/article/pii/S0923596521002423).
These papers motivate the spatial priors. The parameter values above come from
this project's held-out EDA.
