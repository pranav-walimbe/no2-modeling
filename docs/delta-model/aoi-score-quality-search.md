# AOI continuous plume-quality score

## Objective

Rank AOIs by whether a detectable source-localized NO2 plume changes with the
continuous CAMPD emissions label. The score is for AOI selection and analysis,
not a delta-model input.

Raster samples may come from the prior delta dataset, the general TEMPO and
weather caches, or the masked-dataset corpus. Only histories with aligned CAMPD
labels contribute to score-label agreement. Unlabeled histories may contribute
to a separate plume-detectability diagnostic. Deduplicate histories by AOI and
observation time before aggregation.

## Label

For four scans ending at `t3`, overlap-weight CAMPD hourly NOx over each scan
interval and update an irregular two-hour EMA:

```text
r_i = exp(-(time_i - time_{i-1}) / 2 hours)
E_i = r_i E_{i-1} + (1 - r_i) NOx_i
emissions_delta = E_3 - E_2
```

Decrease, steady, and increase categories remain diagnostics. The scoring
target is continuous:

```text
e = tanh(emissions_delta / 100 lb)
```

Thus 99 and 100 lb receive nearly equal targets, and steady histories are
represented by `e` near zero.

## Raster heuristic

Normalize valid NO2 pixels with the fixed values in `AGENTS.md`:

```text
z = clip((NO2 - 1.868138303979520e15) / 1.1997222249899362e15, -8, 8)
```

At each timestep:

1. subtract a mask-normalized Gaussian background;
2. search within 45 degrees of current and preceding local 80 m wind;
3. compare a source-anchored downwind core with crosswind flanks;
4. penalize broad and source-disconnected positive structure.

```text
noise = max(1.4826 * background_MAD, 0.10)
raw_SNR = max((core_response - flank_response) / noise, 0)
SNR = raw_SNR * sqrt(localization * anchored_fraction) * exp(-3 * broad_fraction)
```

Apply the label EMA timing to the four signed plume amplitudes. Robustly scale
their final change:

```text
p = tanh((plume_EMA_3 - plume_EMA_2) / plume_delta_scale)
d = tanh(temporal_SNR / snr_scale)
agreement = 1 - abs(e - p)
record_quality = d * agreement
```

`record_quality` lies in `[-1, 1]`. A visible stable plume scores well for a
steady emissions interval. A blank raster receives little credit because `d`
is near zero. Opposed plume and emissions changes can score below zero.

## AOI aggregation

```text
center = mean(record_quality)
shrunk_center = center * n / (n + pseudo_count)
AOI_score = shrunk_center - uncertainty_penalty * SE(record_quality)
```

Require at least eight finite labeled histories. Search the temporal SNR
summary, saturation scales, pseudo-count, and uncertainty penalty using held-out
record folds. Publish the selected score to a run CSV and atomically merge it
into the JSON path configured by `AOI_SCORE_JSON`.

## Rewarded and penalized characteristics

| Level | Rewarded | Penalized |
|---|---|---|
| Timestep | localized downwind enhancement; source-connected structure; adequate support | broad regional enhancement; crosswind response; weak support |
| History | detectable plume; continuous plume change matching emissions change; stable plume during steady emissions | weak detection; mismatched magnitude or direction; plume change during steady emissions |
| AOI | high mean agreement; many histories; low uncertainty | inconsistent histories; small sample; high standard error |
| Source geometry, empirically | one dominant facility; fewer and more compact sources | many dispersed or independently operating sources |

Fuel type is not a scoring term. Prior coal association weakened after adjusting
for source count, geometry, capacity dominance, and emissions scale.

## Results

Continuous baseline job `39206139` scored 25,507 histories. The selected
non-transport configuration uses median four-timestep SNR, `snr_scale = 0.25`,
`plume_scale = 3.0`, no pseudo-count, and a `0.5 * SE` penalty.

| Metric | Continuous result |
|---|---:|
| AOIs represented in feature analysis | 953 |
| Mean held-out Spearman | 0.512 |
| Mean top-quartile lift | 0.0616 |
| Mean top-minus-bottom separation | 0.1169 |

Cache gap-fill job `39206363` added 65 AOIs with at least eight finite labeled
histories. The persistent mapping now contains 1,018 of 1,339 emissions AOIs.

Directional baseline from job `39195305` and refinement job `39195783`:

| Metric | Result |
|---|---:|
| Sampled histories | 28,937 |
| Final eligible AOIs | 413 |
| Mean held-out Spearman | 0.308 |
| Mean top-quartile lift | 0.066 |
| Mean top-minus-bottom separation | 0.155 |

Interpretation: label attribution, not raw plume strength, was the main
separator. AOIs with fewer, more compact sources were more likely to align the
aggregate CAMPD change with the plume near the selected hotspot.

## Next ablations

1. Continuous agreement on prior delta bundles.
2. Add cache-backed labeled histories only where they expand AOI coverage or
   reduce uncertainty.
3. The tested semi-Lagrangian advection residual was rejected: held-out
   Spearman fell from `0.5110` to `0.4383`, and the combined objective fell by
   `0.0178`. Top-minus-bottom separation increased by `0.0213`, but that was
   insufficient to retain transport.
4. Do not test chemical lifetime unless a later transport formulation first
   beats the non-transport baseline.

Transport literature supports wind rotation and downwind integration, while
also showing sensitivity to wind choice and weak identifiability of lifetime
from individual plumes: [wind rotation and EMG](https://amt.copernicus.org/articles/17/3439/2024/),
[lifetime and spread validation](https://amt.copernicus.org/articles/14/7929/2021/),
and [wind-field and plume-curvature sensitivity](https://acp.copernicus.org/articles/23/4577/2023/).

## Artifacts

- Directional baseline: `/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-39195305/`
- Directional refinement: `/global/home/users/pranavwalimbe/vis/aoi-score-refinement-39195783/`
- Continuous and transport comparison: `/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-39206139/`
- Reliability search: `/global/home/users/pranavwalimbe/vis/aoi-score-refinement-39205972/`
- Final 953-AOI baseline: `/global/home/users/pranavwalimbe/vis/aoi-score-refinement-39206284/`
- Cache gap-fill: `/global/home/users/pranavwalimbe/vis/aoi-score-cache-scan-39206363/`
