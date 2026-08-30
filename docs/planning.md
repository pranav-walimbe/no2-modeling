# Research plan: detecting power-plant anomalies with TEMPO

## Research question

Can hourly TEMPO NO2 and SO2 observations detect departures from expected power-plant operation and emissions, and how do weather, source density, fuel mix, and pixel resolution limit attribution?

The study will estimate continuous deviations from an expectation model. It will serve two applications:

- Atmospheric science: detect unusual emissions, control performance, and fossil-generation changes.
- Commodity research: test whether plant or regional operating surprises add information before standard public reports.

## Targets

### Operating anomaly

Estimate expected gross load or capacity factor from capacity, unit and fuel type, calendar variables, weather, and region. Include regional demand only when the application permits it.

$$
z_{\mathrm{load},i,t}=
\frac{y_{\mathrm{load},i,t}-\hat{\mu}_{\mathrm{load},i,t}}
{\hat{\sigma}_{\mathrm{load},i,t}}
$$

This target covers outages, startups, shutdowns, and dispatch changes. It is the primary commodity target.

### Emissions anomaly

$$
z_{g,i,t}=
\frac{\log(1+y_{g,i,t})-\hat{\mu}_{g,i,t}}
{\hat{\sigma}_{g,i,t}},
\qquad g\in\{\mathrm{NOx},\mathrm{SO2}\}
$$

Use total-emissions surprise as the primary atmospheric target. A version conditioned on gross load will measure emissions-intensity or control-performance surprise. Exclude observed load at market inference time when it has not entered the public information set.

Train on signed continuous residuals and derive high, normal, and low event classes afterward. Add startup, shutdown, and ramp labels when CAMPD supports them.

## Defining expectation

### Fleet model

Fit quantile models using plant characteristics, calendar variables, weather, and region. Estimate the 5th, 50th, and 95th percentiles to capture changes in conditional variance. The fleet model must transfer to plants absent from training and must omit historical facility-emissions priors.

### Plant calibration

Add a past-only correction to the fleet estimate for known-site monitoring:

$$
\mu_{i,t}=f_\theta(x_{i,t})+b_{i,t}
$$

$$
b_{i,t}=(1-\alpha)b_{i,t-1}
+\alpha\,\operatorname{clip}\!\left[
\log(1+y_{i,t-1})-f_\theta(x_{i,t-1})
\right]
$$

Clip large residuals and pause updates during flagged events so sustained anomalies do not enter the baseline. Compare this hybrid with a seasonal plant median and a raw plant-hour exponential moving average.

Map each observation through the estimated conditional distribution to produce a common anomaly scale:

$$
z_{i,t}=\Phi^{-1}\!\left[\widehat{F}(y_{i,t}\mid x_{i,t})\right]
$$

Use a hurdle model for operating state and conditional output. Generate every expectation through rolling-origin or out-of-fold prediction.

Keep behavior and observation features separate. The expectation model receives engineering, calendar, demand, and dispatch-weather variables. The TEMPO model receives plume winds, cloud, air-mass-factor, and retrieval-quality variables.

## Source and fuel scope

Match the target to source identifiability:

1. Isolated plants: plant-level emissions and operating targets.
2. Homogeneous clusters: group emissions and generation targets.
3. Heterogeneous unresolved clusters: aggregate NOx, with fuel-specific generation as an experimental target.

Construct a response kernel for each source from location, wind, stack or plume height, and dispersion. Aggregate sources when their response vectors become too collinear at TEMPO resolution. Permit the model to abstain when the scene cannot support attribution.

Center evaluation on coal and coal-dominant groups because they provide the strongest NOx and SO2 test. Retain other fuels for coverage and transfer tests. A fuel-aware model should use capacity shares, expected emissions-intensity shares, controls, unit counts, source spacing, and a fuel-concentration index. Do not infer aggregate generation from mixed-fuel plumes without these features.

## Data and preparation

### TEMPO and meteorology

- Use V04 Level-2 NO2 and preserve native pixel geometry and exposure time.
- Test SO2 first on large CAMPD sources; retain it if coverage and held-out value justify it.
- Include column uncertainty, quality flags, cloud properties, geometry, pressure, terrain, albedo, and boundary-layer height.
- Replace 10 m wind with components near 100, 200, 300, and 500 m; interpolate at stack and plume-rise heights.
- Add stability, wind shear, and disagreement across vertical levels.

### Plants and labels

Use capacity, fuel, unit type, stack height, controls, commissioning age, heat rate, and member locations. Build CAMPD labels for NOx, SO2, gross load, operating time, and operating-state changes.

Revise the current collection before modeling power. It sums facility NOx but retains one representative unit's gross load. The new aggregation must sum compatible electrical loads and separate steam-load records. The scraper must retain SO2, CO2, heat input, load units, status fields, method-of-determination fields, zero operation, and missingness.

### Physical representation

- Align CAMPD hours to facility time zone and TEMPO exposure time; test plume-travel lags.
- Transform scenes into downwind and crosswind coordinates.
- Derive upwind background, downwind enhancement, plume width and decay, directional derivative, flux divergence, column tendency, matched-filter response, and signal-to-uncertainty ratio.
- Compare single scans with adjacent scans, daily samples, and rolling multi-day samples.
- Apply consistent cloud and quality tests across sequences while retaining missingness indicators.

Defer free-standing super-resolution. Native pixels, response kernels, plume coordinates, and changing winds address resolution without inventing sub-pixel structure. Revisit super-resolution only if a mass-conserving model improves held-out emission or event estimates after downsampling.

## Models and experiments

Compare:

1. Seasonal median and raw exponential-moving-average expectations.
2. Cold-start fleet quantile expectation.
3. Plant-calibrated fleet expectation.
4. Plant characteristics and weather without TEMPO.
5. Physics summaries plus plant characteristics and weather.
6. Wind-conditioned raster model or native-pixel set model.
7. Multi-pollutant and temporal versions of the combined model.

Ablate NO2, SO2, tendency features, vertical winds, uncertainty, plant characteristics, and temporal context. Claim satellite value only if the combined model beats the plant-and-weather baseline on held-out plants or times.

## Existing event support

The coal data contain 5.18 million unit-hours and 2.93 million facility-hours across 178 facilities. An exploratory event rule, an hourly change above 25% of a facility's 95th-percentile level, found:

- 33,123 gross-load changes, including 15,972 during approximate daylight hours.
- 74,597 NOx changes, including 37,197 during approximate daylight hours.
- Daylight load events at 175 facilities and daylight NOx events at all 178.

These counts establish label-stage feasibility. TEMPO coverage, clouds, sequence requirements, and identifiability will reduce support. The final labels will use out-of-sample residuals and require operating time, load, adjacent hours, and status fields where available.

## Evaluation

### Atmospheric science

- Hold out source groups to test transfer and hold out time to test known-site monitoring.
- Report results by source class, fuel, emission magnitude, wind, cloud, spacing, and identifiability.
- Use MAE, normalized MAE, R2, rank correlation, calibration, and plant-macro averages.
- Use precision-recall area, event F1, false alarms per observation, and detection delay for derived events.
- Estimate detection limits for each pollutant, operating change, and source-separation regime.

### Commodity research

Use a walk-forward backtest with observation-availability timestamps. Reproduce cloud gaps and TEMPO product latency, exclude revised or future records, and compare against weather, ISO prices and outages, EIA data, and other public inputs.

Test plant anomalies and regional thermal-availability aggregates against nodal or hub power-price spreads. Statistical detection has economic value only if the signal improves forecasts after public information available at release time.

## Decision rules

- **SO2:** require usable coverage and incremental held-out skill; otherwise retain a large-source case study.
- **Attribution:** report plant estimates only when response kernels pass an identifiability threshold.
- **Fuel:** use coal as the primary detectability stratum and mixed fuels as the transfer and attribution test.
- **Cadence:** select the shortest interval with stable held-out skill.
- **Target:** model continuous residuals; use direct classification only if it improves calibration or represents a distinct status.
- **Expectation:** use fleet quantiles for unseen plants and past-only plant calibration for known sites.
- **Market claim:** require signal availability before, or incremental to, standard public data.

## Paper framing

**Atmospheric-science question:** Under which source and atmospheric conditions can TEMPO detect and attribute anomalous power-sector emissions? Contributions include an expectation-based target, uncertainty-aware aggregation, and fuel-specific detection limits. The climate link concerns fossil-generation changes and co-emitted pollutants; TEMPO NO2 and SO2 do not measure CO2 directly.

**Commodity application:** Can TEMPO anomaly scores improve regional thermal-availability and power-spread forecasts? Keep this as a separate application unless the walk-forward results support an economic claim.

## Literature anchors

- [Sun et al.: hourly TEMPO NOx emissions](https://doi.org/10.1029/2025JD044565)
- [Li et al.: first TEMPO SO2 retrievals](https://doi.org/10.1029/2025GL115788)
- [Beirle et al.: point-source NOx flux divergence](https://doi.org/10.5194/essd-13-2995-2021)
- [Koene et al.: divergence-method limits](https://doi.org/10.1029/2023JD039904)
- [Kuhlmann et al.: temporal variability and source aggregation](https://doi.org/10.5194/acp-26-4405-2026)
- [TEMPO V04 trace-gas guide](https://asdc.larc.nasa.gov/documents/tempo/guide/TEMPO_Level-2-3_trace_gas_clouds_user_guide_V2.1.pdf)
