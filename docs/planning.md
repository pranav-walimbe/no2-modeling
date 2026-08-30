# Research plan: detecting power-plant NOx anomalies with TEMPO

## Research question

Can hourly TEMPO NO2 observations detect departures from expected power-plant operation and NOx emissions, and how do weather, source density, fuel mix, and pixel resolution limit attribution?

NO2/NOx is the sole satellite pollutant in the planned models and the basis for the paper's claims. Reserve SO2 for future work.

The study will estimate continuous deviations from an expectation model. It will serve two applications:

- Atmospheric science: detect unusual emissions, control performance, and fossil-generation changes.
- Commodity research: test whether plant or regional operating surprises add information before standard public reports.

## Targets

### Operating anomaly

Estimate the expected range of gross load or capacity factor from capacity, unit and fuel type, calendar variables, weather, and regional electricity demand. Define the operating anomaly from the signed difference between observed and expected output, scaled by the model's uncertainty.

This target covers outages, startups, shutdowns, and dispatch changes. It is the primary commodity target.

### Emissions anomaly

Estimate the expected range of hourly NOx emissions and define the emissions anomaly from the signed difference between observed and expected emissions. A version conditioned on gross load will measure emissions-intensity or control-performance surprise. Exclude observed load at market inference time when it has not entered the public information set.

Train on signed continuous residuals and derive high, normal, and low event classes afterward. Add startup, shutdown, and ramp labels when CAMPD supports them.

## Defining expectation

### Fleet model

Fit quantile models using plant characteristics, calendar variables, weather, and region. Estimate the 5th, 50th, and 95th percentiles to capture changes in conditional variance. The fleet model must transfer to plants absent from training and must omit historical facility-emissions priors.

### Plant calibration

Add a past-only moving correction to the fleet estimate for known-site monitoring. Limit the effect of large residuals and pause updates during flagged events so sustained anomalies do not enter the baseline. Compare this hybrid with a seasonal plant median and a raw plant-hour exponential moving average.

Convert each observation's position within its expected distribution to a common anomaly scale across plants and operating regimes.

Use a hurdle model for operating state and conditional output. Generate every expectation through rolling-origin or out-of-fold prediction.

Keep behavior and observation features separate. The expectation model receives engineering, calendar, demand, and dispatch-weather variables. The TEMPO model receives plume winds, cloud, air-mass-factor, and retrieval-quality variables.

## Source and fuel scope

Match the target to source identifiability:

1. Isolated plants: plant-level emissions and operating targets.
2. Homogeneous clusters: group emissions and generation targets.
3. Heterogeneous unresolved clusters: aggregate NOx, with fuel-specific generation as an experimental target.

Construct a response kernel for each source from location, wind, stack or plume height, and dispersion. Aggregate sources when their response vectors become too collinear at TEMPO resolution. Permit the model to abstain when the scene cannot support attribution.

Center evaluation on coal and coal-dominant groups because they provide the strongest NOx test. Retain other fuels for NOx coverage and transfer tests. A fuel-aware model should use capacity shares, expected emissions-intensity shares, controls, unit counts, source spacing, and a fuel-concentration index. Do not infer aggregate generation from mixed-fuel plumes without these features.

## Data and preparation

### TEMPO and meteorology

- Use standard V04 Level-3 NO2 as the sole satellite input. Retain its 0.02-degree grid, scan timestamp, retrieval uncertainty, quality information, and coverage indicators.
- Include column uncertainty, quality flags, cloud properties, geometry, pressure, terrain, albedo, and boundary-layer height.
- Use archived HRRR model runs for training and live HRRR for inference. For each observation, select a weather field valid at the TEMPO scan time from a model cycle available by the decision timestamp.
- Use HRRR wind components near 100, 200, 300, and 500 m; interpolate at stack and plume-rise heights.
- Add stability, wind shear, and disagreement across vertical levels.
- Use HRRR temperature in two roles. Plant-local and regional 2 m temperature enter the expectation model to capture weather-driven load and dispatch. Vertical temperature profiles, potential-temperature gradients, and derived stability enter the TEMPO model to represent plume rise and boundary-layer mixing.
- Keep HRRR inputs tabular in the primary experiment. Summarize plant-local, regional, upwind, downwind, and vertical-profile conditions instead of adding a meteorology raster.

### Regional electricity data

- Use the EIA-930 Hourly Electric Grid Monitor as the nationwide source. Map each plant to its balancing authority using EIA-860 metadata.
- Use hourly balancing-authority demand after EIA publishes it, which respondents must provide within 60 minutes after the reported hour. Use the current-day demand forecast only after its publication time.
- Treat generation by fuel and total interchange as recent-history features because EIA publishes them with a one-day lag. Treat interchange with neighboring balancing authorities as a two-day-lag feature.
- Derive demand level, one- and three-hour ramps, demand forecast error, hour-of-week demand percentile, and temperature-conditioned demand surprise. Add lagged coal and gas generation shares and net interchange as secondary features.
- Archive each API response with retrieval and source-publication timestamps. The historical API may contain revisions that were unavailable at the original decision time.
- Keep direct ISO and RTO feeds outside the core paper. They can provide five-minute load, fuel mix, prices, and outages in a later market-specific implementation.

### Plants and labels

Use capacity, fuel, unit type, stack height, controls, commissioning age, heat rate, and member locations. Build CAMPD labels for NOx, gross load, operating time, and operating-state changes.

Revise the current collection before modeling power. It sums facility NOx but retains one representative unit's gross load. The new aggregation must sum compatible electrical loads and separate steam-load records. The scraper should retain SO2 and CO2 for future work, along with heat input, load units, status fields, method-of-determination fields, zero operation, and missingness.

### Physical representation

- Align CAMPD hours to facility time zone and TEMPO exposure time; test plume-travel lags.
- Extract fixed plant-centered patches from the V04 Level-3 grid and transform copies into downwind and crosswind coordinates.
- Derive upwind background, downwind enhancement, plume width and decay, directional derivative, flux divergence, column tendency, matched-filter response, and signal-to-uncertainty ratio.
- Compare single scans with adjacent scans, daily samples, and rolling multi-day samples.
- Apply consistent cloud and quality tests across sequences while retaining missingness indicators.

Do not upsample the V04 Level-3 grid or claim sub-grid structure. Response kernels, plume coordinates, and changing winds can address source resolution within the observed grid. Revisit super-resolution only if a mass-conserving model improves held-out emission or event estimates after downsampling.

## Models and experiments

Compare:

1. Seasonal median and raw exponential-moving-average expectations.
2. Cold-start fleet quantile expectation.
3. Plant-calibrated fleet expectation.
4. Plant characteristics and weather without TEMPO.
5. Plant characteristics, weather, and EIA-930 regional electricity data without TEMPO.
6. Directional-derivative or flux-divergence emission estimates without supervised raster features.
7. Physics summaries plus plant characteristics, weather, and regional electricity data.
8. Wind-conditioned V04 Level-3 raster model.
9. A hybrid model that combines the physical estimator and the raster model.
10. Temporal versions of the combined model.

Ablate NO2, EIA-930 regional demand, tendency features, vertical winds, temperature and stability features, uncertainty, plant characteristics, and temporal context. Claim satellite value only if the combined model beats the plant-weather-electricity baseline on held-out plants or times.

Use two hybrid forms. First, feed the directional-derivative or flux-divergence estimate and its uncertainty diagnostics into the supervised model. Second, fit a past-only residual correction to the physical estimate and stack its out-of-fold prediction with the raster model. Compare these with a scene-quality gate that favors the physical branch under identifiable, high-signal conditions and allows the raster branch to dominate when derivatives become noisy. Keep the uncorrected physical estimate as a benchmark so the hybrid's gain and any loss of transferability remain visible.

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
- Estimate detection limits for NOx, operating changes, and each source-separation regime.
- Report accuracy, calibration, coverage, and inference time separately. Measure latency from the availability of V04 Level-3 and the selected HRRR cycle. Record HRRR initialization, forecast lead, and provider-availability timestamps.

### Commodity research

Use a walk-forward backtest with observation-availability timestamps. Reproduce cloud gaps, V04 Level-3 latency, HRRR model-cycle availability, and EIA-930 publication lags. Exclude revised or future records, and compare against weather, ISO prices and outages, EIA data, and other public inputs.

Test plant anomalies and regional thermal-availability aggregates against nodal or hub power-price spreads. Statistical detection has economic value only if the signal improves forecasts after public information available at release time.

## Decision rules

- **Satellite input:** use standard V04 Level-3 NO2 for all planned models. Leave Level-2, NRT, and SO2 extensions outside the current scope.
- **Attribution:** report plant estimates only when response kernels pass an identifiability threshold.
- **Fuel:** use coal as the primary detectability stratum and mixed fuels as the transfer and attribution test.
- **Cadence:** select the shortest interval with stable held-out skill.
- **Target:** model continuous residuals; use direct classification only if it improves calibration or represents a distinct status.
- **Expectation:** use fleet quantiles for unseen plants and past-only plant calibration for known sites. Include EIA-930 regional demand only when the source had published it by the decision timestamp; use generation and interchange as lagged history.
- **Market claim:** require signal availability before, or incremental to, standard public data.

## Paper framing

**Atmospheric-science question:** Under which source and atmospheric conditions can standard TEMPO V04 Level-3 NO2 detect and attribute anomalous power-sector NOx emissions? Contributions include an expectation-based target, HRRR-informed plume representation, a comparison of physical and learned estimators, and fuel-specific detection limits. The climate link concerns fossil-generation changes and co-emitted pollutants; TEMPO does not measure CO2 directly.

**Commodity application:** Can TEMPO anomaly scores improve regional thermal-availability and power-spread forecasts? Keep this as a separate application unless the walk-forward results support an economic claim.

## Literature anchors

- [Sun et al.: hourly TEMPO NOx emissions](https://doi.org/10.1029/2025JD044565)
- [Beirle et al.: point-source NOx flux divergence](https://doi.org/10.5194/essd-13-2995-2021)
- [Koene et al.: divergence-method limits](https://doi.org/10.1029/2023JD039904)
- [Kuhlmann et al.: temporal variability and source aggregation](https://doi.org/10.5194/acp-26-4405-2026)
- [TEMPO V04 trace-gas guide](https://asdc.larc.nasa.gov/documents/tempo/guide/TEMPO_Level-2-3_trace_gas_clouds_user_guide_V2.1.pdf)
- [NOAA HRRR model and open archive](https://registry.opendata.aws/noaa-hrrr-pds/)
- [NOAA operational model access](https://nomads.ncep.noaa.gov/)
- [EIA-930 reporting schedule](https://www.eia.gov/Survey/)
- [EIA electricity API](https://www.eia.gov/opendata/index.php/browser/electricity/rto)
