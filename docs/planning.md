# Research plan: detecting power-plant operating and emissions anomalies with TEMPO

## Modified research question

Can hourly geostationary observations of NO2 and SO2 detect departures from expected power-plant operation and emissions, and how do weather, source density, and sensor resolution limit attribution?

The project will estimate continuous deviations rather than raw emissions alone. A fleet model will define expected plant or source-group behavior from engineering characteristics, calendar variables, and weather. A satellite model will test whether TEMPO adds information about departures from that expectation.

The project supports two applications:

- Atmospheric science: detect unusual emissions intensity, changes in control performance, and shifts in fossil generation.
- Commodity research: detect unexpected changes in thermal generation, outages, ramping, or fuel use before standard public reports capture them.

The two applications will share data preparation and model features. They will use different targets and evaluation protocols.

## Estimands

### Operating anomaly

The operating target measures whether a plant or source group produces more or less power than a fleet model expects.

The baseline will estimate gross load or capacity factor from:

- Nameplate capacity and unit type.
- Fuel, location, hour, season, and weather.
- Regional demand only when the application permits that input.

The normalized operating residual is

\[
z_{\text{load},t} =
\frac{y_{\text{load},t}-\hat\mu_{\text{load},t}}
{\hat\sigma_{\text{load},t}}.
\]

This target supports outage, startup, shutdown, and dispatch-change detection. It also provides the main commodity-research target.

### Emissions-performance anomaly

The emissions target measures whether NOx or SO2 differs from the level expected for the plant configuration and operating regime.

\[
z_{g,t} =
\frac{\log(1+y_{g,t})-\hat\mu_{g,t}}
{\hat\sigma_{g,t}},
\qquad g \in \{\mathrm{NOx},\mathrm{SO2}\}.
\]

The baseline may condition on gross load when the research question concerns emissions intensity or control performance. A second version will omit observed load and estimate total emission departures from plant characteristics, calendar variables, and weather. The market analysis must exclude gross load at inference if public load data would arrive after the satellite observation.

The model will predict the continuous standardized residual and its uncertainty. The evaluation will derive anomaly probabilities and event classes from that output.

### Target hierarchy

The project will use the following hierarchy:

1. Use the continuous NOx emissions residual as the primary atmospheric target.
2. Use the NOx or SO2 intensity residual conditional on load to study combustion and control performance.
3. Use the continuous gross-load residual as the primary commodity target and an auxiliary atmospheric target.
4. Derive high, normal, and low event classes from the continuous residuals.
5. Add startup, shutdown, and ramp labels when CAMPD supports confident identification.

A multi-task model can predict load, NOx, and SO2 residuals together. The comparison among these tasks will show whether TEMPO responds to plant operation, emissions performance, or both.

The training objective will retain residual sign and magnitude. Classification metrics will use thresholds set from the training residual distribution. Candidate definitions include fixed standardized thresholds, tail quantiles, and thresholds paired with a minimum physical change. Event definitions may require persistence across adjacent observations.

## Definition of expectation

### Fleet expectation

The primary expectation model will estimate the behavior of a plant from characteristics available across the fleet. It will support transfer to facilities absent from training and avoid facility emissions priors.

For gross load, the model will estimate

\[
p(y^{\mathrm{load}}_{i,t}\mid
\mathrm{capacity}_i,\mathrm{fuel}_i,\mathrm{unit\ type}_i,
\mathrm{calendar}_t,\mathrm{weather}_{i,t},\mathrm{region}_i).
\]

For emissions, the project will compare two expectations:

\[
p(y^g_{i,t}\mid
\mathrm{plant\ characteristics}_i,\mathrm{calendar}_t,
\mathrm{weather}_{i,t})
\]

and

\[
p(y^g_{i,t}\mid
y^{\mathrm{load}}_{i,t},\mathrm{controls}_i,
\mathrm{unit\ type}_i,\mathrm{weather}_{i,t}),
\qquad g \in \{\mathrm{NOx},\mathrm{SO2}\}.
\]

The first expectation measures total emissions surprise. The second measures emissions-performance surprise conditional on plant operation.

### Plant-calibrated expectation

A second model will add a past-only plant correction to the fleet estimate:

\[
\mu_{i,t}=f_\theta(x_{i,t})+b_{i,t},
\]

\[
b_{i,t}=(1-\alpha)b_{i,t-1}
+\alpha\,\mathrm{clip}\left[
\log(1+y_{i,t-1})-f_\theta(x_{i,t-1})
\right].
\]

The exponential moving average will correct persistent plant bias rather than define the full expectation. This model will support monitoring at known facilities. The study will report results for both cold-start fleet expectations and plant-calibrated expectations.

A seasonal naive benchmark will use the past plant median by hour, month, and weekday or weekend. A raw plant-hour exponential moving average will provide another benchmark.

### Conditional distribution and anomaly score

The expectation model will estimate a conditional distribution because operating and emissions variance changes across plant sizes and regimes. Initial models will estimate the 5th, 50th, and 95th percentiles with quantile gradient boosting. A hierarchical or state-space model can add partial pooling by fuel and unit type.

The estimated cumulative distribution will define a signed anomaly score:

\[
z_{i,t}=\Phi^{-1}\left[\widehat F(y_{i,t}\mid x_{i,t})\right].
\]

This score expresses each observation on a common probability scale. It avoids a global variance assumption and supports high, normal, and low event classes.

### Operating-state mixture

A hurdle model will separate operating state from output magnitude:

1. Estimate the probability that a facility operates.
2. Estimate load or emissions conditional on operation.

For multi-unit facilities, the model may estimate active-capacity fraction instead of a single on or off state.

### Dynamic-baseline safeguards

A plant correction can absorb a sustained anomaly. The update rule will clip large residuals, pause updates for flagged events, or maintain short- and long-memory states. The analysis will distinguish transient deviations from structural changes such as a retrofit, fuel conversion, or retirement.

All expectation models will generate rolling-origin or out-of-fold predictions. The pipeline will compute each expectation from information available before the target observation.

### Information boundary

The expectation model will describe plant behavior with engineering characteristics, calendar variables, demand inputs, and weather that can affect dispatch. The TEMPO observation model will use wind, plume height, cloud fraction, air mass factors, and retrieval uncertainty. This separation prevents atmospheric observation conditions from defining expected plant behavior.

## Unit of analysis

The unit of analysis will depend on source identifiability:

1. Model isolated sources as individual facilities.
2. Model unresolved groups with similar fuels and operating roles as aggregate sources.
3. Model sources with separable wind responses as individual facilities, even when they occupy the same area.
4. Report only group totals for mixed, unresolved sources.

A fixed distance threshold cannot determine source identity. For each observation, the pipeline will construct a plume response for every source from its location, wind, stack or plume height, and an assumed dispersion kernel. The pipeline will aggregate sources when their response vectors become too collinear at TEMPO resolution.

This rule treats source clustering and coarse imagery as one inverse problem. The model will not claim plant-level attribution when the observation supports only a group total.

### Fuel scope

The study will center its evaluation on coal without restricting the full sample to coal plants. Coal units offer the strongest initial test because their NOx and SO2 enhancements tend to exceed those of lower-emitting generators. A coal-only sample would reduce coverage, favor large and detectable facilities, and leave uncertain whether the method transfers to gas-heavy regions.

The model will remain fuel-aware. It will divide sources into three strata:

1. Isolated plants, with plant-level emissions and operating targets.
2. Homogeneous clusters, with group-level emissions and generation targets.
3. Heterogeneous clusters, with aggregate NOx as the primary target and fuel-specific generation as an experimental target.

The pipeline will predict aggregate emissions for unresolved mixed-fuel groups because TEMPO observes their combined plume. It will predict aggregate generation only after conditioning on fuel mix, control technology, and capacity shares. A common NO2 enhancement can correspond to different amounts of generation across coal and gas units.

Each source group will include coal and gas capacity shares, expected emissions-intensity shares, unit counts, control technologies, source spacing, and a fuel-concentration index. Evaluation will report results by source stratum and fuel composition. The model will abstain from plant-level or fuel-specific attribution when the observation cannot separate group members.

## Data

### TEMPO

Use V04 Level-2 data for the main experiments. Preserve native pixel geometry and exposure time where the workflow permits it. Level-3 data can support comparison runs and faster prototypes.

Candidate NO2 variables include:

- Tropospheric vertical column.
- Fitted slant column and tropospheric air mass factor.
- Column uncertainty and quality flags.
- Effective cloud fraction and cloud pressure.
- Surface pressure, terrain height, albedo, and planetary boundary-layer height.
- Solar and viewing geometry.

TEMPO supports SO2 retrieval from Level-1B spectra. The published retrieval remains a research product rather than a standard archive product. The project will test SO2 on large CAMPD sources before making it a required input. The test will measure coverage, signal-to-noise ratio, relation to CAMPD SO2, and added value after NO2 and weather features.

### Meteorology

Replace the single 10 m wind vector with a vertical wind representation:

- Wind components near 100, 200, 300, and 500 m.
- Interpolated wind at stack height and candidate plume-rise heights.
- Boundary-layer height, temperature, pressure, and stability measures.
- Wind shear and disagreement across levels.

Validation will determine the transport height. Prior TEMPO work found strong performance near 300 m, but the project will test that result across this sample.

### Plant and source-group characteristics

Use static or engineering features in place of historical plant emissions priors:

- Nameplate capacity and unit count.
- Fuel and unit type.
- Stack height and available plume-rise inputs.
- NOx and SO2 control technology.
- Commissioning age and heat-rate information when available.
- Source positions within each crop.

For source groups, aggregate capacity and labels while retaining the member-level positions and characteristics.

### Targets

Build labels from CAMPD records for:

- NOx mass and rate.
- SO2 mass and rate.
- Gross load and operating time.
- Startup, shutdown, full-operation, and off states.

The existing collection contains unit-level hourly `grossLoad` and `opTime`. EPA reports gross electrical generation or load for operating hours. A revised facility target must sum compatible electrical-unit loads and separate steam-load records. The current facility aggregation sums NOx but retains the representative unit's gross load, so it cannot support a facility power target without revision.

The current scraper requests operating hours and does not retain CAMPD SO2, CO2, heat input, load units, or startup and shutdown flags. The revised collection should add those fields and preserve the method-of-determination fields when available. It must retain zero and partial operating hours and distinguish a reported zero from a missing report.

## Preprocessing and physical representation

### Time alignment

- Convert CAMPD hours with an explicit facility time zone.
- Match each facility to the Level-2 exposure time at its longitude.
- Account for the east-to-west TEMPO scan.
- Align emission labels with plume travel time and test a set of lag windows.

### Background and plume coordinates

Transform each scene into downwind and crosswind coordinates. Derive:

- Upwind background and downwind enhancement.
- Near-source and plume-sector column totals.
- Crosswind width and downwind decay.
- Directional derivative and flux-divergence features.
- Column tendency from adjacent scans.
- Matched-filter responses for each candidate source.
- Signal-to-uncertainty ratios.

These features will complement the raster inputs and provide an interpretable physics baseline.

### Temporal context

Create single-scan, multi-scan, daily, and rolling multi-day samples. Changing wind directions can separate sources that one scan cannot resolve. Longer windows can improve signal-to-noise ratio at the cost of event timing.

### Quality and missingness

Apply matching cloud and quality tests to all scans in a temporal sequence. Retain uncertainty, cloud, and missingness indicators as model inputs. Permit the model to abstain when retrieval quality or source identifiability falls below a set threshold.

## Model families

### Expectation baseline

Compare four expectation baselines:

1. Seasonal naive plant median by hour, month, and day type.
2. Raw plant-hour exponential moving average.
3. Fleet quantile model with plant characteristics, calendar variables, and weather.
4. Fleet quantile model with a past-only exponential moving average of plant residuals.

The fleet model must generalize to facilities absent from training. The hybrid model will test the value of plant calibration at known sites. Historical facility emissions will remain outside the primary cold-start baseline.

### Physics-feature model

Fit a small regression or classification model on plume-coordinate summaries, tendency, directional derivatives, meteorology, and plant characteristics. This model will establish whether physical summaries capture the satellite signal without a large image encoder.

### Raster or pixel-set model

Compare two representations:

- A wind-conditioned image model with source-location channels.
- A set model over native Level-2 pixels and source descriptors.

The image encoder must receive wind before spatial pooling. The set model can use native footprints, observation times, uncertainty, and positions without resampling to a square grid.

### Temporal model

Use a small temporal convolution or recurrent model over adjacent scans. The model will predict load, NOx, and SO2 residuals as separate tasks with a shared atmospheric representation.

### Multi-pollutant model

Treat SO2 as an optional channel and auxiliary task. The model will learn from NO2-only samples and use SO2 when the retrieval passes its quality threshold. An ablation will test whether SO2 reduces false alarms or improves event detection for high-emitting coal sources.

### Deferred super-resolution work

The main study will not train a free-standing super-resolution model. Native Level-2 processing, plume coordinates, source response kernels, and temporal diversity address the resolution problem without inventing sub-pixel structure.

A later resolution experiment must conserve column mass, reproduce the original observation after downsampling, and improve held-out emission or event estimates. Visual sharpness will not count as evidence.

## Experimental comparisons

Each comparison will isolate one source of information:

1. Seasonal naive and raw exponential-moving-average expectations.
2. Cold-start fleet expectation.
3. Plant-calibrated fleet expectation.
4. Satellite physics features without plant characteristics.
5. Plant characteristics and weather without satellite inputs.
6. Combined physics features, plant characteristics, and weather.
7. Combined model with raster or native-pixel inputs.
8. Combined model with SO2.
9. Combined temporal model.

Core ablations will remove:

- NO2 imagery.
- SO2.
- Tendency features.
- Vertical wind information.
- Retrieval uncertainty.
- Plant characteristics.
- Temporal context.

The main result will measure the information that TEMPO adds beyond engineering characteristics and weather.

## Event feasibility in the existing data

An exploratory aggregation of the existing coal records found:

- 5.18 million coal unit-hour records.
- 2.93 million facility-hours across 178 facilities.
- 2.93 million consecutive active-hour pairs.
- 24,825 facility-hours with partial unit operation, including 11,004 during approximate daylight hours.

For a sample-size check, an abrupt event was defined as an absolute hourly change above 25% of the facility's 95th-percentile level. This rule identified:

| Exploratory event | All hours | Approximate daylight hours |
| --- | ---: | ---: |
| Gross-load change | 33,123 | 15,972 |
| NOx change | 74,597 | 37,197 |

Gross-load changes represented 1.13% of consecutive active-hour pairs. NOx changes represented 2.55%. Daylight load events occurred at 175 of 178 facilities, and daylight NOx events occurred at all 178.

These counts establish label-stage feasibility. TEMPO coverage, cloud screening, sequence requirements, and source-identifiability tests will reduce the usable event count. The final study will recalculate support after rebuilding the matched dataset.

The exploratory threshold does not define the final anomaly label. The expectation model will generate out-of-sample standardized residuals, and the training residual distribution will set event thresholds. Partial operation at one unit does not prove a facility startup or shutdown, so event labels must combine unit operating time, facility load, adjacent hours, and explicit status fields.

## Evaluation

### Atmospheric-science evaluation

Use two split designs:

- Group-held-out evaluation for transfer to unseen plants or source groups.
- Time-held-out evaluation for monitoring known plants.

Use rolling-origin predictions for time-held-out expectations. Use out-of-fold expectation predictions when training the satellite residual model. These procedures will prevent the expectation model from understating residual variance through in-sample fitting.

Report results by source type, fuel, emission magnitude, wind speed, cloud fraction, source spacing, and identifiability score.

Regression metrics will include MAE, normalized MAE, R2, rank correlation, calibration, and plant-macro averages. Event metrics will include precision-recall area, event F1, false alarms per observation, and detection delay.

The evaluation will score the continuous residual before applying event thresholds. It will then report three-class high, normal, and low results. A binary anomaly score can support comparisons that require one positive class.

The analysis will estimate detection limits for NOx, SO2, operating changes, and source separation.

### Commodity-research evaluation

Use a walk-forward backtest with strict data-availability timestamps. The backtest must:

- Exclude future plant records and revised data.
- Reproduce cloud gaps and product latency.
- Compare the satellite signal with the release time of ISO, EIA, weather, and emissions data.
- Evaluate plant signals and regional aggregates.
- Separate statistical detection from economic value.

Targets will include gross-load deviation, outage or startup events, and regional coal-generation deviations. Regional aggregation can reduce individual-source ambiguity and may fit commodity decisions better than plant-level attribution.

## Decision rules

### SO2

Keep SO2 in the main model if it provides usable coverage across the target plants and improves held-out anomaly detection after uncertainty controls. Retain it as a case-study signal if it works only for large sources.

### Source aggregation

Predict individual facilities only when source response kernels pass an identifiability threshold. Aggregate unresolved sources and report the member composition.

### Fuel scope

Use coal and coal-dominant sources as the primary detectability stratum. Retain other fuels for coverage and transfer tests. For unresolved heterogeneous groups, make aggregate NOx the main estimand and treat fuel-specific generation attribution as experimental.

### Temporal resolution

Choose the shortest cadence that produces stable held-out skill. Hourly estimates may suit strong events; daily or multi-day estimates may suit weaker sources.

### Satellite contribution

Claim satellite value only when the combined model beats the plant-characteristics and weather baseline across held-out groups or times. An image encoder that matches a tabular baseline does not establish satellite information.

### Target form

Train the main models on continuous standardized residuals. Derive event classes after fitting the model. Use a direct classifier only if it improves held-out event calibration or handles a status label that has no meaningful continuous form.

### Expectation model

Use the fleet quantile model as the primary expectation for unseen plants. Use the plant-calibrated model for known-site monitoring. Keep the seasonal naive and raw exponential moving average as benchmarks. Select thresholds from rolling-origin training residuals and verify their event frequency on held-out data.

### Market relevance

Claim an alternative-data signal only when the walk-forward test shows information before or independent of standard public data. A retrospective match to CAMPD alone does not meet that standard.

## Paper framing

### Atmospheric-science framing

**Working title:** Detectability and attribution of anomalous power-sector emissions with hourly geostationary multi-pollutant observations

The paper can contribute:

- A fleet-expectation anomaly target that avoids facility emissions priors.
- Continuous NOx residual prediction with derived high and low event detection.
- An uncertainty-aware source-identifiability and aggregation method.
- A comparison of isolated, homogeneous-cluster, and mixed-cluster cases.
- Fuel-aware detection limits, with coal as the high-signal stratum and mixed-fuel groups as the attribution test.
- Detection limits for NO2, experimental SO2, and operating changes.
- Evidence on the value of native pixels, vertical winds, and temporal context.

The climate link should focus on fossil-generation changes, co-emitted pollutants, and energy-transition monitoring. TEMPO NO2 and SO2 do not measure CO2 emissions on their own.

### Commodity-intelligence framing

**Working title:** Satellite detection of unexpected thermal-generation changes from geostationary trace-gas observations

The application can produce:

- Plant and source-group operating anomaly scores.
- Regional coal-generation and outage indicators.
- Confidence and data-quality measures.
- A record of signal timing relative to public releases.

The market study should remain a separate application unless the evidence supports both the atmospheric and economic claims.

## Literature anchors

- Sun et al. estimate hourly NOx emissions from TEMPO with a directional derivative and tendency term: <https://doi.org/10.1029/2025JD044565>
- Li et al. present the first TEMPO SO2 retrievals and discuss large-source monitoring: <https://doi.org/10.1029/2025GL115788>
- Beirle et al. derive point-source NOx emissions from NO2 flux divergence: <https://doi.org/10.5194/essd-13-2995-2021>
- Koene et al. analyze the theory and limits of the divergence method: <https://doi.org/10.1029/2023JD039904>
- Kuhlmann et al. evaluate temporal variability and source aggregation for power-plant NOx: <https://doi.org/10.5194/acp-26-4405-2026>
- The TEMPO trace-gas guide documents V04 products, variables, quality controls, and known issues: <https://asdc.larc.nasa.gov/documents/tempo/guide/TEMPO_Level-2-3_trace_gas_clouds_user_guide_V2.1.pdf>
