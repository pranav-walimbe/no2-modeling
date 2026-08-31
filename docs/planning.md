# Research plan: near-real-time detection of anomalous power-plant activity with TEMPO

## Goal

Build and evaluate a system that uses TEMPO NO2 observations to detect unusual power-plant activity near the time it occurs. The first study will focus on plant starts and stops because they produce clear operating-state labels and have a direct connection to electricity-market conditions.

The project has two linked questions:

1. Can TEMPO distinguish plant starts and stops from normal operation under realistic cloud, wind, and source-separation conditions?
2. Do TEMPO-detectable events coincide with changes in wholesale power prices after accounting for information that market participants could observe without TEMPO?

The first question establishes atmospheric detection skill. The second tests whether the signal has value as a measure of unexpected thermal-plant availability. The initial analysis will estimate associations and predictive value. It will not interpret price changes as causal effects of one plant unless the research design supports that claim.

## System concept

For each plant and TEMPO scan, the system will:

1. Form an expectation for normal plant activity from plant characteristics, calendar variables, weather, past operating patterns, and regional grid conditions.
2. Extract the plant's NO2 field and quality information from TEMPO.
3. Estimate the probability of a start, stop, or continued operating state.
4. Attach the plant to a fixed ISO price location.
5. Compare the detected event with day-ahead and real-time power prices before and after the scan.

Historical replay will mimic a near-real-time system. Every feature must carry an availability timestamp, and each prediction may use only information available by the simulated decision time.

## Phase-one scope

### Plants and events

Use CAMPD hourly records from August 1, 2023 onward. Start with coal plants and other large NOx sources that have reliable gross-load and operating-time records. Expand to gas units after measuring detection limits for the strongest sources.

Define events at the unit level, then aggregate them to the facility level:

- **Start:** a transition from no operation to sustained positive operation or gross load.
- **Stop:** a transition from sustained positive operation to no operation.
- **Continuation:** stable on or off operation outside an event exclusion window.

Require the new state to persist for a minimum number of hours. Treat ramps, brief interruptions, missing records, and mixed unit states as separate cases rather than forcing them into start or stop labels. Set thresholds from engineering meaning and label audits, then freeze them before outcome analysis.

### Geography

Use CAISO, ERCOT, ISO-NE, MISO, NYISO, and SPP in phase one. Exclude PJM until a stable data-access path is available. Report how this exclusion changes plant, fuel, capacity, and event coverage.

Retain plants only when the analysis can assign a defensible price location. Prefer the plant's load zone or DLAP. Use a liquid hub when the ISO does not publish load-zone prices or when the plant-zone mapping cannot support historical consistency.

### Price outcomes

Collect hourly day-ahead and real-time prices at retained hubs and load zones. Preserve the energy, congestion, and loss components when the ISO publishes them.

Primary outcomes:

- Real-time locational marginal price.
- Day-ahead locational marginal price.
- Real-time minus day-ahead price spread.
- Same-market price change relative to pre-event hours.

The real-time minus day-ahead spread is the main measure of an operating surprise because day-ahead prices capture the market's prior schedule and real-time prices capture updated system conditions. Component analysis can show whether an association comes from system energy, transmission congestion, or marginal losses.

## Data organization and joins

### Plant-hour table

Store CAMPD hourly operation and emissions with facility and unit attributes in one Zstd-compressed Parquet file. Retain stable identifiers, coordinates, fuel, capacity, unit type, controls, operating time, gross load, and NOx mass.

### Power-price tables

Store one Zstd-compressed Parquet file per ISO and calendar month. Each file contains both markets and identifies the ISO, market, location, location type, UTC interval, LMP components, settlement status, native interval length, retrieval time, and source.

### Plant-to-price crosswalk

Create a versioned crosswalk with one row per plant and effective period. Include:

- CAMPD facility ID and coordinates.
- ISO and balancing authority.
- Price location ID, name, and type.
- Mapping method and source.
- Effective start and end dates.
- Distance or other match-quality diagnostics.
- Manual-review status.

Do not derive mappings during each analysis run. Build the crosswalk once, review ambiguous assignments, and use effective dates to handle boundary or market-definition changes.

### Time alignment

Use UTC as the storage and join key. Preserve source time zones and daylight-saving fields when needed for audits. Attach each plant-hour to the price interval with the same UTC start time.

TEMPO scans do not align with the top of each hour. Record scan start, midpoint, and end times. Link the scan to the plant operating hour that overlaps the exposure, then test adjacent hours to account for plume travel and reporting alignment.

## TEMPO detection study

### Observation construction

Extract a fixed plant-centered TEMPO Level-3 NO2 patch for each usable scan. Keep the native grid. Store retrieval quality, cloud information, uncertainty, viewing geometry, and the fraction of valid pixels.

Use wind fields to define upwind and downwind regions. Candidate summaries include background-adjusted column enhancement, downwind signal-to-noise ratio, plume alignment, and matched-filter response. A raster model may complement these summaries after the physical baseline is established.

### Prediction target

Estimate three probabilities for each usable plant-scan observation:

- Start.
- Stop.
- No state change.

Evaluate an operating-versus-off model as a supporting task. Keep continuous gross-load and NOx-change targets for sensitivity analysis, but center phase one on state transitions.

### Baselines

Compare the TEMPO system against these information sets:

1. Calendar and plant characteristics.
2. Calendar, plant characteristics, weather, and historical operating profiles.
3. The second baseline plus regional demand and public grid conditions available by the decision time.
4. An oracle baseline that adds recent CAMPD operation without treating it as a deployable input.
5. The strongest deployable baseline with TEMPO physical summaries.
6. The strongest deployable baseline with a wind-conditioned TEMPO patch encoder.

TEMPO adds value only if it improves held-out event detection over the strongest non-satellite baseline.

### Validation design

Use time-based splits and plant-held-out splits. A time split measures monitoring at known plants; a plant split measures transfer to facilities absent from training.

Report precision-recall area, event precision and recall, false alerts per plant-day, detection delay, and probability calibration. Break results out by fuel, event direction, NOx magnitude, cloud fraction, wind speed, source isolation, and distance to nearby emitters.

Measure coverage as well as accuracy. A system that performs well on a small clear-sky subset may still have limited monitoring value. Allow the model to abstain when cloud, retrieval quality, or source confusion prevents a defensible prediction.

## Power-price analysis

### Descriptive event study

For each confirmed CAMPD start or stop, construct an event window around the transition. Plot day-ahead prices, real-time prices, and their spread in event time. Normalize outcomes against the same location's hour-of-week baseline.

Estimate average event-time changes with plant, price-location, calendar, and hour controls. Cluster uncertainty by price location and date where the sample supports it. Report starts and stops separately.

### Matched comparisons

Match each event hour to non-event observations at the same plant or price location with similar season, hour, weather, regional demand, and prior price conditions. Exclude control hours near another large plant event in the same zone.

This comparison reduces dispatch and seasonality confounding. It does not remove all system-level causes that can affect plant operation and prices at the same time.

### Incremental prediction test

Create a walk-forward model for real-time price changes and real-time minus day-ahead spreads. Use only features available by each forecast timestamp.

Compare:

1. Price, calendar, weather, demand, and public grid-history features.
2. The same model plus CAMPD event labels, as an upper bound that may arrive too late for live use.
3. The first model plus TEMPO event probabilities and scene-quality measures.

The TEMPO signal has prospective value if the third model improves out-of-sample error or tail-event detection and the scan product arrives before the target decision cutoff. Report results with and without CAMPD labels to separate detection skill from operational latency.

### Heterogeneity

Test whether price associations differ by:

- Start versus stop.
- Plant capacity and fuel.
- Load zone versus hub mapping.
- System load and reserve conditions.
- Congested versus uncongested hours.
- Expected versus unexpected events based on day-ahead prices and public grid data.

Pre-specify the main comparisons. Treat the rest as exploratory and control false discoveries when testing many event windows or subgroups.

## Near-real-time constraints

The research system must track when each input becomes available, not only when the underlying event occurred.

For each source, record:

- Event or observation time.
- Source publication time.
- Retrieval time.
- Revision or settlement status.

Historical CAMPD records provide labels but may not support live detection. Final real-time prices support retrospective evaluation; preliminary prices may be required for a deployed monitor. TEMPO product latency must be measured from observed file arrival rather than assumed from nominal schedules.

Run the full evaluation as a historical replay. Delay each feature until its recorded availability time. Report detection accuracy and end-to-end latency as separate outcomes.

## Bias and failure modes

Address these risks before interpreting results:

- **Cloud selection:** observable events may differ by season, region, and weather.
- **Shared causes:** demand, outages, weather, and fuel constraints can move plant operation and prices together.
- **Dispatch endogeneity:** high prices can cause a plant start, while a plant outage can also raise prices.
- **Spatial mismatch:** a hub price can dilute a local plant effect.
- **Source confusion:** nearby emitters can create a TEMPO signal that the model assigns to the wrong plant.
- **Label error:** CAMPD timestamps, unit aggregation, and zero-operation rules can shift event labels.
- **Revisions:** final price and emissions records may contain information unavailable in real time.

Use negative-control hours, placebo event times, alternative event definitions, and mapping-quality filters. Present the price work as correlation and forecasting unless a later design establishes causal identification.

## Decision gates

Proceed in stages:

1. **Label audit:** CAMPD start and stop labels pass manual review across fuels and operating patterns.
2. **Coverage audit:** enough events overlap usable TEMPO scans in the six retained ISOs.
3. **Detection gate:** TEMPO improves event detection over non-satellite baselines on held-out times or plants.
4. **Mapping audit:** each retained plant has a reviewed price-location assignment with effective dates.
5. **Market gate:** TEMPO probabilities improve walk-forward price or spread forecasts after enforcing data-availability times.
6. **Deployment gate:** measured TEMPO latency and coverage support a useful alert window.

Failure at a gate still produces a useful result, such as a detection-limit estimate, a map of observable plants, or evidence that latency prevents market use.

## Initial deliverables

1. Audited facility-level start and stop table from August 1, 2023 onward.
2. Six-ISO hourly price archive with day-ahead, real-time, and price components.
3. Versioned plant-to-price-location crosswalk.
4. Event-count and TEMPO-overlap report by ISO, fuel, and plant.
5. Baseline CAMPD event study of prices without TEMPO.
6. TEMPO event detector with held-out accuracy, coverage, and latency results.
7. Walk-forward test of whether TEMPO event probabilities add price information.

## Primary paper framing

**Main question:** Under which atmospheric and source conditions can TEMPO detect power-plant starts and stops near the time they occur?

**Application question:** Do those detections provide timely information about thermal availability and regional wholesale power-price surprises?

The paper should lead with detection validity, coverage, and latency. Power prices provide a concrete system-level application and a test of incremental information. Economic claims require walk-forward gains after enforcing the publication time of every competing input.

## Literature and data references

- [Sun et al.: hourly TEMPO NOx emissions](https://doi.org/10.1029/2025JD044565)
- [Beirle et al.: point-source NOx flux divergence](https://doi.org/10.5194/essd-13-2995-2021)
- [Koene et al.: limits of divergence-based NOx estimates](https://doi.org/10.1029/2023JD039904)
- [TEMPO trace-gas and cloud product guide](https://asdc.larc.nasa.gov/documents/tempo/guide/TEMPO_Level-2-3_trace_gas_clouds_user_guide_V2.1.pdf)
- [EPA CAMPD](https://campd.epa.gov/)
- [NOAA HRRR archive](https://registry.opendata.aws/noaa-hrrr-pds/)
