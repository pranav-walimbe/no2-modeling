# Dataset design

The masked-pretraining dataset contains single TEMPO scenes with aligned HRRR
weather. Each sample hides a synthetic subset of observed NO2 pixels and retains
the original raster as the reconstruction target. CAMPD emissions values do not
serve as labels or model inputs. Facility coordinates define the candidate AOIs.

## Record contract

Each record represents one unique AOI and TEMPO scan. The published CSV carries
the candidate index, AOI identifier, scan date and number, observation time,
cache key, and a dataset-root-relative raster path.

The raster bundle contains 24 by 24 arrays:

| Array | Type | Meaning |
|---|---|---|
| `no2` | `float32` | Original TEMPO NO2 reconstruction target |
| `no2_mask` | `uint8` | Original TEMPO validity mask |
| `temperature_2m_k` | numeric | Aligned HRRR 2 m temperature |
| `wind_u_80m_mps` | numeric | Aligned geographic eastward wind |
| `wind_v_80m_mps` | numeric | Aligned geographic northward wind |
| `masked_no2` | `float32` | NO2 with synthetic gaps filled by zero |
| `artificial_mask` | `uint8` | One for visible pixels and zero for synthetic gaps |

The model loader reconstructs its normalized masked input from `no2` and
`artificial_mask`. It keeps `no2_mask` separate so synthetic missingness cannot
replace the source-validity record.

## Eligibility

Candidate construction intersects power-plant AOIs with the TEMPO AOI mapping
and deduplicates records by AOI, scan date, and scan number. A scene qualifies
only when all 576 NO2 cells are finite. The generator then aligns HRRR weather
and requires each NO2 and weather value to remain finite with the expected
24 by 24 shape.

The complete-scene rule gives each hidden pixel an observed target. It also
selects a cleaner cohort than the downstream delta dataset. Cohort audits should
compare the selected and downstream populations by geography, time, weather,
and NO2 distribution.

## Geographic splits

The generator groups overlapping 72 km AOIs before assigning train, validation,
and test splits. Each overlap group belongs to one split, which prevents nearby
plants and shared raster footprints from crossing split boundaries.

The configured targets are:

| Split | Records |
|---|---:|
| Train | 500,000 |
| Validation | 50,000 |
| Test | 50,000 |

The target proportions guide a deterministic largest-group-first assignment.
Within each split, seeded hashes order candidates and partition discovery work.
The selector deduplicates valid records by the content-derived cache key before
taking the requested count.

## Synthetic masks

Materialization samples a masking fraction from 1% through 10% for each record.
The sampler gives outer pixels more weight and adds a bounded local weight near
previous selections. This produces exact-size masks with a mixture of isolated
and small clustered gaps.

The split seed, shard task, and record offset determine each record's random
seed. Repeating materialization with the same shard parameters reproduces the
mask. Each run rebuilds masked bundles, so a changed shard size changes the seed
assignment.

The raster bundle stores the resulting mask. The current manifest and bundle do
not store the sampled fraction or seed as separate metadata fields.

## Cache and publication lifecycle

The persistent validity cache uses one content-derived key per AOI-scene pair:

- A positive NPZ stores complete NO2, its validity mask, and aligned weather.
- A negative JSON records a scene that fails complete NO2 coverage.
- Source-read and weather failures remain retryable and do not create negative
  entries.

The pipeline reuses the delta-model TEMPO and weather caches before reading raw
sources. Discovery groups AOIs from the same scan so workers can reuse granule
reads.

Each launch clears candidate work, masked shards, and published dataframes. The
validity cache persists unless the caller passes `--clear-cache` or its
`--refresh-cache` alias. Workers publish cache results, shard manifests, raster
bundles, and final split dataframes with atomic replacements.

The finalizer verifies that each shard stays within its assigned size, contains
unique in-shard paths, references existing bundles, and matches the selected
cache records in order. It then writes `train_df.csv`, `val_df.csv`,
`test_df.csv`, and `generation_summary.json`.

## Leakage controls

Geographic groups cross no pretraining split boundary. Model training fits
normalization on the pretraining train split and reuses those statistics for
validation and test. The generator creates its split assignments independently
from the downstream split files. An inductive transfer experiment must verify
that downstream validation and test AOIs are absent from the pretraining train
split.
