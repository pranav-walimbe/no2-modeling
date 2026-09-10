# Mask-aware EMA benchmark

The dataset generator uses stacked NumPy arrays for per-pixel support counts,
weighted sums, and weight renormalization. It never loops over the 2,304 output
cells.

On September 9, 2026, `scripts/benchmark_masked_ema.py --records 250` compared
the removed nearest-fill implementation with the mask-aware implementation on
250 synthetic records. Each record used fourteen compressed 48 by 48 scan
archives with 10% independently missing cells.

| Implementation | Records/s | Cache-read time | EMA compute time | Peak RSS |
|---|---:|---:|---:|---:|
| Nearest-valid fill | 119.9 | 1.353 s | 0.732 s | 212.6 MiB |
| Mask-aware vectorized EMA | 172.4 | 1.345 s | 0.105 s | 212.8 MiB |

The mask-aware path processed 43.8% more records per second and cut EMA compute
time by 85.7%. Peak memory moved by 0.2 MiB, which sits inside run-to-run noise.
Repeated runs held both throughput figures within 3%.

Archive reads now dominate the record cost, so batching and the existing bounded
worker queue still govern production throughput. This microbenchmark isolates
raster I/O and EMA construction. Complete-record throughput on the real
candidate population comes from the cache-only retention audit instead.
