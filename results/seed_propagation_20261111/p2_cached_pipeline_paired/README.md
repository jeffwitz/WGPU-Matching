# Paired complete scripted pipeline with cached solver pixels

One synthetic 3840 × 2160 pair, one warm-up and alternating-order measured repetitions. Each arm includes all image generation, matching, process startup, diagnostics and result writes. The cached arm also includes encoding, writing, hashing and reading both images. Times are not integrated production or GPU-resident latency.

Median regenerated / cached full script: 14.437 / 10.540 s. Exact state, status and provenance parity: True.

| Run | Warm-up | Regenerated (s) | Cached (s) | Exact parity |
|---:|---|---:|---:|---|
| 0 | True | 14.337 | 10.665 | True |
| 1 | False | 14.982 | 10.540 | True |
| 2 | False | 14.327 | 10.532 | True |
| 3 | False | 14.437 | 10.623 | True |

The saved JavaScript Object Notation (JSON) data contain stage times, quality and per-window outputs. One pair does not establish performance or accuracy in general.
