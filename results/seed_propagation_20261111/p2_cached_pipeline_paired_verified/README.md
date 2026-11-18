# Paired complete scripted pipeline with cached solver pixels

**Timing limitation:** another benchmark was observed running concurrently on
the same graphics processor during this campaign. The output parity checks
remain useful, but the recorded timings must not be treated as isolated hardware
measurements. A separate isolated repeat is used for the performance report.

One synthetic 3840 × 2160 pair, one warm-up and alternating-order measured repetitions. Each arm includes all image generation, matching, process startup, diagnostics and result writes. The cached arm also includes encoding, writing, hashing and reading both images. Times are not integrated production or GPU-resident latency.

Median regenerated / cached full script: 16.715 / 12.043 s. Exact state, status and provenance parity: True.

| Run | Warm-up | Regenerated (s) | Cached (s) | Exact parity |
|---:|---|---:|---:|---|
| 0 | True | 23.140 | 17.027 | True |
| 1 | False | 22.827 | 18.476 | True |
| 2 | False | 16.715 | 11.396 | True |
| 3 | False | 15.803 | 12.043 | True |

The saved JavaScript Object Notation (JSON) data contain stage times, quality and per-window outputs. One pair does not establish performance or accuracy in general.
