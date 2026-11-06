# Sparse DeepMatching: seed-local root evaluation

## Question

The standard INRIA-style path computes every root in the image-wide hierarchy, even when callers pass only a few explicit seeds. This study measures a seed-local variant that evaluates only the root tiles containing those seeds. It is intended for a fallback workflow where another stage has already identified the POIs needing initialization.

## Implementation and scope

Use `DeepMatchingOptions(mode="inria_sparse", seed_points=...)`. The mode maps each seed to its atomic source leaf, selects the non-overlapping hierarchy root containing that leaf, and runs the existing hierarchy only for the unique selected roots. The default `mode="inria"` still evaluates the complete root grid.

Sparse matching restricts reciprocal candidate competition to the selected roots. This can affect `usable` decisions and displacements when an unselected root contains a competing correspondence. Therefore sparse mode is an explicitly local result; it is not guaranteed to reproduce the image-wide reciprocal field for arbitrary seeds and deformation. In the tested frames below, it does reproduce the full-grid results exactly.

## Measured results

Measurements used the 3840 × 2160 Live-DIC images, `wgpu-py`, Vulkan on Intel Graphics (LNL), and the same visible seeds for each full/sparse pair. Images and device were loaded before timing. Each mode was warmed up, then timed twice; reported time is the median. Timings include the complete matcher call, including per-call pipeline and resource setup. Quality uses analytic truth and accepted seeds only.

### Frame 040

| Requested seeds | Roots evaluated | Full-grid median | Sparse median | Speedup | Accepted | ≤2 px | Accepted RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 → 1 | 1.817 s | 0.127 s | 14.3× | 1/1 | 1/1 | 1.944 px |
| 4 | 40 → 4 | 1.801 s | 0.214 s | 8.4× | 4/4 | 4/4 | 1.364 px |
| 8 | 40 → 8 | 1.808 s | 0.351 s | 5.1× | 8/8 | 7/8 | 1.355 px |
| 16 | 40 → 16 | 1.834 s | 0.761 s | 2.4× | 16/16 | 15/16 | 1.325 px |
| 32 | 40 → 32 | 1.855 s | 1.582 s | 1.2× | 32/32 | 30/32 | 1.314 px |

### Frames 100 and 199

| Frame | Requested seeds | Roots evaluated | Full-grid median | Sparse median | Speedup | Accepted | ≤2 px | Accepted RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 40 → 1 | 2.394 s | 0.113 s | 21.2× | 1/1 | 1/1 | 1.285 px |
| 100 | 8 | 40 → 8 | 1.710 s | 0.350 s | 4.9× | 8/8 | 6/8 | 1.702 px |
| 100 | 24 | 40 → 24 | 1.654 s | 1.127 s | 1.5× | 24/24 | 20/24 | 1.556 px |
| 100 | 28 | 40 → 28 | 1.611 s | 1.303 s | 1.2× | 28/28 | 22/28 | 1.625 px |
| 199 | 1 | 40 → 1 | 1.586 s | 0.110 s | 14.4× | 1/1 | 1/1 | 1.128 px |
| 199 | 4 | 40 → 4 | 1.588 s | 0.191 s | 8.3× | 4/4 | 4/4 | 1.262 px |
| 199 | 8 | 40 → 8 | 1.593 s | 0.312 s | 5.1× | 8/8 | 8/8 | 1.238 px |
| 199 | 16 | 40 → 16 | 1.583 s | 0.705 s | 2.2× | 16/16 | 15/16 | 1.211 px |
| 199 | 24 | 40 → 24 | 1.590 s | 1.099 s | 1.4× | 24/24 | 23/24 | 1.148 px |
| 199 | 28 | 40 → 28 | 1.583 s | 1.302 s | 1.2× | 28/28 | 27/28 | 1.165 px |

For every tested subset, the sparse and full-grid calls returned the same accepted/rejected decisions and identical displacement vectors. This agreement is evidence for these seeds and frames, not a general equivalence guarantee. The single-seed frame-100 full-grid timing varied substantially between the two repetitions, so its speedup should be treated as approximate.

## Interpretation

The benefit follows the number of distinct seed roots. One to eight windows avoid 80–97.5% of root evaluations and ran roughly 5–21× faster in this experiment. With 24–32 windows, most of the 40-root grid is needed and the gain falls to about 1.2–1.5×. These runs use visible grid seeds as a controlled workload proxy; they do not yet model an online classifier that identifies difficult POIs.

This provides a practical sparse fallback for a small set of already-known POIs, while preserving the existing full-grid implementation for parity-sensitive runs. A future caller can pass only POIs rejected by HORG, local IC-GN, or another quality check. The caller must still validate sparse results independently.

## Reproduction

Run one frame at a time; default settings compare seed counts 1, 4, 8, 16, and 32 (counts above a frame's visible population should be omitted):

```console
python python/benchmarks/benchmark_sparse_deepmatching.py \
  --frame 040 --seed-counts 1,4,8,16,32 --repeats 2 \
  --output /tmp/sparse_deepmatching_040.csv
```

The combined per-mode timings, root counts, descriptor-comparison estimates, accuracy, and full/sparse parity checks are in [`sparse_deepmatching_study.csv`](sparse_deepmatching_study.csv). The benchmark script is [`benchmark_sparse_deepmatching.py`](../../python/benchmarks/benchmark_sparse_deepmatching.py).
