# Qualification matcher parity at 60% tensile deformation

This paired run used the same 4K image pair, the learned DIC detector, an 8,192-point budget, DeDoDe-B descriptors at 1,024-pixel maximum side, the same seeds, local geometric verification, and affine field projection. Only the reciprocal cosine matcher changed. Each run had one warmup and one timed trial; this is a deterministic parity check, not a timing distribution.

| Matcher | Direct mutual pairs | Pair digest | Correct projected centres <5 px | RMSE (px) | Complete affine trial (s) | Description (s) | Match stage (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| NumPy reference | 1,496 | `9a0e408265e266f9dfaa7a12d873370d132bbc8eca1b5259f9d542d1636852e0` | 16/16 | 0.224077 | 7.790 | 7.154 | 0.405 |
| WGSL WebGPU | 1,496 | `9a0e408265e266f9dfaa7a12d873370d132bbc8eca1b5259f9d542d1636852e0` | 16/16 | 0.224077 | 8.421 | 6.952 | 1.257 |

The observed direct correspondence arrays are byte-identical, and the projected field preserves all 16 correct centres. This establishes parity on this case. At 256 descriptor dimensions, the first WGSL matcher is slower than the NumPy/BLAS path; the 64-dimensional microbenchmark does not predict this workload. The CPU DeDoDe-B descriptor remains the dominant stage, and this matcher change alone does not improve full-pipeline latency.

Reproduction: run `python/benchmarks/qualify_all_initializers.py` once with `--feature-matcher cpu`, then with `--feature-matcher wgpu`, using `--methods dic_learned_b --cases tensile_60pct --repeats 1 --keypoints 8192`. The complete raw outputs and exact correspondences are stored in the sibling `feature_matching_cpu_60_20261108` and `feature_matching_wgpu_60_20261108` directories.
