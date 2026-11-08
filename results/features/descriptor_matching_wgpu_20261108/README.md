# GPU mutual descriptor matching benchmark (8 November 2026)

The common feature pipeline now uploads L2-normalized descriptor rows once per call, computes nearest cosine neighbours in both directions with a WGSL compute shader, and reads back only one `(index, cosine score)` pair per query. Reciprocal filtering and point-coordinate gathering run on the CPU over these small arrays. Local affine RANSAC remains the existing WGSL implementation. Detector inference and descriptor extraction are not moved by this change.

## Measurement

Measured on Vulkan Intel(R) Graphics (LNL), using deterministic random 64-component descriptors. Each size received one untimed warmup followed by five timed repetitions; the JSON file stores every sample.

| Points per image | WGSL matching median (s) | NumPy reference median (s) | Pairs | Same pairs |
|---:|---:|---:|---:|:---:|
| 512 | 0.0056 | 0.0027 | 512 | yes |
| 1,024 | 0.0104 | 0.0066 | 1,024 | yes |
| 2,048 | 0.0231 | 0.0225 | 2,048 | yes |
| 4,096 | 0.0800 | 0.1294 | 4,096 | yes |
| 8,192 | 0.3923 | 0.4342 | 8,192 | yes |

Reproduce with `python python/benchmarks/benchmark_feature_matching_wgpu.py --output results/features/descriptor_matching_wgpu_20261108/benchmark.json`. Timings include descriptor uploads, two dispatches and compact readback. The shader pipeline is created before warmup and reused. At 512 and 1,024 points setup dominates and NumPy is faster; 2,048 is near parity; at 4,096 and 8,192 the GPU path is faster in this run. This microbenchmark does not establish end-to-end detector/descriptor pipeline speed.

## Scope and limits

This is not a GPU-resident pipeline: descriptors cross from their producer to host NumPy and are uploaded to the WebGPU device. The new path eliminates the quadratic CPU similarity matrix and returns only linear-sized nearest-neighbour results. The DeDoDe-B descriptor extraction used by the current benchmark remains CPU-based and dominates end-to-end time. The matching shader assumes descriptors are finite and normalizes each row on the host before upload, matching the NumPy reference convention.

The deterministic parity test compares pair order, feature indices, and cosine scores against the existing CPU reference. Geometric verification remains independent and runs after matching.
