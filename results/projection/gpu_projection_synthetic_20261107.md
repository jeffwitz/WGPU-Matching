# WGPU affine and Gaussian-RBF initialization projection — 2026-11-07

## Protocol

This is a reproducible analytic projection test, not an image-matcher benchmark.
It uses 432 direct point pairs on a 512 × 420 pixel domain, an affine displacement
plus a localized Gaussian displacement, 0.08 px Gaussian observation noise, and
14 deliberately corrupted target points. Four 128 × 96 px DIC patches are
projected. The benchmark warms each method once and reports the median and P95 of
five further runs. Image decoding and feature extraction are excluded.

Hardware: Intel(R) Graphics (LNL) integrated GPU, Vulkan, Mesa 25.2.8.
Software: Python 3.12.3, NumPy 2.5.4, wgpu-py 0.32.0.

Raw measurements and all candidate scores are in
[`gpu_projection_synthetic_20261107.json`](gpu_projection_synthetic_20261107.json).
Run `PYTHONPATH=python/src python3 python/benchmarks/benchmark_projection.py`
to reproduce them in the project environment.

## Results

| Projection | Median / P95 total | GPU final fit | GPU patch projection | CPU selection and pilot | Field error median / RMSE | Points <2 px / <5 px | Patch-center max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Affine | 14.41 / 20.21 ms | 11.26 ms | 0.99 ms | 1.21 ms | 2.60 / 3.99 px | 159/432 / 353/432 | 10.41 px |
| Auto Gaussian-RBF | 384.58 / 400.46 ms | 41.33 ms | 1.47 ms | 340.68 ms | 0.28 / 0.48 px | 431/432 / 432/432 | 1.54 px |

The affine model represents its own projected patches almost exactly (support
RMS below 1.4e-6 px), but cannot represent the fixture's localized displacement.
The selected 16-center Gaussian-RBF reduces the median analytic point error by
about 2.31 px and the RMSE by about 3.51 px. Its four patch support RMS values
are 0.67, 2.12, 0.61, and 1.30 px. These values quantify the local-affine
approximation error over each DIC window; they do not predict IC-GN convergence.

## Execution and limits

The final affine/RBF IRLS solve and patch projection run in WGSL on WebGPU. Input
pairs are uploaded from host arrays, and final residuals, robust weights, model
coefficients, and requested patch parameters are read back. This is not a
zero-copy ONNX-to-wgpu pipeline.

Automatic RBF spatial cross-validation and the robust-scale pilot still run in
NumPy float64 on the CPU. They dominate the RBF runtime (about 341 ms of 385 ms),
so this result does **not** establish an end-to-end GPU speedup for automatic RBF
selection. They are retained for model-selection behavior and reference
stability; moving that stage to GPU is still open work.

The best CPU cross-validation candidate had 64 centers, but its FP32 GPU normal
system failed Cholesky; the 32-center candidate also failed. The recorded fallback
selected 16 centers with ridge 1e-6. The GPU IRLS stopped at the configured
20-iteration limit without satisfying the 1e-4 robust-weight delta tolerance
(final maximum delta 0.383). Its low analytic error is useful evidence for this
fixture, but the fit is explicitly marked non-converged and must not be presented
as a generally reliable RBF result. Higher-ridge candidates and more stable
factorizations need qualification before changing defaults.

This test has no experimental images, image-derived correspondences, cracks,
spatial discontinuities, or downstream IC-GN solve. It validates native-pixel
mapping, GPU fitting/evaluation, patch coefficient order, and a smooth synthetic
field only. The result is not a comparison against DeepMatching, HORG, CPM, or an
IC-GN convergence benchmark.
