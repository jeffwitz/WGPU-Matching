# High-budget DaD and DeDoDe v2 ONNX/WebGPU validation

## Export parity at K=8192

Both official detector models were exported as keypoint-only ONNX graphs with
dynamic spatial dimensions and a static output capacity of K=8192. The export
check used 640 × 360 inputs on Intel Graphics LNL with ONNX Runtime 1.31.1 and
`onnxruntime-ep-webgpu` 0.5.0.

| Detector | Median ONNX `session.run` | Output bytes | PyTorch set points within 1 px | Score max absolute error |
|---|---:|---:|---:|---:|
| DaD | 163.5 ms | 98,304 B | 8130/8192 | 9.73e-7 |
| DeDoDe v2-L | 412.4 ms | 98,304 B | 8192/8192 | 6.55e-11 |

The custom top-K selection itself matches the official PyTorch selection for
all 8192 points within 0.1 px in both exports. ONNX-vs-PyTorch DaD has a few
near-tied points that change order; 8187/8192 points are within 5 px under the
one-way nearest-set metric. The set metric is not a one-to-one assignment.
Raw export profiles and model hashes are in `detector_onnx_webgpu_20261107/`.

## 60% synthetic affine stress test

The source pair is 3840 × 2160 with analytic 60% axial strain. The models see
640 × 360; coordinates are converted back to source pixels. DeDoDe-B
description and mutual cosine matching run on CPU, followed by the same WGSL
local-affine RANSAC. Timing results below are one measured repeat after one
warm-up; quality counts are for the shared analytic seeds.

At budget 4096, candidate pools contain 8192 points:

| Detector | Path | Accepted / correct <5 px | RMSE / P95 accepted error (px) | Detector pair (s) | Full pair (s) |
|---|---|---:|---:|---:|---:|
| DaD | PyTorch CPU | 16/16 / 15 | 4.19 / 5.94 | 1.222 | 4.385 |
| DaD | ONNX/WebGPU | 16/16 / 15 | 2.16 / 3.77 | 0.392 | 3.134 |
| DeDoDe v2-L | PyTorch CPU | 14/16 / 14 | 1.79 / 3.08 | 3.790 | 6.232 |
| DeDoDe v2-L | ONNX/WebGPU | 14/16 / 14 | 1.79 / 3.08 | 1.261 | 3.735 |

At budget 8192, each exported detector supplies exactly 8192 candidates (1×
candidate pool):

| Detector | Path | Accepted / correct <5 px | RMSE / P95 accepted error (px) | Detector pair (s) | Full pair (s) |
|---|---|---:|---:|---:|---:|
| DaD | PyTorch CPU | 16/16 / 16 | 1.21 / 2.11 | 1.070 | 3.353 |
| DaD | ONNX/WebGPU | 16/16 / 16 | 1.11 / 1.80 | 0.375 | 2.824 |
| DeDoDe v2-L | PyTorch CPU | 16/16 / 16 | 1.27 / 1.97 | 2.835 | 5.018 |
| DeDoDe v2-L | ONNX/WebGPU | 16/16 / 16 | 1.17 / 1.85 | 0.977 | 3.601 |

At the maximum budget, both detector/descriptor combinations initialize every
visible analytic seed within 5 px through both execution paths. The matching
outcome is preserved, and the ONNX/WebGPU detector cuts detector-pair time by
about 2.8–2.9×. Full-pair speedup is smaller because DeDoDe-B and CPU mutual
matching remain costly: about 1.19× for DaD and 1.39× for DeDoDe v2 in this
single-repeat run.

Increasing the point budget matters more to coverage than changing the
inference backend: at budget 512, DaD is at 6/16 correct seeds and DeDoDe v2
at 0/16; at budget 8192 both reach 16/16. This is a single globally affine
synthetic case and does not establish performance on heterogeneous or
experimental deformation fields, nor IC-GN convergence.

## Limits and raw data

The graphs execute detector inference through ONNX Runtime's WebGPU provider.
They do not share buffers with `wgpu-py`; keypoint/score arrays are copied back
to NumPy (96 KiB at K=8192). The common descriptor and mutual matcher are CPU
reference stages. Model peak memory was not measured.

Raw data:

- Export/runtime parity: `detector_onnx_webgpu_20261107/dad_k8192.json` and
  `detector_onnx_webgpu_20261107/dedodev2_k8192.json`.
- Budget sweep: `detector_onnx_webgpu_20261107/common_descriptor_60pct_k8192.json`.
- Budget-8192 run: `detector_onnx_webgpu_20261107/common_descriptor_60pct_budget8192.json`.
