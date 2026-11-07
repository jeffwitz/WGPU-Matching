# Feature-method context at 60% tensile strain

## Common outcome metric

This table combines the latest high-budget DaD/DeDoDe ONNX results with
previously published method-specific reports on the same 3840 × 2160 synthetic
global-affine 60% tensile pair. All methods are scored at the same 16 visible
analytic DIC seed centres. “Correct ≤2 px” and “correct <5 px” use the same
analytic displacement truth and denominator. The method-specific acceptance
criteria and timing scopes differ, so this is a system-level context table,
not a controlled backend microbenchmark.

| Initializer | Accepted / usable | Correct ≤2 px / 16 | Correct <5 px / 16 | Pair time | Timing scope |
|---|---:|---:|---:|---:|---|
| DeepMatching INRIA | 10/16 | 4/16 | 8/16 | 16.405 s | Full-resolution `wgpu-py` matcher with 1024 px search radius |
| HORG | 0/16 | 0/16 | 0/16 | 0.146 s | Browser WebGPU matcher |
| XFeat | 12/16 | 2/16 | 8/16 | 0.553 s | ONNX/WebGPU feature inference and local affine fit |
| EfficientLoFTR | 16/16 | 12/16 | 14/16 | 1.501 s | ONNX/WebGPU inference plus local affine fit at 640 × 360 |
| DaD + DeDoDe-B, ONNX/WebGPU | 16/16 | 15/16 | 16/16 | 2.824 s | 640 × 360 detector, CPU descriptor/mutual match, WGSL affine RANSAC; K=8192 |
| DeDoDe v2-L + DeDoDe-B, ONNX/WebGPU | 16/16 | 16/16 | 16/16 | 3.601 s | 640 × 360 detector, CPU descriptor/mutual match, WGSL affine RANSAC; K=8192 |
| HORG/SIFT + DeDoDe-B | 2/16 | 1/16 | 2/16 | 7.241 s | WGSL detection/RANSAC with CPU descriptor/matching; budget 1024 |
| DIC learned WGSL + DeDoDe-B | 16/16 | 16/16 | 16/16 | 7.006 s | Native 4K WGSL detection, CPU descriptor/matching, WGSL RANSAC; budget 8192 |

The two ONNX feature pipelines have the best measured seed coverage in this
set. DeDoDe v2-L + DeDoDe-B is the only listed method with all 16 seeds within
2 px in the ONNX high-budget run; DaD is within 2 px for 15/16 and within 5 px
for all 16. They are slower than EfficientLoFTR and XFeat here, but faster than
DeepMatching at its search radius required for the 60% displacement. The DIC
learned detector also reaches 16/16 within 2 px, with a slower but native-4K
detector stage.

This does not imply the learned detectors will generalize better to
heterogeneous deformation. The 60% case is globally affine, the ONNX feature
detectors see resized inputs, DeDoDe-B and mutual matching still run on CPU,
and none of these tests measures IC-GN convergence. The two external feature
methods and DIC learned baseline have different acceptance and stage-time
definitions; use the table for end-user initializations and approximate
throughput context, not an isolated detector benchmark.

## Provenance

- New K=8192 ONNX/WebGPU results and raw data:
  `detector_onnx_high_budget_20261107.md` and
  `detector_onnx_webgpu_20261107/common_descriptor_60pct_budget8192.json`.
- DeepMatching/XFeat same-seed results and protocol:
  `deepmatching_vs_xfeat.md`.
- EfficientLoFTR same-seed results:
  `detector_method_comparison_20261107.md` and
  `efficientloftr_webgpu_study.json`.
- HORG browser results:
  `../uniaxial/tensile_60pct/three_method_browser_60.md`.
- HORG/SIFT and DIC learned detector results:
  `detector_budget_sweep_60pct_20261107.md`.
