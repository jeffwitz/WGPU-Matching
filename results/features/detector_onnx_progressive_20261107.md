# DaD and DeDoDe v2 ONNX/WebGPU on progressive tensile frames

## Protocol

The comparison uses the Live-DIC progressive tensile frames 040, 100 and 199
with their analytic truth. The detector source images are UHD; both reference
PyTorch and ONNX/WebGPU paths resize them to 640 × 360 and return K=1024
candidates. Each run selects 512 points, extracts the same DeDoDe-B descriptor
on CPU, matches by mutual cosine similarity on CPU, and runs the same
local-affine RANSAC in WGSL. Three timed repeats were made after one warm-up.
The table reports global top-K selection from each detector's returned pool.

Times include the two image detector calls, descriptor calls on both images,
mutual matching and WGSL RANSAC wall time. Model loading is excluded. The FPS
is therefore full-pair throughput for this reduced-resolution experiment, not
detector-only FPS.

## Results

| Frame | Detector | Path | Repeatable ≤2 px | Accepted / correct <5 px | Affine RMSE / P95 (px) | Detect pair (s) | Full pair (s) | FPS |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 040 | DaD | PyTorch CPU | 397/512 | 21/32 (21) | 0.42 / 0.72 | 1.385 | 3.406 | 0.294 |
| 040 | DaD | ONNX/WebGPU | 399/512 | 21/32 (21) | 0.44 / 0.71 | 0.393 | 3.051 | 0.328 |
| 040 | DeDoDe v2-L | PyTorch CPU | 232/512 | 12/32 (12) | 0.85 / 1.43 | 3.979 | 6.719 | 0.149 |
| 040 | DeDoDe v2-L | ONNX/WebGPU | 232/512 | 12/32 (12) | 0.85 / 1.43 | 1.067 | 3.522 | 0.284 |
| 100 | DaD | PyTorch CPU | 238/503 | 19/28 (19) | 0.75 / 1.23 | 1.555 | 4.015 | 0.249 |
| 100 | DaD | ONNX/WebGPU | 239/503 | 19/28 (19) | 0.78 / 1.25 | 0.392 | 2.884 | 0.347 |
| 100 | DeDoDe v2-L | PyTorch CPU | 57/503 | 1/28 (1) | 0.45 / 0.45 | 3.720 | 6.197 | 0.161 |
| 100 | DeDoDe v2-L | ONNX/WebGPU | 57/503 | 1/28 (1) | 0.45 / 0.45 | 1.052 | 3.510 | 0.285 |
| 199 | DaD | PyTorch CPU | 232/484 | 18/28 (17) | 1.61 / 2.28 | 1.648 | 4.169 | 0.240 |
| 199 | DaD | ONNX/WebGPU | 234/484 | 18/28 (17) | 1.54 / 2.30 | 0.393 | 2.960 | 0.338 |
| 199 | DeDoDe v2-L | PyTorch CPU | 46/471 | 0/28 (0) | — | 3.878 | 6.391 | 0.156 |
| 199 | DeDoDe v2-L | ONNX/WebGPU | 46/471 | 0/28 (0) | — | 1.060 | 3.604 | 0.277 |

The ONNX/WebGPU path preserves the selected affine seed outcomes across all
three frames. DaD's repeatability and raw match counts differ by at most a few
points; its accepted under-5-pixel seed counts match. DeDoDe v2's detector,
match and affine outcomes are identical in these rows.

Across the three frames, detector-pair time is about 3.5–4.2× lower for DaD
and 3.5–3.8× lower for DeDoDe v2. End-to-end pair time improves by about
1.1–1.4× for DaD and 1.8–2.0× for DeDoDe v2. DeDoDe v2's detector and
descriptor combination still loses almost all valid affine initializations by
frame 100 and all at frame 199 at this 512 point budget; faster inference does
not fix this scientific limitation.

The alternative spatial-quota selection preserves the same broad conclusion:
DaD obtains 21/32 (PyTorch) and 22/32 (ONNX) correct affine seeds at <5 px on
frame 040, 19/28 through both paths on frame 100, and 15/28 through both paths
on frame 199. DeDoDe v2 obtains 11/32, 0/28 and 0/28 correct seeds through
both backends.

## Limits

- The models see 640 × 360 inputs, not native UHD. Returned point coordinates
  are mapped to the source resolution before the common descriptor/RANSAC
  stages.
- DeDoDe-B description and mutual matching remain CPU reference stages;
  RANSAC is WGSL on the recorded Vulkan Intel Graphics LNL adapter. This is
  not an all-GPU pipeline.
- These are sparse feature detector/matcher measurements, not a direct
  DeepMatching comparison and not an IC-GN convergence test.
- The progressive dataset has increasing tensile deformation, but this test
  does not cover localized heterogeneous motion or experimental damage.

Raw repeated data, including per-seed results and per-stage times:
`detector_onnx_webgpu_20261107/common_descriptor_progressive_repeats3.json`.

Timing qualification: the reported full-pair total is a sum of per-stage
medians, not the measured median of complete end-to-end calls. The raw stage
repetitions remain the evidence for its construction.
