# Shared-descriptor detector comparison

Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 640px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 CPU threads for the reference PyTorch components and are medians over 1 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD-ONNX-WebGPU | 512 | global_topk | 71/346 | 253 | 168 | 6/16 | 6 | 2.14/3.52 | 0.378 | 2.109 | 0.005 | 2.92 | 2.498 | 0.400 |
| synthetic_60pct | DaD-ONNX-WebGPU | 512 | spatial_quotas | 65/333 | 256 | 162 | 4/16 | 4 | 1.75/2.42 | 0.378 | 2.109 | 0.001 | 2.66 | 2.493 | 0.401 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 512 | global_topk | 15/331 | 206 | 27 | 0/16 | 0 | — | 1.173 | 2.175 | 0.001 | 2.98 | 3.355 | 0.298 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 512 | spatial_quotas | 9/318 | 215 | 19 | 0/16 | 0 | — | 1.173 | 2.175 | 0.001 | 2.52 | 3.356 | 0.298 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
