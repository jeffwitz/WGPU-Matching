# Shared-descriptor detector comparison

Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 640px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 CPU threads for the reference PyTorch components and are medians over 3 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 512 | global_topk | 71/345 | 252 | 169 | 6/16 | 6 | 2.24/3.55 | 1.447 | 2.191 | 0.002 | 2.62 | 3.645 | 0.274 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 65/332 | 255 | 163 | 4/16 | 4 | 1.74/2.42 | 1.447 | 2.191 | 0.001 | 2.54 | 3.644 | 0.274 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 15/331 | 206 | 27 | 0/16 | 0 | — | 3.709 | 2.583 | 0.003 | 3.09 | 6.302 | 0.159 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 9/318 | 215 | 19 | 0/16 | 0 | — | 3.709 | 2.583 | 0.002 | 2.32 | 6.300 | 0.159 |
| synthetic_60pct | DaD-ONNX-WebGPU | 512 | global_topk | 71/346 | 253 | 168 | 6/16 | 6 | 2.14/3.52 | 0.391 | 2.569 | 0.002 | 2.75 | 2.969 | 0.337 |
| synthetic_60pct | DaD-ONNX-WebGPU | 512 | spatial_quotas | 65/333 | 256 | 162 | 4/16 | 4 | 1.75/2.42 | 0.391 | 2.569 | 0.002 | 3.35 | 2.970 | 0.337 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 512 | global_topk | 15/331 | 206 | 27 | 0/16 | 0 | — | 1.056 | 2.644 | 0.002 | 2.53 | 3.708 | 0.270 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 512 | spatial_quotas | 9/318 | 215 | 19 | 0/16 | 0 | — | 1.056 | 2.644 | 0.002 | 2.39 | 3.709 | 0.270 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
