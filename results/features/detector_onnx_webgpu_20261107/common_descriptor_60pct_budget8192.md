# Shared-descriptor detector comparison

Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 640px. Keypoint selection candidates: 1× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 CPU threads for the reference PyTorch components and are medians over 1 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 8192 | global_topk | 886/5417 | 4109 | 2301 | 16/16 | 16 | 1.21/2.11 | 1.070 | 1.982 | 0.264 | 33.54 | 3.353 | 0.298 |
| synthetic_60pct | DaD | 8192 | spatial_quotas | 886/5417 | 4109 | 2301 | 16/16 | 16 | 1.21/2.11 | 1.070 | 1.982 | 0.293 | 33.91 | 3.385 | 0.295 |
| synthetic_60pct | DeDoDe-v2-L | 8192 | global_topk | 562/5176 | 3570 | 1411 | 16/16 | 16 | 1.27/1.97 | 2.835 | 1.924 | 0.222 | 33.50 | 5.018 | 0.199 |
| synthetic_60pct | DeDoDe-v2-L | 8192 | spatial_quotas | 562/5176 | 3570 | 1411 | 16/16 | 16 | 1.27/1.97 | 2.835 | 1.924 | 0.231 | 17.61 | 5.011 | 0.200 |
| synthetic_60pct | DaD-ONNX-WebGPU | 8192 | global_topk | 893/5419 | 4118 | 2313 | 16/16 | 16 | 1.11/1.80 | 0.375 | 2.165 | 0.245 | 34.17 | 2.824 | 0.354 |
| synthetic_60pct | DaD-ONNX-WebGPU | 8192 | spatial_quotas | 893/5419 | 4118 | 2313 | 16/16 | 16 | 1.11/1.80 | 0.375 | 2.165 | 0.245 | 35.04 | 2.824 | 0.354 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 8192 | global_topk | 562/5176 | 3570 | 1411 | 16/16 | 16 | 1.17/1.85 | 0.977 | 2.326 | 0.258 | 35.18 | 3.601 | 0.278 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 8192 | spatial_quotas | 562/5176 | 3570 | 1411 | 16/16 | 16 | 1.17/1.85 | 0.977 | 2.326 | 0.268 | 31.81 | 3.608 | 0.277 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
