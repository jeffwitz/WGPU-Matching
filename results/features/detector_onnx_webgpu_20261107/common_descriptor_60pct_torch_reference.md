# Shared-descriptor detector comparison

Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 640px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 CPU threads for the reference PyTorch components and are medians over 1 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 512 | global_topk | 71/345 | 252 | 169 | 6/16 | 6 | 2.24/3.55 | 1.474 | 2.144 | 0.013 | 2.91 | 3.637 | 0.275 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 65/332 | 255 | 163 | 4/16 | 4 | 1.74/2.42 | 1.474 | 2.144 | 0.002 | 4.28 | 3.628 | 0.276 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 15/331 | 206 | 27 | 0/16 | 0 | — | 3.571 | 2.249 | 0.001 | 3.56 | 5.827 | 0.172 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 9/318 | 215 | 19 | 0/16 | 0 | — | 3.571 | 2.249 | 0.004 | 3.50 | 5.831 | 0.172 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
