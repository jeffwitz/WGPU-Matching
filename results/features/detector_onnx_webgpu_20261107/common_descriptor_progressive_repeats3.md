# Shared-descriptor detector comparison

Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 640px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 CPU threads for the reference PyTorch components and are medians over 3 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| progressive_040 | DaD | 512 | global_topk | 397/512 | 465 | 465 | 21/32 | 21 | 0.42/0.72 | 1.385 | 2.012 | 0.001 | 4.94 | 3.406 | 0.294 |
| progressive_040 | DaD | 512 | spatial_quotas | 393/512 | 464 | 459 | 21/32 | 21 | 0.45/0.69 | 1.385 | 2.012 | 0.002 | 3.43 | 3.407 | 0.294 |
| progressive_040 | DeDoDe-v2-L | 512 | global_topk | 232/512 | 294 | 238 | 12/32 | 12 | 0.85/1.43 | 3.979 | 2.729 | 0.001 | 4.19 | 6.719 | 0.149 |
| progressive_040 | DeDoDe-v2-L | 512 | spatial_quotas | 225/512 | 302 | 232 | 11/32 | 11 | 0.86/1.44 | 3.979 | 2.729 | 0.002 | 3.13 | 6.717 | 0.149 |
| progressive_040 | DaD-ONNX-WebGPU | 512 | global_topk | 399/512 | 469 | 469 | 21/32 | 21 | 0.44/0.71 | 0.393 | 2.644 | 0.004 | 3.84 | 3.051 | 0.328 |
| progressive_040 | DaD-ONNX-WebGPU | 512 | spatial_quotas | 394/512 | 464 | 459 | 22/32 | 22 | 0.46/0.71 | 0.393 | 2.644 | 0.001 | 3.56 | 3.051 | 0.328 |
| progressive_040 | DeDoDe-v2-ONNX-WebGPU | 512 | global_topk | 232/512 | 294 | 238 | 12/32 | 12 | 0.85/1.43 | 1.067 | 2.446 | 0.001 | 3.34 | 3.522 | 0.284 |
| progressive_040 | DeDoDe-v2-ONNX-WebGPU | 512 | spatial_quotas | 225/512 | 302 | 232 | 11/32 | 11 | 0.86/1.44 | 1.067 | 2.446 | 0.001 | 2.88 | 3.521 | 0.284 |
| progressive_100 | DaD | 512 | global_topk | 238/503 | 390 | 363 | 19/28 | 19 | 0.75/1.23 | 1.555 | 2.448 | 0.002 | 4.51 | 4.015 | 0.249 |
| progressive_100 | DaD | 512 | spatial_quotas | 236/499 | 379 | 345 | 19/28 | 19 | 0.93/1.62 | 1.555 | 2.448 | 0.003 | 3.22 | 4.014 | 0.249 |
| progressive_100 | DeDoDe-v2-L | 512 | global_topk | 57/503 | 231 | 88 | 1/28 | 1 | 0.45/0.45 | 3.720 | 2.468 | 0.002 | 2.51 | 6.197 | 0.161 |
| progressive_100 | DeDoDe-v2-L | 512 | spatial_quotas | 52/500 | 240 | 86 | 0/28 | 0 | — | 3.720 | 2.468 | 0.003 | 2.39 | 6.198 | 0.161 |
| progressive_100 | DaD-ONNX-WebGPU | 512 | global_topk | 239/503 | 392 | 366 | 19/28 | 19 | 0.78/1.25 | 0.392 | 2.483 | 0.001 | 3.55 | 2.884 | 0.347 |
| progressive_100 | DaD-ONNX-WebGPU | 512 | spatial_quotas | 232/499 | 377 | 344 | 19/28 | 19 | 0.90/1.62 | 0.392 | 2.483 | 0.001 | 3.83 | 2.887 | 0.346 |
| progressive_100 | DeDoDe-v2-ONNX-WebGPU | 512 | global_topk | 57/503 | 231 | 88 | 1/28 | 1 | 0.45/0.45 | 1.052 | 2.449 | 0.002 | 2.60 | 3.510 | 0.285 |
| progressive_100 | DeDoDe-v2-ONNX-WebGPU | 512 | spatial_quotas | 52/500 | 240 | 86 | 0/28 | 0 | — | 1.052 | 2.449 | 0.001 | 3.08 | 3.510 | 0.285 |
| progressive_199 | DaD | 512 | global_topk | 232/484 | 379 | 361 | 18/28 | 17 | 1.61/2.28 | 1.648 | 2.512 | 0.002 | 3.28 | 4.169 | 0.240 |
| progressive_199 | DaD | 512 | spatial_quotas | 223/475 | 355 | 328 | 15/28 | 15 | 0.81/1.39 | 1.648 | 2.512 | 0.001 | 3.12 | 4.168 | 0.240 |
| progressive_199 | DeDoDe-v2-L | 512 | global_topk | 46/471 | 219 | 67 | 0/28 | 0 | — | 3.878 | 2.501 | 0.002 | 2.41 | 6.391 | 0.156 |
| progressive_199 | DeDoDe-v2-L | 512 | spatial_quotas | 45/465 | 220 | 69 | 0/28 | 0 | — | 3.878 | 2.501 | 0.001 | 2.41 | 6.388 | 0.157 |
| progressive_199 | DaD-ONNX-WebGPU | 512 | global_topk | 234/484 | 378 | 359 | 18/28 | 17 | 1.54/2.30 | 0.393 | 2.557 | 0.001 | 3.87 | 2.960 | 0.338 |
| progressive_199 | DaD-ONNX-WebGPU | 512 | spatial_quotas | 223/475 | 357 | 328 | 15/28 | 15 | 0.81/1.27 | 0.393 | 2.557 | 0.003 | 3.99 | 2.964 | 0.337 |
| progressive_199 | DeDoDe-v2-ONNX-WebGPU | 512 | global_topk | 46/471 | 219 | 67 | 0/28 | 0 | — | 1.060 | 2.533 | 0.003 | 4.08 | 3.604 | 0.277 |
| progressive_199 | DeDoDe-v2-ONNX-WebGPU | 512 | spatial_quotas | 45/465 | 220 | 69 | 0/28 | 0 | — | 1.060 | 2.533 | 0.002 | 2.51 | 3.602 | 0.278 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
