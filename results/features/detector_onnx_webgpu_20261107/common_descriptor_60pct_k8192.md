# Shared-descriptor detector comparison

Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 640px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 CPU threads for the reference PyTorch components and are medians over 1 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 512 | global_topk | 71/345 | 252 | 169 | 6/16 | 6 | 2.24/3.55 | 1.222 | 3.002 | 0.005 | 5.86 | 4.241 | 0.236 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 65/332 | 255 | 163 | 4/16 | 4 | 1.74/2.42 | 1.222 | 3.002 | 0.005 | 2.74 | 4.237 | 0.236 |
| synthetic_60pct | DaD | 1024 | global_topk | 138/699 | 521 | 347 | 14/16 | 14 | 2.14/3.83 | 1.222 | 3.002 | 0.015 | 6.98 | 4.251 | 0.235 |
| synthetic_60pct | DaD | 1024 | spatial_quotas | 135/671 | 512 | 337 | 13/16 | 13 | 1.89/3.55 | 1.222 | 3.002 | 0.012 | 3.80 | 4.245 | 0.236 |
| synthetic_60pct | DaD | 2048 | global_topk | 249/1390 | 1048 | 640 | 15/16 | 14 | 2.20/4.59 | 1.222 | 3.002 | 0.063 | 15.37 | 4.305 | 0.232 |
| synthetic_60pct | DaD | 2048 | spatial_quotas | 236/1314 | 1019 | 599 | 16/16 | 16 | 1.58/3.64 | 1.222 | 3.002 | 0.025 | 10.02 | 4.262 | 0.235 |
| synthetic_60pct | DaD | 4096 | global_topk | 499/2770 | 2105 | 1238 | 16/16 | 15 | 4.19/5.94 | 1.222 | 3.002 | 0.133 | 24.13 | 4.385 | 0.228 |
| synthetic_60pct | DaD | 4096 | spatial_quotas | 472/2591 | 2026 | 1141 | 16/16 | 16 | 1.52/2.31 | 1.222 | 3.002 | 0.121 | 21.86 | 4.370 | 0.229 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 15/331 | 206 | 27 | 0/16 | 0 | — | 3.790 | 2.336 | 0.002 | 2.00 | 6.134 | 0.163 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 9/318 | 215 | 19 | 0/16 | 0 | — | 3.790 | 2.336 | 0.001 | 2.19 | 6.133 | 0.163 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | global_topk | 30/657 | 423 | 66 | 1/16 | 1 | 1.91/1.91 | 3.790 | 2.336 | 0.005 | 3.31 | 6.138 | 0.163 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | spatial_quotas | 28/639 | 396 | 58 | 2/16 | 2 | 2.78/3.36 | 3.790 | 2.336 | 0.031 | 2.83 | 6.164 | 0.162 |
| synthetic_60pct | DeDoDe-v2-L | 2048 | global_topk | 78/1321 | 834 | 177 | 6/16 | 6 | 1.76/2.19 | 3.790 | 2.336 | 0.021 | 7.52 | 6.158 | 0.162 |
| synthetic_60pct | DeDoDe-v2-L | 2048 | spatial_quotas | 80/1287 | 841 | 173 | 3/16 | 3 | 1.33/1.67 | 3.790 | 2.336 | 0.030 | 7.09 | 6.167 | 0.162 |
| synthetic_60pct | DeDoDe-v2-L | 4096 | global_topk | 224/2607 | 1731 | 539 | 14/16 | 14 | 1.79/3.08 | 3.790 | 2.336 | 0.084 | 18.82 | 6.232 | 0.160 |
| synthetic_60pct | DeDoDe-v2-L | 4096 | spatial_quotas | 221/2570 | 1698 | 508 | 13/16 | 12 | 3.02/6.20 | 3.790 | 2.336 | 0.082 | 16.56 | 6.228 | 0.161 |
| synthetic_60pct | DaD-ONNX-WebGPU | 512 | global_topk | 71/346 | 253 | 168 | 6/16 | 6 | 2.14/3.52 | 0.392 | 2.555 | 0.004 | 2.27 | 2.959 | 0.338 |
| synthetic_60pct | DaD-ONNX-WebGPU | 512 | spatial_quotas | 65/333 | 256 | 162 | 4/16 | 4 | 1.75/2.42 | 0.392 | 2.555 | 0.003 | 4.11 | 2.966 | 0.337 |
| synthetic_60pct | DaD-ONNX-WebGPU | 1024 | global_topk | 141/699 | 521 | 350 | 13/16 | 13 | 2.04/3.84 | 0.392 | 2.555 | 0.009 | 5.24 | 2.966 | 0.337 |
| synthetic_60pct | DaD-ONNX-WebGPU | 1024 | spatial_quotas | 136/671 | 512 | 337 | 13/16 | 13 | 1.78/3.55 | 0.392 | 2.555 | 0.011 | 3.95 | 2.967 | 0.337 |
| synthetic_60pct | DaD-ONNX-WebGPU | 2048 | global_topk | 252/1389 | 1054 | 642 | 15/16 | 14 | 2.15/4.43 | 0.392 | 2.555 | 0.043 | 13.86 | 3.008 | 0.332 |
| synthetic_60pct | DaD-ONNX-WebGPU | 2048 | spatial_quotas | 241/1315 | 1020 | 605 | 16/16 | 16 | 1.43/2.35 | 0.392 | 2.555 | 0.026 | 8.41 | 2.986 | 0.335 |
| synthetic_60pct | DaD-ONNX-WebGPU | 4096 | global_topk | 506/2775 | 2101 | 1246 | 16/16 | 15 | 2.16/3.77 | 0.392 | 2.555 | 0.161 | 21.87 | 3.134 | 0.319 |
| synthetic_60pct | DaD-ONNX-WebGPU | 4096 | spatial_quotas | 477/2589 | 2021 | 1142 | 16/16 | 16 | 1.33/2.06 | 0.392 | 2.555 | 0.196 | 19.79 | 3.168 | 0.316 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 512 | global_topk | 15/331 | 206 | 27 | 0/16 | 0 | — | 1.261 | 2.299 | 0.010 | 3.02 | 3.577 | 0.280 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 512 | spatial_quotas | 9/318 | 215 | 19 | 0/16 | 0 | — | 1.261 | 2.299 | 0.005 | 3.29 | 3.572 | 0.280 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 1024 | global_topk | 30/657 | 423 | 66 | 1/16 | 1 | 1.91/1.91 | 1.261 | 2.299 | 0.005 | 3.42 | 3.571 | 0.280 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 1024 | spatial_quotas | 28/639 | 396 | 58 | 2/16 | 2 | 2.78/3.36 | 1.261 | 2.299 | 0.004 | 3.95 | 3.572 | 0.280 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 2048 | global_topk | 78/1321 | 834 | 177 | 6/16 | 6 | 1.76/2.19 | 1.261 | 2.299 | 0.021 | 7.54 | 3.593 | 0.278 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 2048 | spatial_quotas | 80/1287 | 841 | 173 | 3/16 | 3 | 1.72/2.31 | 1.261 | 2.299 | 0.019 | 7.63 | 3.591 | 0.278 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 4096 | global_topk | 224/2607 | 1731 | 539 | 14/16 | 14 | 1.79/3.08 | 1.261 | 2.299 | 0.152 | 18.99 | 3.735 | 0.268 |
| synthetic_60pct | DeDoDe-v2-ONNX-WebGPU | 4096 | spatial_quotas | 221/2570 | 1698 | 508 | 13/16 | 12 | 3.02/6.20 | 1.261 | 2.299 | 0.171 | 19.55 | 3.755 | 0.266 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
