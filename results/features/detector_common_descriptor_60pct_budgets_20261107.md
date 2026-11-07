# Shared-descriptor detector comparison

Optional CPU-PyTorch reference detector inference; common DeDoDe-B descriptors; mutual cosine matching on CPU; local-affine RANSAC on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 1024px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 PyTorch CPU threads and are medians over 2 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 2048 | global_topk | 480/1333 | 1089 | 826 | 16/16 | 16 | 0.75/1.36 | 3.375 | 6.068 | 0.019 | 6.44 | 9.472 | 0.106 |
| synthetic_60pct | DaD | 2048 | spatial_quotas | 471/1305 | 1075 | 799 | 16/16 | 16 | 0.84/1.41 | 3.375 | 6.068 | 0.019 | 5.79 | 9.471 | 0.106 |
| synthetic_60pct | DaD | 4096 | global_topk | 901/2725 | 2219 | 1640 | 16/16 | 16 | 0.79/1.33 | 3.375 | 6.068 | 0.073 | 14.49 | 9.533 | 0.105 |
| synthetic_60pct | DaD | 4096 | spatial_quotas | 863/2606 | 2136 | 1535 | 16/16 | 16 | 0.61/1.02 | 3.375 | 6.068 | 0.076 | 14.64 | 9.536 | 0.105 |
| synthetic_60pct | DaD | 8192 | global_topk | 1702/5407 | 4339 | 3142 | 16/16 | 16 | 0.46/0.81 | 3.375 | 6.068 | 0.440 | 35.96 | 9.922 | 0.101 |
| synthetic_60pct | DaD | 8192 | spatial_quotas | 1606/5130 | 4155 | 2916 | 16/16 | 16 | 0.46/0.77 | 3.375 | 6.068 | 0.432 | 25.90 | 9.905 | 0.101 |
| synthetic_60pct | DeDoDe-v2-L | 2048 | global_topk | 109/1281 | 773 | 162 | 3/16 | 3 | 1.17/1.37 | 9.345 | 7.261 | 0.024 | 4.99 | 16.640 | 0.060 |
| synthetic_60pct | DeDoDe-v2-L | 2048 | spatial_quotas | 110/1294 | 770 | 163 | 4/16 | 4 | 2.92/4.17 | 9.345 | 7.261 | 0.023 | 3.88 | 16.638 | 0.060 |
| synthetic_60pct | DeDoDe-v2-L | 4096 | global_topk | 319/2604 | 1560 | 476 | 15/16 | 15 | 0.91/1.39 | 9.345 | 7.261 | 0.248 | 12.06 | 16.871 | 0.059 |
| synthetic_60pct | DeDoDe-v2-L | 4096 | spatial_quotas | 305/2577 | 1541 | 454 | 15/16 | 15 | 1.24/2.24 | 9.345 | 7.261 | 0.241 | 12.46 | 16.864 | 0.059 |
| synthetic_60pct | DeDoDe-v2-L | 8192 | global_topk | 874/5167 | 3321 | 1416 | 16/16 | 16 | 0.42/0.70 | 9.345 | 7.261 | 0.503 | 21.54 | 17.137 | 0.058 |
| synthetic_60pct | DeDoDe-v2-L | 8192 | spatial_quotas | 851/5123 | 3260 | 1383 | 16/16 | 16 | 0.46/0.77 | 9.345 | 7.261 | 0.489 | 23.41 | 17.122 | 0.058 |
| synthetic_60pct | DIC-Tensor-WGSL | 2048 | global_topk | 9/1317 | 786 | 73 | 0/16 | 0 | — | 0.250 | 6.904 | 0.031 | 7.08 | 7.196 | 0.139 |
| synthetic_60pct | DIC-Tensor-WGSL | 2048 | spatial_quotas | 11/1283 | 808 | 79 | 0/16 | 0 | — | 0.250 | 6.904 | 0.027 | 4.59 | 7.190 | 0.139 |
| synthetic_60pct | DIC-Tensor-WGSL | 4096 | global_topk | 54/2619 | 1601 | 271 | 6/16 | 4 | 4.43/6.06 | 0.250 | 6.904 | 0.213 | 14.32 | 7.386 | 0.135 |
| synthetic_60pct | DIC-Tensor-WGSL | 4096 | spatial_quotas | 52/2558 | 1605 | 276 | 3/16 | 2 | 4.72/6.24 | 0.250 | 6.904 | 0.234 | 15.38 | 7.408 | 0.135 |
| synthetic_60pct | DIC-Tensor-WGSL | 8192 | global_topk | 204/5238 | 3269 | 950 | 7/16 | 5 | 3.57/5.63 | 0.250 | 6.904 | 0.410 | 23.81 | 7.593 | 0.132 |
| synthetic_60pct | DIC-Tensor-WGSL | 8192 | spatial_quotas | 202/5125 | 3294 | 945 | 9/16 | 8 | 3.27/5.09 | 0.250 | 6.904 | 0.405 | 23.77 | 7.586 | 0.132 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 2048 | global_topk | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.076 | 6.680 | 0.002 | 3.79 | 6.765 | 0.148 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 2048 | spatial_quotas | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.076 | 6.680 | 0.002 | 3.72 | 6.765 | 0.148 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 4096 | global_topk | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.076 | 6.680 | 0.005 | 4.84 | 6.769 | 0.148 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 4096 | spatial_quotas | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.076 | 6.680 | 0.008 | 5.09 | 6.774 | 0.148 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 8192 | global_topk | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.076 | 6.680 | 0.017 | 3.94 | 6.782 | 0.147 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 8192 | spatial_quotas | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.076 | 6.680 | 0.004 | 2.41 | 6.769 | 0.148 |
| synthetic_60pct | DIC-Learned-WGSL | 2048 | global_topk | 344/1265 | 1010 | 662 | 15/16 | 15 | 1.83/3.86 | 0.261 | 6.160 | 0.019 | 9.73 | 6.453 | 0.155 |
| synthetic_60pct | DIC-Learned-WGSL | 2048 | spatial_quotas | 288/1282 | 941 | 572 | 16/16 | 16 | 0.86/1.46 | 0.261 | 6.160 | 0.032 | 5.07 | 6.463 | 0.155 |
| synthetic_60pct | DIC-Learned-WGSL | 4096 | global_topk | 798/2537 | 2109 | 1486 | 16/16 | 16 | 0.70/1.37 | 0.261 | 6.160 | 0.206 | 17.76 | 6.649 | 0.150 |
| synthetic_60pct | DIC-Learned-WGSL | 4096 | spatial_quotas | 807/2558 | 2117 | 1518 | 16/16 | 16 | 0.71/1.26 | 0.261 | 6.160 | 0.204 | 20.12 | 6.649 | 0.150 |
| synthetic_60pct | DIC-Learned-WGSL | 8192 | global_topk | 1910/5138 | 4320 | 3204 | 16/16 | 16 | 0.38/0.61 | 0.261 | 6.160 | 0.545 | 37.49 | 7.006 | 0.143 |
| synthetic_60pct | DIC-Learned-WGSL | 8192 | spatial_quotas | 1911/5129 | 4317 | 3191 | 16/16 | 16 | 0.40/0.78 | 0.261 | 6.160 | 0.429 | 32.79 | 6.886 | 0.145 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
