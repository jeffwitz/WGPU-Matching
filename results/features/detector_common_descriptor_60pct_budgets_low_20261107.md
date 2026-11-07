# Shared-descriptor detector comparison

Optional CPU-PyTorch reference detector inference; common DeDoDe-B descriptors; mutual cosine matching on CPU; local-affine RANSAC on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 1024px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 PyTorch CPU threads and are medians over 2 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 512 | global_topk | 112/336 | 269 | 200 | 9/16 | 9 | 1.11/1.60 | 4.042 | 6.864 | 0.004 | 2.81 | 10.916 | 0.092 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 106/327 | 251 | 188 | 7/16 | 7 | 1.79/2.99 | 4.042 | 6.864 | 0.001 | 3.91 | 10.915 | 0.092 |
| synthetic_60pct | DaD | 1024 | global_topk | 231/665 | 550 | 422 | 16/16 | 16 | 1.09/1.80 | 4.042 | 6.864 | 0.005 | 5.35 | 10.919 | 0.092 |
| synthetic_60pct | DaD | 1024 | spatial_quotas | 230/650 | 532 | 413 | 16/16 | 16 | 1.22/2.12 | 4.042 | 6.864 | 0.005 | 3.97 | 10.918 | 0.092 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 11/325 | 176 | 16 | 0/16 | 0 | — | 10.004 | 6.522 | 0.007 | 3.33 | 16.540 | 0.060 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 11/314 | 195 | 14 | 0/16 | 0 | — | 10.004 | 6.522 | 0.003 | 2.69 | 16.534 | 0.060 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | global_topk | 36/636 | 381 | 51 | 0/16 | 0 | — | 10.004 | 6.522 | 0.007 | 2.91 | 16.539 | 0.060 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | spatial_quotas | 41/646 | 377 | 56 | 0/16 | 0 | — | 10.004 | 6.522 | 0.009 | 4.74 | 16.542 | 0.060 |
| synthetic_60pct | DIC-Tensor-WGSL | 512 | global_topk | 1/332 | 188 | 4 | 0/16 | 0 | — | 0.052 | 7.042 | 0.001 | 2.51 | 7.100 | 0.141 |
| synthetic_60pct | DIC-Tensor-WGSL | 512 | spatial_quotas | 1/322 | 196 | 3 | 0/16 | 0 | — | 0.052 | 7.042 | 0.001 | 3.07 | 7.101 | 0.141 |
| synthetic_60pct | DIC-Tensor-WGSL | 1024 | global_topk | 2/667 | 383 | 20 | 0/16 | 0 | — | 0.052 | 7.042 | 0.013 | 3.31 | 7.116 | 0.141 |
| synthetic_60pct | DIC-Tensor-WGSL | 1024 | spatial_quotas | 3/647 | 382 | 22 | 0/16 | 0 | — | 0.052 | 7.042 | 0.005 | 3.32 | 7.107 | 0.141 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 512 | global_topk | 26/327 | 194 | 58 | 1/16 | 1 | 0.68/0.68 | 0.073 | 6.967 | 0.001 | 2.77 | 7.047 | 0.142 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 512 | spatial_quotas | 25/316 | 187 | 53 | 0/16 | 0 | — | 0.073 | 6.967 | 0.003 | 3.25 | 7.049 | 0.142 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 1024 | global_topk | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.073 | 6.967 | 0.003 | 3.11 | 7.049 | 0.142 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 1024 | spatial_quotas | 37/443 | 310 | 91 | 2/16 | 2 | 2.98/3.99 | 0.073 | 6.967 | 0.003 | 3.42 | 7.050 | 0.142 |
| synthetic_60pct | DIC-Learned-WGSL | 512 | global_topk | 20/306 | 190 | 39 | 0/16 | 0 | — | 0.063 | 6.875 | 0.006 | 3.09 | 6.953 | 0.144 |
| synthetic_60pct | DIC-Learned-WGSL | 512 | spatial_quotas | 25/314 | 205 | 50 | 0/16 | 0 | — | 0.063 | 6.875 | 0.005 | 3.17 | 6.949 | 0.144 |
| synthetic_60pct | DIC-Learned-WGSL | 1024 | global_topk | 102/639 | 425 | 185 | 8/16 | 8 | 1.77/3.36 | 0.063 | 6.875 | 0.005 | 5.41 | 6.952 | 0.144 |
| synthetic_60pct | DIC-Learned-WGSL | 1024 | spatial_quotas | 84/641 | 429 | 165 | 8/16 | 8 | 1.42/2.38 | 0.063 | 6.875 | 0.005 | 3.84 | 6.950 | 0.144 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
