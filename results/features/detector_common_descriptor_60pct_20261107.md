# Shared-descriptor detector comparison

Optional CPU-PyTorch reference detector inference; common DeDoDe-B descriptors; mutual cosine matching on CPU; local-affine RANSAC on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 1024px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 PyTorch CPU threads and are medians over 2 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 512 | global_topk | 112/336 | 269 | 200 | 9/16 | 9 | 3.367 | 5.760 | 0.001 | 3.43 | 9.135 | 0.109 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 106/327 | 251 | 188 | 7/16 | 7 | 3.367 | 5.760 | 0.001 | 2.81 | 9.136 | 0.109 |
| synthetic_60pct | DaD | 1024 | global_topk | 231/665 | 550 | 422 | 16/16 | 16 | 3.367 | 5.760 | 0.004 | 4.93 | 9.141 | 0.109 |
| synthetic_60pct | DaD | 1024 | spatial_quotas | 230/650 | 532 | 413 | 16/16 | 16 | 3.367 | 5.760 | 0.006 | 5.39 | 9.143 | 0.109 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 11/325 | 176 | 16 | 0/16 | 0 | 8.444 | 5.765 | 0.001 | 2.14 | 14.215 | 0.070 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 11/314 | 195 | 14 | 0/16 | 0 | 8.444 | 5.765 | 0.002 | 3.67 | 14.218 | 0.070 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | global_topk | 36/636 | 381 | 51 | 0/16 | 0 | 8.444 | 5.765 | 0.009 | 3.85 | 14.224 | 0.070 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | spatial_quotas | 41/646 | 377 | 56 | 0/16 | 0 | 8.444 | 5.765 | 0.004 | 3.72 | 14.220 | 0.070 |
| synthetic_60pct | DIC-Tensor-WGSL | 512 | global_topk | 1/332 | 188 | 4 | 0/16 | 0 | 0.044 | 5.649 | 0.001 | 2.16 | 5.698 | 0.175 |
| synthetic_60pct | DIC-Tensor-WGSL | 512 | spatial_quotas | 1/322 | 196 | 3 | 0/16 | 0 | 0.044 | 5.649 | 0.002 | 2.28 | 5.700 | 0.175 |
| synthetic_60pct | DIC-Tensor-WGSL | 1024 | global_topk | 2/667 | 383 | 20 | 0/16 | 0 | 0.044 | 5.649 | 0.005 | 3.21 | 5.703 | 0.175 |
| synthetic_60pct | DIC-Tensor-WGSL | 1024 | spatial_quotas | 3/647 | 382 | 22 | 0/16 | 0 | 0.044 | 5.649 | 0.005 | 3.12 | 5.703 | 0.175 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 512 | global_topk | 26/327 | 194 | 58 | 1/16 | 1 | 0.077 | 5.751 | 0.002 | 2.31 | 5.836 | 0.171 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 512 | spatial_quotas | 25/316 | 187 | 53 | 0/16 | 0 | 0.077 | 5.751 | 0.001 | 2.32 | 5.835 | 0.171 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 1024 | global_topk | 37/443 | 310 | 91 | 2/16 | 2 | 0.077 | 5.751 | 0.004 | 3.77 | 5.839 | 0.171 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 1024 | spatial_quotas | 37/443 | 310 | 91 | 2/16 | 2 | 0.077 | 5.751 | 0.003 | 3.27 | 5.837 | 0.171 |
| synthetic_60pct | DIC-Learned-WGSL | 512 | global_topk | 16/316 | 203 | 34 | 0/16 | 0 | 0.055 | 5.745 | 0.001 | 3.19 | 5.807 | 0.172 |
| synthetic_60pct | DIC-Learned-WGSL | 512 | spatial_quotas | 23/323 | 208 | 39 | 0/16 | 0 | 0.055 | 5.745 | 0.002 | 2.61 | 5.808 | 0.172 |
| synthetic_60pct | DIC-Learned-WGSL | 1024 | global_topk | 63/639 | 418 | 122 | 5/16 | 5 | 0.055 | 5.745 | 0.004 | 4.59 | 5.811 | 0.172 |
| synthetic_60pct | DIC-Learned-WGSL | 1024 | spatial_quotas | 69/638 | 423 | 128 | 6/16 | 6 | 0.055 | 5.745 | 0.006 | 4.30 | 5.816 | 0.172 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
