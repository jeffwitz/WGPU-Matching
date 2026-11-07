# DaD and DeDoDe v2 detector reference experiment

Optional CPU-PyTorch detector inference; common DeDoDe-B descriptors; mutual cosine matching on CPU; local-affine RANSAC on the recorded WebGPU adapter.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 1024px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings are medians over 3 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Detection s | Matching s | WGSL RANSAC ms |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic_60pct | DaD | 512 | global_topk | 112/336 | 269 | 200 | 9/16 | 9 | 3.776 | 0.009 | 2.83 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 106/327 | 251 | 188 | 7/16 | 7 | 3.776 | 0.004 | 3.65 |
| synthetic_60pct | DaD | 1024 | global_topk | 231/665 | 550 | 422 | 16/16 | 16 | 3.776 | 0.012 | 6.20 |
| synthetic_60pct | DaD | 1024 | spatial_quotas | 230/650 | 532 | 413 | 16/16 | 16 | 3.776 | 0.013 | 3.13 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 11/325 | 176 | 16 | 0/16 | 0 | 11.733 | 0.001 | 2.76 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 11/314 | 195 | 14 | 0/16 | 0 | 11.733 | 0.001 | 3.81 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | global_topk | 36/636 | 381 | 51 | 0/16 | 0 | 11.733 | 0.004 | 2.95 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | spatial_quotas | 41/646 | 377 | 56 | 0/16 | 0 | 11.733 | 0.005 | 2.76 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
