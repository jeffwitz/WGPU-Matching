# Shared-descriptor detector comparison

Optional CPU-PyTorch reference detector inference; common DeDoDe-B descriptors; mutual cosine matching on CPU; local-affine RANSAC on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.
These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.

Model maximum side: 1024px. Keypoint selection candidates: 2× each requested budget. Adapter: Vulkan (Intel(R) Graphics (LNL)). Timings use 8 PyTorch CPU threads and are medians over 2 steady-state repeats after one warm-up.

| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| progressive_040 | DaD | 512 | global_topk | 416/512 | 454 | 450 | 25/32 | 25 | 3.306 | 6.017 | 0.002 | 4.75 | 9.332 | 0.107 |
| progressive_040 | DaD | 512 | spatial_quotas | 406/512 | 442 | 437 | 22/32 | 22 | 3.306 | 6.017 | 0.002 | 4.55 | 9.332 | 0.107 |
| progressive_040 | DaD | 1024 | global_topk | 838/1024 | 906 | 904 | 31/32 | 31 | 3.306 | 6.017 | 0.006 | 9.51 | 9.341 | 0.107 |
| progressive_040 | DaD | 1024 | spatial_quotas | 831/1024 | 898 | 895 | 31/32 | 31 | 3.306 | 6.017 | 0.004 | 7.03 | 9.339 | 0.107 |
| progressive_040 | DeDoDe-v2-L | 512 | global_topk | 156/512 | 241 | 160 | 7/32 | 7 | 9.080 | 6.325 | 0.007 | 2.86 | 15.419 | 0.065 |
| progressive_040 | DeDoDe-v2-L | 512 | spatial_quotas | 150/512 | 241 | 152 | 4/32 | 4 | 9.080 | 6.325 | 0.001 | 3.39 | 15.413 | 0.065 |
| progressive_040 | DeDoDe-v2-L | 1024 | global_topk | 340/1024 | 523 | 349 | 15/32 | 15 | 9.080 | 6.325 | 0.006 | 5.76 | 15.421 | 0.065 |
| progressive_040 | DeDoDe-v2-L | 1024 | spatial_quotas | 332/1024 | 523 | 342 | 13/32 | 13 | 9.080 | 6.325 | 0.007 | 5.82 | 15.422 | 0.065 |
| progressive_040 | DIC-Tensor-WGSL | 512 | global_topk | 163/512 | 265 | 179 | 6/32 | 6 | 0.046 | 6.100 | 0.002 | 5.14 | 6.155 | 0.162 |
| progressive_040 | DIC-Tensor-WGSL | 512 | spatial_quotas | 167/512 | 284 | 188 | 7/32 | 7 | 0.046 | 6.100 | 0.002 | 3.19 | 6.155 | 0.162 |
| progressive_040 | DIC-Tensor-WGSL | 1024 | global_topk | 371/1024 | 558 | 424 | 19/32 | 19 | 0.046 | 6.100 | 0.005 | 5.48 | 6.162 | 0.162 |
| progressive_040 | DIC-Tensor-WGSL | 1024 | spatial_quotas | 424/1024 | 604 | 486 | 23/32 | 23 | 0.046 | 6.100 | 0.005 | 5.92 | 6.160 | 0.162 |
| progressive_040 | HORG-SIFT-common-DeDoDe-B | 512 | global_topk | 391/512 | 403 | 392 | 15/32 | 15 | 0.070 | 6.171 | 0.002 | 4.25 | 6.252 | 0.160 |
| progressive_040 | HORG-SIFT-common-DeDoDe-B | 512 | spatial_quotas | 368/512 | 383 | 369 | 18/32 | 18 | 0.070 | 6.171 | 0.001 | 3.49 | 6.251 | 0.160 |
| progressive_040 | HORG-SIFT-common-DeDoDe-B | 1024 | global_topk | 534/685 | 547 | 535 | 22/32 | 22 | 0.070 | 6.171 | 0.002 | 4.53 | 6.252 | 0.160 |
| progressive_040 | HORG-SIFT-common-DeDoDe-B | 1024 | spatial_quotas | 534/685 | 547 | 535 | 22/32 | 22 | 0.070 | 6.171 | 0.003 | 4.88 | 6.254 | 0.160 |
| progressive_040 | DIC-Learned-WGSL | 512 | global_topk | 101/512 | 229 | 108 | 5/32 | 5 | 0.056 | 6.092 | 0.001 | 2.59 | 6.154 | 0.162 |
| progressive_040 | DIC-Learned-WGSL | 512 | spatial_quotas | 92/512 | 241 | 97 | 4/32 | 4 | 0.056 | 6.092 | 0.001 | 2.25 | 6.154 | 0.163 |
| progressive_040 | DIC-Learned-WGSL | 1024 | global_topk | 268/1024 | 507 | 280 | 11/32 | 11 | 0.056 | 6.092 | 0.004 | 4.47 | 6.159 | 0.162 |
| progressive_040 | DIC-Learned-WGSL | 1024 | spatial_quotas | 256/1024 | 501 | 270 | 11/32 | 11 | 0.056 | 6.092 | 0.004 | 5.18 | 6.161 | 0.162 |
| progressive_100 | DaD | 512 | global_topk | 330/491 | 395 | 379 | 16/28 | 16 | 3.287 | 7.010 | 0.003 | 3.65 | 10.306 | 0.097 |
| progressive_100 | DaD | 512 | spatial_quotas | 314/492 | 385 | 364 | 16/28 | 16 | 3.287 | 7.010 | 0.002 | 4.85 | 10.308 | 0.097 |
| progressive_100 | DaD | 1024 | global_topk | 653/993 | 794 | 761 | 27/28 | 27 | 3.287 | 7.010 | 0.005 | 8.26 | 10.315 | 0.097 |
| progressive_100 | DaD | 1024 | spatial_quotas | 645/992 | 785 | 757 | 27/28 | 27 | 3.287 | 7.010 | 0.007 | 5.60 | 10.316 | 0.097 |
| progressive_100 | DeDoDe-v2-L | 512 | global_topk | 63/500 | 211 | 70 | 0/28 | 0 | 13.098 | 6.952 | 0.005 | 4.41 | 20.062 | 0.050 |
| progressive_100 | DeDoDe-v2-L | 512 | spatial_quotas | 61/495 | 209 | 69 | 0/28 | 0 | 13.098 | 6.952 | 0.008 | 2.72 | 20.068 | 0.050 |
| progressive_100 | DeDoDe-v2-L | 1024 | global_topk | 158/999 | 468 | 173 | 9/28 | 8 | 13.098 | 6.952 | 0.006 | 4.93 | 20.064 | 0.050 |
| progressive_100 | DeDoDe-v2-L | 1024 | spatial_quotas | 150/998 | 476 | 167 | 7/28 | 6 | 13.098 | 6.952 | 0.005 | 6.28 | 20.064 | 0.050 |
| progressive_100 | DIC-Tensor-WGSL | 512 | global_topk | 91/496 | 247 | 130 | 0/28 | 0 | 0.049 | 7.158 | 0.002 | 2.17 | 7.214 | 0.139 |
| progressive_100 | DIC-Tensor-WGSL | 512 | spatial_quotas | 80/494 | 245 | 116 | 0/28 | 0 | 0.049 | 7.158 | 0.002 | 2.23 | 7.215 | 0.139 |
| progressive_100 | DIC-Tensor-WGSL | 1024 | global_topk | 212/989 | 502 | 302 | 17/28 | 15 | 0.049 | 7.158 | 0.004 | 4.69 | 7.221 | 0.138 |
| progressive_100 | DIC-Tensor-WGSL | 1024 | spatial_quotas | 208/986 | 510 | 302 | 14/28 | 14 | 0.049 | 7.158 | 0.015 | 4.37 | 7.231 | 0.138 |
| progressive_100 | HORG-SIFT-common-DeDoDe-B | 512 | global_topk | 339/497 | 367 | 341 | 16/28 | 16 | 0.074 | 7.468 | 0.003 | 5.61 | 7.557 | 0.132 |
| progressive_100 | HORG-SIFT-common-DeDoDe-B | 512 | spatial_quotas | 306/495 | 344 | 306 | 15/28 | 15 | 0.074 | 7.468 | 0.002 | 5.08 | 7.557 | 0.132 |
| progressive_100 | HORG-SIFT-common-DeDoDe-B | 1024 | global_topk | 448/673 | 489 | 456 | 20/28 | 20 | 0.074 | 7.468 | 0.002 | 6.50 | 7.556 | 0.132 |
| progressive_100 | HORG-SIFT-common-DeDoDe-B | 1024 | spatial_quotas | 448/673 | 489 | 456 | 20/28 | 20 | 0.074 | 7.468 | 0.003 | 5.48 | 7.555 | 0.132 |
| progressive_100 | DIC-Learned-WGSL | 512 | global_topk | 38/487 | 210 | 48 | 0/28 | 0 | 0.062 | 6.940 | 0.002 | 2.94 | 7.012 | 0.143 |
| progressive_100 | DIC-Learned-WGSL | 512 | spatial_quotas | 37/488 | 209 | 48 | 0/28 | 0 | 0.062 | 6.940 | 0.002 | 4.11 | 7.013 | 0.143 |
| progressive_100 | DIC-Learned-WGSL | 1024 | global_topk | 125/980 | 457 | 169 | 7/28 | 7 | 0.062 | 6.940 | 0.005 | 3.61 | 7.018 | 0.142 |
| progressive_100 | DIC-Learned-WGSL | 1024 | spatial_quotas | 110/979 | 463 | 156 | 6/28 | 5 | 0.062 | 6.940 | 0.008 | 5.14 | 7.021 | 0.142 |
| progressive_199 | DaD | 512 | global_topk | 291/469 | 369 | 351 | 17/28 | 16 | 4.171 | 7.150 | 0.001 | 4.22 | 11.329 | 0.088 |
| progressive_199 | DaD | 512 | spatial_quotas | 273/465 | 356 | 331 | 19/28 | 18 | 4.171 | 7.150 | 0.008 | 5.00 | 11.338 | 0.088 |
| progressive_199 | DaD | 1024 | global_topk | 595/940 | 756 | 720 | 26/28 | 26 | 4.171 | 7.150 | 0.005 | 6.86 | 11.336 | 0.088 |
| progressive_199 | DaD | 1024 | spatial_quotas | 596/935 | 757 | 724 | 27/28 | 27 | 4.171 | 7.150 | 0.013 | 4.63 | 11.342 | 0.088 |
| progressive_199 | DeDoDe-v2-L | 512 | global_topk | 41/477 | 208 | 44 | 0/28 | 0 | 10.075 | 7.161 | 0.001 | 6.44 | 17.247 | 0.058 |
| progressive_199 | DeDoDe-v2-L | 512 | spatial_quotas | 43/463 | 195 | 46 | 0/28 | 0 | 10.075 | 7.161 | 0.005 | 4.05 | 17.249 | 0.058 |
| progressive_199 | DeDoDe-v2-L | 1024 | global_topk | 126/951 | 407 | 138 | 4/28 | 4 | 10.075 | 7.161 | 0.010 | 3.31 | 17.256 | 0.058 |
| progressive_199 | DeDoDe-v2-L | 1024 | spatial_quotas | 117/941 | 415 | 128 | 4/28 | 4 | 10.075 | 7.161 | 0.012 | 5.11 | 17.257 | 0.058 |
| progressive_199 | DIC-Tensor-WGSL | 512 | global_topk | 70/477 | 226 | 111 | 0/28 | 0 | 0.050 | 7.140 | 0.002 | 3.47 | 7.202 | 0.139 |
| progressive_199 | DIC-Tensor-WGSL | 512 | spatial_quotas | 67/471 | 228 | 105 | 0/28 | 0 | 0.050 | 7.140 | 0.002 | 1.94 | 7.199 | 0.139 |
| progressive_199 | DIC-Tensor-WGSL | 1024 | global_topk | 166/947 | 477 | 280 | 11/28 | 9 | 0.050 | 7.140 | 0.005 | 4.55 | 7.204 | 0.139 |
| progressive_199 | DIC-Tensor-WGSL | 1024 | spatial_quotas | 163/935 | 493 | 274 | 11/28 | 9 | 0.050 | 7.140 | 0.005 | 4.41 | 7.204 | 0.139 |
| progressive_199 | HORG-SIFT-common-DeDoDe-B | 512 | global_topk | 257/463 | 300 | 276 | 12/28 | 12 | 0.081 | 7.031 | 0.004 | 4.26 | 7.126 | 0.140 |
| progressive_199 | HORG-SIFT-common-DeDoDe-B | 512 | spatial_quotas | 227/465 | 284 | 242 | 10/28 | 10 | 0.081 | 7.031 | 0.004 | 2.48 | 7.122 | 0.140 |
| progressive_199 | HORG-SIFT-common-DeDoDe-B | 1024 | global_topk | 334/630 | 419 | 380 | 17/28 | 17 | 0.081 | 7.031 | 0.003 | 3.64 | 7.122 | 0.140 |
| progressive_199 | HORG-SIFT-common-DeDoDe-B | 1024 | spatial_quotas | 334/630 | 419 | 380 | 17/28 | 17 | 0.081 | 7.031 | 0.013 | 4.47 | 7.136 | 0.140 |
| progressive_199 | DIC-Learned-WGSL | 512 | global_topk | 41/459 | 219 | 54 | 0/28 | 0 | 0.061 | 7.193 | 0.001 | 3.76 | 7.264 | 0.138 |
| progressive_199 | DIC-Learned-WGSL | 512 | spatial_quotas | 49/462 | 227 | 61 | 0/28 | 0 | 0.061 | 7.193 | 0.008 | 4.00 | 7.270 | 0.138 |
| progressive_199 | DIC-Learned-WGSL | 1024 | global_topk | 134/933 | 441 | 172 | 5/28 | 5 | 0.061 | 7.193 | 0.010 | 5.96 | 7.273 | 0.137 |
| progressive_199 | DIC-Learned-WGSL | 1024 | spatial_quotas | 138/929 | 452 | 177 | 6/28 | 6 | 0.061 | 7.193 | 0.014 | 6.82 | 7.277 | 0.137 |
| synthetic_60pct | DaD | 512 | global_topk | 112/336 | 269 | 200 | 9/16 | 9 | 4.099 | 7.059 | 0.001 | 3.75 | 11.165 | 0.090 |
| synthetic_60pct | DaD | 512 | spatial_quotas | 106/327 | 251 | 188 | 7/16 | 7 | 4.099 | 7.059 | 0.001 | 2.54 | 11.164 | 0.090 |
| synthetic_60pct | DaD | 1024 | global_topk | 231/665 | 550 | 422 | 16/16 | 16 | 4.099 | 7.059 | 0.006 | 4.84 | 11.173 | 0.089 |
| synthetic_60pct | DaD | 1024 | spatial_quotas | 230/650 | 532 | 413 | 16/16 | 16 | 4.099 | 7.059 | 0.007 | 3.96 | 11.171 | 0.090 |
| synthetic_60pct | DeDoDe-v2-L | 512 | global_topk | 11/325 | 176 | 16 | 0/16 | 0 | 10.461 | 7.247 | 0.001 | 2.54 | 17.715 | 0.056 |
| synthetic_60pct | DeDoDe-v2-L | 512 | spatial_quotas | 11/314 | 195 | 14 | 0/16 | 0 | 10.461 | 7.247 | 0.002 | 2.97 | 17.716 | 0.056 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | global_topk | 36/636 | 381 | 51 | 0/16 | 0 | 10.461 | 7.247 | 0.007 | 4.78 | 17.724 | 0.056 |
| synthetic_60pct | DeDoDe-v2-L | 1024 | spatial_quotas | 41/646 | 377 | 56 | 0/16 | 0 | 10.461 | 7.247 | 0.008 | 5.05 | 17.725 | 0.056 |
| synthetic_60pct | DIC-Tensor-WGSL | 512 | global_topk | 1/332 | 188 | 4 | 0/16 | 0 | 0.049 | 6.994 | 0.012 | 3.04 | 7.062 | 0.142 |
| synthetic_60pct | DIC-Tensor-WGSL | 512 | spatial_quotas | 1/322 | 196 | 3 | 0/16 | 0 | 0.049 | 6.994 | 0.013 | 3.62 | 7.064 | 0.142 |
| synthetic_60pct | DIC-Tensor-WGSL | 1024 | global_topk | 2/667 | 383 | 20 | 0/16 | 0 | 0.049 | 6.994 | 0.026 | 4.06 | 7.077 | 0.141 |
| synthetic_60pct | DIC-Tensor-WGSL | 1024 | spatial_quotas | 3/647 | 382 | 22 | 0/16 | 0 | 0.049 | 6.994 | 0.014 | 4.59 | 7.066 | 0.142 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 512 | global_topk | 26/327 | 194 | 58 | 1/16 | 1 | 0.078 | 7.152 | 0.004 | 3.62 | 7.241 | 0.138 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 512 | spatial_quotas | 25/316 | 187 | 53 | 0/16 | 0 | 0.078 | 7.152 | 0.001 | 2.36 | 7.238 | 0.138 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 1024 | global_topk | 37/443 | 310 | 91 | 2/16 | 2 | 0.078 | 7.152 | 0.003 | 3.77 | 7.241 | 0.138 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | 1024 | spatial_quotas | 37/443 | 310 | 91 | 2/16 | 2 | 0.078 | 7.152 | 0.003 | 3.33 | 7.241 | 0.138 |
| synthetic_60pct | DIC-Learned-WGSL | 512 | global_topk | 16/316 | 203 | 34 | 0/16 | 0 | 0.055 | 6.329 | 0.001 | 3.38 | 6.392 | 0.156 |
| synthetic_60pct | DIC-Learned-WGSL | 512 | spatial_quotas | 23/323 | 208 | 39 | 0/16 | 0 | 0.055 | 6.329 | 0.004 | 2.28 | 6.393 | 0.156 |
| synthetic_60pct | DIC-Learned-WGSL | 1024 | global_topk | 63/639 | 418 | 122 | 5/16 | 5 | 0.055 | 6.329 | 0.006 | 4.14 | 6.397 | 0.156 |
| synthetic_60pct | DIC-Learned-WGSL | 1024 | spatial_quotas | 69/638 | 423 | 128 | 6/16 | 6 | 0.055 | 6.329 | 0.004 | 3.85 | 6.395 | 0.156 |

Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.
