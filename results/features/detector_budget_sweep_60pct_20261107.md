# Shared-descriptor detector budget sweep at 60% tensile strain

This experiment covers the same 3840×2160 synthetic global-affine 60% tensile pair at budgets 512, 1024, 2048, 4096, and 8192. Each method receives a candidate pool twice the requested budget; the table uses global top-K from that returned pool. The complete reports also include spatial-quota selection.

All 16 analytic seed centres are visible. Learned DaD/DeDoDe inference, DeDoDe-B description, and mutual-cosine matching run on CPU; HORG/DIC detection and affine RANSAC run on WGSL through Vulkan (Intel(R) Graphics (LNL)). These timings include CPU descriptor/matching and WGSL RANSAC and are not GPU-resident pipeline times. DIC's detector candidate pool is produced with its internal 16×9 spatial quotas, so global top-K is not over the full response map.

| Detector | Budget | Raw mutual matches | Raw correct <5 px | Accepted affine | Correct ≤2 px | Correct <5 px | Accepted RMSE/P95 px | Total s | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DaD | 512 | 269 | 200 | 9/16 | 9/16 | 9/16 | 1.11/1.60 | 10.916 | 0.092 |
| DaD | 1024 | 550 | 422 | 16/16 | 15/16 | 16/16 | 1.09/1.80 | 10.919 | 0.092 |
| DaD | 2048 | 1089 | 826 | 16/16 | 16/16 | 16/16 | 0.75/1.36 | 9.472 | 0.106 |
| DaD | 4096 | 2219 | 1640 | 16/16 | 16/16 | 16/16 | 0.79/1.33 | 9.533 | 0.105 |
| DaD | 8192 | 4339 | 3142 | 16/16 | 16/16 | 16/16 | 0.46/0.81 | 9.922 | 0.101 |
| DeDoDe-v2-L | 512 | 176 | 16 | 0/16 | 0/16 | 0/16 | —/— | 16.540 | 0.060 |
| DeDoDe-v2-L | 1024 | 381 | 51 | 0/16 | 0/16 | 0/16 | —/— | 16.539 | 0.060 |
| DeDoDe-v2-L | 2048 | 773 | 162 | 3/16 | 3/16 | 3/16 | 1.17/1.37 | 16.640 | 0.060 |
| DeDoDe-v2-L | 4096 | 1560 | 476 | 15/16 | 15/16 | 15/16 | 0.91/1.39 | 16.871 | 0.059 |
| DeDoDe-v2-L | 8192 | 3321 | 1416 | 16/16 | 16/16 | 16/16 | 0.42/0.70 | 17.137 | 0.058 |
| DIC-Tensor-WGSL | 512 | 188 | 4 | 0/16 | 0/16 | 0/16 | —/— | 7.100 | 0.141 |
| DIC-Tensor-WGSL | 1024 | 383 | 20 | 0/16 | 0/16 | 0/16 | —/— | 7.116 | 0.141 |
| DIC-Tensor-WGSL | 2048 | 786 | 73 | 0/16 | 0/16 | 0/16 | —/— | 7.196 | 0.139 |
| DIC-Tensor-WGSL | 4096 | 1601 | 271 | 6/16 | 1/16 | 4/16 | 4.43/6.06 | 7.386 | 0.135 |
| DIC-Tensor-WGSL | 8192 | 3269 | 950 | 7/16 | 3/16 | 5/16 | 3.57/5.63 | 7.593 | 0.132 |
| HORG-SIFT-common-DeDoDe-B | 512 | 194 | 58 | 1/16 | 1/16 | 1/16 | 0.68/0.68 | 7.047 | 0.142 |
| HORG-SIFT-common-DeDoDe-B | 1024 | 310 | 91 | 2/16 | 1/16 | 2/16 | 2.98/3.99 | 7.049 | 0.142 |
| HORG-SIFT-common-DeDoDe-B | 2048 | 310 | 91 | 2/16 | 1/16 | 2/16 | 2.98/3.99 | 6.765 | 0.148 |
| HORG-SIFT-common-DeDoDe-B | 4096 | 310 | 91 | 2/16 | 1/16 | 2/16 | 2.98/3.99 | 6.769 | 0.148 |
| HORG-SIFT-common-DeDoDe-B | 8192 | 310 | 91 | 2/16 | 1/16 | 2/16 | 2.98/3.99 | 6.782 | 0.147 |
| DIC-Learned-WGSL | 512 | 190 | 39 | 0/16 | 0/16 | 0/16 | —/— | 6.953 | 0.144 |
| DIC-Learned-WGSL | 1024 | 425 | 185 | 8/16 | 7/16 | 8/16 | 1.77/3.36 | 6.952 | 0.144 |
| DIC-Learned-WGSL | 2048 | 1010 | 662 | 15/16 | 13/16 | 15/16 | 1.83/3.86 | 6.453 | 0.155 |
| DIC-Learned-WGSL | 4096 | 2109 | 1486 | 16/16 | 16/16 | 16/16 | 0.70/1.37 | 6.649 | 0.150 |
| DIC-Learned-WGSL | 8192 | 4320 | 3204 | 16/16 | 16/16 | 16/16 | 0.38/0.61 | 7.006 | 0.143 |

DaD goes from 9/16 accepted seeds at budget 512 to 16/16 by 1024; at 8192 it has 0.46 px accepted RMSE and 0.81 px P95. DeDoDe v2-L improves from 3/16 at 2048 to 16/16 at 8192, but takes about 17.14 s/pair. The discontinuity-aware DIC score model + DeDoDe-B reaches 16/16 accepted, 16/16 within 2 px, and 16/16 within 5 px from budget 4096; at 8192 its accepted RMSE/P95 is 0.38/0.61 px at 7.01 s/pair. HORG/SIFT goes from 1/16 at 512 to 2/16 from 1024 onward, and the tensor-only DIC baseline reaches 7/16 accepted at 8192 with 5 correct under 5 px.

These results show the value of increasing candidate count for this particular analytically affine stress image; they do not establish performance on heterogeneous strain, experimental speckles, cracks, or IC-GN convergence. The descriptor and mutual matching stages are CPU references, and the keypoint repeatability denominators vary with the selected points. Do not treat accepted coverage alone as correctness; the analytic-error columns show both.

Full low-budget data: `detector_common_descriptor_60pct_budgets_low_20261107.json`. Full high-budget data: `detector_common_descriptor_60pct_budgets_20261107.json`.
