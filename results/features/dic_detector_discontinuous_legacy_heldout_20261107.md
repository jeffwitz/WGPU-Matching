# DIC-Detector held-out synthetic evaluation

Adapter: Vulkan (Intel(R) Graphics (LNL)); image size 256×256; 24 held-out texture/warp pairs; budget 512; medians after 3 warmed repeats.

| Detector | Repeatability ≤1 px | ≤2 px | ≤5 px | Pair detection median | Pair FPS |
|---|---:|---:|---:|---:|---:|
| tensor | 554/3757 (14.7%) | 1246/3757 (33.2%) | 2435/3757 (64.8%) | 0.0024 s | 417.15 |
| learned | 1319/4816 (27.4%) | 2608/4816 (54.2%) | 3686/4816 (76.5%) | 0.0040 s | 247.17 |

This detector-only evaluation measures repeatability of selected locations. It does not use a descriptor and does not measure matching or IC-GN convergence.
The breakdown below separates translation, rotation, affine, smooth heterogeneous, and discontinuous motion; a mode with few held-out pairs should be treated as exploratory.

| Motion | Detector | Pairs | Repeatability ≤1 px | ≤2 px | ≤5 px | Median pair time |
|---|---|---:|---:|---:|---:|---:|
| affine | tensor | 2 | 11/253 (4.3%) | 25/253 (9.9%) | 124/253 (49.0%) | 0.0027 s |
| affine | learned | 2 | 84/365 (23.0%) | 191/365 (52.3%) | 283/365 (77.5%) | 0.0043 s |
| rigid | tensor | 8 | 123/1302 (9.4%) | 373/1302 (28.6%) | 828/1302 (63.6%) | 0.0024 s |
| rigid | learned | 8 | 356/1751 (20.3%) | 899/1751 (51.3%) | 1327/1751 (75.8%) | 0.0041 s |
| translation | tensor | 7 | 176/1170 (15.0%) | 432/1170 (36.9%) | 782/1170 (66.8%) | 0.0024 s |
| translation | learned | 7 | 386/1462 (26.4%) | 740/1462 (50.6%) | 1088/1462 (74.4%) | 0.0041 s |
| heterogeneous | tensor | 2 | 5/203 (2.5%) | 33/203 (16.3%) | 107/203 (52.7%) | 0.0023 s |
| heterogeneous | learned | 2 | 87/284 (30.6%) | 165/284 (58.1%) | 233/284 (82.0%) | 0.0041 s |
| discontinuous | tensor | 5 | 239/829 (28.8%) | 383/829 (46.2%) | 594/829 (71.7%) | 0.0023 s |
| discontinuous | learned | 5 | 406/954 (42.6%) | 613/954 (64.3%) | 755/954 (79.1%) | 0.0040 s |

Per-case metrics, motion parameters, timing repeats, and checkpoint SHA-256 are in the adjacent JSON file.
