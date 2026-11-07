# DIC-Detector held-out synthetic evaluation

Adapter: Vulkan (Intel(R) Graphics (LNL)); image size 256×256; 24 held-out texture/warp pairs; budget 512; medians after 3 warmed repeats.

| Detector | Repeatability ≤1 px | ≤2 px | ≤5 px | Pair detection median | Pair FPS |
|---|---:|---:|---:|---:|---:|
| tensor | 554/3757 (14.7%) | 1246/3757 (33.2%) | 2435/3757 (64.8%) | 0.0024 s | 423.49 |
| learned | 1403/4756 (29.5%) | 2711/4756 (57.0%) | 3715/4756 (78.1%) | 0.0041 s | 246.86 |

This detector-only evaluation measures repeatability of selected locations. It does not use a descriptor and does not measure matching or IC-GN convergence.
The breakdown below separates translation, rotation, affine, smooth heterogeneous, and discontinuous motion; a mode with few held-out pairs should be treated as exploratory.

| Motion | Detector | Pairs | Repeatability ≤1 px | ≤2 px | ≤5 px | Median pair time |
|---|---|---:|---:|---:|---:|---:|
| affine | tensor | 2 | 11/253 (4.3%) | 25/253 (9.9%) | 124/253 (49.0%) | 0.0024 s |
| affine | learned | 2 | 86/359 (24.0%) | 201/359 (56.0%) | 287/359 (79.9%) | 0.0043 s |
| rigid | tensor | 8 | 123/1302 (9.4%) | 373/1302 (28.6%) | 828/1302 (63.6%) | 0.0024 s |
| rigid | learned | 8 | 426/1734 (24.6%) | 935/1734 (53.9%) | 1334/1734 (76.9%) | 0.0040 s |
| translation | tensor | 7 | 176/1170 (15.0%) | 432/1170 (36.9%) | 782/1170 (66.8%) | 0.0023 s |
| translation | learned | 7 | 412/1453 (28.4%) | 792/1453 (54.5%) | 1113/1453 (76.6%) | 0.0040 s |
| heterogeneous | tensor | 2 | 5/203 (2.5%) | 33/203 (16.3%) | 107/203 (52.7%) | 0.0023 s |
| heterogeneous | learned | 2 | 89/286 (31.1%) | 183/286 (64.0%) | 244/286 (85.3%) | 0.0040 s |
| discontinuous | tensor | 5 | 239/829 (28.8%) | 383/829 (46.2%) | 594/829 (71.7%) | 0.0024 s |
| discontinuous | learned | 5 | 390/924 (42.2%) | 600/924 (64.9%) | 737/924 (79.8%) | 0.0041 s |

Per-case metrics, motion parameters, timing repeats, and checkpoint SHA-256 are in the adjacent JSON file.
