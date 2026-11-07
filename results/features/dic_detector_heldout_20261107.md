# DIC-Detector held-out synthetic evaluation

Adapter: Vulkan (Intel(R) Graphics (LNL)); image size 256×256; 24 held-out texture/warp pairs; budget 512; medians after 3 warmed repeats.

| Detector | Repeatability ≤1 px | ≤2 px | ≤5 px | Pair detection median | Pair FPS |
|---|---:|---:|---:|---:|---:|
| tensor | 402/3749 (10.7%) | 1080/3749 (28.8%) | 2295/3749 (61.2%) | 0.0023 s | 426.57 |
| learned | 1220/4768 (25.6%) | 2530/4768 (53.1%) | 3650/4768 (76.6%) | 0.0040 s | 249.39 |

This detector-only evaluation measures repeatability of selected locations. It does not use a descriptor and does not measure matching or IC-GN convergence.
Per-case metrics, motion parameters, timing repeats, and checkpoint SHA-256 are in the adjacent JSON file.
