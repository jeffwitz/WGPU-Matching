# INRIA and WebGPU DeepMatching: global dense hierarchy

Comparison of INRIA CPU v1.2.2 and WebGPU outputs on Live-DIC uniaxial tensile images. Fixture images, INRIA outputs, and analytic truth were read from `tests_gpuio`; that source repository was not modified.

## Protocol

- Pairs `reference.png` → `frame_040.png`, `frame_100.png`, and `frame_199.png`, each 3840 × 2160.
- 32 requested bins per frame: `x = 63.5 + 512 i`, `y = 63.5 + 512 j`, `i=0…7`, `j=0…3`. WebGPU projects each center onto the 16 px atomic source grid; INRIA is compared at the same bin.
- WebGPU: pixelwise HOG, ×4 average pooling, 16 px atomic patch, 32 px DIC window, 256 px radius, and a 512 px root hierarchy over 6 levels. It computes the full 40-root INRIA grid (8 × 5), then global reciprocal consistency; only the 32 requested positions are exported.
- INRIA: CPU executable v1.2.2 with `-downscale 2 -ngh_rad 256`, default overlap, layers, and scoring options. Its `max_psize=999` builds the 512 px root level here; roots are spaced 512 px apart. Dense maps are compared at the same atomic source center, without nearest-neighbor search.
- Analytic truth: `du=(stretch_x−1)(x−1920)`, `dv=(stretch_y−1)(y−1080)`, evaluated at the atomic source center.

## Results

| Frame | INRIA / WebGPU matches out of 32 | Analytic error, INRIA = WebGPU (RMSE / median / P95 / max, px) | Matches ≤2 px, INRIA / WebGPU | INRIA time, 1 thread | INRIA time, 8 threads | WebGPU time |
|---:|---:|---:|---:|---:|---:|---:|
| 040 | 32/32 · 32/32 | 1.314 / 1.177 / 2.002 / 2.102 | 30/32 (93.8%) · 30/32 (93.8%) | 33.93 s | 8.87 s | 1.45 s |
| 100 | 28/32 · 28/32 | 1.625 / 1.530 / 2.332 / 2.466 | 22/28 (78.6%) · 22/28 (78.6%) | 33.32 s | 9.26 s | 1.32 s |
| 199 | 28/32 · 28/32 | 1.165 / 0.893 / 1.787 / 2.433 | 27/28 (96.4%) · 27/28 (96.4%) | 33.37 s | 11.07 s | 1.30 s |
| **Mean / total** | **88/96 · 88/96** | **1.378 / 1.356 / 2.094 / 2.466** | **79/88 (89.8%) · 79/88 (89.8%)** | **33.54 s** | **9.73 s** | **1.36 s** |

Vector error relative to analytic truth is `sqrt((dx − du)² + (dy − dv)²)` in pixels; the table reports RMSE, median, P95, and maximum among found matches. Precision is the fraction of present/reciprocal matches with error ≤2 px. For the 88 bins returned by both methods, all 88 displacement vectors are identical at pixel precision (WebGPU–INRIA RMSE = 0 px), so both implementations have identical error statistics on this dataset. Each finds a match at 88 of the 96 requested positions, and 79 of those 88 vectors are within 2 px of analytic truth. Matches above this threshold remain in the count and are not hidden.

The calculation covers all non-overlapping roots at the top level and combines the best candidates toward source and target before reciprocal checking. Thus it reproduces the dense correspondence field at the configured root level for the tested setup without rotation or scale variation. Application output is then sampled at the requested seeds. CPU rectification of incomplete border patches (`rectify_conv`) and rotation/scale-invariant variants are not yet ported.

INRIA timings were measured on CPU binary v1.2.2 with `-downscale 2 -ngh_rad 256`, one run per frame: one thread by default, then `-nt 0` for the eight available cores. CLI timing includes PNG loading and writing outputs to `/dev/null`. WebGPU timing includes HOG preparation, pipeline creation, and all 40 grid roots; PNG decoding and grayscale conversion are excluded. Measured November 5, 2026, with Chromium headless 156, WebGPU/Vulkan, and Intel `xe-2lpg`. On average, WebGPU is about 24.7× faster than single-threaded INRIA and 7.2× faster than eight-thread INRIA. Timing boundaries are not strictly identical, so these ratios are indicative.

Recursive argmax descent runs on the GPU; only candidate coordinates and scores are read back instead of complete dense maps. Two roots are encoded per submission. The timings do not separately compare the precomputed INRIA CPU field.

## Data

- `inria_webgpu_hierarchy_comparison.csv`: atomic centers, winning roots, vectors, reciprocity, INRIA differences, and per-seed analytic errors.
- `inria_webgpu_patch_comparison.md` and its CSV: historical results from the earlier direct-correlation implementation.
