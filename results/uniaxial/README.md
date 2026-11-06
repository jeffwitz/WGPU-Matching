# Initializer comparison on the UHD uniaxial tensile test

> **Historical results:** these measurements use earlier matcher versions. The current DeepMatching/HORG accuracy and timing comparison is in the [current report](deepmatching_horg_current_comparison.md) and [its measurement CSV](deepmatching_horg_current_comparison.csv). The current comparison of DeepMatching displacements with exact INRIA outputs is in the [hierarchical report](inria_webgpu_hierarchy_comparison.md) and [its per-seed CSV](inria_webgpu_hierarchy_comparison.csv).

The earlier direct full-patch correlation is documented in [its archived report](inria_webgpu_patch_comparison.md) and [associated CSV](inria_webgpu_patch_comparison.csv). Historical frame-by-frame files below are retained for provenance.

## Data and analytic truth

Images come from `tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile`: a 3840 × 2160 reference at frame 0 and target images at frames 40, 100, and 199. This is the archived virtual Live-DIC dataset, generated before video encoding. The source repository was read without modification.

The generator deforms the reference around `(1920,1080)`. Given stretches `sx`, `sy` in `machine_truth.csv`, the reference-coordinate displacement field is:

```text
du = (sx - 1) * (x - 1920)
dv = (sy - 1) * (y - 1080)
```

Coordinates are zero-based pixel centers, following the dataset convention. Target images are resampled with bilinear interpolation and quantized to 8 bits; the kinematic field is analytic, while pixels remain discrete samples.

## Results

Each method returns up to 32 POIs on an 8 × 4 grid. “Accurate ≤2 px” counts returned POIs whose vector error relative to the analytic field is at most 2 px, out of the full 32 POIs. RMSE is calculated only for accepted initializations. Timings are medians of three calls to the initialization function, with images already loaded.

| Frame | Axial strain | Method | Accepted initializations | Accurate ≤2 px | Accepted RMSE | Median time |
|---:|---:|---|---:|---:|---:|---:|
| 40 | 0.112% | DeepMatching | 21/32 | 21/32 (65.6%) | 1.175 px | 0.14 s |
| 40 | 0.112% | HORG | 31/32 | 31/32 (96.9%) | 0.872 px | 0.20 s |
| 100 | 3.843% | DeepMatching | 21/32 | 1/32 (3.1%) | 17.811 px | 0.14 s |
| 100 | 3.843% | HORG | 26/32 | 26/32 (81.3%) | 0.725 px | 0.18 s |
| 199 | 10.000% | DeepMatching | 27/32 | 1/32 (3.1%) | 881.356 px | 0.14 s |
| 199 | 10.000% | HORG | 5/32 | 4/32 (12.5%) | 1.947 px | 0.17 s |

At 10% deformation, DeepMatching score/uniqueness thresholds pass three seeds with errors above 2500 px. HORG is much more selective: coverage drops, but its five retained affine matches remain close to the exact field (P95 = 4.05 px). Coverage and accuracy among retained seeds must therefore be considered separately.

HORG finds 265, 116, and 49 pairs passing the Lowe ratio for frames 40, 100, and 199. It detects 887 keypoints in the reference; the three targets contain 887, 935, and 979. The number of consistent pairs falls as strain increases, explaining the loss of coverage at 10%.

The [analytical comparison figure](analytical_comparison.png) plots coverage within 2 px, RMSE of retained seeds, and timings. Source CSVs and analytic-error columns are retained frame by frame in this directory.

## Settings and comparison limits

- DIC window 128 px, stride 512 px, same grid and images for both methods.
- DeepMatching: ×8 reduction, score threshold 0.12, and minimum uniqueness margin 0.01.
- HORG: ×4 reduction, maximum 2048 descriptors per image, Lowe ratio 0.75; local affine RANSAC requires at least 4 inliers, 8 candidates, and 20% inliers. Its 1.5 px threshold in the reduced image corresponds to 6 px in original coordinates.
- HORG timing includes local RANSAC on the JavaScript CPU. Both GPU pipelines use WebGPU in headless Chromium; the adapter is Intel `xe-2lpg`.
- Acceptance criteria differ and images were synthesized with resampling; rates describe initialization on this known dataset, not generalization to physical experiments.

## Result files

- `frame_040|100|199_deepmatching.csv` and `frame_040|100|199_horg.csv`: raw application outputs.
- `frame_040|100|199_deepmatching_analytic_error.csv` and `frame_040|100|199_horg_analytic_error.csv`: analytic displacement and per-POI error.
- `summary.csv`: coverage, RMSE, and median time by method and frame.
