# INRIA / WebGPU DeepMatching comparison on the uniaxial tensile test

> Archive of the earlier direct patch-correlation implementation. It does not describe the current hierarchical implementation; measurements comparable with INRIA are now in the [hierarchical report](inria_webgpu_hierarchy_comparison.md).

This historical measurement compares displacements on three Live-DIC frames with known analytic truth. CSV files are retained to track implementation changes.

## Protocol

- Images `reference.png` to `frame_040.png`, `frame_100.png`, and `frame_199.png` from `uhd_tensile` (3840 × 2160), read from `tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile` without modifying it.
- Same grid of 32 centers for WebGPU and INRIA: `x = 63.5 + 512 i`, `y = 63.5 + 512 j`, with `i = 0…7` and `j = 0…3`.
- WebGPU: 128 px window, ×8 reduction, 256 px radius, minimum score 0.12, and minimum uniqueness margin 0.04. Direct correlation of the full patch, with parabolic peak interpolation; maxima at the search-radius boundary and ambiguous scores are rejected.
- INRIA: DeepMatching CPU v1.2.2, `-downscale 2`, `-ngh_rad 256`, dense outputs. For each seed, compare with the nearest official correspondence within 16 px.
- Analytic field: `du = (sx − 1)(x − 1920)`, `dv = (sy − 1)(y − 1080)`, using stretches from `machine_truth.csv`.

Analytic RMSE and WebGPU/INRIA differences are calculated on the seeds retained by WebGPU. Selection criteria differ: INRIA outputs are unfiltered, and spatial alignment uses the nearest official correspondence. The direct WebGPU search does not implement the complete INRIA hierarchical aggregation or top-down traversal.

## Results

| Frame | WebGPU seeds retained | WebGPU analytic error (RMSE; ≤2 px) | INRIA on same seeds (RMSE; ≤2 px) | WebGPU–INRIA vector difference (RMSE; ≤2 px) | WebGPU time |
|---:|---:|---:|---:|---:|---:|
| 40 | 21/32 | 0.71 px; 21/21 | 1.19 px; 20/21 | 0.78 px; 20/21 | 0.24 s |
| 100 | 27/32 | 1.14 px; 26/27 | 1.55 px; 24/27 | 1.85 px; 16/27 | 0.18 s |
| 199 | 23/32 | 0.77 px; 23/23 | 1.69 px; 18/23 | 1.68 px; 18/23 | 0.18 s |

The retained WebGPU field is therefore close to the INRIA field on this test, with vector-difference RMSE from 0.78 to 1.85 px depending on the frame. At 10% strain, 23 of 32 seeds are retained, all within 2 px of analytic truth. The margin and radius-boundary checks reject unreliable edge seeds.

WebGPU timings are from a headless Chromium/WebGPU run on the integrated Intel adapter, with images already loaded; they include descriptor generation and search. They are not a speed ratio against INRIA: INRIA returns a dense field at ×2 reduction, while this adaptation computes only 32 seeds at ×8 reduction.

## Files

- `frame_040|100|199_webgpu_direct_patch.csv`: complete WebGPU outputs, scores, margins, and `usable` flag.
- `inria_webgpu_patch_comparison.csv`: analytic truth, nearest INRIA correspondence, and per-seed errors.
