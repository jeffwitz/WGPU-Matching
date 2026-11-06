# Current WebGPU comparison: DeepMatching and HORG-DIC

Comparison of the two initializers on the Live-DIC uniaxial tensile test, which has known analytic ground truth. DeepMatching is unchanged. HORG uses 3D interpolation of DoG extrema, subpixel-sampled descriptors, floating-point affine displacements, Lowe ratio filtering and GPU candidate preparation, GPU affine RANSAC, and GPU leave-one-out affine consistency with eight neighbors.

## Protocol

- UHD images, 3840 × 2160: `reference.png` and frames 040, 100, and 199 from `uhd_tensile`.
- The same logical grid of 32 POIs, 8 × 4, spaced 512 px apart and centered at `(63.5 + 512 i, 63.5 + 512 j)`.
- DeepMatching: 32 px window, ×4 reduction, 256 px search radius, 512 px roots; all 40 roots are computed, then global reciprocal consistency is checked. Centers are projected onto the 16 px atomic grid.
- HORG-DIC: local 128 px window, ×4 reduction, 512 px stride, up to 1024 descriptors per image, Lowe ratio filtering, and local affine RANSAC verification. Its window center coincides with the requested logical center. The new localization retains 693 reference keypoints and 685, 699, and 735 in targets 040, 100, and 199. Eight-neighbor leave-one-out values are exported as diagnostics and do not affect acceptance.
- Timing: images are decoded before timing; one warm-up call followed by the median of three calls per frame. HORG includes extraction, kNN + ratio + WGSL candidate preparation, WGSL RANSAC + LOO diagnostics, and result reads. The matching result contains only diagnostic ratios, not descriptors or kNN pairs. Measured on November 5, 2026, using Chromium headless 156 and WebGPU/Vulkan; the session API did not expose the adapter identifier. Modified or added HORG kernels compile without errors.
- Analytic truth: `du = (sx − 1) (x − 1920)`, `dv = (sy − 1) (y − 1080)`. Vector error is `sqrt((dx − du)² + (dy − dv)²)` at the source center. RMSE includes accepted initializations only; “≤2 px” counts accepted results within that threshold.

## Error results

| Frame | DeepMatching accepted / 32 | DeepMatching ≤2 px among accepted | DeepMatching RMSE | HORG subpixel accepted / 32 | HORG ≤2 px among accepted | HORG RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 040 | 32/32 | 30/32 (93.8%) | 1.314 px | 32/32 | 32/32 (100%) | 0.088 px |
| 100 | 28/32 | 22/28 (78.6%) | 1.625 px | 32/32 | 32/32 (100%) | 0.216 px |
| 199 | 28/32 | 27/28 (96.4%) | 1.165 px | 23/32 | 21/23 (91.3%) | 1.047 px |
| **Total** | **88/96** | **79/88 (89.8%)** | **1.378 px** | **87/96** | **85/87 (97.7%)** | **0.557 px** |

Across these 96 seeds, subpixel HORG produces 85 accepted initializations within 2 px, compared with 79 for DeepMatching, and six more accepted seeds overall. At frame 199, HORG coverage rises from 5 to 23 accepted POIs compared with the previous HORG measurement; two of the 23 exceed 2 px. The methods have different acceptance criteria. At frame 199, DeepMatching still retains more accurate seeds (27 versus 21).

The eight-neighbor leave-one-out diagnostic is available for all 87/87 accepted POIs and places 85/87 below 1.5 px across these three frames. It is not used as a filter: on 18 holdout frames, the same threshold still passes seven false positives and rejects six correct POIs ([detailed analysis](horg_quality_indicators.md)).

Compared with the previous HORG revision, accepted points increase from 62 to 87 and results within 2 px increase from 61 to 85. Pooled RMSE falls from 0.952 to 0.557 px. Precision among accepted results decreases slightly from 98.4% to 97.7% because the method now accepts more difficult cases.

Per-seed results with analytic displacement and residual are in the [frame 040 CSV](frame_040_horg_subpixel_analytic_error.csv), [frame 100 CSV](frame_100_horg_subpixel_analytic_error.csv), and [frame 199 CSV](frame_199_horg_subpixel_analytic_error.csv).

An exploratory false-positive screen examines RANSAC support and leave-one-out affine consistency with the eight neighboring POIs. The consistency check identifies some errors, but not enough to act as a standalone filter; the expanded protocol and limitations are detailed in the [confidence-indicator study](horg_quality_indicators.md).

## Processing time

| Frame | DeepMatching median | HORG subpixel median | HORG speedup |
|---:|---:|---:|---:|
| 040 | 1.335 s | 0.153 s | 8.7× |
| 100 | 1.333 s | 0.149 s | 8.9× |
| 199 | 1.290 s | 0.150 s | 8.6× |
| **Mean of medians** | **1.319 s** | **0.151 s** | **8.8×** |

RANSAC + LOO kernels take median times of 7.5 ms, 7.6 ms, and 2.7 ms. The full HORG pipeline has a mean median time of 0.151 s, close to the previous revision, with the same displacements and coverage; Lowe ratio filtering and candidate preparation now run on the GPU. Under this protocol, HORG takes about 8.8 times less time than DeepMatching.

## Raw measurements

[`deepmatching_horg_current_comparison.csv`](deepmatching_horg_current_comparison.csv) contains aggregates, the three timings per frame, and median GPU RANSAC + LOO time. Per-POI results are provided in the three subpixel CSVs linked above. Historical results before subpixel localization remain in the older CSVs `frame_040|100|199_horg_analytic_error.csv`.
