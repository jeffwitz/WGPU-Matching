# HORG-DIC vs. DeepMatching on the masked ROI

> **Historical results:** the DeepMatching measurements below use the earlier simplified descriptor computed directly on the reduced image. They do not describe the later pixelwise HOG descriptor pipeline; rerun the benchmark before drawing conclusions about current timing or match counts.

Measured November 5, 2026, on `img_000000_0.09422.tiff` and `img_000075_1608.88508.tiff`. `Mask.tiff` defines the ROI `x=[384,7576)`, `y=[888,5088)`, which is `7192 × 4200` pixels. Both methods ran in the same Chromium 148 headless session with WebGPU/Vulkan; the browser reported an Intel adapter, architecture `xe-2lpg`.

## Settings

- DIC grid: 128 px window, 1024 px stride, 28 POIs.
- DeepMatching: ×8 reduction, score threshold 0.12, uniqueness margin 0.01.
- HORG: ×4 reduction, maximum 2048 descriptors per image, Lowe ratio 0.75. The RANSAC threshold of 1.5 px in reduced coordinates corresponds to 6 px in the original image. An affine is retained with at least four inliers, eight local candidates, and 20% inliers.

## Observed processing time

Each method ran three times after loading the images. Time includes extraction and initialization; for HORG it also includes local RANSAC on the JavaScript CPU.

| Method | Three run times | Median | Difference from DeepMatching |
|---|---:|---:|---:|
| DeepMatching WebGPU | 0.41 · 0.43 · 0.42 s | 0.42 s | — |
| HORG-DIC WebGPU | 0.55 · 0.52 · 0.51 s | 0.52 s | +0.10 s, or +23.8% |

## Results on this image pair

| Method | Correspondences / seeds | POIs with retained initialization |
|---|---:|---:|
| DeepMatching | 25 of 28 seeds pass score and uniqueness thresholds | 25/28 |
| HORG-DIC | 131 pairs pass the Lowe ratio | 0/28 local affines pass RANSAC consensus |

Passing DeepMatching thresholds is not validation against ground truth. HORG detects 8,069 and 8,315 keypoints in the two images, but retained pairs do not provide enough local consensus here. The PNG shows all 28 rejected POIs without drawing outlier affine estimates. There is no usable HORG field for this pair despite a processing time close to DeepMatching.

These timings compare useful settings on the same grid and ROI, not strictly equal amounts of computation: the reduction factors differ and HORG compares 2,048 descriptors per image. The implementation stops at initialization and does not run IC-GN refinement.
