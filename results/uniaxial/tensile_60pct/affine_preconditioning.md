# DeepMatching affine compensation

Experiment: `python/benchmarks/experiment_affine_deepmatching.py`; use `--ecc` for photometric refinement. Raw outputs: `affine_deepmatching_experiment.json` and `affine_ecc_deepmatching_experiment.json`.

The first WGPU DeepMatching pass uses a denser grid (128 px stride) over the full image. RANSAC tests 10,000 deterministic triplets (configurable seed, default 42) and scores hypotheses on WGPU with a 4 px threshold; selecting the best consensus and refitting least squares also run on the GPU. The target is warped back into the reference coordinate system. ECC then optimizes an affine transform on the central half of each dimension, selected independently of analytic truth. A second DeepMatching pass measures the residual; its target coordinates are composed with the affine transform to recover the original displacement.

ECC accumulates Gauss–Newton equations in WGSL over the image, with free photometric gain and bias, reduces the sums, and solves the 8 × 8 system on the GPU. This optimizes correlation while allowing a linear intensity transform. Python drives the iterations and updates parameters from the eight increments read back each round. On these images, the practical stopping rule limits optimization to 5–6 iterations.

Analytic truth is used only for the visibility mask and for calculating errors after matching. No analytic deformation parameters are supplied to affine estimation, ECC, or DeepMatching. Final queries cover the usual grid; only the same visible seeds are evaluated.

## Results at 60%

| Variant | Accepted / 16 visible | Error ≤2 px | Accepted RMSE | Time |
|---|---:|---:|---:|---:|
| Initial DeepMatching, previous benchmark | 6 | 2 | 147.423 px | ≈1.82 s |
| Dense DeepMatching + GPU affine RANSAC + second DeepMatching | 16 | 13 | 1.394 px | 3.681 s |
| Pipeline with WGSL RANSAC and ECC | 16 | 16 | 0.000854 px | 4.107 s |

At 60%, the first pass produces 178 reciprocal correspondences and affine consensus retains 63 inliers. After ECC, the affine transform is approximately `[[1.59999955, 0.00000105, -1152.00024], [0.00000020, 0.79102582, 225.69176]]`.

## Checks at other strain levels

| Strain | Visible | Accepted with ECC | ≤2 px with ECC | Accepted RMSE | Total time |
|---|---:|---:|---:|---:|---:|
| 9.90% | 28 | 24 | 24 | 0.000803 px | 4.111 s |
| 20.07% | 24 | 24 | 24 | 0.000952 px | 3.743 s |
| 29.87% | 24 | 24 | 24 | 0.001054 px | 3.794 s |
| 40.04% | 24 | 20 | 20 | 0.000914 px | 3.808 s |
| 49.83% | 20 | 20 | 20 | 0.000718 px | 3.908 s |
| 60.00% | 16 | 16 | 16 | 0.000854 px | 4.107 s |

This prototype exploits the exact global affine field in the synthetic dataset. Subpixel accuracy comes mainly from the affine model refined by ECC; it does not demonstrate equivalent local DeepMatching accuracy on heterogeneous fields. Rejections at 10% and 40% still need analysis. Remapping uses constant borders; a production version must propagate a valid support mask.

Affine remapping is implemented as a bilinear WGSL kernel in `python/src/wgpu_matching/shaders/affine-warp.wgsl`. Compared with `cv2.warpAffine` on the UHD image at 60%, 99.964% of pixels are identical and the maximum difference is one grayscale level. The RANSAC/ECC-WGSL pipeline keeps 16/16 points within 2 px, with 0.000854 px RMSE.

Single-pass timings per level, with images and device already loaded, on Intel Graphics LNL / Vulkan. At 60%, measured phases are: dense DeepMatching 1.893 s, GPU RANSAC 0.0062 s, GPU ECC 0.323 s (6 iterations), remapping 0.048 s, second DeepMatching 1.836 s; total 4.107 s. The historical CPU RANSAC/ECC comparison was 4.112 s at this level: times are equivalent within measurement precision, with no speed gain established. These are single runs, not medians. Subpixel quality is confirmed on this synthetic dataset, but optimization and validation on heterogeneous fields with occlusions remain outstanding.

Next steps: compare repeated runs with GPU-event timing; reduce readback of eight increments per iteration; validate on heterogeneous fields and occlusions with a support mask; and add an independent photometric check before applying the global affine transform.
