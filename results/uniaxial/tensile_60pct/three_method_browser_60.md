# Three-method comparison at 60% tensile strain

This report uses the existing synthetic UHD tensile case, frame 199 at 60.000% engineering strain. Sixteen of the original 32 DIC seeds remain inside the target image. The transform is `stretch_x=1.6000`, `stretch_y=0.7910266`; material behavior above 10% is an unvalidated synthetic extrapolation, as documented in the [dataset report](README.md).

DeepMatching and HORG values were already recorded in `matcher_comparison.csv`. I reran all three methods in one Chromium/Vulkan session on the same 16 visible seeds to get comparable timings. The accepted DeepMatching and HORG quality counts match the existing Python benchmark. The raw per-seed browser run is in [the JSON data](three_method_browser_60.json), and the compact table is in [the CSV summary](three_method_browser_60.csv).

| Method | Accepted | Accepted with error <5 px | Repairable coverage (<5 px, /16) | Precision among accepted (<5 px) | RMSE of accepted matches <5 px | Strict ≤2 px (/16) | Median runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepMatching INRIA | 6/16 | 5/6 | 31.25% | 83.3% | 2.078 px | 2/16 | 1.308 s |
| HORG | 0/16 | 0 | 0% | — | — | 0/16 | 0.154 s |
| CPM-WGPU | 0/16 | 0 | 0% | — | — | 0/16 | 1.611 s |

Following the DIC use case, this comparison treats an accepted seed error strictly below 5 px as repairable by subsequent DIC refinement. Repairable coverage divides those accepted matches by all 16 visible seeds; precision is the repairable fraction of accepted matches. The stricter ≤2 px count is retained for reference. DeepMatching's six accepted matches include one 361.080 px outlier, so its RMSE over all accepted matches is 147.423 px even though five are below 5 px. The RMSE of those five repairable matches is 2.078 px. The browser timings are medians of three full matcher calls with decoded pixels and one WebGPU device already available. Each call includes descriptor preparation, pipeline creation, GPU matching, synchronization, and output readback. Image loading/decoding is excluded. Runs were deterministic across all three repeats for each method.

## CPM outcome

The CPM-HOG settings were: 128 px DIC patches, 512 px requested-seed spacing, 32 px internal work-grid spacing, four effective pyramid levels, six propagation passes per level, 16 random trials per pass, 1024 px global search radius, 64 px local refinement radius, seed 17, and an 8 px forward/backward threshold. All 16 CPM vectors failed the reciprocity acceptance check. Their diagnostic errors against analytic truth had RMSE 691.960 px, median 601.412 px, P95 1053.759 px, maximum 1084.121 px, and minimum 9.440 px. These rejected vectors are not counted as matches; the accepted-match RMSE is undefined because there are no accepted CPM results.

This shows the current CPM prototype does not initialize the 60% case reliably: none of its vectors are accepted, and none of the rejected proposals is even within 5 px of analytic truth. The large-displacement coarse search is falling into poor matches before fine-level refinement. Its median 1.611 s runtime is also slower than the 1.308 s DeepMatching browser run under this configuration. Do not use these CPM settings for IC-GN initialization yet.

## Protocol and limits

- Browser: Headless Chromium 156; WebGPU enabled with the same Vulkan/ignore-blocklist flags used by the `tests_gpuio` browser server.
- Vulkan reports the host devices as Intel Graphics (LNL), Mesa 25.2.8, and llvmpipe. Chromium did not expose `GPUAdapterInfo`, so the exact adapter selected by the page could not be confirmed independently.
- DeepMatching: 512 px INRIA root, ×4 descriptor reduction, 256 px search radius. HORG: ×4 reduction and 1024 features per image. Their quality decisions were not changed.
- CPM’s larger 1024 px radius is required to cover the 60% analytic displacement range. The CPM implementation and its settings are experimental; no claim of parity with original CPM is made.
- The prior global affine-RANSAC/ECC preconditioning experiment is a separate DeepMatching pipeline variant, not one of these three standalone matchers. Its prior result (16/16 accepted, 13/16 within 2 px, 1.394 px accepted RMSE, 3.681 s single run) is documented in [`affine_preconditioning.md`](affine_preconditioning.md); it relies on the synthetic case’s global affine structure and should not be conflated with standalone matcher performance.
