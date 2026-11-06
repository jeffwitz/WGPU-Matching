# CPM progressive tensile qualification: frames 040, 100, and 199

This browser comparison reuses the original UHD Live-DIC tensile images from the neighboring `tests_gpuio` checkout without modifying that project. It measures DeepMatching, HORG, and CPM on the same analytically visible seed centers for each frame. CPM's returned vector may be a local-affine projection from nearby directly measured work-grid vectors; its direct support and fit diagnostics remain in the raw data.

## Results

Counts are `accepted / visible seeds`; “<5 px” and “≤2 px” count accepted vectors only. RMSE is conditional on accepted vectors. “All CPM proposals <5 px” is shown separately to make clear when the validity filter rejects geometrically accurate proposals.

| Frame | Axial strain | Method | Accepted | Accepted <5 px | Accepted ≤2 px | Accepted RMSE (px) | Median runtime (s) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 040 | 0.11% | DeepMatching | 32/32 | 32 | 30 | 1.314 | 1.308 |
|  |  | HORG | 32/32 | 32 | 32 | 0.088 | 0.144 |
|  |  | CPM-WGPU | 32/32 | 32 | 32 | 1.078 | 0.565 |
| 100 | 3.84% | DeepMatching | 28/28 | 28 | 22 | 1.625 | 1.272 |
|  |  | HORG | 28/28 | 28 | 28 | 0.204 | 0.150 |
|  |  | CPM-WGPU | 2/28 | 2 | 2 | 1.554 | 0.550 |
| 199 | 10.00% | DeepMatching | 28/28 | 28 | 27 | 1.165 | 1.212 |
|  |  | HORG | 23/28 | 23 | 21 | 1.047 | 0.148 |
|  |  | CPM-WGPU | 0/28 | 0 | 0 | — | 0.526 |

At 3.84%, all 28 CPM proposals are within 5 px and 22 are within 2 px, yet only two pass reciprocal/local-affine validity. At 10%, 13/28 unfiltered CPM proposals are within 5 px and 10/28 within 2 px, but none is accepted; the all-proposal RMSE is 168.039 px because the other vectors are gross outliers. Thus CPM's rejection rule is conservative, but the search still has a substantial false-vector problem and the raw proposal set cannot be treated as a successful initializer.

At 10%, the known affine displacement has lower HOG cost than the found candidate for 15/28 seeds at ×4 and 18/28 at ×8. This is evidence that candidate search is a major limitation in this frame, although it does not exclude cost ambiguity or descriptor-pyramid effects. Increasing the local affine radius from 64 to 128 px did not materially increase accepted coverage; the ablation is recorded in [`progressive_affine_radius_128.json`](progressive_affine_radius_128.json).

## Protocol and interpretation

Each method was run three times per frame in the same headless Chromium session. Images and WebGPU device were already loaded; each runtime includes per-call preprocessing, GPU execution, pipeline creation, synchronization, and result readback. Image decode is excluded. Chromium used Vulkan/WebGPU flags; the browser did not expose adapter identity. The result records settings, run times, visible seeds, per-seed error, and the CPM oracle diagnostics: [`three_method_progressive_040_100_199.json`](three_method_progressive_040_100_199.json).

The compared CPM settings use a 128 px DIC window but a separate 32 px rigid matching support; the work grid is 32 px, radius 1024 px, stratified initialization count 16, and local affine radius 64 px. The 040 case is a successful low-strain smoke result. Coverage collapses by frame 100 and frame 199 despite many proposals being analytically close. CPM remains a prototype and is not qualified as a reliable DIC initializer for progressively larger strain. DeepMatching and HORG results are retained unchanged.

The synthetic tensile series is an affine image warp of a speckle reference and contains no additional sensor noise. It is not evidence for heterogeneous strain fields, crack boundaries, occlusion, or plastic localization. Source and truth provenance are included in the raw record; the external checkout was read only.
