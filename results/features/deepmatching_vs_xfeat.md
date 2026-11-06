# DeepMatching vs XFeat on Live-DIC tensile images

## Result summary

This comparison evaluates the two methods as initializers at the same visible DIC seed centers. A returned initialization is counted as correct when its displacement error against the analytic affine truth is below 5 px (the stated repairable threshold). The stricter ≤2 px count is shown separately. “Accepted” means DeepMatching's own validity flag or, for XFeat, a local affine fit passing the shared GPU RANSAC criteria.

| Pair | Visible seeds | DeepMatching accepted | DeepMatching ≤2 px | DeepMatching <5 px | DM time / pair FPS | XFeat raw pairs | XFeat accepted | XFeat ≤2 px | XFeat <5 px | XFeat time / pair FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frame 040 (~0.11%) | 32 | 32 | 30 | 32 | 1.720 s / 0.58 | 3,087 | 32 | 14 | 32 | 0.724 s / 1.38 |
| Frame 100 (3.84%) | 28 | 28 | 22 | 28 | 1.708 s / 0.59 | 1,945 | 26 | 10 | 26 | 0.567 s / 1.76 |
| Frame 199 (10%) | 28 | 28 | 27 | 28 | 1.686 s / 0.59 | 1,743 | 26 | 7 | 22 | 0.575 s / 1.74 |
| Synthetic tensile stress (60%) | 16 | 10 | 4 | 8 | 16.405 s / 0.061 | 893 | 12 | 2 | 8 | 0.553 s / 1.81 |

The FPS values are complete image-pair calls per second, with image decoding and one-time device creation excluded. XFeat time includes ONNX inference and the local affine fit; DeepMatching time includes the full Python matcher call. At 60%, both methods produce eight fits under 5 px, while XFeat is about 29.6× faster. DeepMatching produces four fits within 2 px versus two for XFeat. Among accepted fits, 8/12 XFeat results and 8/10 DeepMatching results are under 5 px.

Across the four cases, DeepMatching yields 83/104 visible seeds at ≤2 px and 96/104 below 5 px. XFeat yields 33/104 at ≤2 px and 88/104 below 5 px. Thus XFeat increases pair throughput, but DeepMatching retains substantially better precise coverage, especially through the progressive 10% series.

## What is being counted

- DeepMatching evaluates the requested DIC centers directly and returns one displacement estimate per center. It returned 32, 28, 28, and 16 seed estimates in these cases.
- XFeat's dense ONNX matcher returned 3,087, 1,945, 1,743, and 893 raw point pairs. Those counts are not directly comparable to the number of DIC centers. The local WGSL affine RANSAC then converted the pairs into estimates at the same requested centers.
- Every error is evaluated against the analytic displacement at the requested center. XFeat's subpixel local affine result is evaluated at that exact center.
- The 60% image is a synthetic global affine extrapolation, not a heterogeneous experimental strain field.

## Protocol and provenance

- Data: UHD Live-DIC uniaxial tensile reference with frames 040, 100, and 199, plus the repository's synthetic 60% tensile pair.
- Both methods use the same visible 128 px DIC windows on a 512 px seed grid. DeepMatching uses the INRIA hierarchy (`patch_size=128`, `root_patch_size=512`, `downscale_levels=2`). Its search radius is 256 px through 10% strain and 1024 px for the 60% stress case, where a 256 px limit would exclude much of the known displacement field.
- XFeat uses the community `xfeat_dense_e2e.onnx` export, ONNX Runtime 1.31.1 with `WebGpuExecutionProvider`, and a 256 px neighborhood for GPU local affine fitting. The model SHA-256 is recorded on <a href="xfeat_onnx_tensile_webgpu.json.html">the XFeat raw-data page</a>. The provider reports the Intel LNL GPU, but ONNX Runtime falls back to CPU for some operators; “WebGPU” therefore does not mean the full XFeat graph ran on GPU.
- Both paths ran on the `Vulkan (Intel(R) Graphics (LNL))` adapter. DeepMatching ran through `wgpu-py`; XFeat ran through the native ONNX Runtime WebGPU provider, then passed its point pairs to the repository's WGSL local RANSAC. The APIs and GPU runtimes differ, so the throughput comparison describes these implementations on this machine, not an isolated shader-to-shader speed comparison.
- DeepMatching timing: one warm-up plus three timed calls; report the median of the timed calls. XFeat timing: ONNX inference median of three calls after the first, plus the wall time of its GPU affine-fit call. Loading and image decoding are excluded.

The 60% DeepMatching search-radius increase has a large cost: its median rose to 16.405 s per pair (0.061 FPS). This is the cost of covering the large known displacement range under the tested INRIA hierarchy settings. The former 256 px browser result is not used for the fair 60% row because its search bound was too small.

## Reproduction artifacts

- XFeat experiment and its per-seed output: [`experiment_xfeat.py`](../../python/benchmarks/experiment_xfeat.py) and <a href="xfeat_onnx_tensile_webgpu.json.html">its raw-data page</a>.
- DeepMatching same-center measurements: <a href="deepmatching_same_protocol.json.html">the combined raw-data page</a> and <a href="deepmatching_60pct_radius1024.json.html">the 60% radius-ablation data page</a>.
- The ONNX model remains an external input to the experiment script; it is not bundled in this repository.
