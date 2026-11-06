# RaCo-ALIKED-LightGlue+ on Live-DIC speckle images

## Summary

The published RaCo-ALIKED-LightGlue+ end-to-end ONNX model produces accurate local affine initializations on these speckle images, including the 60% synthetic tensile case. The reduced-resolution experiment is promising for accuracy, but it is slow on the tested integrated GPU and does not yet establish full-resolution performance.

| Pair | Visible DIC seeds | Accepted affine seeds | Correct ≤2 px | Correct <5 px | Median time / pair | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Frame 040 (~0.11%) | 32 | 26 | 26 | 26 | 2.403 s | 0.416 |
| Frame 100 (3.84%) | 28 | 23 | 23 | 23 | 2.546 s | 0.393 |
| Frame 199 (10%) | 28 | 23 | 23 | 23 | 2.796 s | 0.358 |
| Synthetic tensile stress (60%) | 16 | 13 | 6 | 12 | 2.578 s | 0.388 |

An accepted seed means the repository's GPU local-affine RANSAC passed its configured support and residual checks. Correctness is evaluated separately against analytic truth at the requested DIC center. At 60%, one accepted estimate is a false positive at 9.13 px; three seeds are rejected, and one more accepted seed is outside the 5 px repairability threshold. The experiment therefore reports coverage and conditional precision separately.

## Context against existing measurements

The table below places the new result beside the previous DeepMatching/XFeat study. The same analytically visible DIC centers and the same 128 px windows on a 512 px grid were used. RaCo-ALIKED was run on images resized from 3840×2160 to 640×360, then its coordinates were mapped back to the original image. The other two columns are full-resolution results from the existing study, so their times are not a like-for-like resolution comparison.

| Pair | Method | Accepted | ≤2 px | <5 px | Time / pair | FPS | Input size |
|---|---|---:|---:|---:|---:|---:|---:|
| 040 | DeepMatching | 32/32 | 30 | 32 | 1.720 s | 0.58 | 3840×2160 |
| 040 | XFeat | 32/32 | 14 | 32 | 0.724 s | 1.38 | 3840×2160 |
| 040 | RaCo-ALIKED-LightGlue+ | 26/32 | 26 | 26 | 2.403 s | 0.416 | 640×360 |
| 100 | DeepMatching | 28/28 | 22 | 28 | 1.708 s | 0.59 | 3840×2160 |
| 100 | XFeat | 26/28 | 10 | 26 | 0.567 s | 1.76 | 3840×2160 |
| 100 | RaCo-ALIKED-LightGlue+ | 23/28 | 23 | 23 | 2.546 s | 0.393 | 640×360 |
| 199 | DeepMatching | 28/28 | 27 | 28 | 1.686 s | 0.59 | 3840×2160 |
| 199 | XFeat | 26/28 | 7 | 22 | 0.575 s | 1.74 | 3840×2160 |
| 199 | RaCo-ALIKED-LightGlue+ | 23/28 | 23 | 23 | 2.796 s | 0.358 | 640×360 |
| 60% | DeepMatching | 10/16 | 4 | 8 | 16.405 s | 0.061 | 3840×2160 |
| 60% | XFeat | 12/16 | 2 | 8 | 0.553 s | 1.81 | 3840×2160 |
| 60% | RaCo-ALIKED-LightGlue+ | 13/16 | 6 | 12 | 2.578 s | 0.388 | 640×360 |

At 10%, RaCo-ALIKED has better precision among its accepted estimates than XFeat, but returns five fewer accepted seeds than DeepMatching. At 60%, it gives four more correct-under-5-px seeds than either previous method in this experiment. These results make it worth further work on quality, while its measured latency and reduced input resolution prevent a performance win claim.

## Protocol

- Data: UHD Live-DIC uniaxial tensile frames 040, 100, and 199, plus the repository's synthetic 60% tensile pair. All are 3840×2160 before resizing.
- The RaCo-ALIKED-LightGlue+ model is the published `raco_aliked_lightglue_pipeline_k1024.onnx` release (about 64.5 MB). SHA-256: `575e1746138204e19027cb37541cd840c7fc9f2c932b844e9685354a3913bda1`.
- Input images were resized to 640×360, converted to RGB float values in [0, 1], and padded at the bottom to 640×384 to satisfy the model's multiple-of-32 image shape. Predicted coordinates were rescaled by 6 to the original coordinate system.
- The same visible 128 px DIC-window centers on a 512 px seed grid were evaluated. The local affine fit uses HORG's WGSL RANSAC implementation with a 256 px radius, 3 px residual threshold, six minimum inliers, and eight minimum candidates.
- ONNX Runtime 1.31.1, native `onnxruntime-ep-webgpu` 0.5.0, `wgpu-py` 0.32.0, Intel Graphics (LNL), Mesa 25.2.8, Vulkan. Each pair had one warm-up followed by three timed repeats. Time includes inference and local affine fitting; image decoding and resizing are excluded.
- Across the profiled calls, ORT recorded 37,024 WebGPU node events and 8,704 CPU node events (16 model runs total). Thus this configuration is a mixed WebGPU/CPU execution, with transfer and partition overhead. The provider name alone does not mean the whole ONNX graph runs on GPU.
- A separate attempt at 1920×1080 did not finish within 1 minute 45 seconds and reached about 3.9 GB process RSS; it was stopped. No full-resolution latency is reported.
- The 60% case is a synthetic globally affine field. It does not establish similar performance around cracks, localization, or spatially varying strain.

## Reproduction

The optional model and runtime are not package dependencies. Download the release model from [LightGlue-ONNX v3.0](https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v3.0), then run:

```bash
python -m pip install 'onnxruntime==1.31.1' 'onnxruntime-ep-webgpu==0.5.0' 'wgpu>=0.32,<0.33' numpy pillow
PYTHONPATH=python/src python python/benchmarks/experiment_aliked_lightglue.py \
  --model /path/to/raco_aliked_lightglue_pipeline_k1024.onnx \
  --width 640 --height 360 --repeats 3
```

See the <a href="raco_aliked_lightglue_webgpu_study.json.html">per-seed JSON data</a> and the existing <a href="deepmatching_vs_xfeat.html">DeepMatching/XFeat report</a> for the earlier full-resolution measurements.

## Assessment

The follow-on <a href="modern_matcher_speckle_study.html">EfficientLoFTR comparison</a> recovered 16/16 visible seeds at 60%, with 14 under 5 px and 12 under 2 px, at 640×360 input. A full-resolution ALIKED benchmark would need an optimized WebGPU graph or tiling strategy first.
