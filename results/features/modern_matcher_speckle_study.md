# Modern learned matchers on Live-DIC speckle images

## Main result

EfficientLoFTR is the strongest of the two new candidates in this first study. After local affine fitting, it recovers all 16 visible DIC seeds on the synthetic 60% tensile pair; 14 are within 5 px and 12 within 2 px of analytic truth. It also recovers all visible seeds through the 10% Live-DIC series, with 26–32 of them within 2 px.

This is an exploratory accuracy result, not a performance win. Both ONNX models were evaluated at 640×360, with a 640×384 padded tensor, while the existing DeepMatching and XFeat times below used 3840×2160 input. EfficientLoFTR takes about 1.1–1.7 s per pair at the reduced size; RaCo-ALIKED-LightGlue+ takes about 2.4–2.8 s.

## Comparison

All methods use the same visible DIC seed centers, a 128 px DIC window and a 512 px seed grid. “Correct” is measured against the analytic displacement at the requested center. For the learned matchers, the returned feature pairs are converted to center estimates using the same HORG WGSL local-affine RANSAC settings: 256 px radius, 3 px residual threshold, six minimum inliers and eight minimum candidates.

| Pair | Method | Accepted / visible | Correct ≤2 px | Correct <5 px | Median time / pair | FPS | Input size |
|---|---|---:|---:|---:|---:|---:|---:|
| 040 | DeepMatching | 32/32 | 30 | 32 | 1.720 s | 0.58 | 3840×2160 |
| 040 | XFeat | 32/32 | 14 | 32 | 0.724 s | 1.38 | 3840×2160 |
| 040 | RaCo-ALIKED-LightGlue+ | 26/32 | 26 | 26 | 2.403 s | 0.416 | 640×360 |
| 040 | EfficientLoFTR | 32/32 | 32 | 32 | 1.139 s | 0.878 | 640×360 |
| 100 | DeepMatching | 28/28 | 22 | 28 | 1.708 s | 0.59 | 3840×2160 |
| 100 | XFeat | 26/28 | 10 | 26 | 0.567 s | 1.76 | 3840×2160 |
| 100 | RaCo-ALIKED-LightGlue+ | 23/28 | 23 | 23 | 2.546 s | 0.393 | 640×360 |
| 100 | EfficientLoFTR | 28/28 | 28 | 28 | 1.271 s | 0.787 | 640×360 |
| 199 | DeepMatching | 28/28 | 27 | 28 | 1.686 s | 0.59 | 3840×2160 |
| 199 | XFeat | 26/28 | 7 | 22 | 0.575 s | 1.74 | 3840×2160 |
| 199 | RaCo-ALIKED-LightGlue+ | 23/28 | 23 | 23 | 2.796 s | 0.358 | 640×360 |
| 199 | EfficientLoFTR | 28/28 | 26 | 28 | 1.667 s | 0.600 | 640×360 |
| 60% | DeepMatching | 10/16 | 4 | 8 | 16.405 s | 0.061 | 3840×2160 |
| 60% | XFeat | 12/16 | 2 | 8 | 0.553 s | 1.81 | 3840×2160 |
| 60% | RaCo-ALIKED-LightGlue+ | 13/16 | 6 | 12 | 2.578 s | 0.388 | 640×360 |
| 60% | EfficientLoFTR | 16/16 | 12 | 14 | 1.501 s | 0.666 | 640×360 |

EfficientLoFTR's 60% accepted set includes two estimates above 5 px; the largest accepted error is 11.30 px. These two false accepts need better confidence filtering or a stronger local consistency check. Raw semi-dense pairs also contain large outliers: at 60%, their RMSE is 264 px before robust local fitting. The robust affine stage is essential to the result.

The comparison supports a promising accuracy hypothesis: a detector-free semi-dense matcher plus robust local geometry can recover more difficult DIC windows than the tested XFeat or DeepMatching configurations. It does not establish a speed comparison because the learned models were run on only 1/36 as many input pixels. A future performance comparison must either run those models at UHD or measure all methods at a shared scale and report the changed seed-window support explicitly.

## Protocol and provenance

- Images: 3840×2160 Live-DIC frames 040, 100 and 199, plus the synthetic 60% tensile pair. The same analytic truth and visible seeds as the previous <a href="deepmatching_vs_xfeat.html">DeepMatching/XFeat study</a> are used.
- EfficientLoFTR weights: `eloftr_outdoor_opt.onnx` from the [SpatialHub ONNX model card](https://huggingface.co/SpatialHub/efficient-loftr-onnx), SHA-256 `a2cbdcfef0ddb5cdab062ae34bf7b09da6cac79baf97fc5155033c8cb1eab602`. The model card declares Apache-2.0. The upstream EfficientLoFTR project says its pretrained model is trained on MegaDepth and works best for outdoor scenes, so the speckle results are a cross-domain test.
- RaCo-ALIKED-LightGlue+ weights: `raco_aliked_lightglue_pipeline_k1024.onnx` from [LightGlue-ONNX v3.0](https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v3.0), SHA-256 `575e1746138204e19027cb37541cd840c7fc9f2c932b844e9685354a3913bda1`.
- Input to both learned methods: resized RGB or grayscale at 640×360 and padded at the bottom to 640×384. The mapping back to original pixel coordinates is ×6 on both axes. The padded region is excluded from truth scoring.
- Runtime: ONNX Runtime 1.31.1 and `onnxruntime-ep-webgpu` 0.5.0 on Intel Graphics (LNL), Mesa 25.2.8, Vulkan. Local affine fitting uses the repository's `wgpu-py` / WGSL RANSAC path. Each pair had one warm-up and three timed repeats; timing includes inference and affine fit, excludes image decoding and resizing.
- EfficientLoFTR's ORT trace recorded 20,736 WebGPU node events and 20,096 CPU node events over 16 model calls. The WebGPU provider executes much of the graph, but approximately half of the node events are on CPU. RaCo-ALIKED-LightGlue+'s trace recorded 37,024 WebGPU and 8,704 CPU node events over the same number of calls.
- A 1920×1080 attempt for RaCo-ALIKED-LightGlue+ did not finish within 1 minute 45 seconds and reached approximately 3.9 GB process RSS. No full-resolution learned-matcher timing is available.
- The 60% test is a synthetic global affine deformation. It does not validate performance across cracks or spatially heterogeneous deformation.

## Reproduction

The optional model files and ONNX Runtime packages are not dependencies of `wgpu-matching`. Use the scripts and per-seed outputs:

- [EfficientLoFTR benchmark script](../../python/benchmarks/experiment_efficientloftr.py) and <a href="efficientloftr_webgpu_study.json.html">raw results</a>.
- [RaCo-ALIKED-LightGlue+ benchmark script](../../python/benchmarks/experiment_aliked_lightglue.py) and <a href="raco_aliked_lightglue_webgpu_study.json.html">raw results</a>.
- <a href="deepmatching_vs_xfeat.html">Existing DeepMatching/XFeat report</a>.

EfficientLoFTR command:

```bash
python -m pip install 'onnxruntime==1.31.1' 'onnxruntime-ep-webgpu==0.5.0' 'wgpu>=0.32,<0.33' numpy pillow
PYTHONPATH=python/src python python/benchmarks/experiment_efficientloftr.py \
  --model /path/to/eloftr_outdoor_opt.onnx --width 640 --height 360 --repeats 3
```

## Next step

EfficientLoFTR should move to a more detailed validation before any WGSL port: test confidence filtering against the two false accepts at 60%, vary input resolution while preserving the original aspect ratio, and evaluate heterogeneous displacement if available. If coverage stays high, profile the CPU/WebGPU partition and memory transfers. Then measure at the highest practical resolution before deciding whether ONNX Runtime WebGPU or a WGSL implementation is the better deployment path.
