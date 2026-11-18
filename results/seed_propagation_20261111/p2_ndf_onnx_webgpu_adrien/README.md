# Learned coarse matching through ONNX Runtime WebGPU

The NDF (neural deformation field) pair matcher and its shared feature encoder can now be exported to fixed-shape ONNX (Open Neural Network Exchange) graphs and executed by the ONNX Runtime WebGPU execution provider. This is an experimental opt-in path: input and output arrays still traverse CPU memory between ONNX Runtime, PyTorch and wgpu-py. Runtime selection and preprocessing remain on CPU. It is not a GPU-resident pipeline and still requires PyTorch for the surrounding experimental code and initial export. The graphs use a fixed image shape and support soft or hard correspondences, not the top-8 mode. The all-pairs feature grid is limited to 4,096 positions.

## Diagnostic placement

The profiled warm-up used ONNX Runtime 1.31.1 and the native WebGPU provider 0.5.0. The discovered provider device reports Intel vendor ID 32902; the wgpu-py solver separately reports Intel Graphics Lunar Lake. The profile attributes all recorded node events to WebGPU: 506 encoder events and 138 whole-pair events, with zero events attributed to the CPU execution provider. This is an operator-placement observation, not proof of zero transfers or a measurement of pure GPU kernel time. The two execution providers remain independently allocated and no buffers are shared between them. Graph results return through `session.run()` as NumPy arrays.

The checkpoint SHA-256 is `150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`. The `model_sha256` field in the archived diagnostic report is the composite export fingerprint, not the checkpoint hash; newer output uses separate named fields. Export uses PyTorch's legacy fixed-shape tracer, which emits deprecation and trace warnings. Do not infer support for different input sizes from this trace; regenerate the graphs for each size.

## Paired Adrien measurement

Both methods use the native 7920 × 6004 reference and deformed images, with a 192 × 146 network raster, the same pretrained checkpoint, 10 coarse orientations, affine prefit, local normalized cross-correlation (NCC), and inverse-compositional Gauss–Newton (IC-GN) settings. The profiler was enabled only in the warm-up; the following three repetitions were unprofiled and alternated backend order. These are staged prefit diagnostics, without region-of-interest masking or subsequent anchor propagation. No independent truth is available.

| Measure | PyTorch CPU | ONNX Runtime WebGPU |
| --- | ---: | ---: |
| Median network and orientation-search wall time | 0.073 s | 0.152 s |
| Median full scripted process time | 8.019 s | 8.283 s |
| IC-GN windows accepted in one pass | 6,045 | 6,045 |

All four runs, including warm-up, select the same rotation and return bitwise-identical initial parameters, affine-only initial parameters, IC-GN status and IC-GN parameters on this pair. A slower GPU result is retained: this experiment does not demonstrate an acceleration. Process time includes input loading, model setup, solve and output, but not the subsequent seed-confirmation/propagation pipeline. It is not production latency.

The separate two-case synthetic check in `../p2_ndf_onnx_webgpu_synthetic/README.md` and the staged matching-to-propagation check in `../p2_ndf_onnx_end_to_end/README.md` provide additional, limited evidence. Raw timing and per-window differences are in `summary.json`; ONNX Runtime profile events are in `graphs/`.

Reproduce in a new directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_onnx_webgpu.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --reference /home/jeff/CNRS/DIC/AdrienVariational/img_000000_0.09422.tiff \
  --target /home/jeff/CNRS/DIC/AdrienVariational/img_000075_1608.88508.tiff \
  --output-dir results/seed_propagation_20261111/p2_ndf_onnx_webgpu_adrien_new \
  --repetitions 3 --diagnostic-profile
```
