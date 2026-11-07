# DaD and DeDoDe v2 keypoint ONNX/WebGPU path

Date: 2026-11-07\
Runtime: ONNX Runtime 1.31.1 + `onnxruntime-ep-webgpu` 0.5.0\
Adapter: Intel vendor `0x8086`, device `0x64a0` (Intel Graphics LNL)\
Model input: 640 × 360 pixels\
Output: K=1024 floating-point keypoints and scores

## Scope

This experiment exports the official DaD and DeDoDe-v2 detector weights to
keypoint-only ONNX graphs and runs them with ONNX Runtime's WebGPU execution
provider. A custom export wrapper moves top-K selection and coordinate
generation into the graph, so dense response maps are not copied to the host.
The keypoint selection was compared with each official PyTorch detector before
measuring ONNX inference.

The report is about detector export and inference. It does not yet measure
descriptor matching, affine RANSAC, IC-GN convergence, or native-resolution 4K
inference. The matching experiment can now accept these graphs, but a full
common-descriptor benchmark remains pending.

## Inference timings

One warm-up followed by five measured `session.run` calls, using the same
640 × 360 random normalized input for each run:

| Detector | Median `session.run` | WebGPU node duration in ORT profile | CPU node-kernel duration | Output bytes returned |
|---|---:|---:|---:|---:|
| DaD | 119.63 ms | 120.15 ms | 0.516 ms | 12,288 B |
| DeDoDe v2-L | 405.26 ms | 409.07 ms | 1.034 ms | 12,288 B |

ORT listed both `WebGpuExecutionProvider` and `CPUExecutionProvider`. The
profile reports the CPU node events and summed operator durations shown above;
this does not prove that there are no unprofiled copies or waits. `session.run`
returns CPU NumPy arrays. There is no shared allocation with `wgpu-py`, and the
peak GPU/RAM memory was not measured.

The point output is only 12 KiB for K=1024 (coordinates and scores). The large
dense confidence map stays inside ONNX Runtime's WebGPU graph and is not an
output. These timings exclude Python-side grayscale conversion, resize and
normalization; the public adapter records those separately in its per-call
metadata.

## Numerical checks

The custom selection export matched the official PyTorch detector on the
exporter's 640 × 360 check input:

| Detector | Reference points within 1 px (one-way nearest set) | Score max absolute difference |
|---|---:|---:|
| DaD | 1024/1024 | 0 |
| DeDoDe v2-L | 1024/1024 | 0 |

Those rows compare custom PyTorch selection with official PyTorch selection.
Comparing ONNX Runtime output to PyTorch output, DeDoDe v2-L had 1024/1024
points within 1 px and maximum score error `4.92e-11`. DaD had 1012/1024
reference points within 1 px, 1018/1024 within 5 px and maximum score error
`5.00e-7`; this is consistent with output-order changes among near-tied
points. The metric is one-way nearest-neighbour set distance, not one-to-one
assignment, so it should not be read as a complete parity proof.

A separate API smoke on a generated speckle pair returned 1019/1024 DaD and
1024/1024 DeDoDe-v2 points within 1 px of the PyTorch outputs by the same
one-way nearest-set measure. This confirms coordinate mapping through the
public adapter for that case; it is not a keypoint matching accuracy result.

## Graph and checkpoint provenance

| Model | Official weights SHA-256 | ONNX SHA-256 | ONNX size | Nodes |
|---|---|---|---:|---:|
| DaD | `7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f` | `f68a6bae97bbbc9745bf81693fc375f31f5bf396434ea5a990d9795bd88384fe` | 13.0 MB | 400 |
| DeDoDe v2-L | `4113809dd9e0367af013a45fc2255a6b243ff241cd06520d17a65d9e231bdc17` | `a62de67958710088b08dc508bfa71705e2254e47a720844d91feaf2d402267ea` | 58.2 MB | 409 |

The exported graphs carry detector/checkpoint metadata, dynamic spatial axes,
and a static K=1024 output dimension. Model files remain outside the Git
repository; the raw JSON records and ORT profiles are stored next to this
report.

## Interpretation and limitations

- The detector convolution and keypoint graph execute predominantly on
  WebGPU, but ONNX Runtime's normal Python `session.run` API returns the point
  tensors to CPU memory.
- These graphs are not a shared-memory bridge between ONNX Runtime and
  `wgpu-py`. A 12 KiB output readback is small, but the remaining descriptor
  extraction and mutual matching stages are separate CPU reference operations.
- DaD and DeDoDe v2 were evaluated on reduced 640 × 360 inputs. A 4K source
  passed through the adapter would first be resized to the configured
  `max_side`; it is not native 4K feature inference.
- DaD's top-K output can reorder close-score points because floating-point
  differences alter ties. The set check is not injective and does not assess
  descriptor match correctness.
- No peak memory, end-to-end per-pair time, IC-GN convergence, or quality
  comparison to the existing DeepMatching result is reported here.

## Reproduction

Install the optional dependencies and export keypoint-mode models using the
official upstream packages/checkpoints. The benchmark accepts the resulting
paths with `--dad-onnx` and `--dedode-onnx`; fixed candidate K means the product
of the largest requested point budget and `--candidate-multiplier` cannot
exceed the export capacity. The code records preprocessing, ONNX run,
coordinate conversion, descriptor, matching and WGSL RANSAC timings
independently.

Raw output and profiles: `results/features/detector_onnx_webgpu_20261107/`.
