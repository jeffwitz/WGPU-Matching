# GPU performance audit — 2026-11-06

## Scope and setup

This is the first P0 audit for the recent ONNX feature matchers. It profiles one
diagnostic inference per network and separately benchmarks all four existing
uniaxial cases with profiling disabled. The historical study JSON files were
left untouched; new machine-readable data is in
[`gpu_performance_audit_20261106.json`](gpu_performance_audit_20261106.json).

| Item | Configuration |
| --- | --- |
| GPU | Intel Graphics (LNL), integrated |
| Backend / driver | Vulkan / Mesa 25.2.8-0ubuntu0.24.04.4 |
| Input images | 3840 × 2160, resized to 640 × 360 and padded to 640 × 384 |
| ONNX Runtime | 1.31.1; native `onnxruntime-ep-webgpu` 0.5.0 |
| Python WebGPU | wgpu-py 0.32.0 |
| Timing protocol | One warm-up, then three unprofiled repeats per pair; median reported |
| Timed scope | `session.run` + coordinate conversion + WGPU local affine fit |
| Separately measured | Session creation, reference decode/resize, target decode/resize |

The test set contains frames 040, 100, 199, and the synthetic 60% tensile case.
Each model was run on the same existing visible seeds and analytic truth as the
published feature study. Inputs, preprocessing, and the external model files
were read only. Model SHA-256 values and complete per-seed results are in the
JSON report.

## Production timings and quality

All times below exclude the one-time session creation and image decode/resize.
The affine-fit stage itself is small; ORT inference dominates both methods.

| Method | Case | ORT inference median | Affine/postprocess median | Pair median | FPS | Accepted | ≤2 px | <5 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientLoFTR | frame 040 | 1.243 s | 0.024 s | 1.267 s | 0.789 | 32/32 | 32 | 32 |
| EfficientLoFTR | frame 100 | 1.179 s | 0.020 s | 1.199 s | 0.834 | 28/28 | 28 | 28 |
| EfficientLoFTR | frame 199 | 1.286 s | 0.021 s | 1.305 s | 0.766 | 28/28 | 26 | 28 |
| EfficientLoFTR | 60% tensile | 1.212 s | 0.021 s | 1.233 s | 0.811 | 16/16 | 12 | 14 |
| RaCo-ALIKED-LightGlue+ | frame 040 | 2.344 s | 0.013 s | 2.357 s | 0.424 | 26/32 | 26 | 26 |
| RaCo-ALIKED-LightGlue+ | frame 100 | 2.353 s | 0.013 s | 2.366 s | 0.423 | 23/28 | 23 | 23 |
| RaCo-ALIKED-LightGlue+ | frame 199 | 2.435 s | 0.009 s | 2.443 s | 0.409 | 23/28 | 23 | 23 |
| RaCo-ALIKED-LightGlue+ | 60% tensile | 2.341 s | 0.009 s | 2.350 s | 0.425 | 13/16 | 6 | 12 |

The EfficientLoFTR 60% result is unchanged from its earlier baseline: 16/16
accepted, 12 at ≤2 px, 14 at <5 px. RaCo is also unchanged in coverage and
precision counts. The end-to-end accuracy did not regress when timing was run
with profiling disabled.

Target decode/resize took 0.076–0.203 s per image for EfficientLoFTR and
0.112–0.201 s for RaCo in these runs. The reference preparation, which can be
amortized over a sequence, took 0.088–0.094 s. Session creation took 1.008 s
for EfficientLoFTR and 0.734 s for RaCo. These one-time costs are recorded
separately from pair FPS. Host process peak RSS was 326,504,448 bytes for
EfficientLoFTR and 856,084,480 bytes for RaCo. This is process RAM high-water,
not GPU memory usage.

Neither model is close to the stated pair-time targets (<0.5 s EfficientLoFTR,
<0.8 s RaCo). The current 4K coarse-to-local pipeline has not yet been built or
measured.

## ONNX Runtime profile findings

The following are ORT trace event durations from one diagnostic inference at
640 × 360 (padded to 640 × 384). They are not GPU hardware timestamps. Copy
events can include queue waits and synchronization, so their durations should
not be interpreted as raw copy bandwidth or added to independent kernel times.
The instrumented wall times for these single diagnostic calls were 1.512 s for
EfficientLoFTR and 3.280 s for RaCo; do not compare these with the unprofiled
production medians above.

| Model | Provider | Node events | Cumulative ORT event duration |
| --- | --- | ---: | ---: |
| EfficientLoFTR | WebGPU | 1,296 | 1,389 ms |
| EfficientLoFTR | CPU fallback | 1,256 | 110 ms |
| RaCo-ALIKED-LightGlue+ | WebGPU | 2,314 | 2,956 ms |
| RaCo-ALIKED-LightGlue+ | CPU fallback | 544 | 285 ms |

### EfficientLoFTR

- `MemcpyToHost`: 12 events, 33.5 MB estimated cumulative tensor volume,
  1,102 ms cumulative event duration. The largest event transfers a 14.7 MB
  boolean tensor (`[1, 48, 80, 48, 80]`) and dominates the observed copy duration.
- CPU `ScatterND`: 8 calls, 48.8 ms cumulative. They form a chain in the
  coarse-matching stage, each producing a full 14.7 MB boolean tensor with
  shape `[1, 48, 80, 48, 80]`; the index tensors are rank 6 and contain
  five-coordinate indices. This is more than an isolated final selection
  kernel, so a replacement must preserve the entire staged mask construction.
- Other visible CPU fallbacks include `Expand` (46 calls, 32.5 ms) and
  `Concat` (141 calls, 22.7 ms).
- The event profile reports 1,296 WebGPU and 1,256 CPU node events. Event
  counts are not a measure of time share.

Top 20 operator/provider pairs, ranked by summed ORT event duration:

| Rank | Operator | Provider | Calls | Cumulative event time |
| ---: | --- | --- | ---: | ---: |
| 1 | MemcpyToHost | WebGPU | 12 | 1,102.0 ms |
| 2 | ScatterND | CPU | 8 | 48.8 ms |
| 3 | LeakyRelu | WebGPU | 16 | 32.6 ms |
| 4 | MemcpyFromHost | WebGPU | 47 | 32.0 ms |
| 5 | Softmax | WebGPU | 19 | 32.0 ms |
| 6 | Transpose | WebGPU | 139 | 24.9 ms |
| 7 | Expand | CPU | 46 | 22.3 ms |
| 8 | Conv | WebGPU | 44 | 22.0 ms |
| 9 | Concat | CPU | 141 | 20.8 ms |
| 10 | Gather | WebGPU | 11 | 20.1 ms |
| 11 | Slice | WebGPU | 74 | 17.6 ms |
| 12 | MatMul | WebGPU | 128 | 15.5 ms |
| 13 | Div | WebGPU | 28 | 9.1 ms |
| 14 | Concat | WebGPU | 36 | 9.0 ms |
| 15 | Mul | WebGPU | 77 | 9.0 ms |
| 16 | Expand | WebGPU | 11 | 6.8 ms |
| 17 | Add | WebGPU | 37 | 6.7 ms |
| 18 | Resize | WebGPU | 19 | 5.5 ms |
| 19 | GatherND | WebGPU | 3 | 5.2 ms |
| 20 | GatherElements | WebGPU | 2 | 4.7 ms |

### RaCo-ALIKED-LightGlue+

- `MemcpyToHost`: 145 events, about 1.37 GB estimated cumulative tensor volume,
  1,665 ms cumulative event duration.
- `MemcpyFromHost`: 168 events, about 1.23 GB estimated cumulative tensor
  volume, 758 ms cumulative event duration.
- The largest single copied tensor is 251.7 MB (240 MiB), shape
  `[2, 128, 384, 640]`.
- CPU `Selu`: 83 calls, 193 ms; CPU `GridSample`: 39 calls, 65 ms. These are
  real fallback operators, but copy/synchronization overhead is the larger
  issue in this trace.

Top 20 operator/provider pairs, ranked by summed ORT event duration:

| Rank | Operator | Provider | Calls | Cumulative event time |
| ---: | --- | --- | ---: | ---: |
| 1 | MemcpyToHost | WebGPU | 145 | 1,665.3 ms |
| 2 | MemcpyFromHost | WebGPU | 168 | 758.4 ms |
| 3 | Selu | CPU | 83 | 193.3 ms |
| 4 | Conv | WebGPU | 121 | 70.2 ms |
| 5 | GridSample | CPU | 39 | 64.9 ms |
| 6 | Div | WebGPU | 108 | 60.6 ms |
| 7 | Transpose | WebGPU | 170 | 56.4 ms |
| 8 | Gather | WebGPU | 140 | 38.7 ms |
| 9 | MatMul | WebGPU | 122 | 36.5 ms |
| 10 | Expand | WebGPU | 22 | 34.5 ms |
| 11 | Softmax | WebGPU | 19 | 32.0 ms |
| 12 | Concat | WebGPU | 76 | 26.1 ms |
| 13 | Sub | WebGPU | 114 | 18.0 ms |
| 14 | Add | WebGPU | 239 | 16.0 ms |
| 15 | Slice | WebGPU | 58 | 15.7 ms |
| 16 | Mul | WebGPU | 139 | 12.7 ms |
| 17 | Cast | WebGPU | 96 | 12.6 ms |
| 18 | Expand | CPU | 15 | 10.8 ms |
| 19 | Max | WebGPU | 13 | 9.8 ms |
| 20 | Cos | WebGPU | 1 | 8.0 ms |

The current process high-water RSS was 311 MiB for EfficientLoFTR and 816 MiB
for RaCo. The ORT profile does not expose a GPU allocation high-water mark in
this configuration. The tensor sizes above are shape-based estimates; they do
not imply that all listed tensors are simultaneously resident.

## P1 experiments

| Experiment | Result | Decision |
| --- | --- | --- |
| ORT IOBinding with ORT-owned WebGPU input values, RaCo | `session.run`: 2.560 s median; IOBinding: 2.576 s, three repeats each; all three outputs exactly equal | No observed speedup; it does not remove internal CPU fallbacks or WebGPU↔CPU copies |
| WebGPU buffer cache `bucket`, RaCo | Alternating warmed sessions, four runs each: default 2.639 s median; `bucket` 2.751 s | No gain in this sample; do not enable globally on this evidence |
| EfficientLoFTR preferred layout | Alternating, four runs each: NHWC 1.365 s; NCHW 1.563 s; outputs exactly equal | Keep NHWC, already used by the benchmark |
| EfficientLoFTR validation mode | Alternating, four runs each: default 1.328 s; disabled 1.325 s; outputs exactly equal | 0.3% delta is within run-to-run noise |
| EfficientLoFTR flattened `ScatterND` graph rewrite | Frame 100 outputs exactly equal; 8 `ScatterND` nodes remained CPU; CPU events rose 1,256→1,416 and `MemcpyToHost` events 12→27 | Rejected: no measured speed gain, more CPU work and copies |
| Graph capture | Not enabled | The measured graphs have CPU fallback nodes, and output match counts vary by pair. Native WebGPU EP graph capture is intended for static-shape graphs whose kernels all run on WebGPU; this pipeline does not satisfy those preconditions ([ORT docs](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)). |

IOBinding avoided no measurable time in this test, and the output equality
check protects the interpretation of that comparison. The test reused an
ORT-owned WebGPU `OrtValue`; it did **not** share a `wgpu-py` buffer. On this
stack, the native ORT WebGPU EP is built on Dawn, while wgpu-py wraps
wgpu-native ([ORT](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html),
[wgpu-py](https://github.com/pygfx/wgpu-py)). IOBinding therefore does not
establish zero-copy interoperability between the two runtimes. A cross-runtime
buffer path remains unproven and should only be attempted through a documented
shared-device API, not by passing opaque pointers between implementations.

The buffer-cache result is likewise not a performance improvement. Both remain
configuration options, not optimizations to recommend for these models.

The temporary EfficientLoFTR graph experiment flattened each rank-5
`ScatterND` into one-dimensional data and updates, converted its indices to
flat offsets, then restored the original tensor shape. It produced exactly the
same three outputs for frame 100, but the native EP still assigned all eight
`ScatterND` nodes to CPU; the added shape/index graph increased CPU assignments
and host copies. The candidate was not integrated. The temporary model resides
under `/tmp`, outside the repository.

## EfficientLoFTR memory bound at native 4K

At a 1/8 feature stride, a 3840 × 2160 image has a 480 × 270 coarse grid,
or 129,600 tokens. A dense FP32 all-pairs similarity matrix would contain
16,796,160,000 values and occupy **67.2 GB (62.6 GiB)** for one pairwise matrix,
before activations, masks, and scratch space. Therefore the global model cannot
simply be run at native 4K with the current dense matching graph. The proposed
global-coarse plus bounded local-4K refinement still needs implementation.

## Reproduction

The diagnostic profile path is explicit; production timing leaves ORT profiling
off by default. Model exports are optional external files and are not included
in the repository:

```bash
PYTHONPATH=python/src python python/benchmarks/experiment_efficientloftr.py \
  --model /path/to/eloftr_outdoor_opt.onnx --repeats 3 \
  --output /tmp/efficientloftr.json

PYTHONPATH=python/src python python/benchmarks/experiment_aliked_lightglue.py \
  --model /path/to/raco_aliked_lightglue_pipeline_k1024.onnx --repeats 3 \
  --output /tmp/raco.json

# Diagnostic only; adds profiling overhead and reports ORT trace events.
PYTHONPATH=python/src python python/benchmarks/experiment_efficientloftr.py \
  --model /path/to/eloftr_outdoor_opt.onnx --profile --output /tmp/eloftr-profile.json
```

The native WebGPU provider documents both the buffer cache and graph-capture
preconditions in its [configuration reference](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html).
ORT's [I/O Binding documentation](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
also distinguishes device-bound inputs/outputs from ordinary host-array calls.

## Next implementation step

The evidence points to the model graph and device boundaries, not just allocator
overhead. Next, isolate the EfficientLoFTR coarse matching `ScatterND` subgraph
and test an equivalent WebGPU-compatible formulation against exact outputs;
for RaCo, investigate the export's `Selu` and `GridSample` CPU nodes and why
feature tensors are copied back and forth. Do not start the 4K tiled/local
pipeline until its coarse candidate and memory budget are explicit. Any rewrite
must preserve the current accepted-seed counts and analytic errors before its
speed is reported.
