# EfficientLoFTR native WebGPU graph optimization

Date: 7 November 2026.

## Result

The 640 × 360 EfficientLoFTR export is measurably faster after two graph changes, with exact output parity on all four checked image pairs.

| Pair | Original median | Optimized median | Original FPS | Optimized FPS | Speedup | Output arrays |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Live-DIC frame 040 | 1.227 s | 1.069 s | 0.815 | 0.935 | 1.15× | Exact |
| Live-DIC frame 100 | 1.201 s | 1.132 s | 0.833 | 0.883 | 1.06× | Exact |
| Live-DIC frame 199 | 1.194 s | 1.058 s | 0.838 | 0.945 | 1.13× | Exact |
| Synthetic tensile 60% | 1.131 s | 0.956 s | 0.884 | 1.046 | 1.18× | Exact |

These are **model inference times**. They include ONNX Runtime input/output handling, but exclude image decoding/resizing and the following local affine fit. The baseline and candidate were timed in alternating order with one warm-up and three timed repetitions per pair; profiling was disabled during those measurements. The profile runs were separate. The GPU is the Intel Graphics LNL integrated GPU using Vulkan/Mesa 25.2.8, with ONNX Runtime 1.31.1 and its native WebGPU EP 0.5.0.

The output coordinates and confidence values are bit-for-bit identical for every reported pair. The previously reported analytic DIC quality therefore remains unchanged: 32/32 accepted on frame 040, 28/28 on frame 100, 28/28 on frame 199, and 16/16 at 60% strain. The 60% case remains 12/16 at ≤2 px and 14/16 at <5 px. Those quality counts are from the existing [EfficientLoFTR benchmark](efficientloftr_webgpu_study.json); this optimization changes only how its ONNX graph evaluates.

## Changes to the graph

The export represented the eight border slices from EfficientLoFTR's `mask_border` operation as eight sequential `ScatterND` nodes. Each scatter materialized a 14.7 MB boolean tensor. The rewrite replaces that construction with one same-shape boolean `And` and a precomputed border-validity mask. This preserves the reference implementation's first/last border removal on all four coarse-grid axes ([upstream `mask_border`](https://github.com/zju3dv/EfficientLoFTR/blob/main/src/loftr/utils/coarse_matching.py)). The old slice/index/update subgraphs are pruned.

The coarse binary mask was also cast from int8 to float32 before `ArgMax` and `ReduceMax`. With int8, these reductions fell back to CPU; with float32, the native WebGPU EP assigns the cast and both reductions to WebGPU. Since the input values are only zero and one, the ordering and maximum results are unchanged. Output parity was checked after this conversion.

The implementation is an opt-in rewrite in [`optimize_onnx_webgpu.py`](../../python/benchmarks/optimize_onnx_webgpu.py). The candidate is specialized to NCHW tensors of **640 × 384** (the 640 × 360 resized image padded at the bottom); it must be regenerated for another inference tensor size. The 14.7 MB dense boolean initializer increases the serialized model by about 14.6 MB. A smaller pair of broadcast masks was tested first, but chained broadcasting produced mismatches on this provider and was discarded.

The ONNX graph shrank from 5,630 to 4,703 nodes. The rewrite prunes 928 now-dead nodes, including all eight CPU `ScatterND` nodes, then adds one `And`; the net change is −927 nodes. This is not a fully GPU-resident model: small shape, indexing, and compaction operations still run on CPU.

## Profile evidence

One diagnostic run on frame 100 gives the following representative provider profile. These are ONNX Runtime event durations, not GPU hardware timestamps.

| Diagnostic | Original | Optimized |
| --- | ---: | ---: |
| CPU node events | 1,256 | 92 |
| CPU event duration sum | 107.3 ms | 1.6 ms |
| Coarse-mask `ScatterND` on CPU | 8 | 0 |
| Coarse-selection `ArgMax` / `ReduceMax` on CPU | 1 / 1 | 0 / 0 |
| `MemcpyToHost` events | 12 | 4 |
| Estimated `MemcpyToHost` volume | 33.45 MB | 0.063 MB |
| `MemcpyFromHost` events | 47 | 14 |
| Estimated `MemcpyFromHost` volume | 19.18 MB | 0.485 MB |

The reported `MemcpyToHost` event duration remains around 1.05 s after the transfer volume falls. ONNX Runtime notes that such events may include queue waits and synchronization, so this duration is not a measurement of copying 63 KB over the bus. It is evidence that synchronization is still a major latency source, and explains why the real speedup is smaller than the volume reduction suggests.

## Scope and next bottleneck

This is still a **640 × 360 network pass over images whose source resolution is 3840 × 2160**. It is not the native-4K local refinement stage, and the inference median remains near one second. The earlier target of <0.5 s has not been reached. No full-GPU claim is made.

The next optimization should inspect the remaining synchronization boundaries and the small CPU-side shape/index/compaction chain. The comparison script [`compare_efficientloftr_border_mask.py`](../../python/benchmarks/compare_efficientloftr_border_mask.py) preserves output-parity, timing, and profiling checks. The machine-readable run is [`efficientloftr_border_mask_optimization_20261107.json`](efficientloftr_border_mask_optimization_20261107.json).

## Reproduction

With the same optional ONNX Runtime environment and the external model file:

```bash
python python/benchmarks/optimize_onnx_webgpu.py \
  /path/to/eloftr_outdoor_opt.onnx \
  /tmp/eloftr_border_mask_640x384.onnx \
  --operation loftr-border-mask \
  --tensor-width 640 --tensor-height 384 --coarse-stride 8 --border 2

python python/benchmarks/compare_efficientloftr_border_mask.py \
  --baseline-model /path/to/eloftr_outdoor_opt.onnx \
  --optimized-model /tmp/eloftr_border_mask_640x384.onnx \
  --repeats 3 --profile \
  --output results/features/efficientloftr_border_mask_optimization_20261107.json
```

The source ONNX weights were read-only and are not included in this repository. The baseline SHA-256 is `a2cbdcfef0ddb5cdab062ae34bf7b09da6cac79baf97fc5155033c8cb1eab602`; the specialized candidate SHA-256 is `b9dbf3e6b30158f33a7fa3e89cdcac319df5fc874903151891bc1a491dff5724`.
