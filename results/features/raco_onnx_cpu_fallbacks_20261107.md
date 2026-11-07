# RaCo ONNX CPU-fallback reduction — 2026-11-07

## Result

An opt-in rewrite changes RaCo's sparse feature-index subgraph from int64 to
int32. It casts the compact row/column coordinates before broadcasting, runs the
large `Add`/`Mul` index-grid construction in WebGPU, and passes int32 indices to
`GatherElements`. This also lets the two large channel-wise index expansions
run in WebGPU instead of materializing expanded int64 tensors on the CPU.

The model output is unchanged: all per-seed records are exactly equal to the
existing SELU + `GridSample` optimized graph on all four benchmark pairs.
That baseline is the previous optimized graph, not the raw download: source
model SHA-256
`575e1746138204e19027cb37541cd840c7fc9f2c932b844e9685354a3913bda1`, baseline
runtime graph SHA-256
`c41f27ab1585f53e7e8f4709c657fc419d32f113964d6f2802bed0db710728b3`, and
rewritten runtime graph SHA-256
`eb29f297206e7eba2a488224d2378d047331fe334b023e39e3c85067401a9e3e`.

| Case | Seeds accepted | Correct ≤2 px | Correct <5 px | RMSE accepted (px) | Baseline | Rewritten | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Live-DIC frame 040 | 26/32 | 26 | 26 | 0.220 | 1.187 s | 1.071 s | 1.108× |
| Live-DIC frame 100 | 23/28 | 23 | 23 | 0.451 | 1.172 s | 1.073 s | 1.092× |
| Live-DIC frame 199 | 23/28 | 23 | 23 | 0.740 | 1.180 s | 1.066 s | 1.107× |
| Synthetic tensile 60% | 13/16 | 6 | 12 | 3.449 | 1.105 s | 1.070 s | 1.033× |

Timings are medians of five unprofiled repeats, after warm-up, on the same
Intel Graphics LNL adapter. They include ONNX inference and the WGSL local
affine fit, and exclude image decode and resize. The source photos are UHD
3840×2160, but the global network input remains 640×360, padded to 640×384.
The measured speedup is 3.3–10.8% for this run; it is hardware- and load-
dependent, not a general performance guarantee. Each optimized timing comes
from the composite flag applied to the original model export; the operator
profile was captured from the same graph before a metadata-only wording
correction, which did not change executable nodes or tensor types.

## What changed in the profile

The diagnostic run profiled 12 model calls per graph (warm-up plus two timed
calls on each of four pairs). The table normalizes ORT trace values per call.
Byte counts are estimates from tensor shapes; event durations include runtime
overheads and are not hardware GPU timestamp measurements.

| CPU fallback | Before: time / output per call | After: time / output per call |
|---|---:|---:|
| `Expand` | 8.019 ms / 62.69 MB | 0.166 ms / 0.43 MB |
| `Add` | 2.333 ms / 44.43 MB | 0.256 ms / 1.08 MB |
| `Mul` | 1.559 ms / 14.78 MB | 0.493 ms / 0.33 MB |
| All CPU provider nodes | 23.28 ms | 13.37 ms |

The CPU trace duration falls by 42.6% per inference. Estimated
`MemcpyFromHost` input to WebGPU falls from 76.74 MB to 14.59 MB per inference
(81.0%); estimated `MemcpyToHost` remains 10.85 MB. The count of CPU node
events is almost unchanged, so this specifically removes the large int64
fallback work rather than eliminating CPU execution from the graph.

A separate one-call ORT trace confirms the target nodes are actually assigned
to `WebGpuExecutionProvider` and emit int32 tensors:

| Node | Operator | Output shape |
|---|---|---|
| `node_add_495` | `Add` | `[2, 512, 42, 42]` int32 |
| `node_mul_378` | `Mul` | `[2, 512, 42, 42]` int32 |
| `node_add_480` | `Add` | `[2, 512, 42, 42]` int32 |
| `node_add_506` | `Add` | `[2, 512, 42, 42]` int32 |
| `node_expand_4` | `Expand` | `[2, 3, 903168]` int32 |
| `node_expand_18` | `Expand` | `[2, 128, 9216]` int32 |

The opt-in rewrite can be enabled from the original export with:

```bash
PYTHONPATH=python/src:python/benchmarks python \
  python/benchmarks/experiment_aliked_lightglue.py \
  --model /path/to/raco_aliked_lightglue_pipeline_k1024.onnx \
  --decompose-selu --decompose-grid-sample --optimize-raco-indices \
  --repeats 5 --output /tmp/raco-index-fallback-benchmark.json
```

The downloaded model is read-only, and the default benchmark path is unchanged.
The implementation and exact comparison data are in
[`optimize_onnx_webgpu.py`](../../python/benchmarks/optimize_onnx_webgpu.py),
[`experiment_aliked_lightglue.py`](../../python/benchmarks/experiment_aliked_lightglue.py),
and the [machine-readable report](raco_onnx_cpu_fallbacks_20261107.json).

## Remaining CPU work and limits

The graph is still not GPU-resident. `LogSoftmax`, dynamic shape/index
operations (`Concat`, `Reshape`, `Gather`, `Unsqueeze` and others), and two
small border-padded `GridSample` nodes still use CPUExecutionProvider. A prior
`LogSoftmax` decomposition reduced estimated host transfers but made the
unprofiled benchmark slower, so it remains excluded. The remaining work should
be considered only with a same-protocol timing and output-parity check.

This experiment also does not share GPU buffers between ONNX Runtime's native
WebGPU provider and the separate `wgpu-py` affine-fit context. The benchmark
still returns model outputs through the existing Python path. The profile's
transfer events do not prove GPU allocation sharing or report peak GPU memory.

Versions: ONNX Runtime 1.31.1, WebGPU EP 0.5.0, ONNX 1.23.2, and `wgpu-py`
0.32.0; Vulkan backend with Mesa 25.2.8 on Intel Graphics LNL.
