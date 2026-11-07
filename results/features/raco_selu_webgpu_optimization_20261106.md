# RaCo SELU placement and transfer optimization — 2026-11-06

## Scope

This addendum records the measured operator-placement experiment completed before starting native-4K local refinement. It leaves the published baseline audit and external model/image inputs unchanged. The source model is read-only; the transform writes a temporary copy and is opt-in.

| Item | Configuration |
|---|---|
| GPU | Intel Graphics LNL integrated |
| Runtime | ONNX Runtime 1.31.1; native WebGPU EP 0.5.0 |
| Input | 640 × 360, padded to 640 × 384; source images 3840 × 2160 |
| Model | RaCo-ALIKED-LightGlue+ K=1024, SHA-256 `575e1746138204e19027cb37541cd840c7fc9f2c932b844e9685354a3913bda1` |
| Transform | 83 `Selu` nodes decomposed to `Exp`, `Sub`, `Mul`, `Greater`, `Where`, `Mul` |

## Production timing and analytic accuracy

Three unprofiled repeats per pair after warm-up. Reported pair time includes ONNX inference and local affine fit but excludes file decoding/resizing. All cases use the same seeds, input resolution, affine fit and thresholds as the baseline audit.

| Case | Pair time before → after | Time reduction | Accepted | Correct ≤2 px | Correct <5 px | RMSE before → after | Max accepted seed delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| frame_040 | 2.357 → 1.420 s | 39.7% | 26/32 → 26/32 | 26 → 26 | 26 → 26 | 0.219718 → 0.219714 px | 0.00025 px |
| frame_100 | 2.366 → 1.624 s | 31.3% | 23/28 → 23/28 | 23 → 23 | 23 → 23 | 0.450780 → 0.450720 px | 0.00074 px |
| frame_199 | 2.443 → 1.508 s | 38.3% | 23/28 → 23/28 | 23 → 23 | 23 → 23 | 0.740336 → 0.740286 px | 0.00064 px |
| stress_60pct | 2.350 → 1.596 s | 32.1% | 13/16 → 13/16 | 6 → 6 | 12 → 12 | 3.449368 → 3.449398 px | 0.00037 px |

No seed usability state changed. The maximum difference between corresponding accepted seed target estimates was below 0.00075 px. Accepted/correct counts are exactly unchanged. The raw model tensors are not bitwise identical after decomposition, so the rewrite remains opt-in; its measured DIC outputs are numerically close on these four cases.

## Placement and transfer evidence

One profiled frame-100 inference per model. These trace durations include runtime synchronization and are diagnostic, not hardware GPU timestamps or production latency.

| Profile item | Original graph | SELU-decomposed graph |
|---|---:|---:|
| WebGpuExecutionProvider node events / summed event duration | 2314 / 2776.7 ms | 2648 / 1263.1 ms |
| CPUExecutionProvider node events / summed event duration | 544 / 230.0 ms | 459 / 86.2 ms |
| MemcpyToHost estimated aggregate tensor bytes / events | 1.373 GB / 145 | 0.262 GB / 64 |
| MemcpyFromHost estimated aggregate tensor bytes / events | 1.231 GB / 168 | 0.118 GB / 85 |

The Selu rewrite removes all 83 CPU `Selu` calls (about 159 ms in the original diagnostic profile) and reduces the aggregate estimated host-transfer volumes by about 81% toward host and 90% from host in this trace. These are sums of trace tensor sizes, not unique allocation traffic or measured bus bandwidth. The largest single host-bound tensor remains 251.7 MB.

## Remaining `GridSample` fallback

`GridSample` remains CPU for all 39 nodes in the exported graph. Isolated ONNX probes using `mode=linear`, `align_corners=1`, `padding_mode=zeros` also ran on CPU across six NCHW sizes from `[1,1,16,16]` through `[2,32,96,160]`, including the representative graph shape `[2,32,48,80]` with grid `[2,48,80,2]`. The earlier interpretation that this operator was WebGPU-capable on this native EP build was not reproducible; treat it as a CPU fallback on ONNX Runtime 1.31.1 / native WebGPU EP 0.5.0. Four ORT graph-optimization levels were measured on the original model. None changed the 39 CPU assignments or approximately 1.37 GB estimated aggregate `MemcpyToHost` volume. `ORT_ENABLE_BASIC` reduced observed event counts by fusion, but did not eliminate transfer volume; its single profiled wall time is not comparable as a production speed result.

| ORT graph optimization | CPU GridSample | CPU Selu | D2H estimate | H2D estimate |
|---|---:|---:|---:|---:|
| ORT_DISABLE_ALL | 39 calls, 98.1 ms | 83 calls, 292.1 ms | 1.373 GB | 1.231 GB |
| ORT_ENABLE_BASIC | 39 calls, 83.6 ms | 83 calls, 166.1 ms | 1.373 GB | 1.231 GB |
| ORT_ENABLE_EXTENDED | 39 calls, 74.2 ms | 83 calls, 166.0 ms | 1.373 GB | 1.231 GB |
| ORT_ENABLE_ALL | 39 calls, 76.7 ms | 83 calls, 193.6 ms | 1.373 GB | 1.231 GB |

The fallback is consistent in the full graph and isolated operator probes, so the next optimization requires replacing or restructuring these `GridSample` operations, not merely changing ORT graph-optimization level. Do not claim the current model is GPU-resident.

## Decision before native-4K refinement

- Retain the SELU rewrite only as an explicit opt-in transform for this model, with source model untouched.
- Keep NHWC as the measured layout choice; earlier NCHW runs were slower and had no transfer-volume benefit.
- Do not start 4K native refinement yet: one 251.7 MB host-bound tensor and all 39 `GridSample` CPU fallbacks remain unresolved.
- The measured pair time improves from 2.35–2.44 s to 1.42–1.62 s on these cases, but still misses the <0.8 s RaCo target.
- No performance conclusion is generalized beyond this exported model, GPU/runtime stack and four cases.

## Reproduction

```bash
PYTHONPATH=python/src python python/benchmarks/experiment_aliked_lightglue.py \
  --model /path/to/raco_aliked_lightglue_pipeline_k1024.onnx \
  --decompose-selu --repeats 3 \
  --output /tmp/raco-selu-webgpu.json
```

Enable `--profile` separately for placement diagnostics. Profiling overhead must not be included in the production medians. Machine-readable before/after data and trace summaries are in [`raco_selu_webgpu_optimization_20261106.json`](raco_selu_webgpu_optimization_20261106.json).

## Limitations

- One GPU generation and one ONNX export were measured.
- ONNX Runtime trace durations are not GPU timestamp queries; copy events can include queue waits.
- Estimated transfer byte sums can count repeated tensors and do not indicate peak allocation.
- GPU peak allocation was not exposed by this runtime profile.
- Raw ONNX output tensors can differ due to floating-point reassociation; only the downstream analytic DIC results were verified across the four cases.
- This work improves graph placement but does not establish a shared allocation between ONNX Runtime WebGPU and wgpu-py.
