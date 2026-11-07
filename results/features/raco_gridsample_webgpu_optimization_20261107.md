# RaCo `GridSample` placement optimization — 2026-11-07

## Result

An opt-in ONNX graph rewrite moves the 37 zero-padded bilinear `GridSample`
operations in the RaCo-ALIKED-LightGlue+ export off the CPU. The rewrite computes
the bilinear coordinates and weights and gathers the four neighboring feature
vectors with `GatherND(batch_dims=1)`. The two border-padded `GridSample` nodes
remain unchanged because this rewrite implements zero padding only.

On four analytic DIC pairs, the rewritten graph is 1.34–1.43× faster than the
same model with only the previously measured SELU rewrite. Every per-seed DIC
record is identical between those two runs, including usability, target
coordinates, and analytic errors.

| Case | Visible seeds | Accepted | Correct ≤2 px | Correct <5 px | RMSE accepted (px) | Pair time before → after | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frame 040 | 32 | 26 | 26 | 26 | 0.220 | 1.454 → 1.081 s | 1.34× |
| Frame 100 | 28 | 23 | 23 | 23 | 0.451 | 1.453 → 1.058 s | 1.37× |
| Frame 199 | 28 | 23 | 23 | 23 | 0.740 | 1.502 → 1.070 s | 1.40× |
| Synthetic tensile, 60% | 16 | 13 | 6 | 12 | 3.449 | 1.506 → 1.052 s | 1.43× |

The original images are 3840×2160. Both models receive 640×360 RGB images padded
to 640×384. Timings are medians of three unprofiled repetitions after one warm-up;
they include ONNX inference and the local affine fit, but exclude image decoding
and resizing. The local affine settings and seeds are unchanged. This is a
reduced-resolution global matcher benchmark, not native-4K inference or the
proposed local 4K refinement pipeline.

## Operator placement and transfers

Each diagnostic profile contains eight model calls: one warm-up and one timed
call for each of the four pairs. Profiling is separate from production timing.
The values below are ONNX Runtime trace summaries, not GPU timestamp queries or
measured bus bandwidth.

| Profile item, 8 calls | SELU-only baseline | GridSample rewrite |
|---|---:|---:|
| CPU `GridSample` calls | 312 | 16 |
| CPU `GridSample` accumulated trace duration | 457.5 ms | 1.4 ms |
| CPU provider node events | 3,672 | 7,696 |
| CPU provider accumulated node duration | 622.1 ms | 176.9 ms |
| Estimated `MemcpyToHost` bytes | 2.092 GB | 86.8 MB |
| Estimated `MemcpyFromHost` bytes | 941.0 MB | 613.9 MB |
| Maximum single `MemcpyToHost` tensor | 251.7 MB | 4.2 MB |
| WebGPU `GatherND` calls | 8 | 1,192 |

`MemcpyToHost` falls by 95.9% and `MemcpyFromHost` by 34.8%. The new graph has
more CPU node events overall because each `GridSample` expands into primitive
operations, but the accumulated CPU operator duration falls by 71.6%. `Expand`
and other CPU work remain. The graph is therefore **not fully GPU-resident**.

The CPU `GridSample` profile entries remaining after the rewrite are the two
border-padded operations per inference. Across eight calls, the CPU `GridSample`
input tensor sum falls from 2.269 GB to 5.6 MB and its output sum from 373.2 MB
to 0.13 MB. The largest input to one CPU `GridSample` event falls from 251.9 MB
to 0.44 MB. The WebGPU gathers read the feature tensor on-device and produce
sampled features; the full feature tensor is not sent to the CPU for those 37
operations.

## Follow-up measured and rejected

A second opt-in experiment replaced the two CPU `LogSoftmax` nodes with a stable
`ReduceMax/Sub/Exp/ReduceSum/Log/Sub` decomposition. It reduced estimated
host-bound bytes further, but its three-repeat pair times were 1.16–1.55 s,
slower than the 1.05–1.08 s `GatherND`-only variant. The rewrite was removed from
the implementation; it is not enabled by the benchmark or exposed as a supported
option. The profile suggests extra GPU work and synchronization may outweigh the
smaller transfer volume, but this trace does not isolate that as a causal result.

## Reproduction

The transformations are opt-in and operate on temporary copies. They do not
modify the downloaded source model or change the default RaCo path.

```bash
PYTHONPATH=python/src:python/benchmarks python python/benchmarks/experiment_aliked_lightglue.py \
  --model /path/to/raco_aliked_lightglue_pipeline_k1024.onnx \
  --decompose-selu --decompose-grid-sample --repeats 3 \
  --output /tmp/raco-gathernd-webgpu.json
```

Enable `--profile` in a separate diagnostic run. The source model SHA-256 is
`575e1746138204e19027cb37541cd840c7fc9f2c932b844e9685354a3913bda1`; the
optimized runtime model SHA-256 for this run is
`c41f27ab1585f53e7e8f4709c657fc419d32f113964d6f2802bed0db710728b3`.
The complete timing, per-seed results, profile summaries and limitations are in
[`raco_gridsample_webgpu_optimization_20261107.json`](raco_gridsample_webgpu_optimization_20261107.json).

## Remaining performance work

The optimized matcher is around 1.05 s per pair, still above the 0.8 s RaCo
target. The profile continues to show CPU `Expand` work, two small border-mode
`GridSample` calls, and data transfers between ONNX Runtime's WebGPU and CPU
providers. The transfer report is an ORT estimate, not proof of memory sharing
with `wgpu-py`. The next step should target these measured costs independently;
native-4K local refinement remains unimplemented.
