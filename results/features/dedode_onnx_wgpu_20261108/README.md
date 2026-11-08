# DeDoDe-B ONNX/WebGPU descriptor qualification (8 November 2026)

## Export and runtime

`python/benchmarks/export_dedode_descriptor_onnx.py` exports the official DeDoDe-B checkpoint as a fixed-shape keypoint descriptor graph. This experiment used a 1,024 by 576 input and 8,192 normalized sample points. The graph is 54,054,082 bytes; SHA-256 `e265973d70f74b63e4fcd3715e7565cc3cb05617c0ea8aacf78a2266e4c28e42`. The graph/checkpoint stay external to Git. The adapter pads short feature sets to the graph's fixed capacity and returns descriptors for actual points only.

ONNX Runtime 1.31.1 used its WebGPU execution provider on Vulkan Intel Graphics LNL. In an isolated profiled graph run, all 121 operator events were assigned to WebGPU; summed operator event duration was 782.4 ms and total `session.run` time was 1,074.7 ms. There were no CPU-assigned ONNX operator nodes in this trace. Input preprocessing, output allocation/copy, runtime queue waits, and Python-side coordinate preparation remain host work. This is an operator placement claim for this exported graph and shape, not zero-copy sharing with `wgpu-py`.

PyTorch/WebGPU descriptor parity was checked on 32 points from a deterministic UHD grayscale image using the same official checkpoint and preprocessing: output shape `(32, 256)`, maximum absolute component difference `5.46e-7`, mean absolute difference `9.72e-8`, minimum row cosine `0.99999976`. PyTorch reference ran on CPU; ONNX Runtime ran with WebGPU plus CPU fallback registered. The separate 8,192-point profile confirms all nodes used WebGPU.

## Full 60% tensile qualification

The same learned DIC detector, 8,192-point budget, image pair, seeds, local consensus, and affine projection were run with one timed repeat after warmup. The CPU baseline uses PyTorch DeDoDe-B and NumPy matching. The GPU path uses the fixed-shape ONNX/WebGPU descriptor and WGSL mutual matching. Both create exactly the same 1,496 direct matches (identical pair digest), pass all 16 visible centres below 5 pixels, and have the same 0.224077 pixel projected-centre RMSE.

| Path | Detection pair (s) | Description pair (s) | Mutual match + local API (s) | Geometric verification (s) | Complete affine call (s) | Correct centres <5 px |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch CPU descriptor + NumPy | 0.146 | 7.154 | 0.405 | 0.047 | 7.790 | 16/16 |
| ONNX/WebGPU descriptor + WGSL | 0.153 | 2.025 | 1.532 | 0.025 | 3.736 | 16/16 |

The descriptor stage is 3.5 times faster in this single paired trial, and complete extraction plus projection is 2.08 times faster. WGSL matching at 256 dimensions is slower than NumPy in this full workload, despite parity. The WebGPU descriptor path reduces the overall pair time because it removes most of the prior seven-second CPU descriptor cost. These are single-run qualification timings, not a latency distribution. Re-run with `--repeats 3` for production estimates.

Reproduce the export:

```bash
python python/benchmarks/export_dedode_descriptor_onnx.py /tmp/dedode_b_descriptor_1024x576_k8192.onnx --height 576 --width 1024 --keypoints 8192
```

Reproduce the paired qualification:

```bash
python python/benchmarks/qualify_all_initializers.py --output results/qualification/dedode_onnx_wgpu_60_20261108 --methods dic_learned_b --cases tensile_60pct --repeats 1 --keypoints 8192 --feature-matcher wgpu --descriptor-backend wgpu --descriptor-onnx /tmp/dedode_b_descriptor_1024x576_k8192.onnx
```

The ONNX graph is static-shape. This export covers the 16:9 tensile images at 1,024 by 576; Adrien's 7,920 by 6,004 images need a separate 1,024 by 776 export or a dynamic-shape graph. The two model stages still cross host memory: ONNX Runtime returns descriptors to NumPy, and `wgpu-py` uploads them into its own WebGPU allocation. No shared allocation between runtimes is claimed.
