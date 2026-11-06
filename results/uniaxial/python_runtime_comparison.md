# Native Python benchmark: HORG and DeepMatching

Measurements use 3840 × 2160 UHD images from the Live-DIC uniaxial tensile test, the `wgpu-matching` Python package, and `wgpu-py` on Vulkan / Intel Graphics (LNL). Each frame uses the same 32-POI grid. HORG uses `patch=128`, `stride=512`, reduction 4, and 1024 features. INRIA-mode DeepMatching uses `patch=128`, root 512, `stride=512`, 2 levels, and a 256 px radius.

Images are loaded before timing. Accuracy is evaluated only for POIs whose analytically transformed centers remain inside the target image; timing covers the complete detector call through GPU result readback. It includes matcher resource and pipeline creation but excludes device initialization. “First call” shows the first-use cost. The median is calculated from the next five calls. Raw data, repetitions, and per-frame accuracy are in [`python_runtime_comparison.csv`](python_runtime_comparison.csv); the reproducible script is [`benchmark_live_dic.py`](../../python/benchmarks/benchmark_live_dic.py).

| Frame | HORG first call | HORG repeated median | DeepMatching first call | DeepMatching repeated median | HORG speedup | HORG accepted / visible (≤2 px) | DeepMatching accepted / visible (≤2 px) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 040 | 354 ms | 105 ms | 1,822 ms | 1,826 ms | 17.5× | 32/32 (32) | 32/32 (30) |
| 100 | 102 ms | 113 ms | 1,820 ms | 1,759 ms | 15.5× | 28/28 (28) | 28/28 (22) |
| 199 | 97 ms | 102 ms | 1,738 ms | 1,839 ms | 18.1× | 23/28 (21) | 28/28 (27) |

Analytic RMSE, computed only for accepted POIs:

| Frame | HORG RMSE | DeepMatching RMSE |
|---:|---:|---:|
| 040 | 0.089 px | 1.314 px |
| 100 | 0.204 px | 1.625 px |
| 199 | 1.048 px | 1.165 px |

With these settings on these three frames, HORG takes about 0.11 s per repeated call and DeepMatching about 1.81 s, making HORG roughly 17× faster. HORG has lower RMSE among its accepted points on each frame; at frame 199, DeepMatching accepts more visible points and more are within 2 px. These measurements characterize this machine and dataset; image loading and device initialization are excluded.

DeepMatching errors are evaluated at the source coordinate actually reported by the matcher: its engine may move the requested center to the leaf grid. The corrected analytic-truth calculation is included in the raw CSV.
