# Native 4K patch-CNN descriptor benchmark

- Adapter: `Intel(R) Graphics (LNL)` / Vulkan.
- Images: native 3840 × 2160 generated speckle; target shifted 37 pixels horizontally.
- One warm-up, three timed repetitions; medians exclude warm-up.

| Stage | HORG-64 | Patch CNN |
|---|---:|---:|
| Detector, two images | 86.7 ms | 86.7 ms |
| Descriptor, two images | 22.7 ms | 69.1 ms |
| Mutual matching | 92.6 ms | 81.6 ms |

The descriptor reads native-resolution grayscale pixels on GPU; its feature map stays in a storage buffer between WGSL passes. Only the final descriptor vectors return to Python, and the existing reciprocal matcher uploads them again. This is not an end-to-end GPU-resident pipeline. The translated roll is a throughput input, not an analytic accuracy benchmark.
