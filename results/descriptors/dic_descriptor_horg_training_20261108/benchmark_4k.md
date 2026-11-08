# Native 4K descriptor benchmark

- Adapter: `Intel(R) Graphics (LNL)`, Vulkan / Vulkan
- Source and target are generated at 3840 × 2160 and kept at native resolution. Target is rolled by 37 pixels horizontally.
- First repetition is warm-up; medians use the following three repetitions.
- DIC detector: median 90.7 ms for two images.
- HORG-64 descriptor: median 22.7 ms; reciprocal WGSL matching 92.6 ms.
- Trained 64D multiscale descriptor: median 83.9 ms; reciprocal WGSL matching 93.4 ms.
- Matching outputs only compact correspondence results to Python; descriptor extraction currently reads HORG features back to host before uploading for projection, and matching uploads projected descriptors again. This is GPU-kernel execution, but not a GPU-resident end-to-end path.
