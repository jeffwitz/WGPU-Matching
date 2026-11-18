# Paired complete scripted pipeline with cached solver pixels

One synthetic 3840 × 2160 pair, one warm-up and alternating-order measured repetitions, on Intel Graphics Lunar Lake using Vulkan. Each arm includes all image generation, matching, process startup, diagnostics and result writes. The cached arm also includes encoding, writing, hashing and reading both images. Times are not integrated production or graphics-processing-unit (GPU) resident latency. No other project benchmark was observed when this run started or at a later spot check; the GPU was not reserved exclusively.

Median regenerated / cached full script: 16.451 / 11.541 s. Exact state, status and provenance parity: True.

| Run | Warm-up | Regenerated (s) | Cached (s) | Exact parity |
|---:|---|---:|---:|---|
| 0 | True | 15.861 | 11.451 | True |
| 1 | False | 16.012 | 11.435 | True |
| 2 | False | 16.451 | 11.541 | True |
| 3 | False | 17.140 | 11.800 | True |

The saved JavaScript Object Notation (JSON) data contain stage times, quality and per-window outputs. One pair does not establish performance or accuracy in general.

Each arm recovered 1,060 correct windows out of 1,121 visible, with no false
accepted visible window. The paired runs have exactly equal parameters,
acceptance status, seed masks and propagation provenance. The cached image
files occupied about 9.3 MiB (mebibytes, units of 2²⁰ bytes). This saving is
specific to avoiding synthetic-pair regeneration; the [Adrien check](../p2_solver_images_adrien/README.md)
did not demonstrate a reading-speed benefit on real images.

Reproduce in a **new** output directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_cached_pipeline_paired.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_cached_pipeline_paired_new --repetitions 3
```
