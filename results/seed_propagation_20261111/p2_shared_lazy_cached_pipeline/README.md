# Shared prefit and deferred rotations: full scripted pair

Synthetic 3840 × 2160 heterogeneous pair, one warm-up and repeated full scripted runs. Times include synthetic-image generation, model and process startup, truth/oracle diagnostics, graphics processing unit (GPU) solver and final result writes. With image caching enabled, image encoding, writing, fingerprinting and reading are included in these times. They do not establish integrated production or GPU-resident latency.

Median prefit / verification / total: 7.928 / 2.443 / 10.288 s.

| Run | Warm-up | Total (s) | Correct / visible | False accepted | Lost / gained vs eager | Rotations requested |
|---:|---|---:|---:|---:|---:|---|
| 0 | True | 10.517 | 1060/1121 | 0 | 0 / 0 | False |
| 1 | False | 10.267 | 1060/1121 | 0 | 0 / 0 | False |
| 2 | False | 10.288 | 1060/1121 | 0 | 0 / 0 | False |
| 3 | False | 10.603 | 1060/1121 | 0 | 0 / 0 | False |

The eager reference was measured in an earlier run, not a contemporaneous paired timing. The saved JavaScript Object Notation (JSON) file retains per-run stage times and outputs; one case cannot establish general performance.

Reproduce in a new directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_shared_lazy_pipeline.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json --reference-evaluation results/seed_propagation_20261111/p2_scripted_timing_isolated/run1/evaluation \
  --output-dir results/seed_propagation_20261111/p2_shared_lazy_cached_pipeline_new --seed-offset 25000000 --repetitions 3 --cache-solver-images
```
