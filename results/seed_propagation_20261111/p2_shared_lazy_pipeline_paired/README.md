# Shared prefit and deferred rotations: full scripted pair

Synthetic 3840 × 2160 heterogeneous pair, one warm-up and repeated full scripted runs on Intel Graphics Lunar Lake through Vulkan. Times include synthetic-image generation, model and process startup, truth/oracle diagnostics, graphics processing unit (GPU) solve and final result writes. They do not establish integrated production or GPU-resident latency.

Median prefit / verification / total: 7.364 / 7.102 / 14.465 s.

| Run | Warm-up | Total (s) | Correct / visible | False accepted | Lost / gained vs eager | Rotations requested |
|---:|---|---:|---:|---:|---:|---|
| 0 | True | 14.321 | 1060/1121 | 0 | 0 / 0 | False |
| 1 | False | 14.465 | 1060/1121 | 0 | 0 / 0 | False |
| 2 | False | 14.272 | 1060/1121 | 0 | 0 / 0 | False |
| 3 | False | 14.546 | 1060/1121 | 0 | 0 / 0 | False |

The eager reference was measured in an earlier run, not a contemporaneous paired timing. The saved JSON retains per-run stage times and outputs; one case cannot establish general performance.

The earlier eager reference has a 45.37 s median on the same pair, with three
repetitions and comparable script boundaries. Its measurements were taken in a
separate campaign; the comparison is indicative, not a paired speedup experiment.
On each of the four new runs, the accepted-window mask and all parameters at
accepted centres equal the saved eager reference exactly. No extra rotations
were requested. The first inference still runs at 192 × 108, while the
inverse-compositional Gauss–Newton (IC-GN) solve uses the native pixels.

Reproduce in a **new** output directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_shared_lazy_pipeline.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --reference-evaluation results/seed_propagation_20261111/p2_scripted_timing_isolated/run1/evaluation \
  --output-dir results/seed_propagation_20261111/p2_shared_lazy_pipeline_new --repetitions 3
```
