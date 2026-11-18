# Shared global prefit: paired script benchmark

One synthetic 3840 × 2160 heterogeneous pair; one warm-up and measured alternating-order repetitions. The scripts include image synthesis, process/model startup, truth/oracle controls, the graphics processing unit (GPU) solve and disk output. The inverse-compositional Gauss–Newton (IC-GN) solver and the local normalized cross-correlation (NCC) use the same settings in both arms. These are prefit script times, not complete initializer or production latency. The graphics adapter is Intel Graphics Lunar Lake through Vulkan.

Exact equality for both initial fields, solver results, status, visibility and truth: True.
Median separate / shared: 14.704 / 7.549 s; ratio 1.95.

| Run | Warm-up | Separate (s) | Shared (s) | Exact parity |
|---:|---|---:|---:|---|
| 0 | True | 14.432 | 7.397 | True |
| 1 | False | 14.675 | 7.393 | True |
| 2 | False | 15.362 | 7.549 | True |
| 3 | False | 14.704 | 7.551 | True |

The detailed JSON records every comparison and all times. One pair does not establish general quality or runtime.

The model weights have SHA-256 (Secure Hash Algorithm with a 256-bit digest)
`150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`.
Reproduce in a **new** output directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_shared_prefit.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_shared_prefit_new --repetitions 3
```
