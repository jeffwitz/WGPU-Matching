# Serial and batched anchor confirmation

One image pair, one warm-up and alternating-order repetitions with identical cached solver pixels. Each run includes process startup, image-cache verification, graphics setup, solver work and output writes. The timers are wall-clock observations, not graphics processing unit (GPU) timestamps. The input cache and hypotheses were prepared beforehand; this is not full pipeline latency.

Median verification time, serial / batched: 0.775 / 0.167 s. Exact output parity across runs: True.

| Run | Warm-up | Serial verification (s) | Batched verification (s) | Calls serial / batched | Active windows serial / batched | Acceptance changes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | True | 0.810 | 0.177 | 233 / 17 | 2830 / 3869 | 0 |
| 1 | False | 0.775 | 0.169 | 233 / 17 | 2830 / 3869 | 0 |
| 2 | False | 0.682 | 0.165 | 233 / 17 | 2830 / 3869 | 0 |
| 3 | False | 0.794 | 0.167 | 233 / 17 | 2830 / 3869 | 0 |

All saved parameters and provenance are compared in the JavaScript Object Notation (JSON) output. Batching may perform speculative solves that the serial path would skip. The method is experimental and requires a window-independent deterministic solver.

Both modes returned 1,060 correct visible windows out of 1,121, with zero false
accepted visible windows. This is one previously inspected synthetic pair;
the images and initial hypotheses were prepared before the timing. For the
comparison on four newly generated pairs, see `../p2_anchor_batch_new_pairs/README.md`.

Reproduce in a **new** directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_anchor_confirmation.py \
  --inputs results/seed_propagation_20261111/p2_solver_images_pilot/inputs \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_anchor_batch_synthetic_new --repetitions 3
```
