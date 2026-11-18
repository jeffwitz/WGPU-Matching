# Reuse of fingerprinted solver images

One 3840 × 2160 synthetic pair, one warm-up and measured alternating-order repetitions. Each run starts a fresh process and graphics context. The cached image files were created earlier; these verification-only times exclude their creation and the initial matching. They are not end-to-end production or GPU-resident timings.

Median verification process: regenerated 6.941 s; cached 2.332 s. Exact state, status and provenance parity: True.

| Run | Warm-up | Regenerated (s) | Cached (s) | Exact parity |
|---:|---|---:|---:|---|
| 0 | True | 7.010 | 2.472 | True |
| 1 | False | 6.941 | 2.316 | True |
| 2 | False | 6.812 | 2.541 | True |
| 3 | False | 7.038 | 2.332 | True |

The source window archive and both image files are checked by SHA-256 (Secure Hash Algorithm with a 256-bit digest) before reuse. Cached values are solver inputs, not accepted measurements. Full stage details are in the saved JavaScript Object Notation (JSON) file.

Reproduce in a **new** directory after preparing inputs with
`--shared-prefit --export-solver-images`:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_solver_image_cache.py \
  --inputs results/seed_propagation_20261111/p2_solver_images_pilot/inputs \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_solver_images_benchmark_new --repetitions 3
```
