# Campaign-runner integration of deferred rotations

The campaign runner's optional `--lazy-rotations` path was exercised on the ten
previously inspected translation pairs in group 24000000. It selected exactly
the requested family and seed group, obtained 14,199 correct visible windows,
and reported no visible false acceptance. No extra rotation was requested.
Global affine and normalized cross-correlation (NCC) inputs were precomputed;
these are not end-to-end timing measurements or independent qualification.

Reproduce:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/run_image_seed_test.py \
  --inputs results/seed_propagation_20261111/p2_confirmatory_inputs \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --output-dir results/seed_propagation_20261111/p2_lazy_runner_translation24000000 \
  --seed-offsets 24000000 --families translation --cases-per-family 10 --skip-g2-summary --lazy-rotations
```
