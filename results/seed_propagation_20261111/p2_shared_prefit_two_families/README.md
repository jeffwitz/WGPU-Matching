# Shared-prefit parity on two additional pairs

On one translation and one heterogeneous synthetic 3840 × 2160 pair, the
shared and separate preparations produce exactly equal affine-only and
affine-plus-local-correlation initial parameters, inverse-compositional
Gauss–Newton (IC-GN) solver states, solver status, visibility and analytic
truth. Each point is compared, not just aggregate coverage. These pairs had
been examined in prior experiments and do not constitute a new independent
qualification. The detailed comparisons are in `parity.json`.

Reproduce in a new directory:

```bash
for variant in separate shared; do
  extra=()
  if [[ $variant == shared ]]; then extra+=(--shared-prefit); fi
  PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
    --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
    --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
    --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
    --output-dir results/seed_propagation_20261111/p2_shared_prefit_two_families_new/$variant \
    --seed-offsets 26000000 --families translation heterogeneous --cases-per-family 1 \
    --retain-translation-affine "${extra[@]}"
done
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/compare_shared_prefit_archives.py \
  --separate results/seed_propagation_20261111/p2_shared_prefit_two_families_new/separate \
  --shared results/seed_propagation_20261111/p2_shared_prefit_two_families_new/shared \
  --seed-offset 26000000 --output results/seed_propagation_20261111/p2_shared_prefit_two_families_new/parity.json
```
