# Cached solver pixels with local-affine fallback

The previously examined synthetic 3840 × 2160 pair 17051007 / 17091007 has no
initial global anchors. In otherwise identical image-only runs, one regenerates
the solver images and one reads the fingerprint-verified cache. Both request
additional rotations and local-affine fallback, and both produce **736 correct
visible windows out of 787**, with zero visible false accepted windows. Saved
parameters, accepted masks, parents, origins, generations and confirmed seeds
are exactly equal. The second run reuses the verified rotation cache; these runs
are a quality check, not a timing comparison. Analytical truth was used only
to score results after initialization.

Reproduce in a **new** directory:

```bash
set -euo pipefail
task_dir=results/seed_propagation_20261111/p2_solver_images_fallback_reproduction
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
export PYTHONPATH=python/src:python/benchmarks
.venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
  --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
  --output-dir "$task_dir/inputs" --seed-offsets 17000000 --families heterogeneous \
  --cases-per-family 1 --case-start-index 3 --retain-translation-affine --shared-prefit --export-solver-images
for mode in regenerated cached; do
  extra=()
  if [[ $mode == cached ]]; then extra+=(--use-cached-solver-images); fi
  .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
    --live-dic-source "$solver" --archive "$task_dir/inputs" --output-dir "$task_dir/$mode" \
    --sparse-stride 4 --lazy-rotation-weights "$weights" --lazy-rotation-output "$task_dir/rotations" \
    --textures 17051007 --escalation --local-affine-escalation "${extra[@]}"
done
```
