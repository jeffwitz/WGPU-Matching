# Predeclared shifted-opening transfer validation

Defined before generation: test five new deterministic discontinuous-opening pairs from texture/warp seed group 63000000 at 3840 × 2160. The analytical opening boundary is x = 0.553 × 3840 = 2123.52 pixels, deliberately offset from the 64-pixel grid. The boundary is used only after inference to classify supports and provenance; no boundary, visibility or ground-truth labels are supplied to seed selection, propagation or fine transfer. The same frozen P2 model, 64-pixel solver and nearest-accepted 32-pixel re-solve protocol and thresholds are used, including the equal-work control.

Compare accepted/correct/incorrect visible windows at identical original centers, accepted invisible windows, and source provenance. Separately count fine subsets crossing the material boundary, 64-pixel source subsets crossing it, and fine centers whose accepted source is on the other side. Do not interpret a correct center displacement as proof of a physically valid support. A group of five synthetic pairs is limited evidence and does not validate unknown fissures or overturn the original G3 failure. Preserve all negative results and do not tune thresholds.

## Provenance and reproduction

The five-pair run used Intel Graphics Lunar Lake through WebGPU/Vulkan, repository HEAD `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93` with uncommitted changes, and checkpoint SHA-256 `150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`. Cached solver inputs include image hashes. The same core source hashes as the multifamily run are listed in `../p3_fine_transfer_multifamily_independent/PROTOCOL.md`.

```bash
base=results/seed_propagation_20261111/p3_fine_transfer_shifted_opening_independent_new
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
  --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" --output-dir "$base/inputs" \
  --seed-offsets 63000000 --families discontinuous --cases-per-family 5 --discontinuity-boundary-fraction .553 \
  --retain-translation-affine --shared-prefit --export-solver-images
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --archive "$base/inputs" --live-dic-source "$solver" --output-dir "$base/coarse" \
  --lazy-rotation-weights "$weights" --lazy-rotation-output "$base/rotations" --frozen-parameters "$frozen" \
  --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3 --discontinuity-boundary-fraction .553 \
  --escalation --local-affine-escalation --batch-anchor-confirmation --same-budget-control --use-cached-solver-images
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_fine_same_centres.py \
  --archive "$base/inputs" --coarse "$base/coarse" --seed-offset 63000000 --live-dic-source "$solver" --output-dir "$base/fine"
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_shifted_opening.py \
  --directory "$base/fine" --boundary 2123.52 --archive "$base/inputs" --coarse "$base/coarse" --seed-offset 63000000
```
