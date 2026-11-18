# Predeclared independent shear-band transfer test

Defined before image generation and evaluation. Select five new synthetic texture/warp pairs from seed group 61000000, previously unused in this campaign, for the smooth localized shear-band family with peak displacement gradient 2.0. Pair identity, exact analytical truth, reference and target are deterministic. This is a narrow test of one deformation family, not a statistically broad qualification of all DIC conditions.

Apply the frozen image-only initialization and 64-pixel/64-step IC-GN propagation without changes; use the same 30 iterations, tolerance 0.001, minimum normalized cross-correlation 0.65, `tau_u=8`, `tau_g=.25`, sparse step 2 and minimum three anchors. Enable the same-work re-solve control. Without using truth or visibility, initialize 32-pixel supports at the **same original 64-grid centers** with nearest accepted 64-pixel states and the existing 96-pixel maximum transport distance; re-solve using 32/16 linear IC-GN and the same solver thresholds. Count only accepted re-solves. Use exact analytical truth and recomputed exact-center visibility solely after each solve. Preserve all source indices, per-window results and false acceptances.

Compare correct coverage, accepted false count and fraction, accepted invisible, the error among accepted windows, formerly correct lost and formerly wrong recovered. Report any spatial concentration, source correctness by post-hoc truth and errors >=2 px. Do not change thresholds or select samples after seeing results. No extrapolation to real experimental accuracy. The original G3 gate and its failure remain unchanged.

## Reproduction

Executed on the Intel Graphics Lunar Lake WebGPU/Vulkan device; the source repository contained uncommitted changes on HEAD `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93`. The fixed pretrained checkpoint has SHA-256 `150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`. The input caches contain the exact quantized solver images and their hashes. Independent means the texture/warp seeds were not used in the preceding shear-band development runs; this single family remains a limited scientific sample.

```bash
base=results/seed_propagation_20261111/p3_shear_32_independent_new
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
  --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
  --output-dir "$base/g2.0/inputs" --seed-offsets 61000000 --families shear_band --cases-per-family 5 \
  --shear-band-peak-gradient 2.0 --retain-translation-affine --shared-prefit --export-solver-images
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --archive "$base/g2.0/inputs" --live-dic-source "$solver" --output-dir "$base/g2.0/evaluation" \
  --lazy-rotation-weights "$weights" --lazy-rotation-output "$base/g2.0/rotations" \
  --frozen-parameters "$frozen" --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3 \
  --shear-band-peak-gradient 2.0 --escalation --local-affine-escalation --batch-anchor-confirmation \
  --same-budget-control --use-cached-solver-images
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_shear_exact_truth.py \
  --base "$base" --live-dic-source "$solver" --output-dir "$base/exact_center_analysis" \
  --seed-offset 61000000 --gradients 2.0 --control-directory-name evaluation
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/diagnose_shear_fine_transfer.py \
  --base "$base" --live-dic-source "$solver" --output-dir "$base/transfer" --seed-offset 61000000 --gradient 2.0
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_shear_fine_transfer.py \
  --base "$base" --directory "$base/transfer" --seed-offset 61000000 --gradient 2.0 --independent
```
