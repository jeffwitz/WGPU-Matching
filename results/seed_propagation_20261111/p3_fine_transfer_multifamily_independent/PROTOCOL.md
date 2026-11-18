# Predeclared independent multifamily fine-subset transfer

Defined before generation and inference. Use five new texture/warp seeds from group 62000000 in each of five synthetic families: translation, rigid rotation, affine, heterogeneous and discontinuous opening, for 25 deterministic pairs at 3840 × 2160. The default opening is aligned with the 64-pixel grid; results cannot establish behavior for shifted cracks or unknown boundaries. No periodic textures are included, per project scope.

Generate inputs using the existing affine + local normalized cross-correlation prefit and the frozen parameters. Analytic ground truth and its Jacobian are evaluated at exact requested window centers; compute visibility at those centers. Image-only 64/64 initialization and affine propagation use the existing frozen criteria, no truth or visibility in hypothesis choice. Use accepted 64-pixel states to initialize 32-pixel supports at the same original 64-grid centers via the unchanged nearest-accepted transfer with radius 96 px; IC-GN at 32/16 is run only for available candidates, with linear interpolation, 30 iterations, tolerance 0.001 and zncc_min 0.65. Accepted fine outputs count only after re-solving. No source or threshold is selected using truth or family labels.

For every pair and family, record visible, accepted correct, accepted incorrect >=2 px, invisible accepted, originally incorrect recovered, originally correct lost, and coverage after each method. For the opening family, distinguish fine and source supports crossing the known discontinuity and sources from the opposite side; use boundary labels only after inference. The sample is paired by image, not by window; intervals will bootstrap pairs with a fixed random seed. This is a limited independent test of the fixed transfer procedure, not a wholesale replacement of the original G3 protocol or a claim about experimental accuracy. Preserve negative results without tuning.

## Provenance and reproduction

Execution used Intel Graphics Lunar Lake with WebGPU/Vulkan, repository HEAD `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93` plus uncommitted working-tree changes, and checkpoint SHA-256 `150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`. Source SHA-256 at execution: `dic_synthetic.py` `12caffd3d505cbb38d5b5a9f4174604be42eb4a268b517f061a25591754f509c`; `qualify_ndf_pair_icgn.py` `1af48eb9fbac66a4a19101da7b820689a89f292bcf9a499cd14037ac2d38c10c`; `qualify_image_seed_pilot.py` `6cde3d588bc431ec73e5e3d0284e03ae6c146799ed024fd1ca64cd0675a1a3a8`; `qualify_fine_same_centres.py` `99b644dc13957fd4f8335f8668327c567ed76e7342be6636497aeb931f5c5fdd`; external `live_dic/wgpu_backend.py` `f0e7cad5d0cb5e017cb23b341c05197ecaf91ff683b34fbd0b8b24d4fe30b238`. Solver input caches retain image hashes. The same texture seed can occur in different families, while each family has five distinct pairs.

```bash
base=results/seed_propagation_20261111/p3_fine_transfer_multifamily_independent_new
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
  --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" --output-dir "$base/inputs" \
  --seed-offsets 62000000 --families translation rigid affine heterogeneous discontinuous --cases-per-family 5 \
  --retain-translation-affine --shared-prefit --export-solver-images
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --archive "$base/inputs" --live-dic-source "$solver" --output-dir "$base/coarse" \
  --lazy-rotation-weights "$weights" --lazy-rotation-output "$base/rotations" --frozen-parameters "$frozen" \
  --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3 --escalation --local-affine-escalation \
  --batch-anchor-confirmation --same-budget-control --use-cached-solver-images
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_fine_same_centres.py \
  --archive "$base/inputs" --coarse "$base/coarse" --seed-offset 62000000 --live-dic-source "$solver" --output-dir "$base/fine"
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_fine_multifamily.py \
  --directory "$base/fine" --archive "$base/inputs" --coarse "$base/coarse" --seed-offset 62000000
```
