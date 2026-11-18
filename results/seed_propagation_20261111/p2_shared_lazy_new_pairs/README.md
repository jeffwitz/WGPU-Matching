# Shared prefit and deferred rotations on four new synthetic pairs

The [protocol](../p2_shared_lazy_new_pairs_protocol.md) was fixed before the
experiment. It uses two translation and two heterogeneous 3840 × 2160 image
pairs from seed groups 51000000 and 52000000. The original eager path and the
shared-prefit/deferred-rotation path use the same images, windows, matching
settings and inverse-compositional Gauss–Newton (IC-GN) solver settings.
Analytical truth is used only for the final score.

| Family and seeds | Visible | Correct eager / deferred | Incorrect accepted eager / deferred | Additional rotations requested |
|---|---:|---:|---:|---|
| Heterogeneous 51051001 / 51091001 | 1,091 | 1,028 / 1,028 | 0 / 0 | no |
| Translation 51051000 / 51091000 | 1,560 | 1,450 / 1,450 | 0 / 0 | no |
| Heterogeneous 52051001 / 52091001 | 1,107 | 1,049 / 1,049 | 0 / 0 | no |
| Translation 52051000 / 52091000 | 1,484 | 1,484 / 1,484 | 0 / 0 | no |
| **Total** | **5,242** | **5,011 / 5,011** | **0 / 0** | **0 / 4 pairs** |

All four accepted-window masks are identical. The independently generated
shared and separate inputs also give exactly equal affine-only and
affine-plus-normalized cross-correlation (NCC) initial parameters, solver
parameters, solver status, visibility and truth on every window. The detailed
array checks are in `parity_51000000.json` and `parity_52000000.json`; the
pairwise comparison and per-window archives are in `lazy/evaluation/`.

This is a four-pair independent regression sample, not a statistically
conclusive robustness study. It does not cover independent rotations,
discontinuities or experimental truth. The affine and NCC image inputs still
include a coarse global image-derived ranking. No complete production or
GPU-resident latency is claimed. The separate eager reference generated all
12 extra rotation hypotheses; the deferred path generated none for these four
pairs. The adapter was Intel Graphics Lunar Lake using Vulkan.

The checkout started at commit `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93`
with local changes. SHA-256 (Secure Hash Algorithm with a 256-bit digest)
fingerprints at the run:

- `qualify_ndf_pair_icgn.py`: `171ffaade637e4417427cb84b155f3367a57a44d0f041356b0a7f9f6e53092da`.
- `generate_p2_test_inputs.py`: `d812516ae0890282da2b808d75f32c3f091bf7273dfdaace4aaa491be46a9adb`.
- `qualify_image_seed_pilot.py`: `2c4d60678ef0ab95073e77f7a80626a9c3ee8c6ca2c4c82c4fe7e852df1a2fe1`.
- `run_image_seed_test.py`: `2ed334aea0551179b07db66bab54067385080f160177667baa86e216e122b302`.
- `seed_initialization.py`: `aa7dc227f4c6fc8c598d585d0f01cc7901b754f32aed9a293f8049034a8f5ef4`.
- Network weights: `150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`.

Reproduce in a **new** output directory, with Bash:

```bash
set -euo pipefail
task_dir=results/seed_propagation_20261111/p2_shared_lazy_new_pairs_reproduction
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
export PYTHONPATH=python/src:python/benchmarks
for variant in separate shared; do
  extra=()
  if [[ $variant == shared ]]; then extra+=(--shared-prefit); fi
  .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
    --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
    --output-dir "$task_dir/$variant" --seed-offsets 51000000 52000000 \
    --families translation heterogeneous --cases-per-family 1 --retain-translation-affine "${extra[@]}"
done
for seed in 51000000 52000000; do
  .venv/bin/python python/benchmarks/compare_shared_prefit_archives.py \
    --separate "$task_dir/separate" --shared "$task_dir/shared" \
    --seed-offset "$seed" --output "$task_dir/parity_$seed.json"
done
for mode in eager lazy; do
  input="$task_dir/separate"
  extra=()
  if [[ $mode == lazy ]]; then input="$task_dir/shared"; extra+=(--lazy-rotations); fi
  .venv/bin/python python/benchmarks/run_image_seed_test.py \
    --inputs "$input" --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
    --output-dir "$task_dir/$mode" --seed-offsets 51000000 52000000 \
    --families translation heterogeneous --cases-per-family 1 --skip-g2-summary "${extra[@]}"
done
.venv/bin/python python/benchmarks/summarize_lazy_rotation.py \
  --eager "$task_dir/eager/evaluation" --lazy "$task_dir/lazy/evaluation" --new-pairs
```
