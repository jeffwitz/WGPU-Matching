# Shared prefit and deferred rotations: six new cases

The [protocol](../p2_shared_lazy_rotations_protocol.md) was fixed before
generation. It covers two rigid rotations, two affine deformations and two
discontinuous openings on synthetic 3840 × 2160 images from seed groups
53000000 and 54000000. The eager and deferred paths use identical images,
windows, thresholds and inverse-compositional Gauss–Newton (IC-GN) settings.
Analytical truth is used only for final scoring.

| Family and seeds | Visible | Correct eager / deferred | Incorrect accepted eager / deferred |
|---|---:|---:|---:|
| Affine 53051003 / 53091003 | 1,039 | 987 / 987 | 0 / 0 |
| Opening 53051004 / 53091004 | 1,323 | 1,248 / 1,248 | 0 / 0 |
| Rigid 53051002 / 53091002 | 1,166 | 1,115 / 1,115 | 0 / 0 |
| Affine 54051003 / 54091003 | 1,416 | 1,373 / 1,373 | 0 / 0 |
| Opening 54051004 / 54091004 | 1,250 | 1,225 / 1,225 | 0 / 0 |
| Rigid 54051002 / 54091002 | 1,196 | 1,148 / 1,148 | 0 / 0 |
| **Total** | **7,390** | **7,096 / 7,096** | **0 / 0** |

The true rigid rotation angles are approximately −69.7° and +68.5°; these
values were used only for diagnostic reporting. Accepted-window masks are
identical for all six pairs. The eager path produced 18 forced-rotation
hypotheses; the deferred path required none after checking the initial anchors.
Those initial affine and normalized cross-correlation (NCC) inputs already use
an image-derived global search. This is not evidence that rotation search was
eliminated altogether.

Shared and separate prefit archives agree exactly on initial parameters,
IC-GN outputs, status, visibility and truth for all six pairs. The exact
comparisons are `parity_53000000.json` and `parity_54000000.json`; detailed
pairwise and per-window outputs are in `lazy/evaluation/`. The material
boundary in both opening cases lies on the window-grid boundary, so these
cases do not qualify supports crossing an opening. Six new pairs do not prove
general robustness or experimental-image accuracy. No production or
GPU-resident latency is reported. The adapter was Intel Graphics Lunar Lake
using Vulkan.

The checkout started at commit `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93`
with local modifications. Source and model fingerprints are the same as in
the four-pair report `../p2_shared_lazy_new_pairs/README.md`; no inference or
solver source was changed between these two campaigns.

Reproduce in a **new** output directory, with Bash:

```bash
set -euo pipefail
task_dir=results/seed_propagation_20261111/p2_shared_lazy_rotations_reproduction
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
export PYTHONPATH=python/src:python/benchmarks
for variant in separate shared; do
  extra=()
  if [[ $variant == shared ]]; then extra+=(--shared-prefit); fi
  .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
    --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
    --output-dir "$task_dir/$variant" --seed-offsets 53000000 54000000 \
    --families rigid affine discontinuous --cases-per-family 1 --retain-translation-affine "${extra[@]}"
done
for seed in 53000000 54000000; do
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
    --output-dir "$task_dir/$mode" --seed-offsets 53000000 54000000 \
    --families rigid affine discontinuous --cases-per-family 1 --skip-g2-summary "${extra[@]}"
done
.venv/bin/python python/benchmarks/summarize_lazy_rotation.py \
  --eager "$task_dir/eager/evaluation" --lazy "$task_dir/lazy/evaluation" --new-pairs
```
