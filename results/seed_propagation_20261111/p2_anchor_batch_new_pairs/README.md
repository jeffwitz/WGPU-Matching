# Batched anchor confirmation on four new synthetic pairs

The [protocol](../p2_anchor_batch_new_pairs_protocol.md) was fixed before
generation. Two heterogeneous and two rigid 3840 × 2160 pairs were generated
from seed groups 55000000 and 56000000. Both modes use identical image-derived
initial hypotheses, fingerprint-verified solver pixels, windows, thresholds
and inverse-compositional Gauss–Newton (IC-GN) settings. Truth is used only to
score outputs after inference.

| Pair | Visible | Correct serial / batched | False accepted serial / batched | Solver calls serial / batched | Active windows serial / batched |
|---|---:|---:|---:|---:|---:|
| Heterogeneous 55051001 / 55091001 | 1,545 | 1,492 / 1,492 | 0 / 0 | 337 / 17 | 3,129 / 4,675 |
| Heterogeneous 56051001 / 56091001 | 865 | 808 / 808 | 0 / 0 | 183 / 17 | 2,651 / 3,443 |
| Rigid 55051002 / 55091002 | 1,356 | 1,299 / 1,299 | 0 / 0 | 306 / 15 | 2,848 / 4,225 |
| Rigid 56051002 / 56091002 | 1,134 | 1,096 / 1,096 | 0 / 0 | 243 / 15 | 2,634 / 3,744 |
| **Total** | **4,900** | **4,695 / 4,695** | **0 / 0** | **1,069 / 64** | **11,262 / 16,087** |

All saved centres, parameters, accepted masks, confirmed anchors and
propagation parents, origins and generations are exactly equal for every pair.
The four-pair sample is not a complete robustness qualification. Each timing
is a single diagnostic execution, so no general speed claim is drawn here.
Batching evaluates some proposals the serial algorithm would have skipped;
the work and launch trade-off must be measured on each target device. The
graphics processing unit (GPU) solver is used, while scheduling remains on the
host. Detailed comparisons and per-window outputs are in `comparison.json`,
`serial/` and `batched/`.

The eight-candidate guard described below was added after these runs. The
selected hypotheses had respectively 328, 174, 299 and 236 candidate anchors,
so all four use the same batched branch with the guard in place.

For the difficult previously inspected pair with initially zero anchors, an
initial unconditional batching experiment needed 70 calls instead of 64. The
implementation now retains serial confirmation when there are at most eight
candidate anchors: there can be no benefit from launching up to eight batches
for at most eight serial parent calls. The [fallback check](../p2_anchor_batch_fallback_v2/comparison.json)
then gives 64 calls in both modes and exactly equal output. This is an
orchestration rule, not a change to any convergence or matching threshold.

Reproduce in a **new** output directory:

```bash
set -euo pipefail
task_dir=results/seed_propagation_20261111/p2_anchor_batch_new_pairs_reproduction
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
export PYTHONPATH=python/src:python/benchmarks
.venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
  --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
  --output-dir "$task_dir/inputs" --seed-offsets 55000000 56000000 \
  --families heterogeneous rigid --cases-per-family 1 --retain-translation-affine --shared-prefit --export-solver-images
for mode in serial batched; do
  extra=()
  if [[ $mode == batched ]]; then extra+=(--batch-anchor-confirmation); fi
  .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
    --live-dic-source "$solver" --archive "$task_dir/inputs" --output-dir "$task_dir/$mode" \
    --sparse-stride 2 --lazy-rotation-weights "$weights" --lazy-rotation-output "$task_dir/rotations" \
    --escalation --local-affine-escalation --use-cached-solver-images --frozen-parameters "$frozen" "${extra[@]}"
done
.venv/bin/python python/benchmarks/summarize_anchor_confirmation.py \
  --serial "$task_dir/serial" --batched "$task_dir/batched" --output "$task_dir/comparison.json"
```
