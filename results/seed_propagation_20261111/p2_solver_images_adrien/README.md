# Verified solver pixels on the masked Adrien pair

The [protocol](../p2_solver_images_adrien_protocol.md) was fixed before this
check. The reference and target images are used at their native 7920 × 6004
resolution with the supplied region mask. The coarse image model uses a
192 × 146 raster. One execution reads the original images; the other reads the
exact quantized solver inputs exported during shared-prefit preparation after
checking the image, source-file and window-archive SHA-256 fingerprints
(Secure Hash Algorithm with a 256-bit digest).

Both executions accept **7,097 windows**, including **6,321 confirmed anchors**.
The accepted-window masks, all saved parameters, parents, origins, generations,
confirmed anchors and valid-region masks are exactly equal. There is no
analytical ground truth for these images: these are acceptance and parity
counts, not claims of displacement accuracy or physical validity.

These were single diagnostic executions, not a timing qualification. Reading
and preparing the original images took 0.38 s in that execution; reading and
verifying the fingerprinted cache took 0.94 s. The cache therefore has no
demonstrated performance benefit on this real pair. Reference setup and
verification were also variable. The cached image files occupy approximately
43 MiB (mebibytes, units of 2²⁰ bytes). The cache remains opt-in and does not
make the pipeline graphics-processor-resident.

Reproduce in a **new** directory:

```bash
set -euo pipefail
task_dir=results/seed_propagation_20261111/p2_solver_images_adrien_reproduction
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
reference=/home/jeff/CNRS/DIC/AdrienVariational/img_000000_0.09422.tiff
target=/home/jeff/CNRS/DIC/AdrienVariational/img_000075_1608.88508.tiff
mask=/home/jeff/CNRS/DIC/AdrienVariational/Mask.tiff
export PYTHONPATH=python/src:python/benchmarks
.venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
  --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
  --output-dir "$task_dir/inputs" --seed-offsets 34000000 --families heterogeneous \
  --cases-per-family 1 --retain-translation-affine --shared-prefit --export-solver-images \
  --model-width 192 --model-height 146 --external-reference "$reference" --external-target "$target"
for mode in regenerated cached; do
  extra=()
  if [[ $mode == cached ]]; then extra+=(--use-cached-solver-images); fi
  .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
    --live-dic-source "$solver" --archive "$task_dir/inputs" --output-dir "$task_dir/$mode" \
    --sparse-stride 2 --lazy-rotation-weights "$weights" --lazy-rotation-output "$task_dir/rotations" \
    --escalation --local-affine-escalation --frozen-parameters "$frozen" \
    --external-reference "$reference" --external-target "$target" --roi-mask "$mask" "${extra[@]}"
done
```
