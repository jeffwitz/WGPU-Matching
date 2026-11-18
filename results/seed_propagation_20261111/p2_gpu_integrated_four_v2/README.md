# GPU frontier integrated into image-only initialization

The existing image-only initializer is run on four previously generated
3840 × 2160 synthetic pairs: two heterogeneous and two rigid. The reference
path confirms anchors in batches and propagates from Python; the experimental
path uses the same anchor confirmation and propagates on the existing
WebGPU solver device. Both use the same image pixels, initial hypotheses,
window centres and inverse-compositional Gauss–Newton (IC-GN) solver settings.
The analytical field is used only for the final score.

Both paths return exactly the same accepted windows, parameters, confirmed
anchors, propagation parents, origins, generations, attempts and rejection
reasons on all four pairs. Each obtains **4,695 correct visible windows out
of 4,900**, with zero false accepted visible windows. Per-window comparisons
are in `comparison.json` and the corresponding NPZ (NumPy compressed archive)
files. This is a four-pair diagnostic, not a general robustness qualification.

Within the GPU frontier, the reference image has already been prepared on the
same device. The target and confirmed seed states are uploaded once. The
frontier, proposed affine states, active masks and IC-GN trial states remain
on that device between passes. Each pass reads an eight-byte counter to decide
whether to continue; the final state and provenance are read once each. This
does not make anchor confirmation, feature matching or the complete pipeline
GPU-resident. The solver integration currently uses private `live-dic`
interfaces and supports affine motion, linear interpolation, no outlier
filter, and no partial-subset solver. Unsupported configurations fail rather
than silently falling back.

For the real-image comparison and numerical tolerance, see
`../p2_gpu_frontier_adrien_paired/README.md`. The command below reproduces this
experiment into a **new** directory:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/seed_propagation_20261111/p2_anchor_batch_new_pairs/inputs \
  --output-dir results/seed_propagation_20261111/p2_gpu_integrated_four_new --sparse-stride 2 \
  --lazy-rotation-weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --lazy-rotation-output results/seed_propagation_20261111/p2_gpu_integrated_four_new/rotations \
  --escalation --local-affine-escalation --use-cached-solver-images --batch-anchor-confirmation --gpu-frontier \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_anchor_confirmation.py \
  --serial results/seed_propagation_20261111/p2_anchor_batch_new_pairs/serial \
  --batched results/seed_propagation_20261111/p2_gpu_integrated_four_new \
  --output results/seed_propagation_20261111/p2_gpu_integrated_four_new/comparison.json
```
