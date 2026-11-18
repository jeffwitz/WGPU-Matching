# Staged coarse matching to GPU propagation: two synthetic fields

The same two previously unused deterministic 3840 × 2160 synthetic pairs were generated twice with identical settings. The reference uses the PyTorch CPU coarse network and host-orchestrated confirmation/propagation. The comparison uses the ONNX Runtime WebGPU coarse network followed, through files and host arrays, by GPU hypothesis selection, anchor confirmation and frontier propagation. These are separate stages, not a GPU-resident end-to-end execution. No thresholds were adjusted and analytical truth is used only for final scoring.

| Family | Correct CPU | Correct GPU path | Visible | False accepted visible |
| --- | ---: | ---: | ---: | ---: |
| Heterogeneous | 994 | 994 | 1,038 | 0 in both |
| Rigid | 1,429 | 1,429 | 1,474 | 0 in both |

The coarse initial parameter arrays match bitwise. Final acceptance masks, confirmed seeds and propagation provenance also match. Final accepted displacement components differ by at most 0.0000611 pixel and gradients by at most 0.000000835; these are observed floating-point differences, not a general bound. Neither pair required deferred rotation generation. The experiment is too small and the separate stages were not timed as one process, so it does not establish a general quality or speed claim.

`comparison.json` and the `cpu/` and `gpu/` directories retain the raw per-window results. The ONNX provider profile and paired real-image benchmark are documented in `../p2_ndf_onnx_webgpu_adrien/README.md`.

Reproduce in a new directory:

```bash
base=results/seed_propagation_20261111/p2_ndf_onnx_end_to_end_new
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
for backend in cpu gpu; do
  extra=()
  if [ "$backend" = gpu ]; then extra=(--ndf-onnx-webgpu-cache "$base/graphs"); fi
  PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
    --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
    --output-dir "$base/$backend/inputs" --seed-offsets 57000000 --families heterogeneous rigid \
    --cases-per-family 1 --retain-translation-affine --shared-prefit --export-solver-images "${extra[@]}"
  if [ "$backend" = gpu ]; then
    extra=(--ndf-onnx-webgpu-cache "$base/graphs" --gpu-frontier --gpu-anchor-frontier \
      --gpu-hypothesis-selection --gpu-resident-selection)
  fi
  PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
    --archive "$base/$backend/inputs" --live-dic-source "$solver" --output-dir "$base/$backend/evaluation" \
    --lazy-rotation-weights "$weights" --lazy-rotation-output "$base/$backend/rotations" \
    --frozen-parameters "$frozen" --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3 \
    --escalation --local-affine-escalation --batch-anchor-confirmation --use-cached-solver-images "${extra[@]}"
done
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_anchor_confirmation.py \
  --serial "$base/cpu/evaluation" --batched "$base/gpu/evaluation" --output "$base/comparison.json"
```
