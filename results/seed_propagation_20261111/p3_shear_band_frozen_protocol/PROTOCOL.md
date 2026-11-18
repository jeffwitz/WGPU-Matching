# Fixed protocol: localized shear bands

Defined before execution. Assess the frozen image-only initializer on smooth synthetic shear bands at peak displacement gradients 0.33, 0.67 and 2.0. Use seed group 59000000, three paired textures per gradient, separate output directories and unchanged solver/frozen P2 thresholds. Images are 3840 × 2160, model raster 192 × 108, subset 64, stride 64. Use pretrained coarse matcher, affine + local normalized cross-correlation, status-only sparse selection, confirmed anchors, and Jacobi affine propagation. The three magnitudes share texture identities to allow paired diagnostics; this is a small exploratory cohort, not a claim of statistical generalization.

Ground truth and the local band coordinate are used only for after-the-fact scoring, oracle controls and regional diagnostics; neither chooses hypotheses nor active windows. Report per case: visible, accepted, correct displacement <2 px, false accepted visible, accepted invisible, oracle, accepted anchors, passes, IC-GN work and the windows and errors versus distance from the band. Compute both G3 false acceptance (<=0.5% per family) and separately the stricter prior 0.1% threshold. A family with no accepted visible windows has undefined false-acceptance fraction, not a pass. Quantify performance without tuning on these results.

Cache the exact solver images and bind them by hash; generate at each gradient separately to avoid reusing a mismatched source. Use the existing CPU coarse model and host-orchestrated solver propagation to match prior P3 qualification; GPU measurements are a separate experiment. Preserve raw JSON and NPZ files and provide reproduction commands and device details.

## Provenance and reproduction

Executed with Intel Graphics Lunar Lake, WebGPU through Vulkan, against repository HEAD `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93` plus the uncommitted working-tree changes containing this implementation. The checkpoint SHA-256 is `150acb9437463034ff879750d2ea8b2d675667ea533a22633a4b56e3dc15cfaf`. At execution, `qualify_ndf_pair_icgn.py` SHA-256 was `82bae3680b3fe85643029eb4e0365253de112d85bf2bef95fcb05681b891bc35`, `qualify_image_seed_pilot.py` was `ed1e59ed45097988e2d75e97d3e69022eab9bf6219c7b482803f15bfb7067af9`, `seed_propagation.py` was `a5d3e9df2863ca4832ef47db9aa24f1ec3025b9289ecc2f619271c38670777a3`, and the external `live_dic/wgpu_backend.py` was `f0e7cad5d0cb5e017cb23b341c05197ecaf91ff683b34fbd0b8b24d4fe30b238`. Exact solver pixels and their hashes are retained in each `inputs` directory.

The original `evaluation` run omitted the equal-work control; `evaluation_control` and `evaluation_control_exact_truth` are separate reruns with `--same-budget-control`. Their accepted states and provenance agree. No thresholds were changed. The original archived truth sampled rounded pixel coordinates; the current qualifier now uses exact analytical shear truth. The commands below reproduce the corrected protocol in a new directory, not the superseded rounded scoring:

```bash
base=results/seed_propagation_20261111/p3_shear_band_frozen_protocol_new
weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt
solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src
frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
for gradient in 0.33 0.67 2.0; do
  PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_p2_test_inputs.py \
    --frozen-parameters "$frozen" --weights "$weights" --live-dic-source "$solver" \
    --output-dir "$base/g$gradient/inputs" --seed-offsets 59000000 --families shear_band \
    --cases-per-family 3 --shear-band-peak-gradient "$gradient" --retain-translation-affine \
    --shared-prefit --export-solver-images
  for run in evaluation evaluation_control_exact_truth; do
    extra=()
    if [ "$run" = evaluation_control_exact_truth ]; then extra=(--same-budget-control); fi
    PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
      --archive "$base/g$gradient/inputs" --live-dic-source "$solver" --output-dir "$base/g$gradient/$run" \
      --lazy-rotation-weights "$weights" --lazy-rotation-output "$base/g$gradient/rotations" \
      --frozen-parameters "$frozen" --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3 \
      --shear-band-peak-gradient "$gradient" --escalation --local-affine-escalation \
      --batch-anchor-confirmation --use-cached-solver-images "${extra[@]}"
  done
done
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_shear_exact_truth.py \
  --base "$base" --live-dic-source "$solver" --output-dir "$base/exact_center_analysis"
```

The analysis and plotting script needs the optional `research` dependencies, including Matplotlib.

To rescore the **existing historical rounded-truth archives**, rather than regenerate the now-corrected inputs, use `qualify_shear_exact_truth.py` with `--baseline-rounded` and `--control-directory-name evaluation_control_exact_truth`.
