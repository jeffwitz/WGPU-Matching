# ONNX Runtime WebGPU coarse matching: two synthetic fields

The pair matcher is run on the same two deterministic 3840 × 2160 synthetic pairs and 192 × 108 input raster with the same affine prefit, local normalized cross-correlation and one-pass inverse-compositional Gauss–Newton (IC-GN) settings. Only the learned matcher backend changes. Truth is accessed solely for scoring.

| Family | Correct visible windows, CPU | Correct visible windows, WebGPU | Visible |
| --- | ---: | ---: | ---: |
| Heterogeneous | 1,307 | 1,307 | 1,545 |
| Rigid | 1,197 | 1,197 | 1,356 |

Initial parameters, solver statuses and solver parameters are bitwise identical on these two cases. This is one pass, before anchor confirmation or propagation, and is not a broad validation. Per-window arrays and raw report files are in `torch/` and `webgpu/`. The full network still exchanges CPU arrays with the remaining pipeline. See `../p2_ndf_onnx_webgpu_adrien/README.md` for provider profiling, performance and limitations.

Reproduce in a new directory, running the loop serially:

```bash
base=results/seed_propagation_20261111/p2_ndf_onnx_webgpu_synthetic_new
for backend in torch webgpu; do
  mkdir -p "$base/$backend"
  extra=()
  if [ "$backend" = webgpu ]; then extra=(--ndf-onnx-webgpu-cache "$base/graphs"); fi
  PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_ndf_pair_icgn.py \
    --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
    --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
    --output "$base/$backend/report.json" --window-diagnostics-dir "$base/$backend/windows" \
    --native-scale 20 --model-width 192 --model-height 108 --speckle-size .35 \
    --families heterogeneous rigid --cases-per-family 1 --seed-offset 55000000 \
    --contextual-features --rotation-bank-steps 9 --rotation-bank-probe-query-budget 256 \
    --global-affine-prefit --field-residual none --retain-translation-affine \
    --local-ncc-radius 8 --local-ncc-patch-radius 4 --local-ncc-backend wgpu \
    --local-ncc-geometry translation --local-consensus-radius 40 --local-consensus-threshold 5 \
    --projection raw_bilinear --export-pre-ncc "${extra[@]}"
done
```
