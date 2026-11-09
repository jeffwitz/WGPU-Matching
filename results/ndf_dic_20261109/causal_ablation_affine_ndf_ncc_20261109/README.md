# NDF causal ablation: image-derived affine, nonlinear residual and NCC

All variants use paired heterogeneous synthetic image pairs, the same rotation bank, the same coarse matcher, the same image-derived affine prefit, and identical WebGPU IC-GN settings.
The affine-only label means no NDF nonlinear residual after the image-derived affine. It still uses the learned coarse matcher to select rotation and estimate that affine.
The generated images are native 3840 × 2160, but matching, affine prefit, and local NCC use the 192 × 108 model raster (20× reduction); only IC-GN uses native pixels. This is not a native-resolution matching benchmark.

## Pooled results

| Variant | Correct / visible | Coverage | Pair median | Init translation RMSE (px) | Init gradient RMSE (median) | Median pipeline (s) | Correct windows/s | Oracle correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Image-derived affine only | 2,070/22,416 | 9.23% | 7.15% | 186.73 | 0.2054 | 0.092 | 927 | 21,367 |
| Affine + local NCC | 16,266/22,416 | 72.56% | 81.29% | 185.00 | 1.0311 | 0.078 | 12152 | 21,367 |
| Affine + NDF residual | 1,748/22,416 | 7.80% | 6.76% | 405.14 | 2.4386 | 0.109 | 773 | 21,367 |
| Affine + NDF + local NCC | 13,172/22,416 | 58.76% | 64.55% | 404.16 | 2.9579 | 0.105 | 6982 | 21,367 |

Exclusive recovery relative to affine-only:

- Affine + local NCC: 14,263 windows recovered only by this variant; 67 affine-only successes lost.
- Affine + NDF residual: 1,230 windows recovered only by this variant; 1,552 affine-only successes lost.
- Affine + NDF + local NCC: 11,418 windows recovered only by this variant; 316 affine-only successes lost.
- Adding the NDF residual when NCC is already enabled: 245 extra windows, 3,339 NCC-only windows lost.

## Paired changes relative to affine-only

Coverage differences are percentage points; intervals resample the 20 image pairs, not individual windows.

| Contrast | Mean coverage change | Pair bootstrap 95% interval | Better / tied / worse pairs |
|---|---:|---:|---:|
| Affine + local NCC | +60.08% | [+47.80%, +70.21%] | 19 / 1 / 0 |
| Affine + NDF residual | -1.66% | [-4.21%, +0.78%] | 7 / 2 / 11 |
| Affine + NDF + local NCC | +46.93% | [+37.25%, +55.35%] | 18 / 2 / 0 |

Conditional contrasts:

- Add NDF when NCC is enabled: -13.15 percentage points (95% pair-bootstrap interval -16.43 to -9.69); 1 pairs improve, 1 are tied, 18 regress.
- Add NCC when NDF is enabled: +48.58 percentage points (95% pair-bootstrap interval +38.80 to +56.75); 18 pairs improve, 2 are tied, 0 regress.
- NDF–NCC interaction: -11.50 percentage points (95% pair-bootstrap interval -15.05 to -8.05).

## Initial affine-parameter errors

Pair-balanced root mean square error over visible centres; `u` and `v` are native-pixel displacements, and the other four entries are displacement gradients.

| Variant | u (px) | du/dx | du/dy | v (px) | dv/dx | dv/dy | Median translation error (px) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Image-derived affine only | 112.56 | 0.1900 | 0.1957 | 148.99 | 0.2932 | 0.2061 | 35.35 |
| Affine + local NCC | 109.65 | 0.7113 | 0.7454 | 149.00 | 0.6345 | 0.6479 | 3.43 |
| Affine + NDF residual | 320.36 | 1.4938 | 1.2992 | 248.00 | 1.0588 | 1.0267 | 45.82 |
| Affine + NDF + local NCC | 319.80 | 1.7669 | 1.5817 | 247.14 | 1.2865 | 1.2702 | 4.50 |

The JSON retains each pair’s six initial affine parameters, per-window IC-GN outcomes, windows recovered or lost versus affine-only, and a 3 × 3 spatial breakdown. The oracle count is a diagnostic ceiling, not an available pipeline.

Conditional gains from adding the NDF residual to NCC, and adding NCC to the NDF residual, plus their interaction, are reported in `summary.json`.

## Spatial change when adding the NDF residual to affine + NCC

| Region | Visible windows | NCC correct | NCC + NDF correct | Coverage change |
|---|---:|---:|---:|---:|
| r1c1 | 1,580 | 1,005 | 771 | -14.81% |
| r1c2 | 3,305 | 2,164 | 1,735 | -12.98% |
| r1c3 | 1,852 | 1,036 | 788 | -13.39% |
| r2c1 | 2,282 | 1,797 | 1,346 | -19.76% |
| r2c2 | 4,712 | 3,837 | 3,332 | -10.72% |
| r2c3 | 2,610 | 2,165 | 1,808 | -13.68% |
| r3c1 | 1,351 | 904 | 679 | -16.65% |
| r3c2 | 2,924 | 1,945 | 1,570 | -12.82% |
| r3c3 | 1,800 | 1,413 | 1,143 | -15.00% |

## Reproduction

Set `live_dic_source` to the `python/src` directory of the Live-DIC checkout used by this repository’s IC-GN adapter, then run:

```console
python python/benchmarks/benchmark_ndf_causal_ablation.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source "$live_dic_source" \
  --output-dir results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109
```

The runner uses seed offsets 16,000,000 and 17,000,000, ten heterogeneous pairs per offset, and checks that all variants retain the same selected orientation, affine map, visibility mask, and DIC window centres.
