# Shear-band qualification with exact-center analytical truth

This report supersedes the numerical scoring in `../analysis/README.md`. The previous truth was sampled at rounded integer pixels instead of the actual half-pixel window centers. Inference outputs, frozen thresholds and solver settings are unchanged. The synthetic shear map and its Jacobian are now evaluated analytically at each exact center; visibility is recomputed from that map. Oracle and zero-initialization controls were re-solved on the GPU.

| Peak gradient | Correct / visible | False / accepted visible | Exact oracle | One pass | Same-budget control | G3 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.33 | 4663/4849 | 0/4663 (0.00%) | 4662 | 4200 | 4224 | pass |
| 0.67 | 4673/4833 | 0/4673 (0.00%) | 4673 | 3920 | 3954 | pass |
| 2.0 | 4023/4761 | 57/4080 (1.40%) | 4240 | 3205 | 3258 | fail |

| Pair | Peak gradient | Correct / visible | False / accepted visible | Oracle |
| --- | ---: | ---: | ---: | ---: |
| shear_band_texture59051005_warp59091005 | 0.33 | 1637/1723 | 0/1637 | 1637 |
| shear_band_texture59051007_warp59091007 | 0.33 | 1411/1428 | 0/1411 | 1411 |
| shear_band_texture59051009_warp59091009 | 0.33 | 1615/1698 | 0/1615 | 1614 |
| shear_band_texture59051005_warp59091005 | 0.67 | 1635/1711 | 0/1635 | 1635 |
| shear_band_texture59051007_warp59091007 | 0.67 | 1411/1424 | 0/1411 | 1411 |
| shear_band_texture59051009_warp59091009 | 0.67 | 1627/1698 | 0/1627 | 1627 |
| shear_band_texture59051005_warp59091005 | 2.0 | 1388/1670 | 22/1410 | 1454 |
| shear_band_texture59051007_warp59091007 | 2.0 | 1232/1416 | 18/1250 | 1306 |
| shear_band_texture59051009_warp59091009 | 2.0 | 1403/1675 | 17/1420 | 1480 |

At peak gradient 2.0, 57 windows are falsely accepted; 57 of them are also accepted and remain incorrect after re-solving from exact analytical local affine parameters. This oracle is a diagnostic only. The fixed-size affine subset under a curved field is a plausible source of bias, but this observation alone does not establish its cause.

The three textures are paired across gradients. This small synthetic cohort does not establish performance on experimental images or other deformation families. The exact oracle is a reference, not a mathematical upper bound.

![Exact-center shear band 0.33](shear_exact_0.33.png)

![Exact-center shear band 0.67](shear_exact_0.67.png)

![Exact-center shear band 2.0](shear_exact_2.0.png)

`summary.json` records per-pair values and region counts; `false_accepted_windows.json` lists each false acceptance. NPZ files retain exact truth and re-solved oracle outputs. The protocol and reproduction commands are in `../PROTOCOL.md`.
