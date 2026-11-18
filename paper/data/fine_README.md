# Independent multifamily transfer to 32-pixel subsets

Five predeclared new pairs per family, native 3840 × 2160 pixels. Each fine solve uses a nearest accepted 64-pixel image-derived state at the same requested center; truth is used only for scoring. Window positions and thresholds are frozen. This evaluates the transfer on the original 64-grid centers, not all centers of a dense 32/16 field.

| Family | 64 px correct / visible | 32 px correct / visible | Fine false / accepted visible | Lost 64 correct | Paired gain CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| translation | 6322/6467 | 6395/6467 | 0/6395 (0.00%) | 0 | [0.00; 2.66] points |
| rigid | 6008/6278 | 6144/6278 | 0/6144 (0.00%) | 0 | [1.70; 2.74] points |
| affine | 5616/5845 | 5718/5845 | 0/5718 (0.00%) | 0 | [1.28; 2.45] points |
| heterogeneous | 5804/6061 | 5920/6061 | 0/5920 (0.00%) | 0 | [1.40; 2.64] points |
| discontinuous | 6414/6556 | 6443/6556 | 0/6443 (0.00%) | 0 | [0.00; 1.41] points |

| Family | Zero 64 | One pass 64 | Same-budget control 64 | Analytical oracle 64 |
| --- | ---: | ---: | ---: | ---: |
| translation | 0 | 5380 | 5415 | 6322 |
| rigid | 4 | 4703 | 4727 | 6008 |
| affine | 0 | 5192 | 5202 | 5616 |
| heterogeneous | 0 | 4765 | 4789 | 5804 |
| discontinuous | 0 | 4972 | 4996 | 6414 |

| Family | Fine displacement median / RMSE (px) | Displacement RMSE, coarse → fine on common correct | Fine gradient median / RMSE | Gradient RMSE, coarse → fine on common correct |
| --- | ---: | ---: | ---: | ---: |
| translation | 0.0073 / 0.0106 | 0.0033 → 0.0106 | 0.0017 / 0.0022 | 0.0004 → 0.0022 |
| rigid | 0.0077 / 0.0099 | 0.0033 → 0.0099 | 0.0017 / 0.0020 | 0.0003 → 0.0020 |
| affine | 0.0083 / 0.0109 | 0.0038 → 0.0109 | 0.0018 / 0.0022 | 0.0004 → 0.0022 |
| heterogeneous | 0.0151 / 0.0191 | 0.0589 → 0.0191 | 0.0021 / 0.0025 | 0.0009 → 0.0025 |
| discontinuous | 0.0051 / 0.0087 | 0.0036 → 0.0087 | 0.0012 / 0.0018 | 0.0004 → 0.0018 |

Gradient error is the dimensionless Frobenius norm of the difference between the four displacement-gradient components. The last column uses only windows correct in both methods; it does not establish a production-quality threshold. The smaller support increases the gradient error on this cohort despite recovering additional centers.


| Pair | 64 correct | 32 correct | Visible | 32 false | Lost 64 correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| affine_texture62051003_warp62091003 | 761 | 787 | 815 | 0 | 0 |
| affine_texture62051005_warp62091005 | 1717 | 1736 | 1755 | 0 | 0 |
| affine_texture62051007_warp62091007 | 880 | 897 | 928 | 0 | 0 |
| affine_texture62051009_warp62091009 | 1098 | 1121 | 1148 | 0 | 0 |
| affine_texture62051011_warp62091011 | 1160 | 1177 | 1199 | 0 | 0 |
| discontinuous_texture62051004_warp62091004 | 1218 | 1247 | 1247 | 0 | 0 |
| discontinuous_texture62051006_warp62091006 | 1056 | 1056 | 1100 | 0 | 0 |
| discontinuous_texture62051008_warp62091008 | 1188 | 1188 | 1215 | 0 | 0 |
| discontinuous_texture62051010_warp62091010 | 1650 | 1650 | 1650 | 0 | 0 |
| discontinuous_texture62051012_warp62091012 | 1302 | 1302 | 1344 | 0 | 0 |
| heterogeneous_texture62051001_warp62091001 | 1370 | 1400 | 1430 | 0 | 0 |
| heterogeneous_texture62051003_warp62091003 | 766 | 793 | 819 | 0 | 0 |
| heterogeneous_texture62051005_warp62091005 | 1691 | 1710 | 1734 | 0 | 0 |
| heterogeneous_texture62051007_warp62091007 | 877 | 893 | 925 | 0 | 0 |
| heterogeneous_texture62051009_warp62091009 | 1100 | 1124 | 1153 | 0 | 0 |
| rigid_texture62051002_warp62091002 | 1185 | 1208 | 1231 | 0 | 0 |
| rigid_texture62051004_warp62091004 | 1093 | 1128 | 1157 | 0 | 0 |
| rigid_texture62051006_warp62091006 | 971 | 998 | 1028 | 0 | 0 |
| rigid_texture62051008_warp62091008 | 1052 | 1078 | 1103 | 0 | 0 |
| rigid_texture62051010_warp62091010 | 1707 | 1732 | 1759 | 0 | 0 |
| translation_texture62051000_warp62091000 | 1568 | 1568 | 1568 | 0 | 0 |
| translation_texture62051002_warp62091002 | 1100 | 1144 | 1144 | 0 | 0 |
| translation_texture62051004_warp62091004 | 1305 | 1334 | 1334 | 0 | 0 |
| translation_texture62051006_warp62091006 | 1080 | 1080 | 1125 | 0 | 0 |
| translation_texture62051008_warp62091008 | 1269 | 1269 | 1296 | 0 | 0 |

Intervals resample whole image pairs 10,000 times with seed 62001111; only five pairs per family are available, so the intervals do not establish broad generalization. Different families can reuse the same texture seeds; they are not independent across families. Invisible accepted windows are counted separately in `analysis.json`. G3 from the original 64-pixel protocol remains failed. This qualification does not measure complete-pipeline execution time.

For discontinuous opening, 0 fine accepted supports cross the analytical boundary; 0 visible initializations use a source on the opposite side, and 0 use a source whose 64-pixel support crosses it. These labels are evaluated after inference, never used for selection. The boundary is aligned with this grid; shifted cracks remain unqualified.

`summary.json` gives each pair's counts and `analysis.json` lists every visible false acceptance and its source. The figure selects the worst case per family for diagnosis after scoring; it is not used to select the algorithm. Reproduction commands are in `../PROTOCOL.md`.

![Representative diagnostic maps](representative_maps.png)
