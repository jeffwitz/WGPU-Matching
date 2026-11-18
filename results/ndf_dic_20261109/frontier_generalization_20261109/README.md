# Held-out frontier propagation generalization

Digital image correlation (DIC) is initialized by the learned neural deformation field (NDF). Inverse-compositional Gauss-Newton (IC-GN) is the local solver. Normalized cross-correlation (NCC), here scored as zero-mean normalized cross-correlation (ZNCC), supplies the local photometric refinement. The solver's accepted status alone defines seeds. Oracle fields are used only for scoring.

| Seed group | Family | Peak shear gradient | Seeded pairs | Pairs at >=97% oracle | Seeded oracle regions below 97% | Pairs with every oracle region seeded | G2 bad accepted / total |
|---:|---|---:|---:|---:|---:|---:|---:|
| 18000000 | rigid | — | 5/5 | 5/5 | 0 | 5/5 | 0/5830 (PASS) |
| 18000000 | affine | — | 5/5 | 5/5 | 0 | 5/5 | 0/5710 (PASS) |
| 18000000 | discontinuous | — | 4/5 | 4/5 | 0 | 4/5 | 0/5366 (PASS) |
| 18000000 | shear_band | 0.33 | 3/5 | 3/5 | 0 | 3/5 | 0/4010 (PASS) |
| 18000000 | shear_band | 0.67 | 5/5 | 5/5 | 0 | 5/5 | 0/6973 (PASS) |
| 18000000 | shear_band | 2.0 | 5/5 | 0/5 | 5 | 5/5 | 59/6142 (FAIL) |
| 19000000 | rigid | — | 5/5 | 5/5 | 0 | 5/5 | 0/6079 (PASS) |
| 19000000 | affine | — | 5/5 | 5/5 | 0 | 5/5 | 0/7092 (PASS) |
| 19000000 | discontinuous | — | 4/5 | 4/5 | 0 | 4/5 | 0/5921 (PASS) |
| 19000000 | shear_band | 0.33 | 3/5 | 3/5 | 0 | 3/5 | 0/5046 (PASS) |
| 19000000 | shear_band | 0.67 | 5/5 | 5/5 | 0 | 5/5 | 0/7967 (PASS) |
| 19000000 | shear_band | 2.0 | 4/5 | 0/5 | 4 | 4/5 | 71/5908 (FAIL) |

Accepted visible windows with displacement error >= 2 px:

- Seed 18000000, shear_band 2.0, texture 18051005, window 1094 at (927.5, 1183.5) px: 2.256 px at ring 2.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1095 at (991.5, 1183.5) px: 2.210 px at ring 1.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1099 at (1247.5, 1183.5) px: 2.333 px at ring 1.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1104 at (1567.5, 1183.5) px: 2.132 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1106 at (1695.5, 1183.5) px: 2.054 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1107 at (1759.5, 1183.5) px: 2.115 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1108 at (1823.5, 1183.5) px: 2.160 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1109 at (1887.5, 1183.5) px: 2.080 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1110 at (1951.5, 1183.5) px: 2.403 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1111 at (2015.5, 1183.5) px: 2.101 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1112 at (2079.5, 1183.5) px: 2.267 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1113 at (2143.5, 1183.5) px: 2.023 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1114 at (2207.5, 1183.5) px: 2.411 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1115 at (2271.5, 1183.5) px: 2.151 px at ring 1.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1122 at (2719.5, 1183.5) px: 2.012 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1123 at (2783.5, 1183.5) px: 2.534 px at ring 1.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1124 at (2847.5, 1183.5) px: 2.386 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1126 at (2975.5, 1183.5) px: 2.038 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051005, window 1128 at (3103.5, 1183.5) px: 2.341 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 920 at (1311.5, 991.5) px: 2.106 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1088 at (543.5, 1183.5) px: 2.538 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1098 at (1183.5, 1183.5) px: 2.157 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1103 at (1503.5, 1183.5) px: 2.284 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1108 at (1823.5, 1183.5) px: 2.038 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1109 at (1887.5, 1183.5) px: 2.016 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1111 at (2015.5, 1183.5) px: 2.437 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1112 at (2079.5, 1183.5) px: 2.022 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1113 at (2143.5, 1183.5) px: 2.224 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1120 at (2591.5, 1183.5) px: 2.049 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1121 at (2655.5, 1183.5) px: 2.554 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1129 at (3167.5, 1183.5) px: 2.215 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1130 at (3231.5, 1183.5) px: 2.339 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051007, window 1132 at (3359.5, 1183.5) px: 2.349 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1102 at (1439.5, 1183.5) px: 2.301 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1112 at (2079.5, 1183.5) px: 2.190 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1118 at (2463.5, 1183.5) px: 2.186 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1119 at (2527.5, 1183.5) px: 2.072 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1121 at (2655.5, 1183.5) px: 2.278 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1122 at (2719.5, 1183.5) px: 2.095 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1123 at (2783.5, 1183.5) px: 2.171 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1124 at (2847.5, 1183.5) px: 2.281 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1130 at (3231.5, 1183.5) px: 2.397 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051009, window 1132 at (3359.5, 1183.5) px: 2.057 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 935 at (2271.5, 991.5) px: 2.173 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1058 at (2463.5, 1119.5) px: 3.155 px at ring 3.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1059 at (2527.5, 1119.5) px: 3.145 px at ring 2.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1060 at (2591.5, 1119.5) px: 2.246 px at ring 1.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1061 at (2655.5, 1119.5) px: 3.698 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1084 at (287.5, 1183.5) px: 2.604 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1085 at (351.5, 1183.5) px: 2.050 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1087 at (479.5, 1183.5) px: 2.062 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1088 at (543.5, 1183.5) px: 2.236 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1105 at (1631.5, 1183.5) px: 2.084 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051011, window 1114 at (2207.5, 1183.5) px: 2.370 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051013, window 1098 at (1183.5, 1183.5) px: 2.154 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051013, window 1103 at (1503.5, 1183.5) px: 2.111 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051013, window 1105 at (1631.5, 1183.5) px: 2.432 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051013, window 1107 at (1759.5, 1183.5) px: 2.212 px at ring 0.
- Seed 18000000, shear_band 2.0, texture 18051013, window 1116 at (2335.5, 1183.5) px: 2.242 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1031 at (735.5, 1119.5) px: 2.683 px at ring 36.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1032 at (799.5, 1119.5) px: 2.055 px at ring 35.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1033 at (863.5, 1119.5) px: 3.600 px at ring 34.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1034 at (927.5, 1119.5) px: 3.655 px at ring 33.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1036 at (1055.5, 1119.5) px: 3.328 px at ring 31.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1044 at (1567.5, 1119.5) px: 3.596 px at ring 23.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1048 at (1823.5, 1119.5) px: 3.085 px at ring 19.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1049 at (1887.5, 1119.5) px: 2.935 px at ring 18.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1050 at (1951.5, 1119.5) px: 2.781 px at ring 17.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1051 at (2015.5, 1119.5) px: 3.294 px at ring 16.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1052 at (2079.5, 1119.5) px: 3.334 px at ring 15.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1059 at (2527.5, 1119.5) px: 3.811 px at ring 8.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1065 at (2911.5, 1119.5) px: 2.831 px at ring 3.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1066 at (2975.5, 1119.5) px: 3.165 px at ring 2.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1067 at (3039.5, 1119.5) px: 2.506 px at ring 2.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1068 at (3103.5, 1119.5) px: 2.407 px at ring 3.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1083 at (223.5, 1183.5) px: 2.149 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1087 at (479.5, 1183.5) px: 2.161 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1098 at (1183.5, 1183.5) px: 2.153 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1106 at (1695.5, 1183.5) px: 2.255 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1114 at (2207.5, 1183.5) px: 2.047 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1117 at (2399.5, 1183.5) px: 2.322 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1122 at (2719.5, 1183.5) px: 2.484 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1123 at (2783.5, 1183.5) px: 2.252 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1124 at (2847.5, 1183.5) px: 2.548 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1126 at (2975.5, 1183.5) px: 2.011 px at ring 3.
- Seed 19000000, shear_band 2.0, texture 19051005, window 1128 at (3103.5, 1183.5) px: 2.017 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1092 at (799.5, 1183.5) px: 2.364 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1094 at (927.5, 1183.5) px: 2.338 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1095 at (991.5, 1183.5) px: 2.209 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1096 at (1055.5, 1183.5) px: 2.123 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1108 at (1823.5, 1183.5) px: 2.162 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1109 at (1887.5, 1183.5) px: 2.051 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051009, window 1110 at (1951.5, 1183.5) px: 2.236 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 922 at (1439.5, 991.5) px: 2.203 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 924 at (1567.5, 991.5) px: 2.389 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 948 at (3103.5, 991.5) px: 2.043 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1081 at (95.5, 1183.5) px: 3.145 px at ring 1.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1082 at (159.5, 1183.5) px: 2.254 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1083 at (223.5, 1183.5) px: 2.381 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1085 at (351.5, 1183.5) px: 2.008 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1086 at (415.5, 1183.5) px: 2.073 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1087 at (479.5, 1183.5) px: 2.337 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1089 at (607.5, 1183.5) px: 2.025 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1093 at (863.5, 1183.5) px: 2.070 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1094 at (927.5, 1183.5) px: 2.232 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1095 at (991.5, 1183.5) px: 2.571 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1096 at (1055.5, 1183.5) px: 2.258 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1102 at (1439.5, 1183.5) px: 2.007 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1103 at (1503.5, 1183.5) px: 2.101 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1104 at (1567.5, 1183.5) px: 2.268 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1105 at (1631.5, 1183.5) px: 2.206 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1108 at (1823.5, 1183.5) px: 2.659 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1116 at (2335.5, 1183.5) px: 2.522 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1121 at (2655.5, 1183.5) px: 2.116 px at ring 1.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1123 at (2783.5, 1183.5) px: 2.142 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1124 at (2847.5, 1183.5) px: 2.323 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1125 at (2911.5, 1183.5) px: 2.110 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1130 at (3231.5, 1183.5) px: 2.057 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051011, window 1131 at (3295.5, 1183.5) px: 2.490 px at ring 1.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1089 at (607.5, 1183.5) px: 2.238 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1090 at (671.5, 1183.5) px: 2.025 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1092 at (799.5, 1183.5) px: 2.312 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1106 at (1695.5, 1183.5) px: 2.119 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1107 at (1759.5, 1183.5) px: 2.177 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1108 at (1823.5, 1183.5) px: 2.199 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1110 at (1951.5, 1183.5) px: 2.181 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1125 at (2911.5, 1183.5) px: 2.321 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1128 at (3103.5, 1183.5) px: 2.401 px at ring 0.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1129 at (3167.5, 1183.5) px: 2.375 px at ring 1.
- Seed 19000000, shear_band 2.0, texture 19051013, window 1130 at (3231.5, 1183.5) px: 2.391 px at ring 0.

Per-pair stopping rings, error counts, region seed maps, timing, and source exports are retained in `summary.json` and each group's `windows/` directory.

A seedless pair or oracle region without a seed is outside propagation's reach. The exact seedless pairs and their oracle support are listed in each group in `summary.json`; they are plausible decoder targets, though this does not establish that a learned field model would recover them. Per-ring attempt and acceptance counts identify where propagation stops.

Reproduce the full campaign with:

```console
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_frontier_generalization.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /path/to/Live-DIC/python/src \
  --output-dir results/ndf_dic_20261109/frontier_generalization_20261109
```
