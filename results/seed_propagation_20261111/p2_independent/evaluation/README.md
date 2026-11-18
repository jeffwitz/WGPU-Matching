# Held-out image-only initialization

The thresholds and sampling stride were frozen on the development group before these test cases. The inverse-compositional Gauss–Newton (IC-GN) solver accepts anchors; analytic truth is used only after selection and propagation for scoring.

G2: **INCOMPLETE**. Heterogeneous correct 23920/24994 (95.70%); paired mean gain +23.53%, 95% pair bootstrap [+15.11%, +34.24%]; 0/20 pairs below 20%; 0/23920 false accepted; translation correct-count difference +28698.

Translation baseline diagnostic: 20/20 global affine prefits were rejected. The pre-existing minimum linear-deviation criterion suppresses near-identity maps; with the neural residual disabled this can discard the translation. The baseline difference is not a valid non-regression conclusion.

Qualification limitations: end-to-end hypothesis-generation timing not measured; local affine fallback not calibrated on the full development set; translation baseline rejected one or more global affine prefits.

Heterogeneous mean gain versus the matched re-solve control: +23.33%, 95% pair bootstrap [+14.78%, +34.13%]. Accepted windows outside analytical visibility across both families: 0.

| Pair | Selected | Confirmed seeds | Correct / visible | Baseline | Matched control | False accepted |
|---|---|---:|---:|---:|---:|---:|
| heterogeneous_texture21051001_warp21091001 | affine_ncc | 856 | 962/1023 | 796 | 798 | 0 |
| heterogeneous_texture21051003_warp21091003 | affine_ncc | 1114 | 1227/1299 | 1042 | 1044 | 0 |
| heterogeneous_texture21051005_warp21091005 | affine_ncc | 330 | 892/965 | 295 | 295 | 0 |
| heterogeneous_texture21051007_warp21091007 | affine_ncc | 972 | 1093/1154 | 916 | 916 | 0 |
| heterogeneous_texture21051009_warp21091009 | affine_ncc | 1298 | 1532/1581 | 1162 | 1165 | 0 |
| heterogeneous_texture21051011_warp21091011 | affine_ncc | 770 | 823/876 | 721 | 727 | 0 |
| heterogeneous_texture21051013_warp21091013 | rotation_rank3 | 91 | 833/886 | 3 | 3 | 0 |
| heterogeneous_texture21051015_warp21091015 | affine_ncc | 1153 | 1330/1374 | 1087 | 1088 | 0 |
| heterogeneous_texture21051017_warp21091017 | affine_ncc | 889 | 987/1041 | 834 | 836 | 0 |
| heterogeneous_texture21051019_warp21091019 | affine_ncc | 1078 | 1310/1366 | 944 | 948 | 0 |
| translation_texture21051000_warp21091000 | local_rotation_rank3 | 51 | 1176/1204 | 0 | 0 | 0 |
| translation_texture21051002_warp21091002 | local_rotation_rank2 | 129 | 1456/1508 | 0 | 0 | 0 |
| translation_texture21051004_warp21091004 | local_rotation_rank3 | 120 | 1650/1736 | 0 | 0 | 0 |
| translation_texture21051006_warp21091006 | local_rotation_rank2 | 93 | 1416/1475 | 0 | 0 | 0 |
| translation_texture21051008_warp21091008 | local_rotation_rank3 | 126 | 1296/1350 | 0 | 0 | 0 |
| translation_texture21051010_warp21091010 | local_rotation_rank1 | 1229 | 1288/1288 | 0 | 0 | 0 |
| translation_texture21051012_warp21091012 | local_rotation_rank2 | 108 | 1705/1760 | 0 | 0 | 0 |
| translation_texture21051014_warp21091014 | local_rotation_rank2 | 99 | 1325/1404 | 0 | 0 | 0 |
| translation_texture21051016_warp21091016 | local_rotation_rank2 | 177 | 1624/1682 | 0 | 0 | 0 |
| translation_texture21051018_warp21091018 | local_rotation_rank3 | 129 | 1590/1643 | 0 | 0 | 0 |
| heterogeneous_texture22051001_warp22091001 | affine_ncc | 1089 | 1217/1266 | 1021 | 1024 | 0 |
| heterogeneous_texture22051003_warp22091003 | affine_ncc | 1585 | 1770/1806 | 1505 | 1508 | 0 |
| heterogeneous_texture22051005_warp22091005 | affine_ncc | 1196 | 1277/1320 | 1133 | 1134 | 0 |
| heterogeneous_texture22051007_warp22091007 | affine_ncc | 1240 | 1275/1322 | 1200 | 1200 | 0 |
| heterogeneous_texture22051009_warp22091009 | affine_ncc | 347 | 826/872 | 279 | 282 | 0 |
| heterogeneous_texture22051011_warp22091011 | affine_ncc | 1044 | 1273/1326 | 954 | 958 | 0 |
| heterogeneous_texture22051013_warp22091013 | affine_ncc | 1003 | 1054/1107 | 969 | 972 | 0 |
| heterogeneous_texture22051015_warp22091015 | affine_ncc | 1283 | 1357/1405 | 1223 | 1228 | 0 |
| heterogeneous_texture22051017_warp22091017 | affine_ncc | 1346 | 1419/1486 | 1278 | 1282 | 0 |
| heterogeneous_texture22051019_warp22091019 | affine_ncc | 1301 | 1463/1519 | 1239 | 1241 | 0 |
| translation_texture22051000_warp22091000 | local_rotation_rank2 | 69 | 1305/1380 | 0 | 0 | 0 |
| translation_texture22051002_warp22091002 | local_rotation_rank2 | 116 | 1344/1376 | 0 | 0 | 0 |
| translation_texture22051004_warp22091004 | local_rotation_rank2 | 126 | 1161/1232 | 0 | 0 | 0 |
| translation_texture22051006_warp22091006 | local_rotation_rank2 | 135 | 1326/1404 | 0 | 0 | 0 |
| translation_texture22051008_warp22091008 | rotation_rank2 | 9 | 1539/1596 | 0 | 0 | 0 |
| translation_texture22051010_warp22091010 | local_rotation_rank3 | 150 | 1581/1612 | 0 | 0 | 0 |
| translation_texture22051012_warp22091012 | local_rotation_rank2 | 123 | 1410/1440 | 0 | 0 | 0 |
| translation_texture22051014_warp22091014 | local_rotation_rank2 | 138 | 1326/1352 | 0 | 0 | 0 |
| translation_texture22051016_warp22091016 | local_rotation_rank3 | 159 | 1590/1674 | 0 | 0 | 0 |
| translation_texture22051018_warp22091018 | local_rotation_rank2 | 81 | 1590/1590 | 0 | 0 | 0 |

The saved JSON and NPZ outputs give the per-hypothesis counts, window states and provenance. Reported wall times exclude generation of the input hypotheses and are not full pipeline times.

Paired maps: green means both methods correct, red means only the baseline correct, blue means only propagation correct, gray means neither correct, and black is invisible.
