# Held-out image-only initialization

**Clarification, 11 November:** the original numerical G2 criteria in `CdC_Codex.md` are met by this fresh-group experiment. The `INCOMPLETE` verdict below refers to the additional end-to-end timing condition in the separately recorded confirmatory protocol, not to an original G2 timing criterion. See `../../cdc_completion_audit_20261111.md`.

The thresholds and sampling stride were frozen on the development group before these test cases. The inverse-compositional Gauss–Newton (IC-GN) solver accepts anchors; analytic truth is used only after selection and propagation for scoring.

G2: **INCOMPLETE**. Heterogeneous correct 23570/24641 (95.65%); paired mean gain +23.65%, 95% pair bootstrap [+14.78%, +35.19%]; 0/20 pairs below 20%; 0/23570 false accepted; translation correct-count difference +3468.

G2 numerical criteria: **PASS**; the qualification limitations below are reported separately.

Translation baseline diagnostic: 0/20 global affine prefits were rejected; the near-identity affine retention option was used.

Qualification limitations: end-to-end hypothesis-generation timing not measured.

Heterogeneous mean gain versus the matched re-solve control: +23.40%, 95% pair bootstrap [+14.69%, +34.61%]. Accepted windows outside analytical visibility across both families: 0.

| Pair | Selected | Confirmed seeds | Correct / visible | Baseline | Matched control | False accepted |
|---|---|---:|---:|---:|---:|---:|
| heterogeneous_texture23051001_warp23091001 | affine_ncc | 943 | 1029/1092 | 897 | 897 | 0 |
| heterogeneous_texture23051003_warp23091003 | affine_ncc | 763 | 968/1023 | 670 | 671 | 0 |
| heterogeneous_texture23051005_warp23091005 | affine_ncc | 980 | 1436/1482 | 871 | 875 | 0 |
| heterogeneous_texture23051007_warp23091007 | affine_ncc | 1140 | 1255/1297 | 1077 | 1078 | 0 |
| heterogeneous_texture23051009_warp23091009 | affine_ncc | 992 | 1080/1127 | 941 | 944 | 0 |
| heterogeneous_texture23051011_warp23091011 | affine_ncc | 1203 | 1338/1396 | 1158 | 1159 | 0 |
| heterogeneous_texture23051013_warp23091013 | affine_ncc | 1231 | 1336/1397 | 1169 | 1170 | 0 |
| heterogeneous_texture23051015_warp23091015 | affine_ncc | 1205 | 1344/1387 | 1137 | 1139 | 0 |
| heterogeneous_texture23051017_warp23091017 | affine_ncc | 944 | 1023/1084 | 892 | 898 | 0 |
| heterogeneous_texture23051019_warp23091019 | affine_ncc | 1163 | 1242/1306 | 1116 | 1118 | 0 |
| translation_texture23051000_warp23091000 | affine_ncc | 1387 | 1568/1568 | 1330 | 1340 | 0 |
| translation_texture23051002_warp23091002 | affine_ncc | 1462 | 1537/1537 | 1360 | 1368 | 0 |
| translation_texture23051004_warp23091004 | affine_ncc | 1630 | 1728/1760 | 1527 | 1533 | 0 |
| translation_texture23051006_warp23091006 | affine_ncc | 1732 | 1792/1881 | 1610 | 1611 | 0 |
| translation_texture23051008_warp23091008 | affine_ncc | 1215 | 1288/1316 | 1145 | 1145 | 0 |
| translation_texture23051010_warp23091010 | affine_ncc | 1008 | 1092/1134 | 964 | 967 | 0 |
| translation_texture23051012_warp23091012 | affine_ncc | 1201 | 1404/1404 | 1144 | 1148 | 0 |
| translation_texture23051014_warp23091014 | affine_ncc | 1498 | 1500/1581 | 1439 | 1441 | 0 |
| translation_texture23051016_warp23091016 | affine_ncc | 1087 | 1176/1225 | 1013 | 1013 | 0 |
| translation_texture23051018_warp23091018 | affine_ncc | 1262 | 1357/1416 | 1177 | 1185 | 0 |
| heterogeneous_texture24051001_warp24091001 | affine_ncc | 965 | 1077/1133 | 913 | 914 | 0 |
| heterogeneous_texture24051003_warp24091003 | affine_ncc | 9 | 901/958 | 16 | 20 | 0 |
| heterogeneous_texture24051005_warp24091005 | affine_ncc | 181 | 1380/1426 | 129 | 133 | 0 |
| heterogeneous_texture24051007_warp24091007 | affine_ncc | 1264 | 1367/1410 | 1207 | 1213 | 0 |
| heterogeneous_texture24051009_warp24091009 | affine_ncc | 1143 | 1424/1469 | 1010 | 1018 | 0 |
| heterogeneous_texture24051011_warp24091011 | affine_ncc | 556 | 637/690 | 509 | 511 | 0 |
| heterogeneous_texture24051013_warp24091013 | affine_ncc | 1055 | 1169/1223 | 986 | 988 | 0 |
| heterogeneous_texture24051015_warp24091015 | affine_ncc | 1350 | 1462/1513 | 1284 | 1289 | 0 |
| heterogeneous_texture24051017_warp24091017 | affine_ncc | 1050 | 1162/1230 | 1001 | 1009 | 0 |
| heterogeneous_texture24051019_warp24091019 | affine_ncc | 870 | 940/998 | 805 | 805 | 0 |
| translation_texture24051000_warp24091000 | affine_ncc | 1205 | 1350/1395 | 1135 | 1142 | 0 |
| translation_texture24051002_warp24091002 | affine_ncc | 1575 | 1652/1652 | 1445 | 1447 | 0 |
| translation_texture24051004_warp24091004 | affine_ncc | 1311 | 1378/1458 | 1268 | 1276 | 0 |
| translation_texture24051006_warp24091006 | affine_ncc | 1413 | 1568/1568 | 1329 | 1335 | 0 |
| translation_texture24051008_warp24091008 | affine_ncc | 1018 | 1118/1161 | 960 | 970 | 0 |
| translation_texture24051010_warp24091010 | affine_ncc | 1664 | 1792/1881 | 1573 | 1583 | 0 |
| translation_texture24051012_warp24091012 | affine_ncc | 1544 | 1632/1664 | 1458 | 1462 | 0 |
| translation_texture24051014_warp24091014 | affine_ncc | 1141 | 1323/1323 | 1116 | 1124 | 0 |
| translation_texture24051016_warp24091016 | affine | 1144 | 1144/1215 | 1087 | 1087 | 0 |
| translation_texture24051018_warp24091018 | affine_ncc | 1119 | 1242/1320 | 1093 | 1094 | 0 |

The saved JSON and NPZ outputs give the per-hypothesis counts, window states and provenance. Reported wall times exclude generation of the input hypotheses and are not full pipeline times.

Paired maps: green means both methods correct, red means only the baseline correct, blue means only propagation correct, gray means neither correct, and black is invisible.
