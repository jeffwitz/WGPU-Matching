# Deferred versus eager rotation hypotheses

These previously inspected pairs are a post-hoc diagnostic. Both paths start from precomputed global affine and local normalized cross-correlation (NCC) inputs. This excludes their generation, the initial global image search and complete pipeline latency. Analytical truth is used only to score outputs.

Pairs: 40; eager / deferred correct: 52211 / 52211; lost / gained correct windows: 0 / 0.
Eager / deferred false accepted: 0 / 0.
Pairs requesting extra rotations: 0 / 40; additional rotations generated: 0.

| Pair | Visible | Correct eager / deferred | Correct lost / gained | False eager / deferred | Rotations requested |
|---|---:|---:|---:|---:|---|
| heterogeneous_texture23051001_warp23091001 | 1092 | 1029 / 1029 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051003_warp23091003 | 1023 | 968 / 968 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051005_warp23091005 | 1482 | 1436 / 1436 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051007_warp23091007 | 1297 | 1255 / 1255 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051009_warp23091009 | 1127 | 1080 / 1080 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051011_warp23091011 | 1396 | 1338 / 1338 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051013_warp23091013 | 1397 | 1336 / 1336 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051015_warp23091015 | 1387 | 1344 / 1344 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051017_warp23091017 | 1084 | 1023 / 1023 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture23051019_warp23091019 | 1306 | 1242 / 1242 | 0 / 0 | 0 / 0 | False |
| translation_texture23051000_warp23091000 | 1568 | 1568 / 1568 | 0 / 0 | 0 / 0 | False |
| translation_texture23051002_warp23091002 | 1537 | 1537 / 1537 | 0 / 0 | 0 / 0 | False |
| translation_texture23051004_warp23091004 | 1760 | 1728 / 1728 | 0 / 0 | 0 / 0 | False |
| translation_texture23051006_warp23091006 | 1881 | 1792 / 1792 | 0 / 0 | 0 / 0 | False |
| translation_texture23051008_warp23091008 | 1316 | 1288 / 1288 | 0 / 0 | 0 / 0 | False |
| translation_texture23051010_warp23091010 | 1134 | 1092 / 1092 | 0 / 0 | 0 / 0 | False |
| translation_texture23051012_warp23091012 | 1404 | 1404 / 1404 | 0 / 0 | 0 / 0 | False |
| translation_texture23051014_warp23091014 | 1581 | 1500 / 1500 | 0 / 0 | 0 / 0 | False |
| translation_texture23051016_warp23091016 | 1225 | 1176 / 1176 | 0 / 0 | 0 / 0 | False |
| translation_texture23051018_warp23091018 | 1416 | 1357 / 1357 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051001_warp24091001 | 1133 | 1077 / 1077 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051003_warp24091003 | 958 | 901 / 901 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051005_warp24091005 | 1426 | 1380 / 1380 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051007_warp24091007 | 1410 | 1367 / 1367 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051009_warp24091009 | 1469 | 1424 / 1424 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051011_warp24091011 | 690 | 637 / 637 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051013_warp24091013 | 1223 | 1169 / 1169 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051015_warp24091015 | 1513 | 1462 / 1462 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051017_warp24091017 | 1230 | 1162 / 1162 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture24051019_warp24091019 | 998 | 940 / 940 | 0 / 0 | 0 / 0 | False |
| translation_texture24051000_warp24091000 | 1395 | 1350 / 1350 | 0 / 0 | 0 / 0 | False |
| translation_texture24051002_warp24091002 | 1652 | 1652 / 1652 | 0 / 0 | 0 / 0 | False |
| translation_texture24051004_warp24091004 | 1458 | 1378 / 1378 | 0 / 0 | 0 / 0 | False |
| translation_texture24051006_warp24091006 | 1568 | 1568 / 1568 | 0 / 0 | 0 / 0 | False |
| translation_texture24051008_warp24091008 | 1161 | 1118 / 1118 | 0 / 0 | 0 / 0 | False |
| translation_texture24051010_warp24091010 | 1881 | 1792 / 1792 | 0 / 0 | 0 / 0 | False |
| translation_texture24051012_warp24091012 | 1664 | 1632 / 1632 | 0 / 0 | 0 / 0 | False |
| translation_texture24051014_warp24091014 | 1323 | 1323 / 1323 | 0 / 0 | 0 / 0 | False |
| translation_texture24051016_warp24091016 | 1215 | 1144 / 1144 | 0 / 0 | 0 / 0 | False |
| translation_texture24051018_warp24091018 | 1320 | 1242 / 1242 | 0 / 0 | 0 / 0 | False |

Per-window arrays retain the location of all changed decisions. This experiment does not validate non-regression on new pairs, rotations or real images. See the five-pair report and fallback test for timing limits.

Reproduce this 40-pair comparison:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/seed_propagation_20261111/p2_confirmatory_inputs \
  --output-dir results/seed_propagation_20261111/p2_lazy_rotation_all_forty --sparse-stride 2 \
  --lazy-rotation-weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --lazy-rotation-output results/seed_propagation_20261111/p2_lazy_rotation_all_forty/rotations --escalation --local-affine-escalation \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
PYTHONPATH=python/src .venv/bin/python python/benchmarks/summarize_lazy_rotation.py \
  --eager results/seed_propagation_20261111/p2_confirmatory/evaluation --lazy results/seed_propagation_20261111/p2_lazy_rotation_all_forty
```
