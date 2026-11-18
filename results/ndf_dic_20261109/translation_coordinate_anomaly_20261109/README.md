# Per-window analysis of the raw-bilinear translation regression

Inverse-compositional Gauss-Newton (IC-GN) is the local solver. This comparison reruns the same ten seeded synthetic translations with code before and after the coordinate correction. It changes no threshold or solver setting. Green map cells are correct in both runs, red cells only before the correction, blue cells only after it, gray cells are visible but incorrect in both, and black cells lie outside analytical visibility.

| Correct before / visible | Correct after / visible | Lost | Gained |
|---:|---:|---:|---:|
| 11,033/13,444 | 10,485/13,444 | 553 | 5 |

All ten pairs regress in net correct windows. The net loss is 548 windows; five individually lost windows become correct in the corrected run while 553 others are lost. The per-window maps and parameter exports show that the change starts in the initial field: the initial displacement root mean square error increases in all ten pairs. The geometric cause is the corrected mapping from native pixel centres to the reduced feature grid's true half-cell centres, following the half-pixel resize convention. This reveals the learned field's sensitivity to spatial sample position. It does not justify reverting the coordinate correction or changing thresholds. These synthetic measurements do not establish accuracy on experimental images.

Reproduce from the repository root (the pre-fix qualifier is archived from commit `4b575de`):

```console
mkdir -p /tmp/wgpu-old-4b575de
git archive 4b575de -- python/benchmarks | tar -x -C /tmp/wgpu-old-4b575de
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_translation_coordinate_anomaly.py \
  --pre-fix-qualifier /tmp/wgpu-old-4b575de/python/benchmarks/qualify_ndf_pair_icgn.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /path/to/Live-DIC/python/src \
  --output-dir results/ndf_dic_20261109/translation_coordinate_anomaly_20261109
```
