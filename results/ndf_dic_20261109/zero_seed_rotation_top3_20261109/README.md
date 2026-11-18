# Image-only rotation verification for zero-seed pairs

Inverse-compositional Gauss-Newton (IC-GN) status is the only selection criterion after taking the three highest image-derived rotation scores. Each angle is evaluated on the same fixed 128-window subset with no learned nonlinear field residual and local normalized cross-correlation (NCC). Truth and visibility are used only for post-selection diagnostics.

| Pair | Rank | Hypothesis angle (degrees) | Image score | Accepted IC-GN seeds / 128 | Selected |
|---:|---:|---:|---:|---:|---|
| 17051005 | 1 | 20.0 | 0.02858 | 0/128 | yes |
| 17051005 | 2 | 60.0 | 0.02312 | 0/128 | no |
| 17051005 | 3 | -100.0 | 0.02183 | 0/128 | no |
| 17051007 | 1 | 20.0 | 0.01887 | 0/128 | yes |
| 17051007 | 2 | 60.0 | 0.01679 | 0/128 | no |
| 17051007 | 3 | -140.0 | 0.01621 | 0/128 | no |

All attempted hypotheses, selected angles, report paths, and timing are in `summary.json`. A tie is resolved by the existing image-score rank order; the solver-status count itself is the selection value.

Reproduce with:

```console
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_zero_seed_rotation.py \
  --ranking-report results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109/affine_ncc/heterogeneous_seed17000000.json \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /path/to/Live-DIC/python/src \
  --output-dir results/ndf_dic_20261109/zero_seed_rotation_top3_20261109
```
