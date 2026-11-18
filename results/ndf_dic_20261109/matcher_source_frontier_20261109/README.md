# Coarse matcher source ablation with NCC and IC-GN propagation

Normalized cross-correlation (NCC) is the local photometric search. Inverse-compositional Gauss-Newton (IC-GN) is the window solver. Histogram of oriented and rotated gradients (HORG) is compared with DeepMatching and the learned coarse matcher. Truth is used only after selection, for scoring. All methods use the same NCC search, robust affine-fit acceptance threshold, IC-GN settings, and status-only frontier propagation. Matcher extraction settings are method-specific and listed in `summary.json`.

| Source | Pairs with affine | Matcher failures | Pairs with seeds | One-pass correct | Correct after propagation | Oracle correct | Seeded-pair oracle fraction | Median pipeline (s) | Bad accepted | Invisible accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned coarse matcher | 20/20 | 0 | 19/20 | 16,266 | 20,651 | 21,367 | 99.98% | 0.102 | 0 | 0 |
| HORG | 0/20 | 20 | 1/20 | 6 | 1,099 | 21,367 | 100.00% | 0.083 | 0 | 0 |
| DeepMatching | 10/20 | 10 | 10/20 | 10,313 | 11,989 | 21,367 | 99.97% | 0.101 | 0 | 0 |

Per-pair counts, affine fit support, bad-acceptance coordinates, and timings are in `summary.json`.

Reproduce the full comparison with:

```console
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_matcher_source.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /path/to/Live-DIC/python/src \
  --learned-baseline-dir results/ndf_dic_20261109/causal_ablation_frontier_corrected_20261109 \
  --output-dir results/ndf_dic_20261109/matcher_source_frontier_20261109
```
