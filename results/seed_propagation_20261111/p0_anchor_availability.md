# Archived anchor availability

Inverse-compositional Gauss–Newton (IC-GN) acceptance and correctness are scored on analytically visible windows. Eight-connected components use the archived grid.

| Variant | Pairs | Visible | Accepted | Correct | False accepted | Median correct | Pairs with at most one | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| affine_ncc | 20 | 22416 | 16266 | 16266 | 0 | 938 | 1 | 21367 |
| affine_ndf | 20 | 22416 | 1748 | 1748 | 0 | 84.5 | 3 | 21367 |
| affine_ndf_ncc | 20 | 22416 | 13172 | 13172 | 0 | 716 | 2 | 21367 |
| affine_only | 20 | 22416 | 2071 | 2070 | 1 | 81 | 3 | 21367 |

Per-pair and per-component counts, including accepted invisible windows, are in the accompanying JSON file.

The archived totals match section 2.2 of the specification. The affine plus neural deformation field variant has an exact median of 84.5 correct anchors, displayed as 84 in the original table. This rounding difference does not affect the totals.

Reproduce from the repository root:

```console
python python/benchmarks/analyze_anchor_availability.py results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 --output results/seed_propagation_20261111/p0_anchor_availability
```
