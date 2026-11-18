# Weighted estimates on discontinuities and shear bands

Only accepted inverse-compositional Gauss–Newton (IC-GN) neighbours are sources. Estimates use inverse-distance weights, at least two sources and an 8 px maximum displacement disagreement. They are not solver-accepted measurements. Truth is used only for scoring; these archived cases were previously inspected.

| Family | Pairs | Unresolved visible | Estimated visible | Correct <2 px | Correct <5 px | Largest error (px) |
|---|---:|---:|---:|---:|---:|---:|
| discontinuous | 10 | 2703 | 322 | 322 | 322 | 0.03 |
| shear_band_g0.33 | 10 | 6314 | 177 | 173 | 177 | 2.77 |
| shear_band_g0.67 | 10 | 483 | 417 | 406 | 413 | 10.83 |
| shear_band_g2 | 10 | 3194 | 973 | 303 | 306 | 35.56 |

Estimates with error ≥2 px despite every source being visible and correct to <2 px:

- discontinuous: 0
- shear_band_g0.33: 4
- shear_band_g0.67: 11
- shear_band_g2: 536

Per-window errors and every estimate with error at least 2 px are in the saved JSON and NPZ outputs.

Reproduce:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/analyze_weighted_frontier_families.py \
  --archive results/ndf_dic_20261109/frontier_generalization_20261109 --output-dir results/seed_propagation_20261111/p2_weighted_discontinuities
```
