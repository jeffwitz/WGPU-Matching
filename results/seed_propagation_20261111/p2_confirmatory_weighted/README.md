# Weighted estimates for unresolved windows

Validated neighbouring inverse-compositional Gauss–Newton (IC-GN) states are transported to each unresolved centre and averaged with inverse-distance weights. At least two sources are required and candidate displacements must agree within 8 px. These estimates have not been accepted by the solver and are not used as new sources.

| Family | Visible unresolved | Estimated visible | Correct <2 px | Correct <5 px | Estimated invisible |
|---|---:|---:|---:|---:|---:|
| heterogeneous | 1071 | 1058 | 1044 | 1058 | 277 |
| translation | 818 | 812 | 812 | 812 | 770 |

The analytical field is used only for the reported errors. See `summary.json` and the per-window NPZ archives for support counts, disagreement and errors.

The 8 px disagreement check is only a preliminary guard. It does not prove that source windows share a material region and has not been qualified on cracks or discontinuities. Estimated windows remain marked as interpolated.

Reproduce from the repository root:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/analyze_weighted_seed_fill.py \
  --evaluation results/seed_propagation_20261111/p2_confirmatory/evaluation --archive results/seed_propagation_20261111/p2_confirmatory_inputs \
  --output-dir results/seed_propagation_20261111/p2_confirmatory_weighted
```
