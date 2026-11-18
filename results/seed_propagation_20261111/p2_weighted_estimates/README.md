# Weighted estimates for unresolved windows

Validated neighbouring inverse-compositional Gauss–Newton (IC-GN) states are transported to each unresolved centre and averaged with inverse-distance weights. At least two sources are required and candidate displacements must agree within 8 px. These estimates have not been accepted by the solver and are not used as new sources.

| Family | Visible unresolved | Estimated visible | Correct <2 px | Correct <5 px | Estimated invisible |
|---|---:|---:|---:|---:|---:|
| heterogeneous | 1074 | 1067 | 1065 | 1067 | 389 |
| translation | 1008 | 1002 | 1002 | 1002 | 587 |

The analytical field is used only for the reported errors. See `summary.json` and the per-window NPZ archives for support counts, disagreement and errors.

The 8 px disagreement check is only a preliminary guard. It does not prove that the source windows belong to the same physical material region and has not been qualified on cracks or discontinuities. The estimated windows must remain marked as interpolated.

Reproduce from the repository root:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/analyze_weighted_seed_fill.py \
  --evaluation results/seed_propagation_20261111/p2_independent/evaluation \
  --archive results/seed_propagation_20261111/p2_test_inputs \
  --output-dir results/seed_propagation_20261111/p2_weighted_estimates
```
