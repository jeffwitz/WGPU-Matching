# Weighted estimates for unresolved windows

ORACLE CONTROL: the analytically known material boundary at x = 1932 px restricts which neighbours may be used. This boundary is unavailable to the unknown-image pipeline. Validated neighbouring inverse-compositional Gauss–Newton (IC-GN) states are transported to each unresolved centre and averaged with inverse-distance weights. At least two sources are required and candidate displacements must agree within 8 px. These estimates have not been accepted by the solver and are not used as new sources.

| Family | Visible unresolved | Estimated visible | Correct <2 px | Correct <5 px | Estimated invisible |
|---|---:|---:|---:|---:|---:|
| discontinuous | 392 | 131 | 131 | 131 | 202 |

The analytical field is used only for the reported errors. See `summary.json` and the per-window NPZ archives for support counts, disagreement and errors.

The 8 px disagreement check is only a preliminary guard. It does not prove that source windows share a material region and has not been qualified on cracks or discontinuities. Estimated windows remain marked as interpolated.

Post-hoc scoring at boundary x = 1932 px finds 0 estimates with error ≥5 px and at least one accepted source across the boundary. This boundary also restricts sources in the oracle control.

Reproduce from the repository root:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/analyze_weighted_seed_fill.py \
  --evaluation results/seed_propagation_20261111/p3_perturbations/shifted_opening/qualification/evaluation --archive results/seed_propagation_20261111/p3_perturbations/shifted_opening/inputs \
  --output-dir results/seed_propagation_20261111/p3_perturbations/shifted_opening/weighted_oracle_regions --diagnostic-boundary-x 1932 --oracle-region-boundary-x 1932
```
