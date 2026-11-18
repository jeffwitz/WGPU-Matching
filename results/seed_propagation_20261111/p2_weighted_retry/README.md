# Weighted estimates followed by a solver retry

The WebGPU inverse-compositional Gauss–Newton (IC-GN) solver retries weighted estimates using unchanged settings. The active mask comes exclusively from the estimates; analytical visibility is used only for scoring.

| Family | Attempted visible | Accepted visible | Correct <2 px | False accepted | Accepted invisible |
|---|---:|---:|---:|---:|---:|
| heterogeneous | 1067 | 0 | 0 | 0 | 0 |
| translation | 1002 | 0 | 0 | 0 | 0 |

These retries are a post-hoc experiment on the previously inspected test set; they do not change the frozen G2 results. See `summary.json` and the per-window archives.

Only 2 of the 1,067 estimated visible heterogeneous windows converge correctly with the archived analytical oracle initialization; neither of these two is accepted by the weighted retry. Their initial translation errors are 0.39 and 0.65 px, and their displacement-gradient errors in Frobenius norm are 0.014 and 0.020. None of the 1,002 estimated visible translation windows converges under the analytical oracle. This locates most of the limitation in the solver/support qualification rather than in the interpolated centre displacement alone; the exact reason for the two remaining rejections is not established.

Reproduce from the repository root:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_weighted_seed_retry.py \
  --estimates results/seed_propagation_20261111/p2_weighted_estimates \
  --archive results/seed_propagation_20261111/p2_test_inputs \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --output-dir results/seed_propagation_20261111/p2_weighted_retry
```
