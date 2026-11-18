# Transfer from 64 px accepted windows to the 32 px grid

Each fine-grid initial affine state is transported from its nearest solver-accepted 64 px neighbour within 96 px. It counts as a fine result only if the WebGPU inverse-compositional Gauss–Newton (IC-GN) solver accepts it. The source mask does not use analytical visibility. These synthetic pairs were already inspected.

Visible fine windows: 102615.
Available initializations: 98592.
Correct fine results: 96969; false accepted: 0 / 96969.
Direct fine-grid baseline correct: 79646.

Per-pair results and per-window source indices are retained in JSON and NPZ files. No material-boundary constraint is imposed; this experiment does not validate transfers across cracks. Its timings are not performance measurements.

Reproduce:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_coarse_to_fine_transfer.py \
  --coarse results/seed_propagation_20261111/p3_perturbations/shifted_opening/qualification/evaluation --fine results/seed_propagation_20261111/p3_32px_shifted \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src --output-dir results/seed_propagation_20261111/p3_32px_shifted_transfer \
  --discontinuity-boundary-fraction 0.503125
```
