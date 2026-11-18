# Transfer from 64 px accepted windows to the 32 px grid

Each fine-grid initial affine state is transported from its nearest solver-accepted 64 px neighbour within 96 px. It counts as a fine result only if the WebGPU inverse-compositional Gauss–Newton (IC-GN) solver accepts it. The source mask does not use analytical visibility. These synthetic pairs were already inspected in the coarse-grid experiment.

Visible fine windows: 397903.
Available initializations: 397821.
Correct fine results: 388983; false accepted: 0 / 388983.
Direct fine-grid baseline correct: 283437.

Per-pair results and per-window source indices are retained in JSON and NPZ files. No material-boundary constraint is imposed; this experiment does not validate transfers across cracks. Its timings are not performance measurements.

Reproduce:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_coarse_to_fine_transfer.py \
  --coarse results/seed_propagation_20261111/p2_confirmatory/evaluation --fine results/seed_propagation_20261111/p3_32px_confirmatory \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src --output-dir results/seed_propagation_20261111/p3_32px_confirmatory_transfer
```
