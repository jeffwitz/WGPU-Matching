# Image-only anchor selection: development pilot

This exploratory run evaluates the existing image-derived zero, affine, and affine plus local normalized cross-correlation (NCC) initial states on the 20 archived heterogeneous development pairs. It ranks them by accepted inverse-compositional Gauss–Newton (IC-GN) solver statuses on a stride-four subgrid, confirms anchors by consistent solved neighbours, and propagates accepted affine states. Analytical truth is accessed only for scoring after selection and propagation. The initial states were computed before this run and the reported wall time excludes their creation.

The fixed provisional parameters are displacement tolerance 8 px, gradient tolerance 0.25 and subgrid stride 4. This is a development experiment, not the independent P2 test or a calibration of those parameters.

| Measure | Result |
|---|---:|
| Visible windows | 22,416 |
| Correct after propagation | 19,919 |
| Baseline affine plus NCC, one solve | 16,266 |
| Oracle correct | 21,367 |
| False accepted visible windows | 0 |
| Pairs with zero confirmed seeds | 2/20 |

The two seedless pairs are `17051005` and `17051007`; their respective oracle counts are 712 and 735. The algorithm selects the first hypothesis in a tie, which is zero here. The current pilot does not perform the requested three-orientation verification, escalation by connected region, or threshold calibration. It has not been tested on the held-out seed groups and does not establish gate G2.

Per-pair status counts and per-window outputs are in `summary.json` and the accompanying NPZ files. Regenerate using:

```console
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 \
  --output-dir results/seed_propagation_20261111/p2_development
```
