# Oracle-seeded propagation qualification

The inverse-compositional Gauss–Newton (IC-GN) solver runs on WebGPU. Analytical truth is used to choose the explicitly oracle-labelled seeds and score results. The direct-neighbour-first frontier uses two attempts per pass and fixed consistency limits of 8 pixels and 0.25 for the displacement-gradient Frobenius norm.

| Anchors | Affine transport | Consistency | Correct / oracle | False accepted / accepted | Median passes | Control correct |
|---:|---|---|---:|---:|---:|---:|
| 1 | False | False | 4252/106835 (3.98%) | 0/4252 | 1 | 5525 |
| 1 | False | True | 327/106835 (0.31%) | 0/327 | 1 | 883 |
| 1 | True | False | 106831/106835 (100.00%) | 0/106831 | 34 | 81357 |
| 1 | True | True | 106831/106835 (100.00%) | 0/106831 | 34 | 81357 |
| 4 | False | False | 4879/106835 (4.57%) | 0/4879 | 1 | 8677 |
| 4 | False | True | 704/106835 (0.66%) | 0/704 | 1 | 2983 |
| 4 | True | False | 106834/106835 (100.00%) | 0/106834 | 21 | 81431 |
| 4 | True | True | 106834/106835 (100.00%) | 0/106834 | 21 | 81431 |
| 16 | False | False | 8817/106835 (8.25%) | 0/8817 | 1 | 20452 |
| 16 | False | True | 2231/106835 (2.09%) | 0/2231 | 1 | 10846 |
| 16 | True | False | 106835/106835 (100.00%) | 0/106835 | 12 | 81763 |
| 16 | True | True | 106835/106835 (100.00%) | 0/106835 | 12 | 81763 |

The no-propagation control repeats the same active-window schedules, starting each window from its own archived affine-plus-local-correlation initial state; it does not use the transported state. Both arms include the same correct oracle seeds.

## Gate G1 by draw

- Draw 0: 21367/21367 correct, 20/20 pairs at 80% of oracle, 0/21367 false accepted: PASS.
- Draw 1: 21366/21367 correct, 20/20 pairs at 80% of oracle, 0/21366 false accepted: PASS.
- Draw 2: 21367/21367 correct, 20/20 pairs at 80% of oracle, 0/21367 false accepted: PASS.
- Draw 3: 21367/21367 correct, 20/20 pairs at 80% of oracle, 0/21367 false accepted: PASS.
- Draw 4: 21367/21367 correct, 20/20 pairs at 80% of oracle, 0/21367 false accepted: PASS.

All five independent anchor draws must satisfy the acceptance criteria for this conservative verdict: PASS.

Per-window states, provenance, rejected-front reasons and matched control outputs are stored in the NPZ files; per-pair measurements and archive hashes are in JSON. The solver's private `_register_seeded_frame` adapter is an integration limitation.

Reproduce with:

```console
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_seed_propagation.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 \
  --output-dir results/seed_propagation_20261111/p1_oracle --draws 5
```

## Paired comparison by image pair

The difference is the mean coverage difference over the five seed draws; bootstrap resampling uses image pairs as the independent units.

| Pair | Correct, mean | Control, mean | Visible | Mean difference (percentage points) |
|---|---:|---:|---:|---:|
| heterogeneous_texture16051001_warp16091001 | 1043.0 | 756.8 | 1102 | +25.97 |
| heterogeneous_texture16051003_warp16091003 | 1219.0 | 1072.0 | 1282 | +11.47 |
| heterogeneous_texture16051005_warp16091005 | 1057.0 | 938.4 | 1118 | +10.61 |
| heterogeneous_texture16051007_warp16091007 | 1303.0 | 1154.2 | 1353 | +11.00 |
| heterogeneous_texture16051009_warp16091009 | 939.0 | 713.8 | 995 | +22.63 |
| heterogeneous_texture16051011_warp16091011 | 1470.0 | 1321.4 | 1506 | +9.87 |
| heterogeneous_texture16051013_warp16091013 | 869.0 | 752.0 | 923 | +12.68 |
| heterogeneous_texture16051015_warp16091015 | 799.0 | 633.8 | 863 | +19.14 |
| heterogeneous_texture16051017_warp16091017 | 1099.0 | 931.2 | 1158 | +14.49 |
| heterogeneous_texture16051019_warp16091019 | 1235.0 | 1004.6 | 1281 | +17.99 |
| heterogeneous_texture17051001_warp17091001 | 1207.0 | 1008.6 | 1252 | +15.85 |
| heterogeneous_texture17051003_warp17091003 | 1305.0 | 1153.4 | 1353 | +11.20 |
| heterogeneous_texture17051005_warp17091005 | 712.0 | 4.0 | 768 | +92.19 |
| heterogeneous_texture17051007_warp17091007 | 736.0 | 10.0 | 787 | +92.25 |
| heterogeneous_texture17051009_warp17091009 | 846.0 | 31.0 | 902 | +90.35 |
| heterogeneous_texture17051011_warp17091011 | 1154.0 | 981.8 | 1208 | +14.25 |
| heterogeneous_texture17051013_warp17091013 | 1177.8 | 1032.0 | 1224 | +11.91 |
| heterogeneous_texture17051015_warp17091015 | 1013.0 | 895.8 | 1064 | +11.02 |
| heterogeneous_texture17051017_warp17091017 | 1073.0 | 938.0 | 1122 | +12.03 |
| heterogeneous_texture17051019_warp17091019 | 1110.0 | 953.4 | 1155 | +13.56 |

Mean paired difference: +26.02 points; 95% pair bootstrap interval [+14.66, +39.13]. Improved/equal/regressed pairs: 20/0/0. Bootstrap seed: 20261111.


The `frontier_maps` images show oracle anchors in blue, correctly resolved visible windows in green, solver rejection in orange, displacement inconsistency in red, gradient inconsistency in purple, incorrect accepted windows in white, unattempted visible windows in gray and invisible windows in black.

## Source fingerprints

SHA-256 hashes of the current local sources:

- `python/benchmarks/qualify_seed_propagation.py`: `9c09462da16ccce02486dd28492230a3bac4dbefa385f9ebd1c5594155aa4cde`
- `python/src/wgpu_matching/seed_propagation.py`: `262a88ed339da5be9612b6d072a0a237954e877272aa61b2cc1233bc3a0a9563`
- `python/src/wgpu_matching/dic_synthetic.py`: `735834a2e048d425811daf6c6577b208388193b69ae3d7c22f1e7693e993a85c`
- `/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src/live_dic/wgpu_backend.py`: `f0e7cad5d0cb5e017cb23b341c05197ecaf91ff683b34fbd0b8b24d4fe30b238`