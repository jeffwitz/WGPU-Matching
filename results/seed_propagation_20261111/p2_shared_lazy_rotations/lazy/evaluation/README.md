# Deferred versus eager rotation hypotheses

These pairs were newly generated under a fixed protocol; this small sample is not final qualification. Both paths start from precomputed global affine and local normalized cross-correlation (NCC) inputs. This excludes their generation, the initial global image search and complete pipeline latency. Analytical truth is used only to score outputs.

Pairs: 6; eager / deferred correct: 7096 / 7096; lost / gained correct windows: 0 / 0.
Eager / deferred false accepted: 0 / 0.
Pairs requesting extra rotations: 0 / 6; additional rotations generated: 0.

| Pair | Visible | Correct eager / deferred | Correct lost / gained | False eager / deferred | Rotations requested |
|---|---:|---:|---:|---:|---|
| affine_texture53051003_warp53091003 | 1039 | 987 / 987 | 0 / 0 | 0 / 0 | False |
| discontinuous_texture53051004_warp53091004 | 1323 | 1248 / 1248 | 0 / 0 | 0 / 0 | False |
| rigid_texture53051002_warp53091002 | 1166 | 1115 / 1115 | 0 / 0 | 0 / 0 | False |
| affine_texture54051003_warp54091003 | 1416 | 1373 / 1373 | 0 / 0 | 0 / 0 | False |
| discontinuous_texture54051004_warp54091004 | 1250 | 1225 / 1225 | 0 / 0 | 0 / 0 | False |
| rigid_texture54051002_warp54091002 | 1196 | 1148 / 1148 | 0 / 0 | 0 / 0 | False |

Per-window arrays retain the location of all changed decisions. This experiment does not establish general non-regression beyond these pairs or on real images. See the five-pair report and fallback test for timing limits.

Reproduce this comparison of the saved outputs:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/summarize_lazy_rotation.py \
  --eager results/seed_propagation_20261111/p2_shared_lazy_rotations/eager/evaluation --lazy results/seed_propagation_20261111/p2_shared_lazy_rotations/lazy/evaluation --new-pairs
```
