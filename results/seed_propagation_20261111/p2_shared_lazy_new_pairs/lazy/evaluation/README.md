# Deferred versus eager rotation hypotheses

These pairs were newly generated under a fixed protocol; this small sample is not final qualification. Both paths start from precomputed global affine and local normalized cross-correlation (NCC) inputs. This excludes their generation, the initial global image search and complete pipeline latency. Analytical truth is used only to score outputs.

Pairs: 4; eager / deferred correct: 5011 / 5011; lost / gained correct windows: 0 / 0.
Eager / deferred false accepted: 0 / 0.
Pairs requesting extra rotations: 0 / 4; additional rotations generated: 0.

| Pair | Visible | Correct eager / deferred | Correct lost / gained | False eager / deferred | Rotations requested |
|---|---:|---:|---:|---:|---|
| heterogeneous_texture51051001_warp51091001 | 1091 | 1028 / 1028 | 0 / 0 | 0 / 0 | False |
| translation_texture51051000_warp51091000 | 1560 | 1450 / 1450 | 0 / 0 | 0 / 0 | False |
| heterogeneous_texture52051001_warp52091001 | 1107 | 1049 / 1049 | 0 / 0 | 0 / 0 | False |
| translation_texture52051000_warp52091000 | 1484 | 1484 / 1484 | 0 / 0 | 0 / 0 | False |

Per-window arrays retain the location of all changed decisions. This experiment does not establish general non-regression beyond these pairs, or for rotations and real images. See the five-pair report and fallback test for timing limits.

Reproduce this comparison of the saved outputs:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/summarize_lazy_rotation.py \
  --eager results/seed_propagation_20261111/p2_shared_lazy_new_pairs/eager/evaluation --lazy results/seed_propagation_20261111/p2_shared_lazy_new_pairs/lazy/evaluation --new-pairs
```
