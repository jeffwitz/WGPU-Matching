# Wall time of the current scripted P2 experiment

One heterogeneous 3840 × 2160 synthetic pair was run once for warm-up and three more times with no other benchmark intentionally running on the GPU. The adapter was Intel Graphics LNL through Vulkan. The current scripts execute the two baseline prefits, generate three rotation hypotheses and run image-only verification and propagation.

| Script stage | Median of three runs |
|---|---:|
| Two baseline prefits | 15.27 s |
| Three rotation hypothesis scripts | 22.80 s |
| Verification and propagation script | 7.40 s |
| Complete scripted experiment | **45.37 s** |

The complete script time includes repeated synthetic-image generation, process and model initialization, analytical-oracle diagnostics and disk output. It is the wall time of this experimental workflow on one pair, not the production latency of an integrated pipeline. No claim is made about meeting the one-second target. Individual measurements are in `summary.json`.

Reproduce from the repository root:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_p2_scripted_wall.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_scripted_timing_new --repetitions 3
```
