# Serial and batched anchor confirmation

The masked Adrien experimental image pair is evaluated at its native 7920 ×
6004 resolution, with the image-model raster at 192 × 146. There is one warm-up
and three alternating-order repetitions with identical cached solver pixels,
on Intel Graphics Lunar Lake using Vulkan. Each run includes process startup,
image-cache verification, graphics setup, solver work and output writes. The
timers are wall-clock observations, not graphics processing unit (GPU)
timestamps. The input cache and hypotheses were prepared beforehand; this is
not full pipeline latency. No other project benchmark was observed at the start;
the graphics processor was not reserved exclusively.

Median verification time, serial / batched: 15.299 / 0.827 s. Exact output parity across runs: True.

| Run | Warm-up | Serial verification (s) | Batched verification (s) | Calls serial / batched | Active windows serial / batched | Acceptance changes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | True | 15.286 | 0.735 | 1538 / 27 | 13646 / 20730 | 0 |
| 1 | False | 15.049 | 0.797 | 1538 / 27 | 13646 / 20730 | 0 |
| 2 | False | 15.299 | 0.860 | 1538 / 27 | 13646 / 20730 | 0 |
| 3 | False | 15.822 | 0.827 | 1538 / 27 | 13646 / 20730 | 0 |

All saved parameters and provenance are compared in the JavaScript Object Notation (JSON) output. Batching may perform speculative solves that the serial path would skip. The method is experimental and requires a window-independent deterministic solver.

The median complete verification-process time, including image/cache reading,
reference setup and output writes, is 23.32 s serial versus 8.63 s batched.
Both modes accept the same 7,097 windows with exactly equal parameters, anchor
masks, parents, origins and generations. No analytical ground truth is available
for these images, so acceptance and exact parity do not establish measurement
accuracy. The graphics solver remains accessed through its private adapter, and
the batching orchestration remains on the host.

After this benchmark, a no-benefit guard was added for cases with at most
eight candidate anchors. The selected Adrien hypothesis had 1,519 candidate
anchors, so the guard does not change the measured execution branch.

Reproduce in a **new** output directory after generating the cached inputs as
described in `../p2_solver_images_adrien/README.md`:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_anchor_confirmation.py \
  --inputs results/seed_propagation_20261111/p2_solver_images_adrien/inputs \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_anchor_batch_adrien_new \
  --external-reference /home/jeff/CNRS/DIC/AdrienVariational/img_000000_0.09422.tiff \
  --external-target /home/jeff/CNRS/DIC/AdrienVariational/img_000075_1608.88508.tiff \
  --roi-mask /home/jeff/CNRS/DIC/AdrienVariational/Mask.tiff --repetitions 3
```
