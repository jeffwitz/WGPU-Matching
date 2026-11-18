# Host versus graphics processing unit (GPU) frontier in the initialization pilot

The masked Adrien experimental pair is used at its native 7920 × 6004
resolution, with a 192 × 146 coarse model raster. One warm-up and three
alternating-order repetitions were measured on Intel Graphics Lunar Lake using
Vulkan. The two paths use identical cached solver images and batched anchor
confirmation. Host times include process startup, image-cache verification,
device/reference setup and result writes. Input hypotheses and cached images
were prepared earlier; these are not complete end-to-end timings. No GPU
timestamp measurement was available.

Median verification and propagation: host 0.850 s; GPU frontier 0.683 s. Median process: 8.404 / 8.046 s.

| Run | Warm-up | Process host / GPU (s) | Metadata exact | Max displacement difference (px) | Max gradient difference |
|---:|---|---:|---|---:|---:|
| 0 | True | 9.692 / 9.473 | True | 3.05175781e-05 | 3.7252903e-07 |
| 1 | False | 8.404 / 7.942 | True | 3.05175781e-05 | 3.7252903e-07 |
| 2 | False | 8.191 / 8.046 | True | 3.05175781e-05 | 3.7252903e-07 |
| 3 | False | 8.467 / 8.800 | True | 3.05175781e-05 | 3.7252903e-07 |

The GPU frontier reads one eight-byte counter per pass, then reads the final
state and provenance buffers once each. In this case it used eight counter
readbacks totalling 64 bytes and two final readbacks totalling 732,096 bytes.
The reference is prepared beforehand; the target and confirmed anchors are
uploaded at the start of the stage. Anchor selection, initial image matching and some intermediate
transfers remain on the host; the complete pipeline is not GPU-resident.
Consult the saved JavaScript Object Notation (JSON) file for per-pass transfer
counts and all output metadata. A real image without truth cannot establish accuracy.

Both paths accepted the same 7,097 windows with identical confirmed anchors,
parents, origins, generations, attempt counts and rejection reasons. Across
the accepted windows, 129 have small floating-point parameter differences:
at most 0.000030518 pixel for displacement components and 0.000000373 for
gradient components. The GPU frontier computes affine transport in 32-bit
floating point, while the Python reference computes it in 64-bit before
submitting states to the solver. The experiment does not establish a general
error bound or accuracy on this experimental pair.

The median process improvement is small compared with run-to-run variation;
the third run was slower with the GPU frontier. These results do not support a
general end-to-end speedup claim. They do show the transfer boundary of the
propagation stage on this pair, subject to the private solver interface and
the limitations above.

Reproduce in a **new** output directory after preparing the image cache as
described in `../p2_solver_images_adrien/README.md`:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_integrated_gpu_frontier.py \
  --inputs results/seed_propagation_20261111/p2_solver_images_adrien/inputs \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json \
  --output-dir results/seed_propagation_20261111/p2_gpu_frontier_adrien_paired_new \
  --external-reference /home/jeff/CNRS/DIC/AdrienVariational/img_000000_0.09422.tiff \
  --external-target /home/jeff/CNRS/DIC/AdrienVariational/img_000075_1608.88508.tiff \
  --roi-mask /home/jeff/CNRS/DIC/AdrienVariational/Mask.tiff --repetitions 3
```
