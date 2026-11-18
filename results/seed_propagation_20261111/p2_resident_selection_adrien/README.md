# Host versus graphics processing unit (GPU) frontier in the initialization pilot

One image pair, one warm-up and alternating-order measured repetitions. The two paths use identical cached solver images and batched anchor confirmation. Host times include process startup, image-cache verification, device/reference setup and result writes. Input hypotheses and cached images were prepared earlier; these are not complete end-to-end timings. No GPU timestamp measurement was available.

GPU anchor confirmation enabled: True; GPU hypothesis selection enabled: True; resident selection enabled: True.

In the tables, host denotes the previous GPU path with host readback; GPU denotes resident selection.

Median verification and propagation: host 0.692 s; GPU frontier 0.704 s. Median process: 8.802 / 8.575 s.

| Run | Warm-up | Process host / GPU (s) | Metadata exact | Max displacement difference (px) | Max gradient difference |
|---:|---|---:|---|---:|---:|
| 0 | True | 8.444 / 8.576 | True | 0 | 0 |
| 1 | False | 8.802 / 8.575 | True | 0 | 0 |
| 2 | False | 8.767 / 8.972 | True | 0 | 0 |
| 3 | False | 9.104 / 8.380 | True | 0 | 0 |

The GPU frontier reads one small counter per pass, then reads the final state and provenance buffers once each. Initial image matching and intermediate transfers remain host-mediated; the complete pipeline is not GPU-resident. Consult the saved JavaScript Object Notation (JSON) file for per-pass transfer counts and all output metadata. A real image without truth cannot establish accuracy.

The selected field and target remain on the solver device through anchor confirmation.

The input is the native 7920 × 6004 Adrien pair with a 192 × 146 coarse raster. Both paths select `affine_ncc`, confirm 6,321 seeds, and accept 7,097 windows. Their selected results, final parameters, masks and provenance are bitwise identical in all four runs. No ground truth is available for this experimental pair.

For three hypotheses, intermediate selection readback decreases from 366,064 bytes to 16 bytes. The resident path also avoids uploading the selected field and target a second time. It still reads 16 bytes of hypothesis counts to the host, an eight-byte termination counter on each of eight passes, and 732,096 bytes of final state and provenance. Hypotheses and the image are initially supplied by the host; the earlier image matching is not resident on this device.

The verification-stage median is 0.692 s before and 0.704 s after, so this experiment does not establish a speed improvement. Process time includes startup and data verification but excludes the earlier generation of hypotheses and cached images. Three measured repetitions on a single pair are insufficient for a general timing claim.

Reproduce with the command in `../p2_gpu_frontier_adrien_paired/README.md`, a new output directory, and options `--gpu-anchor-frontier --gpu-hypothesis-selection --gpu-resident-selection --compare-nonresident`.
