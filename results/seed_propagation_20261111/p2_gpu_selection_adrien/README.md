# Host versus graphics processing unit (GPU) frontier in the initialization pilot

One image pair, one warm-up and alternating-order measured repetitions. The two paths use identical cached solver images and the same confirmation rules. The reference uses host selection and replay; the experimental path uses GPU selection, confirmation and propagation. Host times include process startup, image-cache verification, device/reference setup and result writes. Input hypotheses and cached images were prepared earlier; these are not complete end-to-end timings. No GPU timestamp measurement was available.

GPU anchor confirmation enabled: True; GPU hypothesis selection enabled: True.

Median verification and propagation: host 0.936 s; GPU frontier 0.691 s. Median process: 8.963 / 9.081 s.

| Run | Warm-up | Process host / GPU (s) | Metadata exact | Max displacement difference (px) | Max gradient difference |
|---:|---|---:|---|---:|---:|
| 0 | True | 9.411 / 8.792 | True | 3.05175781e-05 | 5.36441803e-07 |
| 1 | False | 8.963 / 8.566 | True | 3.05175781e-05 | 5.36441803e-07 |
| 2 | False | 8.857 / 10.791 | True | 3.05175781e-05 | 5.36441803e-07 |
| 3 | False | 9.241 / 9.081 | True | 3.05175781e-05 | 5.36441803e-07 |

The GPU frontier reads one small counter per pass, then reads the final state and provenance buffers once each. Initial image matching and intermediate transfers remain host-mediated; the complete pipeline is not GPU-resident. Consult the saved JavaScript Object Notation (JSON) file for per-pass transfer counts and all output metadata. A real image without truth cannot establish accuracy.

The native 7920 × 6004 Adrien images are used with a 192 × 146 model raster. Both paths select `affine_ncc`: the accepted sparse counts are 9 for zero, 515 for affine, and 1,519 for affine with normalized cross-correlation. They confirm the same 6,321 seeds and accept the same 7,097 windows, with identical acceptance and provenance. Maximum observed differences are 0.000030518 pixel in displacement and 0.000000537 in gradient. No accuracy can be inferred for these real images without ground truth.

The selector reads 366,064 bytes per selection: a full selected window field plus four integer counts. It runs the three solver trials and ranking on WebGPU, but the selected field is still read back to the host and uploaded again to confirm anchors; the target image is also uploaded again. The timing reported for selection in the raw data excludes pipeline compilation and target upload. The GPU propagation then reads eight 8-byte counters and 732,096 bytes of final output. Thus the whole pipeline is not resident on the GPU.

This is a paired comparison after one warm-up and only three measured repetitions. Full process medians are 8.963 s host and 9.081 s GPU, with one GPU run at 10.791 s; there is no demonstrated overall speedup. Input matching, source-image preparation and hypothesis generation were performed in advance.

To reproduce, follow the command in `../p2_gpu_frontier_adrien_paired/README.md`, choose a new output directory, and add `--gpu-anchor-frontier --gpu-hypothesis-selection`.
