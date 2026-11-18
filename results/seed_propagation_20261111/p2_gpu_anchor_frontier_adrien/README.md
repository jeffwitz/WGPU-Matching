# Host versus graphics processing unit (GPU) frontier in the initialization pilot

One image pair, one warm-up and alternating-order measured repetitions. The two paths use identical cached solver images and the same anchor-confirmation rules. The reference batches the solves and replays decisions on the host; the experimental path performs both operations on the GPU. Host times include process startup, image-cache verification, device/reference setup and result writes. Input hypotheses and cached images were prepared earlier; these are not complete end-to-end timings. No GPU timestamp measurement was available.

GPU anchor confirmation enabled: True.

Median verification and propagation: host 0.773 s; GPU frontier 0.508 s. Median process: 8.476 / 7.901 s.

| Run | Warm-up | Process host / GPU (s) | Metadata exact | Max displacement difference (px) | Max gradient difference |
|---:|---|---:|---|---:|---:|
| 0 | True | 8.829 / 8.736 | True | 3.05175781e-05 | 5.36441803e-07 |
| 1 | False | 8.300 / 7.920 | True | 3.05175781e-05 | 5.36441803e-07 |
| 2 | False | 8.652 / 7.881 | True | 3.05175781e-05 | 5.36441803e-07 |

The GPU frontier reads one small counter per pass, then reads the final state and provenance buffers once each. Anchor selection, initial image matching and some intermediate transfers remain on the host; the complete pipeline is not GPU-resident. Consult the saved JavaScript Object Notation (JSON) file for per-pass transfer counts and all output metadata. A real image without truth cannot establish accuracy.

The images are the native 7920 × 6004 Adrien pair; the hypotheses and input cache were prepared before this measurement. Both paths confirmed 6,321 seeds and accepted 7,097 windows, with identical masks and provenance. Across accepted windows the maximum difference is 0.000030518 pixel in displacement and 0.000000537 in a displacement gradient. There is no independent ground truth for this real pair. Two measured repetitions are insufficient for a general performance claim.

After sparse candidate selection on the host, the GPU path uploads candidate states and the target, executes eight directional batches of IC-GN and replays their confirmation on the GPU; it then enters GPU frontier propagation without an intermediate state readback. Eight eight-byte counter readbacks control frontier termination; two final buffer readbacks transfer 732,096 bytes. Global image matching, sparse hypothesis selection and final output remain host-mediated. The GPU adapter uses private live-dic interfaces and currently requires affine motion, linear image interpolation, disabled outlier filtering and no partial-subset solver.

Reproduce with the command documented in `../p2_gpu_frontier_adrien_paired/README.md`, a new output directory, and the additional option `--gpu-anchor-frontier`.
