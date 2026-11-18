# GPU sparse-hypothesis selection: four synthetic pairs

Four previously generated native 3840 × 2160 synthetic pairs were run with GPU sparse-hypothesis selection, GPU anchor confirmation and GPU frontier propagation. The serial host reference and this experimental path both produce **4,695 correct visible windows out of 4,900**, with zero false accepted visible windows. Acceptance masks, confirmed anchors, parents, origins, generations and correctness agree on all four pairs. Parameters are not bitwise identical; see `comparison.json` and the preceding `../p2_gpu_anchor_frontier_four/README.md` for the floating-point differences. Analytical truth is used only to score the final output.

All four select the affine hypothesis with normalized cross-correlation. Each selection reads 63,376 bytes to the host and dispatches three solver trials. This stage still reuploads the selected field for anchor confirmation and does not include initial global matching. No full GPU-resident or general-performance claim follows.

To reproduce, follow the command in `../p2_gpu_integrated_four_v2/README.md`, choose a new output directory and add `--gpu-anchor-frontier --gpu-hypothesis-selection`.
