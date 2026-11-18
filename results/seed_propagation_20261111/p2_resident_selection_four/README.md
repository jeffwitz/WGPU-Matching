# Resident selection through GPU anchor confirmation

The four previously generated native 3840 × 2160 synthetic pairs use GPU sparse-hypothesis selection, anchor confirmation and frontier propagation. The selected state is copied within the same GPU device; the image is uploaded once for these stages. The hypothesis counts and frontier termination counters are read by the host. The preceding image matching and input preparation remain host-mediated.

Compared to the preceding GPU path with selection readback, all final parameters, accepted masks, confirmed anchors, parents, origins, generations and attempts are bitwise identical on all four pairs. Each path yields 4,695 correctly converged visible windows out of 4,900 with zero false accepted visible windows. Compared to the original serial host reference, final floating-point parameters differ slightly as described in `../p2_gpu_anchor_frontier_four/README.md`, while acceptance and provenance match. Synthetic truth is used only for the final score.

The selected-field readback shrinks from 63,376 bytes to 16 bytes per selection for these three-hypothesis cases. Results are limited to four cases and do not establish general robustness or whole-pipeline GPU residency.

To reproduce, follow `../p2_gpu_integrated_four_v2/README.md`, choose a new output directory and add `--gpu-anchor-frontier --gpu-hypothesis-selection --gpu-resident-selection`.
