# GPU anchor confirmation and frontier: four synthetic pairs

The same four 3840 × 2160 synthetic pairs and solver settings used in `../p2_gpu_integrated_four_v2/README.md` were run with GPU anchor confirmation followed directly by the GPU frontier. The serial host reference and the new path both obtain **4,695 correct visible windows out of 4,900**, with zero false accepted visible windows. Acceptance masks, confirmed anchors, propagation parents, origins, generations and correctness agree exactly on all four pairs.

Parameters are not bitwise identical: among accepted windows the largest observed absolute component difference is 0.0001221 pixel for displacement and 0.000001312 for displacement gradients. This reflects different floating-point paths and is an observed difference, not a guaranteed error bound. `comparison.json` contains the per-pair comparison, including the nonidentical parameter arrays. This is a four-pair diagnostic, not a full scientific qualification.

The algorithm does not consult synthetic truth when selecting or confirming anchors. The global feature inference and initial sparse hypothesis ranking remain CPU-mediated. GPU-to-host transfers remain for convergence counters and final outputs, so the whole pipeline is not GPU-resident.

To reproduce, follow the command in `../p2_gpu_integrated_four_v2/README.md` with a new output directory and add `--gpu-anchor-frontier` to the pilot invocation.
