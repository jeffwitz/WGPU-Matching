# Initial P3 robustness protocol

This protocol was fixed before inspecting groups 25000000 and 26000000. Each group has five synthetic pairs in each of the rigid, affine and discontinuous families. The frozen P2 seed-selection settings, solver settings, image-derived hypothesis ranking and affine-retention baseline are unchanged. Interpolated windows do not count as solver-converged windows.

The false-acceptance criterion is at most 0.5% of accepted windows per family, with analytical truth used only after solver selection and propagation. All image pairs and per-window outputs remain available for audit. Discontinuity results will distinguish both sides using the generator's known central material boundary, for scoring only.

This is only the first part of P3. Noise and illumination perturbations, blur, repetitive patterns, 32 px windows, traction series and experimental images require additional experiments. The G3 gate cannot be considered complete solely from this initial set.
