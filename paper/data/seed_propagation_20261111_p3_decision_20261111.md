# P3 decision: propagation-first with explicit limits

## Decision

Use affine preconditioning plus local normalized cross-correlation, followed by accepted 64-pixel IC-GN seeds and first-order neighbor propagation, as the baseline for ordinary random-speckle images. Keep the 10-orientation search as the coverage reference. Do not use the NDF residual in this baseline. Keep periodic patterns out of the requested application; their negative results remain archived. Never fill a rejected window and report it as an IC-GN result without an actual accepted solve.

The opt-in same-centre 32→16 retry can recover unresolved windows, using accepted nonsuspect 64-pixel states. Each output retains its actual subset size and provenance. It does not produce a dense fine grid, and a 16-pixel result is not a 64-pixel measurement. The image-residual-based suspicion rule is a research option only, not a default or a guarantee of correctness. A weak score identifies an ambiguous measurement; do not automatically label it wrong or interpolate across an unknown material boundary. The current threshold overflags correct states in high shear and remains imperfect on accepted wrong states.

## Gate and evidence

**G3 does not pass over the entire specified envelope.** The frozen severe-shear qualification at peak gradient 2.0 has 57 visible false acceptances among 4,080 accepted (1.40%, threshold 0.5%) using corrected exact-center truth; all 57 were also incorrect with analytical local-affine initialization. Moderate shear passed those few tested draws. The experiment does not establish a universal maximum supported gradient. See [exact-center analysis](p3_shear_band_frozen_protocol/exact_center_analysis/README.md).

On five additional heterogeneous and five extreme-shear pairs, the separately frozen residual rule detected 119/120 false accepted coarse results but also flagged 221/6,979 correct coarse states. The guarded 32→16 retry returned 7,070 correct and 1 false out of 8,075 visible shear centers, leaving 199 previously correct flagged centers unresolved. The same-center experiment preserves the original grid and does not overturn G3: [report](p3_adaptive_residual_holdout_20261111/RAPPORT.md).

Discontinuities without a supplied material mask can contaminate supports and interpolation. The Adrien pair provides accepted counts and provenance, not measured physical accuracy. Different family examples, strengths and cameras require independent validation. A bounded loss of coverage is preferable to silently reporting an unsupported estimate as converged.

## Next work

Continue to P4: provide a public, versioned seeded solve and expose final image correlation with explicit validity, retaining defaults; add affine transport as an opt-in propagation behavior and tests in the solver repository. Then pursue the already-authorized P5 resident-GPU integration and timings. P3 is closed as a decision and a recorded failure, not declared scientifically passing. No further threshold tuning, new network training, or repeated P3 parameter sweeps are required to start P4.
