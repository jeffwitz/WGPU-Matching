# Translation comparison added after the baseline audit

The planned affine plus local normalized cross-correlation (NCC) baseline rejected all 20 near-identity affine prefits and yielded no correct windows when its neural residual was disabled. This diagnostic uses the existing neural deformation field (NDF) residual plus local NCC, without changing the affine-prefit threshold or the inverse-compositional Gauss–Newton (IC-GN) solver settings. It was added after observing the baseline failure and is not the preregistered G2 comparator.

On the same 20 independent translation pairs, image-only seed selection and propagation give 28,698 correct windows out of 29,706 visible, versus 23,875 for the existing NDF-residual plus NCC diagnostic. Propagation obtains more correct windows on all 20 pairs. The image-only qualification reports no accepted visible window with at least 2 px error. The diagnostic does not repair the planned translation baseline or establish a full G2 verdict.

The two `ndf_seed*.json` files and corresponding per-window NPZ archives record the diagnostic. The new method's matching per-pair counts are in `../p2_independent/evaluation/summary.json`.
