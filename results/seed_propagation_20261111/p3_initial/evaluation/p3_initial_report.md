# Initial P3 qualification: fixed-parameter motion families

The inverse-compositional Gauss–Newton (IC-GN) solver evaluates image-derived seeds and propagation. Analytic displacement and visibility are used only for scoring. False acceptance means a visible accepted centre with error at least 2 px. Interpolated values are not counted as solver successes.

| Family | Correct / visible | Baseline correct | False / accepted | Pairs below baseline |
|---|---:|---:|---:|---:|
| rigid | 12821/13301 | 11682 | 0/12821 | 0 |
| affine | 10963/12305 | 10077 | 0/10963 | 0 |
| discontinuous | 13450/13941 | 11067 | 0/13450 | 0 |

Discontinuities: 10/10 pairs have confirmed seeds on each side. 0 accepted visible windows have support crossing the material boundary. In this generator the boundary aligns with the 64 px grid; no window support crosses it, so crossing-window reliability was not tested. Centre error alone does not validate full support.


This initial campaign alone does not qualify G3. Subsequent perturbation, traction, fine-window and experimental-image results, together with remaining limitations, are indexed in `../../p3_status.md`. The JSON contains per-pair counts, including discontinuity sides; source and propagation maps are in the NPZ files.
