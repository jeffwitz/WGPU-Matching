# Confirmatory P2 protocol after translation-baseline repair

This protocol was recorded before inspecting results from seed groups 23000000 and 24000000. Each group supplies ten heterogeneous and ten translation image pairs. Texture and deformation seeds are disjoint from the development groups and from the previously examined independent groups 21000000 and 22000000.

The image-only method uses `p2_calibration/frozen_parameters.json`: sparse stride 2, displacement consistency 8 px, gradient consistency 0.25 and three confirmed anchors per region. The three image-ranked rotation hypotheses, local affine fallback, regional escalation and matched re-solve control use the existing scripts. The analytical field is never available to the selection or propagation loops.

The comparator uses affine plus local normalized cross-correlation (NCC), with `--retain-translation-affine` so that a valid near-identity affine retains its translation when the neural residual is disabled. This opt-in repair was implemented after examining the earlier translation baseline, so the earlier test groups remain exploratory. The new test groups are not used to tune the correction or propagation settings.

G2 numerical criteria remain the thresholds in `CdC_Codex.md`. The complete qualification also needs full end-to-end timing and systematic development-set validation of the local-affine fallback; if either remains unavailable, the report must mark G2 incomplete even if its numerical criteria pass. Interpolated values are reported separately from windows accepted by the inverse-compositional Gauss–Newton (IC-GN) solver.
