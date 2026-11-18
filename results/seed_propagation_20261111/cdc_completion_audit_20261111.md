# Completion audit: speckle initialization for IC-GN

The user has excluded periodic textures from the intended application. The prior
periodic-texture experiment remains archived; it is not a criterion for
accepting or rejecting speckle initialization. No measurement was changed.

| Work package | Evidence | Status and limits |
|---|---|---|
| P0: archived anchor audit | `p0_anchor_availability.md` | Reproduced; the exact median 84.5 is reported instead of the rounded 84 in the original table. |
| P1: oracle-seeded propagation | `p1_oracle/README.md` | G1 passes on all five draws; the seeds use analytical truth, only in this explicitly oracle-labelled experiment. |
| P2: image-only initialization | `p2_status.md`, `p2_confirmatory/evaluation/README.md` | The numerical criteria written for G2 pass on the new 23M/24M groups with the corrected translation comparator. The comparator was repaired after inspecting 21M/22M; their corrected translation result is post hoc. The later confirmatory protocol additionally required end-to-end timing, which is still unavailable. |
| P3: speckle robustness and real images | `p3_status.md` | Synthetic rigid, affine, opening, traction, noise, illumination and blur campaigns, fine-window tests and Adrien counts are available. A single complete verdict for the intended operating envelope is not established: unresolved crack-support validity and additional strong-shear qualification remain. |
| P4: public solver API | `CdC_Codex.md`, section 5 | Not implemented: seeded six-parameter public API, per-window final correlation and affine propagation in the external solver remain to be developed and tested in that repository. |
| P5: GPU-resident propagation and timing | `CdC_Codex.md`, section 5 | Not implemented: the current propagation orchestration is NumPy/host-side and calls the existing GPU solver through a private adapter. Full production timing and GPU-residency/parity are not demonstrated. |

The present image-only pilot tests zero, the three rotation hypotheses, the
global affine and affine plus local correlation, verifies sparse seeds, and
supports regional escalation. Its local-affine fallback is only run when the
initial region-anchor requirement fails; the global hypotheses have already
been generated. Previously, the local fallback started a fresh initialization;
the resumed local-affine implementation and a development regression are now
reported in `p2_resume_escalation_17051007_v2/README.md`. An opt-in deferred
path now generates the three forced rotations only when verified anchors are
insufficient: `p2_lazy_rotation_all_forty/README.md`. It reproduced all 52,211
correct windows on the 40 already examined confirmation pairs without producing
additional rotation exports, and recovered the previously seedless development
pair with its fallback. This does not defer generation of the initial global
affine/NCC inputs or constitute independent or end-to-end performance qualification.
The complete ordered, on-demand escalation of expensive sources described in
section 3.1 of the CdC therefore remains unfinished. Provenance of
accepted initial, propagated and escalation windows is saved for the stages
that run.

Local affine fits are now restricted to unresolved valid centres. The optional
`--escalate-unresolved` also retries sizable unresolved components when a
region already has enough confirmed anchors; it is not the default. In the
post-hoc `p2_unresolved_affine_five_pairs/README.md` comparison this preserves
all existing accepted states and adds no correct windows on five previously
inspected pairs. Four were already at their analytical-affine oracle count;
the fifth was one window below it. This is not evidence of a performance or
coverage improvement, and it does not replace independent qualification.

On the original CdC G2 conditions, the fresh-group result is 23,570/24,641
heterogeneous visible windows correct, an image-pair bootstrap gain of +23.65
percentage points [14.78, 35.19], 0/20 heterogeneous pairs below 20% coverage,
0/23,570 false accepted visible windows, and no translation pair regression.
These figures satisfy those numerical conditions. The earlier statement that
the CdC itself made full runtime a G2 condition was incorrect: that condition
was introduced in the separate confirmatory protocol.

Strong-shear results of 59/6,142 and 71/5,908 false accepted windows in the
archived `../ndf_dic_20261109/frontier_generalization_20261109/README.md`
concern an earlier configuration. They motivate a paired experiment with the
current pipeline; they do not by themselves establish its G3 result. Centre
accuracy also cannot validate a window whose material support crosses a crack.

The original CdC reserves P4 and P5 for an explicit user decision after P1/P2;
no changes have been made to the external solver under this audit. No neural
network training has been started. The current implementation is an
experimental initializer, not yet the final resident-GPU production chain.
