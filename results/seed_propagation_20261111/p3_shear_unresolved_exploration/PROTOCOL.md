# Exploratory unresolved-region reseeding after shear-band inspection

This is explicitly post-hoc: pairs 61051005/61091005 and 61051007/61091007 were selected because their observed coverage leaves one side of the band unresolved. This exploration cannot validate a gate or establish an independent effect.

Generate the existing three highest-ranked coarse-rotation hypotheses using images only and the frozen settings, then rerun the existing experimental `--escalate-unresolved` option on these pairs with the original IC-GN settings, no analytical truth, visibility, or band position supplied to hypothesis selection, propagation, or region choice. Compare exact-center scoring, accepted masks, per-window source/stage and false acceptances against the frozen baseline. Use the same hard-match local-affine machinery and do not adjust its thresholds. Report all successes and failures and execution time separately; do not attribute changes solely to one stage if the eager hypothesis set also changed.

After inspecting this exploratory reseeding result, an additional post-hoc diagnostic applies the already used 32-pixel nearest-source re-solve to its accepted outputs at the same original centers. The fine stage receives no truth or band labels. This second step is not a predeclared independent experiment and cannot establish a G3 pass.
