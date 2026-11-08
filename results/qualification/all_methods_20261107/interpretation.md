# Scientific reading of the refreshed comparison

This interpretation concerns the current complete field-initialization protocol. It does not replace the per-case tables or the archived observations. [Return to the measured report](index.md).

## What the new detectors contribute

At 60% homogeneous traction, the explicit affine fields from DaD + DeDoDe-B, DeDoDe v2-L + DeDoDe-B and learned DIC + DeDoDe-B recover all 16 visible centres within 2 pixels. Their conditional RMSE values are 0.435, 0.486 and 0.224 pixels. These are reconstructed field initializers, not 16 independent direct correspondences or measured IC-GN convergences.

The shared-descriptor path is still dominated by CPU description. At 60%, median pair-description times are 7.169 seconds for DaD, 7.014 for DeDoDe v2 and 6.884 for learned DIC. Their complete affine-call medians are 8.066, 8.474 and 7.523 seconds. Detector-pair medians are 0.449, 0.992 and 0.140 seconds. These nested stage medians explain the bottleneck; they must not be summed and substituted for the independently measured whole-call median.

The tensor baseline is distinct from the learned score model: it recovers 16/16 centres below 5 pixels but 14/16 at most 2 pixels. It must not be included in the claim that all new learned-detector paths recover 16/16 at most 2 pixels.

The [complete 60% gallery and per-seed records](case_tensile_60pct.md) retain the measured inputs, errors and stage times.

## Speed and repairable accuracy are different trade-offs

At 60%, XFeat's affine field recovers 16/16 below 5 pixels in a median 0.629 seconds, but 0/16 at most 2 pixels; RMSE is 3.048 pixels. EfficientLoFTR gives 16/16 below 5, 10/16 at most 2 and 1.948-pixel RMSE in 0.863 seconds. RaCo-ALIKED-LightGlue+ gives 16/16 at most 2, 0.706-pixel RMSE and 1.140 seconds. These observations favour different choices depending on the downstream capture range; none measures the actual convergence basin of IC-GN.

DeepMatching gives the smallest measured affine-field RMSE here, 0.177 pixels and 16/16 at most 2 pixels, with a 17.128-second median. This run deliberately uses a 1024-pixel search radius and the complete hierarchy, rather than the historical default-radius or sparse-root workload. Its robust weight update narrowly misses the numerical convergence threshold: final maximum weight change is 0.001020 against 0.001 after 20 iterations. This is recorded as non-convergence; the threshold was not relaxed to turn it green. Numerical convergence and analytic centre accuracy are separate observations.

HORG and CPM provide no accepted projected field at 60% in the primary protocol. HORG/SIFT + DeDoDe-B returns a field but only 5/16 centres below 5 pixels, with 9.135-pixel RMSE. These are failed accuracy rows, and their images remain in the gallery. Diagnostic fits from rejected proposals never enter the usable count.

## The affine prior explains much of the gain

The synthetic target is globally affine. Once enough correct point pairs exist, a global affine fit can initialize centres that had no accepted local estimate. The new coverage must not be presented as an improvement in the underlying descriptor or as independent observation of each centre. In particular, the current CPM affine projection recovers all centres at 20% but only 3/24 below 5 pixels at 30%, and none at 60%. This is different from the historical local-seed result and is not a claim of general large-strain CPM qualification.

The same local verification can penalize sparse pipelines: a model needs at least eight candidates within the fixed radius. HORG has fewer features than the common 8192-point detector paths. The controlled raw-pair ablation therefore retains native acceptance and removes only the added common filter. Read the [raw-pair ablation](../raw_projection_ablation_20261107/index.md) before assigning a failure to the detector itself.

## What the controlled ablation actually shows

Across the 99 analytic case/method pairs, adding the common local geometric verification improves the correct-below-5-pixel count in 31 cases, leaves it unchanged in 68, and reduces it in none. At 60%, raw robust projection alone passes only RaCo-ALIKED-LightGlue+; the verified protocol passes eight pipelines. In particular, DeepMatching's raw affine field has 325.402-pixel RMSE, DaD's 384.057 and learned DIC's 358.521, despite their good verified fields. The observations are identical, and no analytic error selects a pair. A low-frequency basis and IRLS alone do not remove a dominant wrong correspondence population.

The sparse-support caveat above remains a limitation, but it is not a demonstrated coverage penalty in this campaign: removing the added filter does not rescue HORG at 10% or 60%. The original native local-seed pipeline is a different initializer and retains its historical qualification. These findings support verification before the shared projection; they do not prove that a geometrically coherent false cluster could never pass.

The [ablation artifact audit](../raw_projection_ablation_20261107/validation.json) checks 132 field records and 176 animations against the cached observations and main protocol fingerprint.

## Automatic radial bases are not an accuracy oracle

Across all 99 analytic case/method pairs, the automatic affine/Gaussian-RBF request improves the number of centres below 5 pixels in zero cases, leaves it unchanged in 97, and reduces it in two. CPM at 30% changes from 3/24 to 0/24; XFeat at 40% changes from 23/24 to 21/24. These are measured negative results, retained without changing the selection rule. Some Gaussian candidates are selected even though the physical field is affine: cross-validation predicts observed correspondence residuals, which can include descriptor bias. It does not observe physical truth.

The explicit affine basis is therefore the better-supported starting point for this homogeneous tensile series. This conclusion cannot determine the right basis for a heterogeneous experimental field without further evidence.

## A 60% pass does not prove a monotone deformation limit

Visible populations change with strain because centres leaving the target are excluded, as required. At 40%, XFeat has one false estimate among 24 requests, while at 60% its 16 remaining visible requests all meet the 5-pixel criterion. The latter population is smaller and more central. A successful 60% row therefore does not imply success at every lesser strain, nor does a smaller raw correct count prove declining coverage. Use the entire coverage matrix and the exported centre coordinates.

## What Adrien and the animations can establish

The experimental pair has no analytic displacement field. Its masked correspondence counts, residuals, valid-sampling fractions and reference/corrected animations support visual examination, but not a physical-error or pass-rate claim. A smooth field is extrapolated outside observed support in whole-image displays. The fixed native-pixel crops make local drift visible without selecting a favourable region.

The synthetic corrections also have legitimate black side bands where material left the target field of view. The unchanged reference viewport is retained rather than filling those bands with an identity warp. Black borders alone do not imply an inaccurate initializer.

The next qualification layer remains image-based convergence on heterogeneous motion, with independent truth where possible. No actual IC-GN solve, crack handling, or calibrated confidence probability is claimed by this campaign.
