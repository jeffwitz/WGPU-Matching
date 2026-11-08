# Homogeneous traction: 40 percent: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `tensile_40pct`.

Prescribed stretch factors: x = 1.4003732246621525, y = 0.8455061970313845. Axial engineering strain is 40.037322 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.29 / 0.45 / 0.51 | 9964 / 19260 / 30720 | False |
| DeepMatching INRIA hierarchy | gaussian_rbf → gaussian_rbf | Pass | 24 / 24 | 24 | 24 | 0.29 / 0.42 / 0.43 | 9964 / 19260 / 30720 | False |
| HORG | affine → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 32 / 693 | — |
| HORG | gaussian_rbf → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 32 / 693 | — |
| CPM-HOG | affine → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 55 / 32026 | — |
| CPM-HOG | gaussian_rbf → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 55 / 32026 | — |
| XFeat | affine → affine | Fail | 24 / 24 | 0 | 23 | 3.49 / 4.94 / 5.49 | 290 / 1135 / 1135 | True |
| XFeat | gaussian_rbf → gaussian_rbf | Fail | 24 / 24 | 0 | 21 | 3.68 / 5.54 / 6.92 | 290 / 1135 / 1135 | False |
| EfficientLoFTR | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.72 / 0.98 / 1.06 | 877 / 1989 / 1992 | True |
| EfficientLoFTR | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.72 / 0.98 / 1.06 | 877 / 1989 / 1992 | True |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.34 / 0.56 / 0.64 | 247 / 471 / 471 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.34 / 0.56 / 0.64 | 247 / 471 / 471 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.20 / 0.34 / 0.37 | 1632 / 4827 / 4827 | False |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.20 / 0.34 / 0.37 | 1632 / 4827 / 4827 | False |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.43 / 0.59 / 0.59 | 967 / 3841 / 3841 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.43 / 0.59 / 0.59 | 967 / 3841 / 3841 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.99 / 1.18 / 1.26 | 475 / 3658 / 3658 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.99 / 1.18 / 1.26 | 475 / 3658 / 3658 | True |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.10 / 0.16 / 0.19 | 2336 / 4881 / 4881 | False |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.10 / 0.16 / 0.19 | 2336 / 4881 / 4881 | False |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 24 / 24 | 19 | 24 | 1.70 / 2.80 / 2.82 | 39 / 335 / 335 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 19 | 24 | 1.70 / 2.80 / 2.82 | 39 / 335 / 335 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.255 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.4371 / 17.1927 / 17.9947 |
| geometric_verification_s | 0.1099 / 0.0720 / 0.1146 |
| matching_pipeline_s | 17.3559 / 17.1124 / 17.9166 |
| field_fit_s | 0.0759 / 0.0737 / 0.0760 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.001369.

Projection capacity selection: {'input': 9964, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_40pct, DeepMatching INRIA hierarchy, affine: reference and corrected target](tensile_40pct/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/deepmatching/affine/image1_ref.png) · [Target PNG](tensile_40pct/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/deepmatching/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.288 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.4371 / 17.1927 / 17.9947 |
| geometric_verification_s | 0.1099 / 0.0720 / 0.1146 |
| matching_pipeline_s | 17.3559 / 17.1124 / 17.9166 |
| field_fit_s | 0.0759 / 0.0737 / 0.0760 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.406167039065622, 'patch_projection_s': 0.002434975001960993, 'fit_and_patch_s': 2.4086020899703726}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.049375.

Projection capacity selection: {'input': 9964, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_40pct, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](tensile_40pct/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 8.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0860 / 0.0845 / 0.0902 |
| geometric_verification_s | 0.0040 / 0.0036 / 0.0044 |
| matching_pipeline_s | 0.0857 / 0.0842 / 0.0899 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_40pct, HORG, affine: reference and corrected target](tensile_40pct/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 62.00/155.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/horg/affine/image1_ref.png) · [Target PNG](tensile_40pct/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/horg/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_40pct/horg/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0860 / 0.0845 / 0.0902 |
| geometric_verification_s | 0.0040 / 0.0036 / 0.0044 |
| matching_pipeline_s | 0.0857 / 0.0842 / 0.0899 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_40pct, HORG, gaussian_rbf: reference and corrected target](tensile_40pct/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 62.00/155.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/horg/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/horg/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_40pct/horg/gaussian_rbf/rejected_diagnostic_model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.4071 / 0.3857 / 0.4358 |
| geometric_verification_s | 0.0059 / 0.0058 / 0.0060 |
| matching_pipeline_s | 0.4028 / 0.3813 / 0.4316 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_40pct, CPM-HOG, affine: reference and corrected target](tensile_40pct/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/cpm/affine/image1_ref.png) · [Target PNG](tensile_40pct/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/cpm/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_40pct/cpm/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.4071 / 0.3857 / 0.4358 |
| geometric_verification_s | 0.0059 / 0.0058 / 0.0060 |
| matching_pipeline_s | 0.4028 / 0.3813 / 0.4316 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_40pct, CPM-HOG, gaussian_rbf: reference and corrected target](tensile_40pct/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/cpm/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_40pct/cpm/gaussian_rbf/rejected_diagnostic_model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Fail.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 0.958. Conditional precision (<5 px / estimated): 0.958. False estimates at ≥5 px: 1. Median analytic error: 3.235 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6350 / 0.6349 / 0.6388 |
| geometric_verification_s | 0.0171 / 0.0132 / 0.0180 |
| matching_pipeline_s | 0.6157 / 0.6085 / 0.6160 |
| field_fit_s | 0.0177 / 0.0140 / 0.0182 |
| patch_projection_s | 0.0013 / 0.0012 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000055.

Projection capacity selection: {'input': 290, 'selected': 290, 'applied': False}.

![tensile_40pct, XFeat, affine: reference and corrected target](tensile_40pct/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.713. Preview absolute intensity difference median/P95: 21.00/101.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/xfeat/affine/image1_ref.png) · [Target PNG](tensile_40pct/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/xfeat/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Fail.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 0.875. Conditional precision (<5 px / estimated): 0.875. False estimates at ≥5 px: 3. Median analytic error: 3.167 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6350 / 0.6349 / 0.6388 |
| geometric_verification_s | 0.0171 / 0.0132 / 0.0180 |
| matching_pipeline_s | 0.6157 / 0.6085 / 0.6160 |
| field_fit_s | 0.0177 / 0.0140 / 0.0182 |
| patch_projection_s | 0.0013 / 0.0012 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.27528182696551085, 'patch_projection_s': 0.0015479520661756396, 'fit_and_patch_s': 0.2768298409646377}.

Selection reason: gpu_solver_fallback_to_next_spatial_cv_candidate. IRLS converged: False. Final weight change: 0.002410.

Projection capacity selection: {'input': 290, 'selected': 290, 'applied': False}.

![tensile_40pct, XFeat, gaussian_rbf: reference and corrected target](tensile_40pct/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.713. Preview absolute intensity difference median/P95: 22.00/102.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.671 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.8808 / 0.8745 / 0.8899 |
| geometric_verification_s | 0.0152 / 0.0147 / 0.0157 |
| matching_pipeline_s | 0.8664 / 0.8608 / 0.8766 |
| field_fit_s | 0.0127 / 0.0125 / 0.0134 |
| patch_projection_s | 0.0009 / 0.0009 / 0.0009 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000086.

Projection capacity selection: {'input': 877, 'selected': 877, 'applied': False}.

![tensile_40pct, EfficientLoFTR, affine: reference and corrected target](tensile_40pct/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/38.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/efficientloftr/affine/image1_ref.png) · [Target PNG](tensile_40pct/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/efficientloftr/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.671 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.8808 / 0.8745 / 0.8899 |
| geometric_verification_s | 0.0152 / 0.0147 / 0.0157 |
| matching_pipeline_s | 0.8664 / 0.8608 / 0.8766 |
| field_fit_s | 0.0127 / 0.0125 / 0.0134 |
| patch_projection_s | 0.0009 / 0.0009 / 0.0009 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.40328103001229465, 'patch_projection_s': 0.0012696939520537853, 'fit_and_patch_s': 0.404550833045505}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000086.

Projection capacity selection: {'input': 877, 'selected': 877, 'applied': False}.

![tensile_40pct, EfficientLoFTR, gaussian_rbf: reference and corrected target](tensile_40pct/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/38.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.293 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1333 / 1.1307 / 1.1597 |
| geometric_verification_s | 0.0119 / 0.0092 / 0.0161 |
| matching_pipeline_s | 1.1175 / 1.1097 / 1.1438 |
| field_fit_s | 0.0143 / 0.0137 / 0.0195 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0021 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000080.

Projection capacity selection: {'input': 247, 'selected': 247, 'applied': False}.

![tensile_40pct, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](tensile_40pct/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/raco/affine/image1_ref.png) · [Target PNG](tensile_40pct/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/raco/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.293 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1333 / 1.1307 / 1.1597 |
| geometric_verification_s | 0.0119 / 0.0092 / 0.0161 |
| matching_pipeline_s | 1.1175 / 1.1097 / 1.1438 |
| field_fit_s | 0.0143 / 0.0137 / 0.0195 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0021 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.1953480129595846, 'patch_projection_s': 0.001332601998001337, 'fit_and_patch_s': 0.19668072392232716}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000080.

Projection capacity selection: {'input': 247, 'selected': 247, 'applied': False}.

![tensile_40pct, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](tensile_40pct/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/raco/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.165 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.9744 / 7.9685 / 8.4539 |
| description_pair_s | 7.2055 / 6.9657 / 7.3391 |
| detection_pair_s | 0.4375 / 0.4356 / 0.4518 |
| geometric_verification_s | 0.0431 / 0.0404 / 0.0488 |
| matching_pipeline_s | 7.9449 / 7.9305 / 8.4104 |
| mutual_matching_s | 0.5020 / 0.2432 / 0.5695 |
| field_fit_s | 0.0295 / 0.0271 / 0.0399 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0015 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000412.

Projection capacity selection: {'input': 1632, 'selected': 1632, 'applied': False}.

![tensile_40pct, DaD + DeDoDe-B, affine: reference and corrected target](tensile_40pct/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dad_b/affine/image1_ref.png) · [Target PNG](tensile_40pct/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dad_b/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.165 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.9744 / 7.9685 / 8.4539 |
| description_pair_s | 7.2055 / 6.9657 / 7.3391 |
| detection_pair_s | 0.4375 / 0.4356 / 0.4518 |
| geometric_verification_s | 0.0431 / 0.0404 / 0.0488 |
| matching_pipeline_s | 7.9449 / 7.9305 / 8.4104 |
| mutual_matching_s | 0.5020 / 0.2432 / 0.5695 |
| field_fit_s | 0.0295 / 0.0271 / 0.0399 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0015 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5473360610194504, 'patch_projection_s': 0.0017619619611650705, 'fit_and_patch_s': 0.5490982030751184}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000412.

Projection capacity selection: {'input': 1632, 'selected': 1632, 'applied': False}.

![tensile_40pct, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_40pct/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.395 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.0846 / 8.0792 / 8.6208 |
| description_pair_s | 6.8110 / 6.7688 / 7.0661 |
| detection_pair_s | 0.9880 / 0.9495 / 1.0140 |
| geometric_verification_s | 0.0420 / 0.0365 / 0.0426 |
| matching_pipeline_s | 8.0518 / 8.0504 / 8.5988 |
| mutual_matching_s | 0.2521 / 0.2465 / 0.4813 |
| field_fit_s | 0.0233 / 0.0176 / 0.0237 |
| patch_projection_s | 0.0020 / 0.0013 / 0.0037 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000049.

Projection capacity selection: {'input': 967, 'selected': 967, 'applied': False}.

![tensile_40pct, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](tensile_40pct/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/35.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dedode_b/affine/image1_ref.png) · [Target PNG](tensile_40pct/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dedode_b/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.395 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.0846 / 8.0792 / 8.6208 |
| description_pair_s | 6.8110 / 6.7688 / 7.0661 |
| detection_pair_s | 0.9880 / 0.9495 / 1.0140 |
| geometric_verification_s | 0.0420 / 0.0365 / 0.0426 |
| matching_pipeline_s | 8.0518 / 8.0504 / 8.5988 |
| mutual_matching_s | 0.2521 / 0.2465 / 0.4813 |
| field_fit_s | 0.0233 / 0.0176 / 0.0237 |
| patch_projection_s | 0.0020 / 0.0013 / 0.0037 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.44253308000043035, 'patch_projection_s': 0.0022522129584103823, 'fit_and_patch_s': 0.4447854249738157}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000049.

Projection capacity selection: {'input': 967, 'selected': 967, 'applied': False}.

![tensile_40pct, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_40pct/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/35.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.939 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.6107 / 7.4605 / 7.8024 |
| description_pair_s | 7.0503 / 6.8686 / 7.2157 |
| detection_pair_s | 0.1309 / 0.1287 / 0.1405 |
| geometric_verification_s | 0.0381 / 0.0380 / 0.0402 |
| matching_pipeline_s | 7.5895 / 7.4318 / 7.7808 |
| mutual_matching_s | 0.3838 / 0.3693 / 0.3952 |
| field_fit_s | 0.0180 / 0.0178 / 0.0207 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000048.

Projection capacity selection: {'input': 475, 'selected': 475, 'applied': False}.

![tensile_40pct, DIC tensor + DeDoDe-B, affine: reference and corrected target](tensile_40pct/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/44.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dic_tensor_b/affine/image1_ref.png) · [Target PNG](tensile_40pct/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.939 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.6107 / 7.4605 / 7.8024 |
| description_pair_s | 7.0503 / 6.8686 / 7.2157 |
| detection_pair_s | 0.1309 / 0.1287 / 0.1405 |
| geometric_verification_s | 0.0381 / 0.0380 / 0.0402 |
| matching_pipeline_s | 7.5895 / 7.4318 / 7.7808 |
| mutual_matching_s | 0.3838 / 0.3693 / 0.3952 |
| field_fit_s | 0.0180 / 0.0178 / 0.0207 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.29178298206534237, 'patch_projection_s': 0.0016406079521402717, 'fit_and_patch_s': 0.2934237120207399}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000048.

Projection capacity selection: {'input': 475, 'selected': 475, 'applied': False}.

![tensile_40pct, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_40pct/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/44.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.078 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5869 / 7.4694 / 7.6304 |
| description_pair_s | 6.8924 / 6.7784 / 6.9983 |
| detection_pair_s | 0.2156 / 0.1364 / 0.2448 |
| geometric_verification_s | 0.0446 / 0.0436 / 0.0463 |
| matching_pipeline_s | 7.5490 / 7.4315 / 7.5759 |
| mutual_matching_s | 0.3912 / 0.3694 / 0.3917 |
| field_fit_s | 0.0349 / 0.0342 / 0.0515 |
| patch_projection_s | 0.0014 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000545.

Projection capacity selection: {'input': 2336, 'selected': 2336, 'applied': False}.

![tensile_40pct, Learned DIC + DeDoDe-B, affine: reference and corrected target](tensile_40pct/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dic_learned_b/affine/image1_ref.png) · [Target PNG](tensile_40pct/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.078 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5869 / 7.4694 / 7.6304 |
| description_pair_s | 6.8924 / 6.7784 / 6.9983 |
| detection_pair_s | 0.2156 / 0.1364 / 0.2448 |
| geometric_verification_s | 0.0446 / 0.0436 / 0.0463 |
| matching_pipeline_s | 7.5490 / 7.4315 / 7.5759 |
| mutual_matching_s | 0.3912 / 0.3694 / 0.3917 |
| field_fit_s | 0.0349 / 0.0342 / 0.0515 |
| patch_projection_s | 0.0014 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.769717819057405, 'patch_projection_s': 0.002195577952079475, 'fit_and_patch_s': 0.7719135280931368}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000545.

Projection capacity selection: {'input': 2336, 'selected': 2336, 'applied': False}.

![tensile_40pct, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_40pct/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.714. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.524 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.0678 / 7.0549 / 7.0698 |
| description_pair_s | 6.9169 / 6.9085 / 6.9194 |
| detection_pair_s | 0.1227 / 0.1198 / 0.1234 |
| geometric_verification_s | 0.0083 / 0.0082 / 0.0089 |
| matching_pipeline_s | 7.0480 / 7.0425 / 7.0528 |
| mutual_matching_s | 0.0022 / 0.0021 / 0.0022 |
| field_fit_s | 0.0117 / 0.0111 / 0.0197 |
| patch_projection_s | 0.0020 / 0.0013 / 0.0032 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000067.

Projection capacity selection: {'input': 39, 'selected': 39, 'applied': False}.

![tensile_40pct, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](tensile_40pct/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.713. Preview absolute intensity difference median/P95: 11.00/62.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/horg_b/affine/image1_ref.png) · [Target PNG](tensile_40pct/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/horg_b/affine/result.json)
- [Full fitted field and provenance](tensile_40pct/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.524 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.0678 / 7.0549 / 7.0698 |
| description_pair_s | 6.9169 / 6.9085 / 6.9194 |
| detection_pair_s | 0.1227 / 0.1198 / 0.1234 |
| geometric_verification_s | 0.0083 / 0.0082 / 0.0089 |
| matching_pipeline_s | 7.0480 / 7.0425 / 7.0528 |
| mutual_matching_s | 0.0022 / 0.0021 / 0.0022 |
| field_fit_s | 0.0117 / 0.0111 / 0.0197 |
| patch_projection_s | 0.0020 / 0.0013 / 0.0032 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.08690111502073705, 'patch_projection_s': 0.0011776769533753395, 'fit_and_patch_s': 0.08807886601425707}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000067.

Projection capacity selection: {'input': 39, 'selected': 39, 'applied': False}.

![tensile_40pct, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_40pct/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.713. Preview absolute intensity difference median/P95: 11.00/62.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_40pct/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_40pct/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_40pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_40pct/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_40pct/horg_b/gaussian_rbf/model.json.gz)
