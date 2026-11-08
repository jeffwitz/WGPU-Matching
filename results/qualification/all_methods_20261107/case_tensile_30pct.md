# Homogeneous traction: 30 percent: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `tensile_30pct`.

Prescribed stretch factors: x = 1.2986765655277774, y = 0.877969519213963. Axial engineering strain is 29.867657 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.07 / 0.12 / 0.12 | 13588 / 20857 / 30720 | False |
| DeepMatching INRIA hierarchy | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.07 / 0.12 / 0.12 | 13588 / 20857 / 30720 | False |
| HORG | affine → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 28 / 693 | — |
| HORG | gaussian_rbf → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 28 / 693 | — |
| CPM-HOG | affine → affine | Fail | 24 / 24 | 1 | 3 | 23.82 / 41.68 / 48.29 | 30 / 131 / 32026 | True |
| CPM-HOG | gaussian_rbf → gaussian_rbf | Fail | 24 / 24 | 0 | 0 | 81.93 / 148.87 / 155.53 | 30 / 131 / 32026 | False |
| XFeat | affine → affine | Pass | 24 / 24 | 4 | 24 | 2.65 / 3.47 / 3.66 | 415 / 1307 / 1307 | True |
| XFeat | gaussian_rbf → affine | Pass | 24 / 24 | 4 | 24 | 2.65 / 3.47 / 3.66 | 415 / 1307 / 1307 | True |
| EfficientLoFTR | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.61 / 0.94 / 1.04 | 1134 / 2179 / 2180 | True |
| EfficientLoFTR | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.61 / 0.94 / 1.04 | 1134 / 2179 / 2180 | True |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.25 / 0.37 / 0.42 | 310 / 532 / 532 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → gaussian_rbf | Pass | 24 / 24 | 24 | 24 | 0.63 / 1.25 / 1.42 | 310 / 532 / 532 | False |
| DaD + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.18 / 0.30 / 0.35 | 1962 / 5156 / 5156 | False |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.18 / 0.30 / 0.35 | 1962 / 5156 / 5156 | False |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.30 / 0.35 / 0.38 | 1139 / 4122 / 4122 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.30 / 0.35 / 0.38 | 1139 / 4122 / 4122 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.16 / 0.26 / 0.28 | 795 / 3991 / 3991 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.16 / 0.26 / 0.28 | 795 / 3991 / 3991 | True |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.12 / 0.17 / 0.17 | 2685 / 5163 / 5163 | False |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.12 / 0.17 / 0.17 | 2685 / 5163 / 5163 | False |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 24 / 24 | 21 | 24 | 1.35 / 2.30 / 2.72 | 74 / 346 / 346 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 21 | 24 | 1.35 / 2.30 / 2.72 | 74 / 346 / 346 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.056 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.3143 / 17.0813 / 17.4649 |
| geometric_verification_s | 0.0812 / 0.0765 / 0.1034 |
| matching_pipeline_s | 17.2345 / 16.9973 / 17.3757 |
| field_fit_s | 0.0795 / 0.0747 / 0.0844 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000254.

Projection capacity selection: {'input': 13588, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_30pct, DeepMatching INRIA hierarchy, affine: reference and corrected target](tensile_30pct/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/deepmatching/affine/image1_ref.png) · [Target PNG](tensile_30pct/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/deepmatching/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.056 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.3143 / 17.0813 / 17.4649 |
| geometric_verification_s | 0.0812 / 0.0765 / 0.1034 |
| matching_pipeline_s | 17.2345 / 16.9973 / 17.3757 |
| field_fit_s | 0.0795 / 0.0747 / 0.0844 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.618142988998443, 'patch_projection_s': 0.00168050907086581, 'fit_and_patch_s': 2.6198235700139776}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000254.

Projection capacity selection: {'input': 13588, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_30pct, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](tensile_30pct/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0957 / 0.0946 / 0.0963 |
| geometric_verification_s | 0.0043 / 0.0040 / 0.0043 |
| matching_pipeline_s | 0.0954 / 0.0944 / 0.0960 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_30pct, HORG, affine: reference and corrected target](tensile_30pct/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/horg/affine/image1_ref.png) · [Target PNG](tensile_30pct/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/horg/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_30pct/horg/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0957 / 0.0946 / 0.0963 |
| geometric_verification_s | 0.0043 / 0.0040 / 0.0043 |
| matching_pipeline_s | 0.0954 / 0.0944 / 0.0960 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_30pct, HORG, gaussian_rbf: reference and corrected target](tensile_30pct/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/horg/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/horg/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_30pct/horg/gaussian_rbf/rejected_diagnostic_model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Fail.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 0.125. Conditional precision (<5 px / estimated): 0.125. False estimates at ≥5 px: 21. Median analytic error: 18.432 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3855 / 0.3802 / 0.4425 |
| geometric_verification_s | 0.0065 / 0.0065 / 0.0068 |
| matching_pipeline_s | 0.3622 / 0.3606 / 0.4205 |
| field_fit_s | 0.0188 / 0.0163 / 0.0200 |
| patch_projection_s | 0.0013 / 0.0012 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000076.

Projection capacity selection: {'input': 30, 'selected': 30, 'applied': False}.

![tensile_30pct, CPM-HOG, affine: reference and corrected target](tensile_30pct/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.784. Preview absolute intensity difference median/P95: 52.00/153.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/cpm/affine/image1_ref.png) · [Target PNG](tensile_30pct/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/cpm/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/cpm/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Fail.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): 0.000. False estimates at ≥5 px: 24. Median analytic error: 69.629 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3855 / 0.3802 / 0.4425 |
| geometric_verification_s | 0.0065 / 0.0065 / 0.0068 |
| matching_pipeline_s | 0.3622 / 0.3606 / 0.4205 |
| field_fit_s | 0.0188 / 0.0163 / 0.0200 |
| patch_projection_s | 0.0013 / 0.0012 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.07180041703395545, 'patch_projection_s': 0.0013282609870657325, 'fit_and_patch_s': 0.07312876300420612}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.001271.

Projection capacity selection: {'input': 30, 'selected': 30, 'applied': False}.

![tensile_30pct, CPM-HOG, gaussian_rbf: reference and corrected target](tensile_30pct/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.820. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/cpm/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/cpm/gaussian_rbf/model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.572 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6344 / 0.6340 / 0.6433 |
| geometric_verification_s | 0.0170 / 0.0153 / 0.0173 |
| matching_pipeline_s | 0.6157 / 0.6136 / 0.6237 |
| field_fit_s | 0.0157 / 0.0155 / 0.0158 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000036.

Projection capacity selection: {'input': 415, 'selected': 415, 'applied': False}.

![tensile_30pct, XFeat, affine: reference and corrected target](tensile_30pct/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 20.00/95.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/xfeat/affine/image1_ref.png) · [Target PNG](tensile_30pct/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/xfeat/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.572 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6344 / 0.6340 / 0.6433 |
| geometric_verification_s | 0.0170 / 0.0153 / 0.0173 |
| matching_pipeline_s | 0.6157 / 0.6136 / 0.6237 |
| field_fit_s | 0.0157 / 0.0155 / 0.0158 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.2660724649904296, 'patch_projection_s': 0.0012741839746013284, 'fit_and_patch_s': 0.2673467700369656}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000036.

Projection capacity selection: {'input': 415, 'selected': 415, 'applied': False}.

![tensile_30pct, XFeat, gaussian_rbf: reference and corrected target](tensile_30pct/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 20.00/95.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.531 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.8835 / 0.8789 / 0.8920 |
| geometric_verification_s | 0.0153 / 0.0151 / 0.0154 |
| matching_pipeline_s | 0.8694 / 0.8651 / 0.8775 |
| field_fit_s | 0.0131 / 0.0129 / 0.0136 |
| patch_projection_s | 0.0009 / 0.0008 / 0.0010 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000077.

Projection capacity selection: {'input': 1134, 'selected': 1134, 'applied': False}.

![tensile_30pct, EfficientLoFTR, affine: reference and corrected target](tensile_30pct/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 8.00/35.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/efficientloftr/affine/image1_ref.png) · [Target PNG](tensile_30pct/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/efficientloftr/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.531 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.8835 / 0.8789 / 0.8920 |
| geometric_verification_s | 0.0153 / 0.0151 / 0.0154 |
| matching_pipeline_s | 0.8694 / 0.8651 / 0.8775 |
| field_fit_s | 0.0131 / 0.0129 / 0.0136 |
| patch_projection_s | 0.0009 / 0.0008 / 0.0010 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.43250071397051215, 'patch_projection_s': 0.001485146931372583, 'fit_and_patch_s': 0.4339859439060092}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000077.

Projection capacity selection: {'input': 1134, 'selected': 1134, 'applied': False}.

![tensile_30pct, EfficientLoFTR, gaussian_rbf: reference and corrected target](tensile_30pct/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 8.00/35.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.220 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1234 / 1.1188 / 1.1302 |
| geometric_verification_s | 0.0110 / 0.0081 / 0.0118 |
| matching_pipeline_s | 1.1024 / 1.1003 / 1.1173 |
| field_fit_s | 0.0166 / 0.0118 / 0.0175 |
| patch_projection_s | 0.0019 / 0.0011 / 0.0035 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000053.

Projection capacity selection: {'input': 310, 'selected': 310, 'applied': False}.

![tensile_30pct, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](tensile_30pct/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/raco/affine/image1_ref.png) · [Target PNG](tensile_30pct/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/raco/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.424 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1234 / 1.1188 / 1.1302 |
| geometric_verification_s | 0.0110 / 0.0081 / 0.0118 |
| matching_pipeline_s | 1.1024 / 1.1003 / 1.1173 |
| field_fit_s | 0.0166 / 0.0118 / 0.0175 |
| patch_projection_s | 0.0019 / 0.0011 / 0.0035 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.2878350260434672, 'patch_projection_s': 0.0024952120147645473, 'fit_and_patch_s': 0.2903304260689765}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.014590.

Projection capacity selection: {'input': 310, 'selected': 310, 'applied': False}.

![tensile_30pct, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](tensile_30pct/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/41.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/raco/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.136 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.2137 / 8.1432 / 8.3242 |
| description_pair_s | 7.2849 / 7.2371 / 7.3744 |
| detection_pair_s | 0.4590 / 0.4402 / 0.4615 |
| geometric_verification_s | 0.0465 / 0.0448 / 0.0477 |
| matching_pipeline_s | 8.1618 / 8.0996 / 8.2901 |
| mutual_matching_s | 0.3747 / 0.3718 / 0.4052 |
| field_fit_s | 0.0389 / 0.0299 / 0.0448 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000104.

Projection capacity selection: {'input': 1962, 'selected': 1962, 'applied': False}.

![tensile_30pct, DaD + DeDoDe-B, affine: reference and corrected target](tensile_30pct/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dad_b/affine/image1_ref.png) · [Target PNG](tensile_30pct/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dad_b/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.136 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.2137 / 8.1432 / 8.3242 |
| description_pair_s | 7.2849 / 7.2371 / 7.3744 |
| detection_pair_s | 0.4590 / 0.4402 / 0.4615 |
| geometric_verification_s | 0.0465 / 0.0448 / 0.0477 |
| matching_pipeline_s | 8.1618 / 8.0996 / 8.2901 |
| mutual_matching_s | 0.3747 / 0.3718 / 0.4052 |
| field_fit_s | 0.0389 / 0.0299 / 0.0448 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.692842124029994, 'patch_projection_s': 0.0023220879957079887, 'fit_and_patch_s': 0.69516448804643}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000104.

Projection capacity selection: {'input': 1962, 'selected': 1962, 'applied': False}.

![tensile_30pct, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_30pct/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.286 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.5184 / 8.4792 / 8.8329 |
| description_pair_s | 7.1078 / 7.0432 / 7.2794 |
| detection_pair_s | 0.9993 / 0.9694 / 1.0000 |
| geometric_verification_s | 0.0399 / 0.0384 / 0.0434 |
| matching_pipeline_s | 8.4840 / 8.4535 / 8.7250 |
| mutual_matching_s | 0.3717 / 0.3624 / 0.4048 |
| field_fit_s | 0.0295 / 0.0173 / 0.1033 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0041 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000092.

Projection capacity selection: {'input': 1139, 'selected': 1139, 'applied': False}.

![tensile_30pct, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](tensile_30pct/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dedode_b/affine/image1_ref.png) · [Target PNG](tensile_30pct/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dedode_b/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.286 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.5184 / 8.4792 / 8.8329 |
| description_pair_s | 7.1078 / 7.0432 / 7.2794 |
| detection_pair_s | 0.9993 / 0.9694 / 1.0000 |
| geometric_verification_s | 0.0399 / 0.0384 / 0.0434 |
| matching_pipeline_s | 8.4840 / 8.4535 / 8.7250 |
| mutual_matching_s | 0.3717 / 0.3624 / 0.4048 |
| field_fit_s | 0.0295 / 0.0173 / 0.1033 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0041 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5069601500872523, 'patch_projection_s': 0.001878686947748065, 'fit_and_patch_s': 0.5088389560114592}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000092.

Projection capacity selection: {'input': 1139, 'selected': 1139, 'applied': False}.

![tensile_30pct, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_30pct/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.154 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8539 / 7.4499 / 7.9120 |
| description_pair_s | 7.1385 / 6.8591 / 7.1815 |
| detection_pair_s | 0.1479 / 0.1353 / 0.1738 |
| geometric_verification_s | 0.0387 / 0.0373 / 0.0393 |
| matching_pipeline_s | 7.8309 / 7.4155 / 7.8825 |
| mutual_matching_s | 0.4633 / 0.3809 / 0.5306 |
| field_fit_s | 0.0222 / 0.0167 / 0.0234 |
| patch_projection_s | 0.0028 / 0.0023 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000047.

Projection capacity selection: {'input': 795, 'selected': 795, 'applied': False}.

![tensile_30pct, DIC tensor + DeDoDe-B, affine: reference and corrected target](tensile_30pct/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dic_tensor_b/affine/image1_ref.png) · [Target PNG](tensile_30pct/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.154 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8539 / 7.4499 / 7.9120 |
| description_pair_s | 7.1385 / 6.8591 / 7.1815 |
| detection_pair_s | 0.1479 / 0.1353 / 0.1738 |
| geometric_verification_s | 0.0387 / 0.0373 / 0.0393 |
| matching_pipeline_s | 7.8309 / 7.4155 / 7.8825 |
| mutual_matching_s | 0.4633 / 0.3809 / 0.5306 |
| field_fit_s | 0.0222 / 0.0167 / 0.0234 |
| patch_projection_s | 0.0028 / 0.0023 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.35140505398157984, 'patch_projection_s': 0.0043864770559594035, 'fit_and_patch_s': 0.3557916199788451}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000047.

Projection capacity selection: {'input': 795, 'selected': 795, 'applied': False}.

![tensile_30pct, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_30pct/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.121 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7114 / 7.7056 / 8.1893 |
| description_pair_s | 7.0967 / 7.0683 / 7.5122 |
| detection_pair_s | 0.1694 / 0.1399 / 0.2219 |
| geometric_verification_s | 0.0461 / 0.0455 / 0.0475 |
| matching_pipeline_s | 7.6659 / 7.6554 / 8.1497 |
| mutual_matching_s | 0.3712 / 0.3668 / 0.3813 |
| field_fit_s | 0.0356 / 0.0355 / 0.0513 |
| patch_projection_s | 0.0015 / 0.0013 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000173.

Projection capacity selection: {'input': 2685, 'selected': 2685, 'applied': False}.

![tensile_30pct, Learned DIC + DeDoDe-B, affine: reference and corrected target](tensile_30pct/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dic_learned_b/affine/image1_ref.png) · [Target PNG](tensile_30pct/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.121 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7114 / 7.7056 / 8.1893 |
| description_pair_s | 7.0967 / 7.0683 / 7.5122 |
| detection_pair_s | 0.1694 / 0.1399 / 0.2219 |
| geometric_verification_s | 0.0461 / 0.0455 / 0.0475 |
| matching_pipeline_s | 7.6659 / 7.6554 / 8.1497 |
| mutual_matching_s | 0.3712 / 0.3668 / 0.3813 |
| field_fit_s | 0.0356 / 0.0355 / 0.0513 |
| patch_projection_s | 0.0015 / 0.0013 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8589192039798945, 'patch_projection_s': 0.0018494270043447614, 'fit_and_patch_s': 0.8607687059557065}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000173.

Projection capacity selection: {'input': 2685, 'selected': 2685, 'applied': False}.

![tensile_30pct, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_30pct/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.770. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.059 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.0724 / 7.0483 / 7.1473 |
| description_pair_s | 6.9112 / 6.9059 / 6.9954 |
| detection_pair_s | 0.1189 / 0.1129 / 0.1216 |
| geometric_verification_s | 0.0097 / 0.0090 / 0.0099 |
| matching_pipeline_s | 7.0597 / 7.0329 / 7.1280 |
| mutual_matching_s | 0.0040 / 0.0038 / 0.0177 |
| field_fit_s | 0.0136 / 0.0114 / 0.0176 |
| patch_projection_s | 0.0017 / 0.0013 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000056.

Projection capacity selection: {'input': 74, 'selected': 74, 'applied': False}.

![tensile_30pct, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](tensile_30pct/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.769. Preview absolute intensity difference median/P95: 11.00/56.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/horg_b/affine/image1_ref.png) · [Target PNG](tensile_30pct/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/horg_b/affine/result.json)
- [Full fitted field and provenance](tensile_30pct/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.059 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.0724 / 7.0483 / 7.1473 |
| description_pair_s | 6.9112 / 6.9059 / 6.9954 |
| detection_pair_s | 0.1189 / 0.1129 / 0.1216 |
| geometric_verification_s | 0.0097 / 0.0090 / 0.0099 |
| matching_pipeline_s | 7.0597 / 7.0329 / 7.1280 |
| mutual_matching_s | 0.0040 / 0.0038 / 0.0177 |
| field_fit_s | 0.0136 / 0.0114 / 0.0176 |
| patch_projection_s | 0.0017 / 0.0013 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.11205936700571328, 'patch_projection_s': 0.0010306639596819878, 'fit_and_patch_s': 0.11309011001139879}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000056.

Projection capacity selection: {'input': 74, 'selected': 74, 'applied': False}.

![tensile_30pct, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_30pct/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.769. Preview absolute intensity difference median/P95: 11.00/56.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_30pct/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_30pct/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_30pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_30pct/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_30pct/horg_b/gaussian_rbf/model.json.gz)
