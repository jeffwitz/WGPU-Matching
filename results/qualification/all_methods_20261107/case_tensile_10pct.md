# Homogeneous traction: 10 percent: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `tensile_10pct`.

Prescribed stretch factors: x = 1.09904979018993, y = 0.9543134908259364. Axial engineering strain is 9.904979 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.11 / 0.17 / 0.20 | 17924 / 26605 / 30720 | True |
| DeepMatching INRIA hierarchy | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.11 / 0.17 / 0.20 | 17924 / 26605 / 30720 | True |
| HORG | affine → none | Fail | 0 / 28 | 0 | 0 | — / — / — | 0 / 47 / 693 | — |
| HORG | gaussian_rbf → none | Fail | 0 / 28 | 0 | 0 | — / — / — | 0 / 47 / 693 | — |
| CPM-HOG | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.41 / 0.70 / 0.83 | 395 / 610 / 32026 | True |
| CPM-HOG | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.41 / 0.70 / 0.83 | 395 / 610 / 32026 | True |
| XFeat | affine → affine | Pass | 28 / 28 | 20 | 28 | 1.97 / 2.34 / 2.35 | 731 / 1761 / 1761 | True |
| XFeat | gaussian_rbf → gaussian_rbf | Pass | 28 / 28 | 17 | 28 | 1.98 / 2.44 / 2.48 | 731 / 1761 / 1761 | False |
| EfficientLoFTR | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.29 / 0.38 / 0.40 | 1593 / 2825 / 2828 | True |
| EfficientLoFTR | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.29 / 0.38 / 0.40 | 1593 / 2825 / 2828 | True |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.26 / 0.40 / 0.46 | 428 / 681 / 681 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.26 / 0.40 / 0.46 | 428 / 681 / 681 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.14 / 0.22 / 0.24 | 2805 / 6066 / 6066 | True |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.14 / 0.22 / 0.24 | 2805 / 6066 / 6066 | True |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.23 / 0.37 / 0.45 | 1565 / 4660 / 4660 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.23 / 0.37 / 0.45 | 1565 / 4660 / 4660 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.12 / 0.18 / 0.20 | 2102 / 4939 / 4939 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.12 / 0.18 / 0.20 | 2102 / 4939 / 4939 | True |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.06 / 0.10 / 0.11 | 3094 / 5817 / 5817 | True |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.06 / 0.10 / 0.11 | 3094 / 5817 / 5817 | True |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.21 / 0.29 / 0.31 | 207 / 425 / 425 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.21 / 0.29 / 0.31 | 207 / 425 / 425 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.105 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.5879 / 17.1122 / 17.6824 |
| geometric_verification_s | 0.0860 / 0.0848 / 0.1248 |
| matching_pipeline_s | 17.4694 / 17.0363 / 17.6062 |
| field_fit_s | 0.0715 / 0.0712 / 0.1136 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000047.

Projection capacity selection: {'input': 17924, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_10pct, DeepMatching INRIA hierarchy, affine: reference and corrected target](tensile_10pct/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/deepmatching/affine/image1_ref.png) · [Target PNG](tensile_10pct/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/deepmatching/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.105 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.5879 / 17.1122 / 17.6824 |
| geometric_verification_s | 0.0860 / 0.0848 / 0.1248 |
| matching_pipeline_s | 17.4694 / 17.0363 / 17.6062 |
| field_fit_s | 0.0715 / 0.0712 / 0.1136 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.598973558982834, 'patch_projection_s': 0.0016143490793183446, 'fit_and_patch_s': 2.600588017026894}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000047.

Projection capacity selection: {'input': 17924, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_10pct, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](tensile_10pct/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 28 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0859 / 0.0841 / 0.0907 |
| geometric_verification_s | 0.0040 / 0.0038 / 0.0047 |
| matching_pipeline_s | 0.0856 / 0.0838 / 0.0904 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_10pct, HORG, affine: reference and corrected target](tensile_10pct/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/horg/affine/image1_ref.png) · [Target PNG](tensile_10pct/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/horg/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_10pct/horg/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 28 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0859 / 0.0841 / 0.0907 |
| geometric_verification_s | 0.0040 / 0.0038 / 0.0047 |
| matching_pipeline_s | 0.0856 / 0.0838 / 0.0904 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_10pct, HORG, gaussian_rbf: reference and corrected target](tensile_10pct/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/horg/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/horg/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_10pct/horg/gaussian_rbf/rejected_diagnostic_model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.346 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3916 / 0.3898 / 0.3948 |
| geometric_verification_s | 0.0160 / 0.0159 / 0.0162 |
| matching_pipeline_s | 0.3719 / 0.3675 / 0.3726 |
| field_fit_s | 0.0183 / 0.0166 / 0.0187 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000038.

Projection capacity selection: {'input': 395, 'selected': 395, 'applied': False}.

![tensile_10pct, CPM-HOG, affine: reference and corrected target](tensile_10pct/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/35.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/cpm/affine/image1_ref.png) · [Target PNG](tensile_10pct/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/cpm/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/cpm/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.346 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3916 / 0.3898 / 0.3948 |
| geometric_verification_s | 0.0160 / 0.0159 / 0.0162 |
| matching_pipeline_s | 0.3719 / 0.3675 / 0.3726 |
| field_fit_s | 0.0183 / 0.0166 / 0.0187 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.28421221300959587, 'patch_projection_s': 0.0016430859686806798, 'fit_and_patch_s': 0.28585554496385157}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000038.

Projection capacity selection: {'input': 395, 'selected': 395, 'applied': False}.

![tensile_10pct, CPM-HOG, gaussian_rbf: reference and corrected target](tensile_10pct/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/35.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/cpm/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/cpm/gaussian_rbf/model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.893 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6422 / 0.6359 / 0.6765 |
| geometric_verification_s | 0.0166 / 0.0160 / 0.0234 |
| matching_pipeline_s | 0.6197 / 0.6112 / 0.6335 |
| field_fit_s | 0.0179 / 0.0162 / 0.0356 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000070.

Projection capacity selection: {'input': 731, 'selected': 731, 'applied': False}.

![tensile_10pct, XFeat, affine: reference and corrected target](tensile_10pct/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.910. Preview absolute intensity difference median/P95: 18.00/86.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/xfeat/affine/image1_ref.png) · [Target PNG](tensile_10pct/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/xfeat/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.937 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6422 / 0.6359 / 0.6765 |
| geometric_verification_s | 0.0166 / 0.0160 / 0.0234 |
| matching_pipeline_s | 0.6197 / 0.6112 / 0.6335 |
| field_fit_s | 0.0179 / 0.0162 / 0.0356 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.4237551150145009, 'patch_projection_s': 0.0020216209813952446, 'fit_and_patch_s': 0.4257768460083753}.

Selection reason: gpu_solver_fallback_to_next_spatial_cv_candidate. IRLS converged: False. Final weight change: 0.000226.

Projection capacity selection: {'input': 731, 'selected': 731, 'applied': False}.

![tensile_10pct, XFeat, gaussian_rbf: reference and corrected target](tensile_10pct/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.910. Preview absolute intensity difference median/P95: 18.00/86.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.287 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9364 / 0.9327 / 0.9435 |
| geometric_verification_s | 0.0182 / 0.0179 / 0.0205 |
| matching_pipeline_s | 0.9204 / 0.9129 / 0.9277 |
| field_fit_s | 0.0150 / 0.0148 / 0.0186 |
| patch_projection_s | 0.0011 / 0.0010 / 0.0012 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000044.

Projection capacity selection: {'input': 1593, 'selected': 1593, 'applied': False}.

![tensile_10pct, EfficientLoFTR, affine: reference and corrected target](tensile_10pct/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/efficientloftr/affine/image1_ref.png) · [Target PNG](tensile_10pct/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/efficientloftr/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.287 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9364 / 0.9327 / 0.9435 |
| geometric_verification_s | 0.0182 / 0.0179 / 0.0205 |
| matching_pipeline_s | 0.9204 / 0.9129 / 0.9277 |
| field_fit_s | 0.0150 / 0.0148 / 0.0186 |
| patch_projection_s | 0.0011 / 0.0010 / 0.0012 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5292573870392516, 'patch_projection_s': 0.0015021479921415448, 'fit_and_patch_s': 0.5307596561033279}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000044.

Projection capacity selection: {'input': 1593, 'selected': 1593, 'applied': False}.

![tensile_10pct, EfficientLoFTR, gaussian_rbf: reference and corrected target](tensile_10pct/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.230 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1257 / 1.1169 / 1.1486 |
| geometric_verification_s | 0.0108 / 0.0087 / 0.0192 |
| matching_pipeline_s | 1.1099 / 1.0984 / 1.1197 |
| field_fit_s | 0.0160 / 0.0145 / 0.0239 |
| patch_projection_s | 0.0025 / 0.0013 / 0.0050 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000076.

Projection capacity selection: {'input': 428, 'selected': 428, 'applied': False}.

![tensile_10pct, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](tensile_10pct/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/raco/affine/image1_ref.png) · [Target PNG](tensile_10pct/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/raco/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.230 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1257 / 1.1169 / 1.1486 |
| geometric_verification_s | 0.0108 / 0.0087 / 0.0192 |
| matching_pipeline_s | 1.1099 / 1.0984 / 1.1197 |
| field_fit_s | 0.0160 / 0.0145 / 0.0239 |
| patch_projection_s | 0.0025 / 0.0013 / 0.0050 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.22147532796952873, 'patch_projection_s': 0.0015847700415179133, 'fit_and_patch_s': 0.22306031396146864}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000076.

Projection capacity selection: {'input': 428, 'selected': 428, 'applied': False}.

![tensile_10pct, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](tensile_10pct/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/raco/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.121 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.1884 / 7.9965 / 8.5076 |
| description_pair_s | 7.1337 / 7.1018 / 7.2558 |
| detection_pair_s | 0.4533 / 0.4262 / 0.4925 |
| geometric_verification_s | 0.0481 / 0.0450 / 0.0489 |
| matching_pipeline_s | 8.1574 / 7.9596 / 8.4785 |
| mutual_matching_s | 0.3627 / 0.3485 / 0.8740 |
| field_fit_s | 0.0258 / 0.0251 / 0.0322 |
| patch_projection_s | 0.0017 / 0.0016 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000050.

Projection capacity selection: {'input': 2805, 'selected': 2805, 'applied': False}.

![tensile_10pct, DaD + DeDoDe-B, affine: reference and corrected target](tensile_10pct/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dad_b/affine/image1_ref.png) · [Target PNG](tensile_10pct/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dad_b/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.121 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.1884 / 7.9965 / 8.5076 |
| description_pair_s | 7.1337 / 7.1018 / 7.2558 |
| detection_pair_s | 0.4533 / 0.4262 / 0.4925 |
| geometric_verification_s | 0.0481 / 0.0450 / 0.0489 |
| matching_pipeline_s | 8.1574 / 7.9596 / 8.4785 |
| mutual_matching_s | 0.3627 / 0.3485 / 0.8740 |
| field_fit_s | 0.0258 / 0.0251 / 0.0322 |
| patch_projection_s | 0.0017 / 0.0016 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8230408550007269, 'patch_projection_s': 0.0013081659562885761, 'fit_and_patch_s': 0.8243491220055148}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000050.

Projection capacity selection: {'input': 2805, 'selected': 2805, 'applied': False}.

![tensile_10pct, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_10pct/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.192 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.5352 / 8.4983 / 8.5803 |
| description_pair_s | 7.0279 / 6.9329 / 7.0709 |
| detection_pair_s | 0.9882 / 0.9845 / 0.9907 |
| geometric_verification_s | 0.0408 / 0.0408 / 0.0469 |
| matching_pipeline_s | 8.5015 / 8.4612 / 8.5554 |
| mutual_matching_s | 0.4949 / 0.3981 / 0.4983 |
| field_fit_s | 0.0284 / 0.0211 / 0.0340 |
| patch_projection_s | 0.0017 / 0.0014 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000046.

Projection capacity selection: {'input': 1565, 'selected': 1565, 'applied': False}.

![tensile_10pct, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](tensile_10pct/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dedode_b/affine/image1_ref.png) · [Target PNG](tensile_10pct/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dedode_b/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.192 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.5352 / 8.4983 / 8.5803 |
| description_pair_s | 7.0279 / 6.9329 / 7.0709 |
| detection_pair_s | 0.9882 / 0.9845 / 0.9907 |
| geometric_verification_s | 0.0408 / 0.0408 / 0.0469 |
| matching_pipeline_s | 8.5015 / 8.4612 / 8.5554 |
| mutual_matching_s | 0.4949 / 0.3981 / 0.4983 |
| field_fit_s | 0.0284 / 0.0211 / 0.0340 |
| patch_projection_s | 0.0017 / 0.0014 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5576389830093831, 'patch_projection_s': 0.0013681530253961682, 'fit_and_patch_s': 0.5590072351042181}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000046.

Projection capacity selection: {'input': 1565, 'selected': 1565, 'applied': False}.

![tensile_10pct, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_10pct/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.105 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8856 / 7.4881 / 8.0254 |
| description_pair_s | 7.1515 / 7.0368 / 7.3364 |
| detection_pair_s | 0.1962 / 0.1255 / 0.2432 |
| geometric_verification_s | 0.0431 / 0.0430 / 0.0448 |
| matching_pipeline_s | 7.8578 / 7.4568 / 7.9988 |
| mutual_matching_s | 0.4180 / 0.2484 / 0.4219 |
| field_fit_s | 0.0249 / 0.0232 / 0.0294 |
| patch_projection_s | 0.0018 / 0.0017 / 0.0022 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000062.

Projection capacity selection: {'input': 2102, 'selected': 2102, 'applied': False}.

![tensile_10pct, DIC tensor + DeDoDe-B, affine: reference and corrected target](tensile_10pct/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dic_tensor_b/affine/image1_ref.png) · [Target PNG](tensile_10pct/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.105 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8856 / 7.4881 / 8.0254 |
| description_pair_s | 7.1515 / 7.0368 / 7.3364 |
| detection_pair_s | 0.1962 / 0.1255 / 0.2432 |
| geometric_verification_s | 0.0431 / 0.0430 / 0.0448 |
| matching_pipeline_s | 7.8578 / 7.4568 / 7.9988 |
| mutual_matching_s | 0.4180 / 0.2484 / 0.4219 |
| field_fit_s | 0.0249 / 0.0232 / 0.0294 |
| patch_projection_s | 0.0018 / 0.0017 / 0.0022 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.750240579014644, 'patch_projection_s': 0.0044008539989590645, 'fit_and_patch_s': 0.7546415129909292}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000062.

Projection capacity selection: {'input': 2102, 'selected': 2102, 'applied': False}.

![tensile_10pct, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_10pct/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.047 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.6200 / 7.4626 / 7.6884 |
| description_pair_s | 7.0238 / 6.9203 / 7.0376 |
| detection_pair_s | 0.2032 / 0.1351 / 0.2090 |
| geometric_verification_s | 0.0490 / 0.0466 / 0.0494 |
| matching_pipeline_s | 7.5847 / 7.4231 / 7.6437 |
| mutual_matching_s | 0.3097 / 0.2435 / 0.4203 |
| field_fit_s | 0.0379 / 0.0337 / 0.0429 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 3094, 'selected': 3094, 'applied': False}.

![tensile_10pct, Learned DIC + DeDoDe-B, affine: reference and corrected target](tensile_10pct/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dic_learned_b/affine/image1_ref.png) · [Target PNG](tensile_10pct/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.047 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.6200 / 7.4626 / 7.6884 |
| description_pair_s | 7.0238 / 6.9203 / 7.0376 |
| detection_pair_s | 0.2032 / 0.1351 / 0.2090 |
| geometric_verification_s | 0.0490 / 0.0466 / 0.0494 |
| matching_pipeline_s | 7.5847 / 7.4231 / 7.6437 |
| mutual_matching_s | 0.3097 / 0.2435 / 0.4203 |
| field_fit_s | 0.0379 / 0.0337 / 0.0429 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8354142060270533, 'patch_projection_s': 0.005538515048101544, 'fit_and_patch_s': 0.8409528360934928}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 3094, 'selected': 3094, 'applied': False}.

![tensile_10pct, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_10pct/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.201 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.3573 / 7.2473 / 7.3834 |
| description_pair_s | 7.1867 / 7.0923 / 7.2287 |
| detection_pair_s | 0.1217 / 0.1157 / 0.1245 |
| geometric_verification_s | 0.0109 / 0.0102 / 0.0147 |
| matching_pipeline_s | 7.3366 / 7.2331 / 7.3661 |
| mutual_matching_s | 0.0113 / 0.0052 / 0.0132 |
| field_fit_s | 0.0162 / 0.0125 / 0.0187 |
| patch_projection_s | 0.0016 / 0.0011 / 0.0020 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000072.

Projection capacity selection: {'input': 207, 'selected': 207, 'applied': False}.

![tensile_10pct, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](tensile_10pct/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/horg_b/affine/image1_ref.png) · [Target PNG](tensile_10pct/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/horg_b/affine/result.json)
- [Full fitted field and provenance](tensile_10pct/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.201 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.3573 / 7.2473 / 7.3834 |
| description_pair_s | 7.1867 / 7.0923 / 7.2287 |
| detection_pair_s | 0.1217 / 0.1157 / 0.1245 |
| geometric_verification_s | 0.0109 / 0.0102 / 0.0147 |
| matching_pipeline_s | 7.3366 / 7.2331 / 7.3661 |
| mutual_matching_s | 0.0113 / 0.0052 / 0.0132 |
| field_fit_s | 0.0162 / 0.0125 / 0.0187 |
| patch_projection_s | 0.0016 / 0.0011 / 0.0020 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.17787225893698633, 'patch_projection_s': 0.001204875996336341, 'fit_and_patch_s': 0.17907721898518503}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000072.

Projection capacity selection: {'input': 207, 'selected': 207, 'applied': False}.

![tensile_10pct, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_10pct/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_10pct/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_10pct/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_10pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_10pct/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_10pct/horg_b/gaussian_rbf/model.json.gz)
