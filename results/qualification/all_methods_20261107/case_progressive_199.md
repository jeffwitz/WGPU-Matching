# Live-DIC frame 199: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `progressive_199`.

Prescribed stretch factors: x = 1.1, y = 0.9539017454387002. Axial engineering strain is 10.000000 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.08 / 0.12 / 0.13 | 17935 / 26569 / 30720 | True |
| DeepMatching INRIA hierarchy | gaussian_rbf → gaussian_rbf | Pass | 28 / 28 | 28 | 28 | 0.11 / 0.21 / 0.24 | 17935 / 26569 / 30720 | False |
| HORG | affine → none | Fail | 0 / 28 | 0 | 0 | — / — / — | 0 / 45 / 693 | — |
| HORG | gaussian_rbf → none | Fail | 0 / 28 | 0 | 0 | — / — / — | 0 / 45 / 693 | — |
| CPM-HOG | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.52 / 0.90 / 0.97 | 265 / 607 / 32026 | True |
| CPM-HOG | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.52 / 0.90 / 0.97 | 265 / 607 / 32026 | True |
| XFeat | affine → affine | Pass | 28 / 28 | 28 | 28 | 1.75 / 1.84 / 1.85 | 733 / 1743 / 1743 | True |
| XFeat | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 1.75 / 1.84 / 1.85 | 733 / 1743 / 1743 | True |
| EfficientLoFTR | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.43 / 0.57 / 0.62 | 1610 / 2818 / 2820 | True |
| EfficientLoFTR | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.43 / 0.57 / 0.62 | 1610 / 2818 / 2820 | True |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.20 / 0.29 / 0.32 | 432 / 682 / 682 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.20 / 0.29 / 0.32 | 432 / 682 / 682 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.18 / 0.26 / 0.28 | 2811 / 6013 / 6013 | True |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.18 / 0.26 / 0.28 | 2811 / 6013 / 6013 | True |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.22 / 0.35 / 0.41 | 1532 / 4665 / 4665 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.22 / 0.35 / 0.41 | 1532 / 4665 / 4665 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.09 / 0.14 / 0.15 | 2057 / 4904 / 4904 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → gaussian_rbf | Pass | 28 / 28 | 28 | 28 | 0.10 / 0.17 / 0.21 | 2057 / 4904 / 4904 | False |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.13 / 0.20 / 0.22 | 3073 / 5781 / 5781 | True |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.13 / 0.20 / 0.22 | 3073 / 5781 / 5781 | True |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.38 / 0.62 / 0.66 | 207 / 419 / 419 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.38 / 0.62 / 0.66 | 207 / 419 / 419 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.076 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.9746 / 17.6950 / 18.2166 |
| geometric_verification_s | 0.0989 / 0.0909 / 0.1001 |
| matching_pipeline_s | 17.9143 / 17.6309 / 18.1465 |
| field_fit_s | 0.0597 / 0.0558 / 0.0655 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0015 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000098.

Projection capacity selection: {'input': 17935, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_199, DeepMatching INRIA hierarchy, affine: reference and corrected target](progressive_199/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/deepmatching/affine/image1_ref.png) · [Target PNG](progressive_199/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/deepmatching/affine/result.json)
- [Full fitted field and provenance](progressive_199/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.067 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.9746 / 17.6950 / 18.2166 |
| geometric_verification_s | 0.0989 / 0.0909 / 0.1001 |
| matching_pipeline_s | 17.9143 / 17.6309 / 18.1465 |
| field_fit_s | 0.0597 / 0.0558 / 0.0655 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0015 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.9843365349806845, 'patch_projection_s': 0.0027091120136901736, 'fit_and_patch_s': 2.9870457419892773}.

Selection reason: gpu_solver_fallback_to_next_spatial_cv_candidate. IRLS converged: False. Final weight change: 0.016542.

Projection capacity selection: {'input': 17935, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_199, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](progressive_199/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 28 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0862 / 0.0857 / 0.0882 |
| geometric_verification_s | 0.0046 / 0.0042 / 0.0049 |
| matching_pipeline_s | 0.0859 / 0.0850 / 0.0879 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![progressive_199, HORG, affine: reference and corrected target](progressive_199/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/horg/affine/image1_ref.png) · [Target PNG](progressive_199/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/horg/affine/result.json)
- [Rejected-proposal diagnostic field](progressive_199/horg/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 28 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0862 / 0.0857 / 0.0882 |
| geometric_verification_s | 0.0046 / 0.0042 / 0.0049 |
| matching_pipeline_s | 0.0859 / 0.0850 / 0.0879 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![progressive_199, HORG, gaussian_rbf: reference and corrected target](progressive_199/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/horg/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/horg/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](progressive_199/horg/gaussian_rbf/rejected_diagnostic_model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.380 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3921 / 0.3895 / 0.4481 |
| geometric_verification_s | 0.0157 / 0.0157 / 0.0161 |
| matching_pipeline_s | 0.3709 / 0.3660 / 0.4281 |
| field_fit_s | 0.0178 / 0.0165 / 0.0201 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000045.

Projection capacity selection: {'input': 265, 'selected': 265, 'applied': False}.

![progressive_199, CPM-HOG, affine: reference and corrected target](progressive_199/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 8.00/34.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/cpm/affine/image1_ref.png) · [Target PNG](progressive_199/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/cpm/affine/result.json)
- [Full fitted field and provenance](progressive_199/cpm/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.380 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3921 / 0.3895 / 0.4481 |
| geometric_verification_s | 0.0157 / 0.0157 / 0.0161 |
| matching_pipeline_s | 0.3709 / 0.3660 / 0.4281 |
| field_fit_s | 0.0178 / 0.0165 / 0.0201 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.24214810493867844, 'patch_projection_s': 0.0014108570758253336, 'fit_and_patch_s': 0.24355905200354755}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000045.

Projection capacity selection: {'input': 265, 'selected': 265, 'applied': False}.

![progressive_199, CPM-HOG, gaussian_rbf: reference and corrected target](progressive_199/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 8.00/34.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/cpm/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/cpm/gaussian_rbf/model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.753 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6411 / 0.6354 / 0.6430 |
| geometric_verification_s | 0.0169 / 0.0163 / 0.0220 |
| matching_pipeline_s | 0.6176 / 0.6003 / 0.6201 |
| field_fit_s | 0.0203 / 0.0159 / 0.0312 |
| patch_projection_s | 0.0029 / 0.0025 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000082.

Projection capacity selection: {'input': 733, 'selected': 733, 'applied': False}.

![progressive_199, XFeat, affine: reference and corrected target](progressive_199/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 16.00/79.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/xfeat/affine/image1_ref.png) · [Target PNG](progressive_199/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/xfeat/affine/result.json)
- [Full fitted field and provenance](progressive_199/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.753 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6411 / 0.6354 / 0.6430 |
| geometric_verification_s | 0.0169 / 0.0163 / 0.0220 |
| matching_pipeline_s | 0.6176 / 0.6003 / 0.6201 |
| field_fit_s | 0.0203 / 0.0159 / 0.0312 |
| patch_projection_s | 0.0029 / 0.0025 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.4020185499684885, 'patch_projection_s': 0.0013634710339829326, 'fit_and_patch_s': 0.40338210901245475}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000082.

Projection capacity selection: {'input': 733, 'selected': 733, 'applied': False}.

![progressive_199, XFeat, gaussian_rbf: reference and corrected target](progressive_199/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 16.00/79.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.418 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9409 / 0.9375 / 0.9457 |
| geometric_verification_s | 0.0201 / 0.0184 / 0.0207 |
| matching_pipeline_s | 0.9221 / 0.9208 / 0.9252 |
| field_fit_s | 0.0187 / 0.0145 / 0.0190 |
| patch_projection_s | 0.0013 / 0.0009 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000089.

Projection capacity selection: {'input': 1610, 'selected': 1610, 'applied': False}.

![progressive_199, EfficientLoFTR, affine: reference and corrected target](progressive_199/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 8.00/33.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/efficientloftr/affine/image1_ref.png) · [Target PNG](progressive_199/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/efficientloftr/affine/result.json)
- [Full fitted field and provenance](progressive_199/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.418 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9409 / 0.9375 / 0.9457 |
| geometric_verification_s | 0.0201 / 0.0184 / 0.0207 |
| matching_pipeline_s | 0.9221 / 0.9208 / 0.9252 |
| field_fit_s | 0.0187 / 0.0145 / 0.0190 |
| patch_projection_s | 0.0013 / 0.0009 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5214441840071231, 'patch_projection_s': 0.0011925309663638473, 'fit_and_patch_s': 0.5226368099683896}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000089.

Projection capacity selection: {'input': 1610, 'selected': 1610, 'applied': False}.

![progressive_199, EfficientLoFTR, gaussian_rbf: reference and corrected target](progressive_199/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 8.00/33.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.186 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1432 / 1.1393 / 1.1691 |
| geometric_verification_s | 0.0112 / 0.0080 / 0.0124 |
| matching_pipeline_s | 1.1249 / 1.1161 / 1.1553 |
| field_fit_s | 0.0131 / 0.0127 / 0.0215 |
| patch_projection_s | 0.0013 / 0.0010 / 0.0052 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 432, 'selected': 432, 'applied': False}.

![progressive_199, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](progressive_199/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/raco/affine/image1_ref.png) · [Target PNG](progressive_199/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/raco/affine/result.json)
- [Full fitted field and provenance](progressive_199/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.186 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1432 / 1.1393 / 1.1691 |
| geometric_verification_s | 0.0112 / 0.0080 / 0.0124 |
| matching_pipeline_s | 1.1249 / 1.1161 / 1.1553 |
| field_fit_s | 0.0131 / 0.0127 / 0.0215 |
| patch_projection_s | 0.0013 / 0.0010 / 0.0052 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.21975920698605478, 'patch_projection_s': 0.0017696020659059286, 'fit_and_patch_s': 0.22152902896050364}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 432, 'selected': 432, 'applied': False}.

![progressive_199, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](progressive_199/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/raco/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.179 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.2417 / 8.0747 / 8.4090 |
| description_pair_s | 7.2585 / 7.1673 / 7.4323 |
| detection_pair_s | 0.4798 / 0.4437 / 0.5115 |
| geometric_verification_s | 0.0486 / 0.0480 / 0.0534 |
| matching_pipeline_s | 8.2119 / 8.0395 / 8.3669 |
| mutual_matching_s | 0.3925 / 0.3785 / 0.4000 |
| field_fit_s | 0.0312 / 0.0249 / 0.0378 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 2811, 'selected': 2811, 'applied': False}.

![progressive_199, DaD + DeDoDe-B, affine: reference and corrected target](progressive_199/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dad_b/affine/image1_ref.png) · [Target PNG](progressive_199/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dad_b/affine/result.json)
- [Full fitted field and provenance](progressive_199/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.179 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.2417 / 8.0747 / 8.4090 |
| description_pair_s | 7.2585 / 7.1673 / 7.4323 |
| detection_pair_s | 0.4798 / 0.4437 / 0.5115 |
| geometric_verification_s | 0.0486 / 0.0480 / 0.0534 |
| matching_pipeline_s | 8.2119 / 8.0395 / 8.3669 |
| mutual_matching_s | 0.3925 / 0.3785 / 0.4000 |
| field_fit_s | 0.0312 / 0.0249 / 0.0378 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8355311420746148, 'patch_projection_s': 0.0019704089500010014, 'fit_and_patch_s': 0.8375016279751435}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 2811, 'selected': 2811, 'applied': False}.

![progressive_199, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_199/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.196 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.6356 / 8.4625 / 8.8119 |
| description_pair_s | 7.2018 / 6.9846 / 7.3503 |
| detection_pair_s | 0.9891 / 0.9858 / 1.0005 |
| geometric_verification_s | 0.0425 / 0.0423 / 0.0433 |
| matching_pipeline_s | 8.6098 / 8.4302 / 8.7812 |
| mutual_matching_s | 0.3870 / 0.3778 / 0.4130 |
| field_fit_s | 0.0242 / 0.0208 / 0.0261 |
| patch_projection_s | 0.0021 / 0.0019 / 0.0037 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000089.

Projection capacity selection: {'input': 1532, 'selected': 1532, 'applied': False}.

![progressive_199, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](progressive_199/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dedode_b/affine/image1_ref.png) · [Target PNG](progressive_199/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dedode_b/affine/result.json)
- [Full fitted field and provenance](progressive_199/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.196 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.6356 / 8.4625 / 8.8119 |
| description_pair_s | 7.2018 / 6.9846 / 7.3503 |
| detection_pair_s | 0.9891 / 0.9858 / 1.0005 |
| geometric_verification_s | 0.0425 / 0.0423 / 0.0433 |
| matching_pipeline_s | 8.6098 / 8.4302 / 8.7812 |
| mutual_matching_s | 0.3870 / 0.3778 / 0.4130 |
| field_fit_s | 0.0242 / 0.0208 / 0.0261 |
| patch_projection_s | 0.0021 / 0.0019 / 0.0037 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5393042849609628, 'patch_projection_s': 0.0018380919937044382, 'fit_and_patch_s': 0.5411425320198759}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000089.

Projection capacity selection: {'input': 1532, 'selected': 1532, 'applied': False}.

![progressive_199, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_199/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.075 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8739 / 7.7725 / 8.1484 |
| description_pair_s | 7.2391 / 7.1570 / 7.4583 |
| detection_pair_s | 0.1527 / 0.1275 / 0.1850 |
| geometric_verification_s | 0.0437 / 0.0427 / 0.0476 |
| matching_pipeline_s | 7.8340 / 7.7407 / 8.1208 |
| mutual_matching_s | 0.4114 / 0.3935 / 0.4336 |
| field_fit_s | 0.0272 / 0.0232 / 0.0323 |
| patch_projection_s | 0.0017 / 0.0016 / 0.0019 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000093.

Projection capacity selection: {'input': 2057, 'selected': 2057, 'applied': False}.

![progressive_199, DIC tensor + DeDoDe-B, affine: reference and corrected target](progressive_199/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dic_tensor_b/affine/image1_ref.png) · [Target PNG](progressive_199/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](progressive_199/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.084 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8739 / 7.7725 / 8.1484 |
| description_pair_s | 7.2391 / 7.1570 / 7.4583 |
| detection_pair_s | 0.1527 / 0.1275 / 0.1850 |
| geometric_verification_s | 0.0437 / 0.0427 / 0.0476 |
| matching_pipeline_s | 7.8340 / 7.7407 / 8.1208 |
| mutual_matching_s | 0.4114 / 0.3935 / 0.4336 |
| field_fit_s | 0.0272 / 0.0232 / 0.0323 |
| patch_projection_s | 0.0017 / 0.0016 / 0.0019 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.7047297470271587, 'patch_projection_s': 0.0016462750500068069, 'fit_and_patch_s': 0.7063761510653421}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.009238.

Projection capacity selection: {'input': 2057, 'selected': 2057, 'applied': False}.

![progressive_199, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_199/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.106 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7938 / 7.6364 / 7.8922 |
| description_pair_s | 7.1185 / 7.0040 / 7.1408 |
| detection_pair_s | 0.1613 / 0.1385 / 0.2809 |
| geometric_verification_s | 0.0501 / 0.0482 / 0.0503 |
| matching_pipeline_s | 7.6867 / 7.5933 / 7.8568 |
| mutual_matching_s | 0.3801 / 0.3765 / 0.3836 |
| field_fit_s | 0.0381 / 0.0282 / 0.1018 |
| patch_projection_s | 0.0016 / 0.0013 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000052.

Projection capacity selection: {'input': 3073, 'selected': 3073, 'applied': False}.

![progressive_199, Learned DIC + DeDoDe-B, affine: reference and corrected target](progressive_199/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dic_learned_b/affine/image1_ref.png) · [Target PNG](progressive_199/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](progressive_199/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.106 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7938 / 7.6364 / 7.8922 |
| description_pair_s | 7.1185 / 7.0040 / 7.1408 |
| detection_pair_s | 0.1613 / 0.1385 / 0.2809 |
| geometric_verification_s | 0.0501 / 0.0482 / 0.0503 |
| matching_pipeline_s | 7.6867 / 7.5933 / 7.8568 |
| mutual_matching_s | 0.3801 / 0.3765 / 0.3836 |
| field_fit_s | 0.0381 / 0.0282 / 0.1018 |
| patch_projection_s | 0.0016 / 0.0013 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.822849623975344, 'patch_projection_s': 0.0012701579835265875, 'fit_and_patch_s': 0.8241198669420555}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000052.

Projection capacity selection: {'input': 3073, 'selected': 3073, 'applied': False}.

![progressive_199, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_199/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.326 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.1085 / 7.0413 / 7.1908 |
| description_pair_s | 6.9653 / 6.8858 / 7.0497 |
| detection_pair_s | 0.1127 / 0.1079 / 0.1236 |
| geometric_verification_s | 0.0095 / 0.0088 / 0.0099 |
| matching_pipeline_s | 7.0866 / 7.0225 / 7.1748 |
| mutual_matching_s | 0.0033 / 0.0033 / 0.0034 |
| field_fit_s | 0.0134 / 0.0127 / 0.0199 |
| patch_projection_s | 0.0025 / 0.0020 / 0.0060 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000093.

Projection capacity selection: {'input': 207, 'selected': 207, 'applied': False}.

![progressive_199, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](progressive_199/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/horg_b/affine/image1_ref.png) · [Target PNG](progressive_199/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/horg_b/affine/result.json)
- [Full fitted field and provenance](progressive_199/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.326 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.1085 / 7.0413 / 7.1908 |
| description_pair_s | 6.9653 / 6.8858 / 7.0497 |
| detection_pair_s | 0.1127 / 0.1079 / 0.1236 |
| geometric_verification_s | 0.0095 / 0.0088 / 0.0099 |
| matching_pipeline_s | 7.0866 / 7.0225 / 7.1748 |
| mutual_matching_s | 0.0033 / 0.0033 / 0.0034 |
| field_fit_s | 0.0134 / 0.0127 / 0.0199 |
| patch_projection_s | 0.0025 / 0.0020 / 0.0060 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.17164673400111496, 'patch_projection_s': 0.0012914439430460334, 'fit_and_patch_s': 0.17293826199602336}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000093.

Projection capacity selection: {'input': 207, 'selected': 207, 'applied': False}.

![progressive_199, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_199/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.909. Preview absolute intensity difference median/P95: 9.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_199/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_199/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_199/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_199/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_199/horg_b/gaussian_rbf/model.json.gz)
