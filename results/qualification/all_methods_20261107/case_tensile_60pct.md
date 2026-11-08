# Homogeneous traction: 60 percent: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `tensile_60pct`.

Prescribed stretch factors: x = 1.6, y = 0.7910265877644076. Axial engineering strain is 60.000000 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 16 / 16 | 16 | 16 | 0.18 / 0.27 / 0.30 | 5014 / 15266 / 30720 | False |
| DeepMatching INRIA hierarchy | gaussian_rbf → affine | Pass | 16 / 16 | 16 | 16 | 0.18 / 0.27 / 0.30 | 5014 / 15266 / 30720 | False |
| HORG | affine → none | Fail | 0 / 16 | 0 | 0 | — / — / — | 0 / 20 / 693 | — |
| HORG | gaussian_rbf → none | Fail | 0 / 16 | 0 | 0 | — / — / — | 0 / 20 / 693 | — |
| CPM-HOG | affine → none | Fail | 0 / 16 | 0 | 0 | — / — / — | 0 / 45 / 32026 | — |
| CPM-HOG | gaussian_rbf → none | Fail | 0 / 16 | 0 | 0 | — / — / — | 0 / 45 / 32026 | — |
| XFeat | affine → affine | Pass | 16 / 16 | 0 | 16 | 3.05 / 3.22 / 3.23 | 143 / 893 / 893 | True |
| XFeat | gaussian_rbf → affine | Pass | 16 / 16 | 0 | 16 | 3.05 / 3.22 / 3.23 | 143 / 893 / 893 | True |
| EfficientLoFTR | affine → affine | Pass | 16 / 16 | 10 | 16 | 1.95 / 2.23 / 2.29 | 379 / 1431 / 1434 | True |
| EfficientLoFTR | gaussian_rbf → affine | Pass | 16 / 16 | 10 | 16 | 1.95 / 2.23 / 2.29 | 379 / 1431 / 1434 | True |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 16 / 16 | 16 | 16 | 0.71 / 1.01 / 1.15 | 148 / 389 / 389 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 16 / 16 | 16 | 16 | 0.71 / 1.01 / 1.15 | 148 / 389 / 389 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 16 / 16 | 16 | 16 | 0.44 / 0.68 / 0.70 | 856 / 4186 / 4186 | True |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 16 / 16 | 16 | 16 | 0.44 / 0.68 / 0.70 | 856 / 4186 / 4186 | True |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 16 / 16 | 16 | 16 | 0.49 / 0.71 / 0.81 | 563 / 3513 / 3513 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 16 / 16 | 16 | 16 | 0.49 / 0.71 / 0.81 | 563 / 3513 / 3513 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 16 / 16 | 14 | 16 | 1.40 / 2.16 / 2.56 | 194 / 3289 / 3289 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → affine | Pass | 16 / 16 | 14 | 16 | 1.40 / 2.16 / 2.56 | 194 / 3289 / 3289 | True |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 16 / 16 | 16 | 16 | 0.22 / 0.34 / 0.38 | 1496 / 4311 / 4311 | False |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 16 / 16 | 16 | 16 | 0.22 / 0.34 / 0.38 | 1496 / 4311 / 4311 | False |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Fail | 16 / 16 | 3 | 5 | 9.13 / 15.54 / 18.66 | 15 / 310 / 310 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Fail | 16 / 16 | 3 | 5 | 9.13 / 15.54 / 18.66 | 15 / 310 / 310 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.164 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.1284 / 17.0128 / 17.4480 |
| geometric_verification_s | 0.0645 / 0.0577 / 0.0646 |
| matching_pipeline_s | 17.0719 / 16.9562 / 17.3535 |
| field_fit_s | 0.0515 / 0.0514 / 0.0894 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.001020.

Projection capacity selection: {'input': 5014, 'selected': 5014, 'applied': False}.

![tensile_60pct, DeepMatching INRIA hierarchy, affine: reference and corrected target](tensile_60pct/deepmatching/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/deepmatching/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/deepmatching/affine/image1_ref.png) · [Target PNG](tensile_60pct/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/deepmatching/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.164 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.1284 / 17.0128 / 17.4480 |
| geometric_verification_s | 0.0645 / 0.0577 / 0.0646 |
| matching_pipeline_s | 17.0719 / 16.9562 / 17.3535 |
| field_fit_s | 0.0515 / 0.0514 / 0.0894 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 1.4812610970111564, 'patch_projection_s': 0.0018737689824774861, 'fit_and_patch_s': 1.4831350629683584}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.001020.

Projection capacity selection: {'input': 5014, 'selected': 5014, 'applied': False}.

![tensile_60pct, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](tensile_60pct/deepmatching/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/deepmatching/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 16 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0858 / 0.0843 / 0.0916 |
| geometric_verification_s | 0.0039 / 0.0038 / 0.0040 |
| matching_pipeline_s | 0.0855 / 0.0840 / 0.0913 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_60pct, HORG, affine: reference and corrected target](tensile_60pct/horg/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/horg/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/155.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/horg/affine/image1_ref.png) · [Target PNG](tensile_60pct/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/horg/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_60pct/horg/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 16 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0858 / 0.0843 / 0.0916 |
| geometric_verification_s | 0.0039 / 0.0038 / 0.0040 |
| matching_pipeline_s | 0.0855 / 0.0840 / 0.0913 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_60pct, HORG, gaussian_rbf: reference and corrected target](tensile_60pct/horg/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/horg/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/155.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/horg/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/horg/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_60pct/horg/gaussian_rbf/rejected_diagnostic_model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 16 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3986 / 0.3649 / 0.4730 |
| geometric_verification_s | 0.0058 / 0.0057 / 0.0060 |
| matching_pipeline_s | 0.3942 / 0.3604 / 0.4683 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_60pct, CPM-HOG, affine: reference and corrected target](tensile_60pct/cpm/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/cpm/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/cpm/affine/image1_ref.png) · [Target PNG](tensile_60pct/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/cpm/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_60pct/cpm/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 16 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3986 / 0.3649 / 0.4730 |
| geometric_verification_s | 0.0058 / 0.0057 / 0.0060 |
| matching_pipeline_s | 0.3942 / 0.3604 / 0.4683 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_60pct, CPM-HOG, gaussian_rbf: reference and corrected target](tensile_60pct/cpm/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/cpm/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 61.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/cpm/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_60pct/cpm/gaussian_rbf/rejected_diagnostic_model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 3.040 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6293 / 0.5981 / 0.6318 |
| geometric_verification_s | 0.0133 / 0.0096 / 0.0195 |
| matching_pipeline_s | 0.6068 / 0.5833 / 0.6092 |
| field_fit_s | 0.0150 / 0.0109 / 0.0181 |
| patch_projection_s | 0.0014 / 0.0010 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000044.

Projection capacity selection: {'input': 143, 'selected': 143, 'applied': False}.

![tensile_60pct, XFeat, affine: reference and corrected target](tensile_60pct/xfeat/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/xfeat/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 18.00/87.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/xfeat/affine/image1_ref.png) · [Target PNG](tensile_60pct/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/xfeat/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 3.040 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6293 / 0.5981 / 0.6318 |
| geometric_verification_s | 0.0133 / 0.0096 / 0.0195 |
| matching_pipeline_s | 0.6068 / 0.5833 / 0.6092 |
| field_fit_s | 0.0150 / 0.0109 / 0.0181 |
| patch_projection_s | 0.0014 / 0.0010 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.18410120101179928, 'patch_projection_s': 0.0011779370252043009, 'fit_and_patch_s': 0.1852792149875313}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000044.

Projection capacity selection: {'input': 143, 'selected': 143, 'applied': False}.

![tensile_60pct, XFeat, gaussian_rbf: reference and corrected target](tensile_60pct/xfeat/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/xfeat/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 18.00/87.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.937 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.8633 / 0.8626 / 0.8796 |
| geometric_verification_s | 0.0131 / 0.0131 / 0.0157 |
| matching_pipeline_s | 0.8530 / 0.8519 / 0.8660 |
| field_fit_s | 0.0098 / 0.0094 / 0.0123 |
| patch_projection_s | 0.0008 / 0.0008 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000095.

Projection capacity selection: {'input': 379, 'selected': 379, 'applied': False}.

![tensile_60pct, EfficientLoFTR, affine: reference and corrected target](tensile_60pct/efficientloftr/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/efficientloftr/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.624. Preview absolute intensity difference median/P95: 13.00/68.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/efficientloftr/affine/image1_ref.png) · [Target PNG](tensile_60pct/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/efficientloftr/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.937 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.8633 / 0.8626 / 0.8796 |
| geometric_verification_s | 0.0131 / 0.0131 / 0.0157 |
| matching_pipeline_s | 0.8530 / 0.8519 / 0.8660 |
| field_fit_s | 0.0098 / 0.0094 / 0.0123 |
| patch_projection_s | 0.0008 / 0.0008 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.2565465059597045, 'patch_projection_s': 0.0014153319643810391, 'fit_and_patch_s': 0.25796192698180676}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000095.

Projection capacity selection: {'input': 379, 'selected': 379, 'applied': False}.

![tensile_60pct, EfficientLoFTR, gaussian_rbf: reference and corrected target](tensile_60pct/efficientloftr/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/efficientloftr/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.624. Preview absolute intensity difference median/P95: 13.00/68.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.683 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1399 / 1.1240 / 1.1429 |
| geometric_verification_s | 0.0110 / 0.0091 / 0.0153 |
| matching_pipeline_s | 1.1236 / 1.1088 / 1.1276 |
| field_fit_s | 0.0140 / 0.0133 / 0.0150 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000076.

Projection capacity selection: {'input': 148, 'selected': 148, 'applied': False}.

![tensile_60pct, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](tensile_60pct/raco/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/raco/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 10.00/46.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/raco/affine/image1_ref.png) · [Target PNG](tensile_60pct/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/raco/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.683 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1399 / 1.1240 / 1.1429 |
| geometric_verification_s | 0.0110 / 0.0091 / 0.0153 |
| matching_pipeline_s | 1.1236 / 1.1088 / 1.1276 |
| field_fit_s | 0.0140 / 0.0133 / 0.0150 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.14173683302942663, 'patch_projection_s': 0.0011369099374860525, 'fit_and_patch_s': 0.1428738459944725}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000076.

Projection capacity selection: {'input': 148, 'selected': 148, 'applied': False}.

![tensile_60pct, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](tensile_60pct/raco/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/raco/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 10.00/46.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/raco/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.362 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.0657 / 7.9979 / 8.0837 |
| description_pair_s | 7.1686 / 7.0899 / 7.1692 |
| detection_pair_s | 0.4495 / 0.4455 / 0.4548 |
| geometric_verification_s | 0.0418 / 0.0413 / 0.0432 |
| matching_pipeline_s | 8.0321 / 7.9652 / 8.0544 |
| mutual_matching_s | 0.3782 / 0.3707 / 0.3951 |
| field_fit_s | 0.0259 / 0.0249 / 0.0270 |
| patch_projection_s | 0.0020 / 0.0017 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000088.

Projection capacity selection: {'input': 856, 'selected': 856, 'applied': False}.

![tensile_60pct, DaD + DeDoDe-B, affine: reference and corrected target](tensile_60pct/dad_b/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dad_b/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 9.00/37.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dad_b/affine/image1_ref.png) · [Target PNG](tensile_60pct/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dad_b/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.362 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.0657 / 7.9979 / 8.0837 |
| description_pair_s | 7.1686 / 7.0899 / 7.1692 |
| detection_pair_s | 0.4495 / 0.4455 / 0.4548 |
| geometric_verification_s | 0.0418 / 0.0413 / 0.0432 |
| matching_pipeline_s | 8.0321 / 7.9652 / 8.0544 |
| mutual_matching_s | 0.3782 / 0.3707 / 0.3951 |
| field_fit_s | 0.0259 / 0.0249 / 0.0270 |
| patch_projection_s | 0.0020 / 0.0017 / 0.0030 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.365246087894775, 'patch_projection_s': 0.002334585995413363, 'fit_and_patch_s': 0.36758077691774815}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000088.

Projection capacity selection: {'input': 856, 'selected': 856, 'applied': False}.

![tensile_60pct, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_60pct/dad_b/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dad_b/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 9.00/37.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.445 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.4736 / 8.4015 / 8.6379 |
| description_pair_s | 7.0138 / 6.9693 / 7.2144 |
| detection_pair_s | 0.9917 / 0.9826 / 0.9979 |
| geometric_verification_s | 0.0380 / 0.0370 / 0.0381 |
| matching_pipeline_s | 8.4447 / 8.3777 / 8.6159 |
| mutual_matching_s | 0.3774 / 0.3657 / 0.4094 |
| field_fit_s | 0.0179 / 0.0171 / 0.0243 |
| patch_projection_s | 0.0015 / 0.0015 / 0.0024 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000090.

Projection capacity selection: {'input': 563, 'selected': 563, 'applied': False}.

![tensile_60pct, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](tensile_60pct/dedode_b/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dedode_b/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 8.00/34.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dedode_b/affine/image1_ref.png) · [Target PNG](tensile_60pct/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dedode_b/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.445 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.4736 / 8.4015 / 8.6379 |
| description_pair_s | 7.0138 / 6.9693 / 7.2144 |
| detection_pair_s | 0.9917 / 0.9826 / 0.9979 |
| geometric_verification_s | 0.0380 / 0.0370 / 0.0381 |
| matching_pipeline_s | 8.4447 / 8.3777 / 8.6159 |
| mutual_matching_s | 0.3774 / 0.3657 / 0.4094 |
| field_fit_s | 0.0179 / 0.0171 / 0.0243 |
| patch_projection_s | 0.0015 / 0.0015 / 0.0024 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.29622388700954616, 'patch_projection_s': 0.0018462429288774729, 'fit_and_patch_s': 0.2980702279601246}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000090.

Projection capacity selection: {'input': 563, 'selected': 563, 'applied': False}.

![tensile_60pct, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_60pct/dedode_b/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dedode_b/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 8.00/34.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.202 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5395 / 7.3029 / 7.7288 |
| description_pair_s | 6.9136 / 6.7172 / 7.1636 |
| detection_pair_s | 0.1436 / 0.1308 / 0.1876 |
| geometric_verification_s | 0.0360 / 0.0355 / 0.0361 |
| matching_pipeline_s | 7.5174 / 7.2794 / 7.7031 |
| mutual_matching_s | 0.3792 / 0.3719 / 0.3822 |
| field_fit_s | 0.0196 / 0.0182 / 0.0207 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0027 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000081.

Projection capacity selection: {'input': 194, 'selected': 194, 'applied': False}.

![tensile_60pct, DIC tensor + DeDoDe-B, affine: reference and corrected target](tensile_60pct/dic_tensor_b/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dic_tensor_b/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 12.00/70.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dic_tensor_b/affine/image1_ref.png) · [Target PNG](tensile_60pct/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.202 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5395 / 7.3029 / 7.7288 |
| description_pair_s | 6.9136 / 6.7172 / 7.1636 |
| detection_pair_s | 0.1436 / 0.1308 / 0.1876 |
| geometric_verification_s | 0.0360 / 0.0355 / 0.0361 |
| matching_pipeline_s | 7.5174 / 7.2794 / 7.7031 |
| mutual_matching_s | 0.3792 / 0.3719 / 0.3822 |
| field_fit_s | 0.0196 / 0.0182 / 0.0207 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0027 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.1999334159772843, 'patch_projection_s': 0.0011754459701478481, 'fit_and_patch_s': 0.2011089469306171}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000081.

Projection capacity selection: {'input': 194, 'selected': 194, 'applied': False}.

![tensile_60pct, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_60pct/dic_tensor_b/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dic_tensor_b/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 12.00/70.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.202 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5234 / 7.3982 / 7.5857 |
| description_pair_s | 6.8836 / 6.7927 / 6.9385 |
| detection_pair_s | 0.1403 / 0.1381 / 0.2547 |
| geometric_verification_s | 0.0423 / 0.0416 / 0.0437 |
| matching_pipeline_s | 7.4931 / 7.3582 / 7.5523 |
| mutual_matching_s | 0.3732 / 0.3692 / 0.3826 |
| field_fit_s | 0.0294 / 0.0253 / 0.0324 |
| patch_projection_s | 0.0017 / 0.0017 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.001110.

Projection capacity selection: {'input': 1496, 'selected': 1496, 'applied': False}.

![tensile_60pct, Learned DIC + DeDoDe-B, affine: reference and corrected target](tensile_60pct/dic_learned_b/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dic_learned_b/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dic_learned_b/affine/image1_ref.png) · [Target PNG](tensile_60pct/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.202 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5234 / 7.3982 / 7.5857 |
| description_pair_s | 6.8836 / 6.7927 / 6.9385 |
| detection_pair_s | 0.1403 / 0.1381 / 0.2547 |
| geometric_verification_s | 0.0423 / 0.0416 / 0.0437 |
| matching_pipeline_s | 7.4931 / 7.3582 / 7.5523 |
| mutual_matching_s | 0.3732 / 0.3692 / 0.3826 |
| field_fit_s | 0.0294 / 0.0253 / 0.0324 |
| patch_projection_s | 0.0017 / 0.0017 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.6810475500533357, 'patch_projection_s': 0.004310528980568051, 'fit_and_patch_s': 0.6853582030162215}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.001110.

Projection capacity selection: {'input': 1496, 'selected': 1496, 'applied': False}.

![tensile_60pct, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_60pct/dic_learned_b/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/dic_learned_b/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.625. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Fail.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 0.312. Conditional precision (<5 px / estimated): 0.312. False estimates at ≥5 px: 11. Median analytic error: 6.722 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 6.9419 / 6.9142 / 7.1870 |
| description_pair_s | 6.7937 / 6.7471 / 7.0410 |
| detection_pair_s | 0.1198 / 0.1176 / 0.1315 |
| geometric_verification_s | 0.0084 / 0.0080 / 0.0102 |
| matching_pipeline_s | 6.9245 / 6.8936 / 7.1735 |
| mutual_matching_s | 0.0046 / 0.0045 / 0.0046 |
| field_fit_s | 0.0142 / 0.0122 / 0.0191 |
| patch_projection_s | 0.0015 / 0.0012 / 0.0032 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000080.

Projection capacity selection: {'input': 15, 'selected': 15, 'applied': False}.

![tensile_60pct, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](tensile_60pct/horg_b/affine/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/horg_b/affine/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.623. Preview absolute intensity difference median/P95: 43.00/150.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/horg_b/affine/image1_ref.png) · [Target PNG](tensile_60pct/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/horg_b/affine/result.json)
- [Full fitted field and provenance](tensile_60pct/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Fail.** 16 of 16 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 0.312. Conditional precision (<5 px / estimated): 0.312. False estimates at ≥5 px: 11. Median analytic error: 6.722 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 6.9419 / 6.9142 / 7.1870 |
| description_pair_s | 6.7937 / 6.7471 / 7.0410 |
| detection_pair_s | 0.1198 / 0.1176 / 0.1315 |
| geometric_verification_s | 0.0084 / 0.0080 / 0.0102 |
| matching_pipeline_s | 6.9245 / 6.8936 / 7.1735 |
| mutual_matching_s | 0.0046 / 0.0045 / 0.0046 |
| field_fit_s | 0.0142 / 0.0122 / 0.0191 |
| patch_projection_s | 0.0015 / 0.0012 / 0.0032 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.07117203599773347, 'patch_projection_s': 0.0011683069169521332, 'fit_and_patch_s': 0.07234043290372938}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000080.

Projection capacity selection: {'input': 15, 'selected': 15, 'applied': False}.

![tensile_60pct, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_60pct/horg_b/gaussian_rbf/comparison.gif)

**Native-pixel inspection:** a fixed central crop, not a region selected by error.

![Native-resolution reference/corrected crop](tensile_60pct/horg_b/gaussian_rbf/zoom_comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.623. Preview absolute intensity difference median/P95: 43.00/150.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_60pct/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_60pct/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_60pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_60pct/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_60pct/horg_b/gaussian_rbf/model.json.gz)
