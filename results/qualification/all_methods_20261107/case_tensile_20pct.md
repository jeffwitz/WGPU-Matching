# Homogeneous traction: 20 percent: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `tensile_20pct`.

Prescribed stretch factors: x = 1.2007464493243052, y = 0.9130464032558431. Axial engineering strain is 20.074645 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.13 / 0.22 / 0.24 | 15740 / 23623 / 30720 | False |
| DeepMatching INRIA hierarchy | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.13 / 0.22 / 0.24 | 15740 / 23623 / 30720 | False |
| HORG | affine → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 19 / 693 | — |
| HORG | gaussian_rbf → none | Fail | 0 / 24 | 0 | 0 | — / — / — | 0 / 19 / 693 | — |
| CPM-HOG | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.72 / 1.09 / 1.27 | 111 / 237 / 32026 | True |
| CPM-HOG | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.72 / 1.09 / 1.27 | 111 / 237 / 32026 | True |
| XFeat | affine → affine | Pass | 24 / 24 | 18 | 24 | 1.90 / 2.05 / 2.06 | 546 / 1517 / 1517 | True |
| XFeat | gaussian_rbf → gaussian_rbf | Pass | 24 / 24 | 12 | 24 | 2.11 / 2.94 / 2.96 | 546 / 1517 / 1517 | False |
| EfficientLoFTR | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.61 / 0.76 / 0.77 | 1471 / 2494 / 2495 | True |
| EfficientLoFTR | gaussian_rbf → gaussian_rbf | Pass | 24 / 24 | 24 | 24 | 0.61 / 0.80 / 0.83 | 1471 / 2494 / 2495 | False |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.21 / 0.32 / 0.36 | 366 / 605 / 605 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.21 / 0.32 / 0.36 | 366 / 605 / 605 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.17 / 0.30 / 0.36 | 2383 / 5584 / 5584 | False |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.17 / 0.30 / 0.36 | 2383 / 5584 / 5584 | False |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.12 / 0.21 / 0.24 | 1287 / 4274 / 4274 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.12 / 0.21 / 0.24 | 1287 / 4274 / 4274 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.22 / 0.35 / 0.41 | 1281 / 4352 / 4352 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → gaussian_rbf | Pass | 24 / 24 | 24 | 24 | 0.37 / 0.60 / 0.78 | 1281 / 4352 / 4352 | False |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.10 / 0.16 / 0.19 | 2891 / 5467 / 5467 | False |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.10 / 0.16 / 0.19 | 2891 / 5467 / 5467 | False |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 24 / 24 | 24 | 24 | 0.41 / 0.68 / 0.77 | 160 / 383 / 383 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 24 / 24 | 24 | 24 | 0.41 / 0.68 / 0.77 | 160 / 383 / 383 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.100 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.6114 / 17.4493 / 17.6987 |
| geometric_verification_s | 0.0921 / 0.0798 / 0.0973 |
| matching_pipeline_s | 17.5304 / 17.3700 / 17.6172 |
| field_fit_s | 0.0763 / 0.0747 / 0.0766 |
| patch_projection_s | 0.0015 / 0.0013 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.001033.

Projection capacity selection: {'input': 15740, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_20pct, DeepMatching INRIA hierarchy, affine: reference and corrected target](tensile_20pct/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/deepmatching/affine/image1_ref.png) · [Target PNG](tensile_20pct/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/deepmatching/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.100 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.6114 / 17.4493 / 17.6987 |
| geometric_verification_s | 0.0921 / 0.0798 / 0.0973 |
| matching_pipeline_s | 17.5304 / 17.3700 / 17.6172 |
| field_fit_s | 0.0763 / 0.0747 / 0.0766 |
| patch_projection_s | 0.0015 / 0.0013 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.317629622993991, 'patch_projection_s': 0.00185817398596555, 'fit_and_patch_s': 2.3194878780050203}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.001033.

Projection capacity selection: {'input': 15740, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![tensile_20pct, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](tensile_20pct/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0833 / 0.0804 / 0.0869 |
| geometric_verification_s | 0.0041 / 0.0039 / 0.0059 |
| matching_pipeline_s | 0.0830 / 0.0801 / 0.0866 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_20pct, HORG, affine: reference and corrected target](tensile_20pct/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 62.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/horg/affine/image1_ref.png) · [Target PNG](tensile_20pct/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/horg/affine/result.json)
- [Rejected-proposal diagnostic field](tensile_20pct/horg/affine/rejected_diagnostic_model.json.gz)

### Requested gaussian_rbf; effective none

**Fail.** 0 of 24 requested centres have an estimated field.

Failure reason: at least three finite in-image correspondences are required

**Diagnostic only:** accepted pairs did not produce a field. The GIF shows a fit to rejected proposals to make the failure visible. This fit contributes no usable estimates and cannot pass.

Correct coverage (<5 px / all requested): 0.000. Conditional precision (<5 px / estimated): —. False estimates at ≥5 px: 0. Median analytic error: — px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.0833 / 0.0804 / 0.0869 |
| geometric_verification_s | 0.0041 / 0.0039 / 0.0059 |
| matching_pipeline_s | 0.0830 / 0.0801 / 0.0866 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: unavailable. IRLS converged: unavailable. Final weight change: —.

Projection capacity selection: unavailable.

![tensile_20pct, HORG, gaussian_rbf: reference and corrected target](tensile_20pct/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 62.00/156.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/horg/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/horg/gaussian_rbf/result.json)
- [Rejected-proposal diagnostic field](tensile_20pct/horg/gaussian_rbf/rejected_diagnostic_model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.592 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3902 / 0.3826 / 0.4474 |
| geometric_verification_s | 0.0072 / 0.0070 / 0.0074 |
| matching_pipeline_s | 0.3702 / 0.3630 / 0.4278 |
| field_fit_s | 0.0164 / 0.0163 / 0.0167 |
| patch_projection_s | 0.0013 / 0.0012 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000047.

Projection capacity selection: {'input': 111, 'selected': 111, 'applied': False}.

![tensile_20pct, CPM-HOG, affine: reference and corrected target](tensile_20pct/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/43.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/cpm/affine/image1_ref.png) · [Target PNG](tensile_20pct/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/cpm/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/cpm/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.592 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.3902 / 0.3826 / 0.4474 |
| geometric_verification_s | 0.0072 / 0.0070 / 0.0074 |
| matching_pipeline_s | 0.3702 / 0.3630 / 0.4278 |
| field_fit_s | 0.0164 / 0.0163 / 0.0167 |
| patch_projection_s | 0.0013 / 0.0012 / 0.0013 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.06862610497046262, 'patch_projection_s': 0.0013064970262348652, 'fit_and_patch_s': 0.06993265403434634}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000047.

Projection capacity selection: {'input': 111, 'selected': 111, 'applied': False}.

![tensile_20pct, CPM-HOG, gaussian_rbf: reference and corrected target](tensile_20pct/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/43.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/cpm/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/cpm/gaussian_rbf/model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.897 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6396 / 0.6234 / 0.6490 |
| geometric_verification_s | 0.0191 / 0.0132 / 0.0247 |
| matching_pipeline_s | 0.6134 / 0.6081 / 0.6186 |
| field_fit_s | 0.0153 / 0.0116 / 0.0253 |
| patch_projection_s | 0.0017 / 0.0010 / 0.0040 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000060.

Projection capacity selection: {'input': 546, 'selected': 546, 'applied': False}.

![tensile_20pct, XFeat, affine: reference and corrected target](tensile_20pct/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.832. Preview absolute intensity difference median/P95: 15.00/74.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/xfeat/affine/image1_ref.png) · [Target PNG](tensile_20pct/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/xfeat/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.049 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6396 / 0.6234 / 0.6490 |
| geometric_verification_s | 0.0191 / 0.0132 / 0.0247 |
| matching_pipeline_s | 0.6134 / 0.6081 / 0.6186 |
| field_fit_s | 0.0153 / 0.0116 / 0.0253 |
| patch_projection_s | 0.0017 / 0.0010 / 0.0040 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.38828171708155423, 'patch_projection_s': 0.003194478922523558, 'fit_and_patch_s': 0.39147632103413343}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.001015.

Projection capacity selection: {'input': 546, 'selected': 546, 'applied': False}.

![tensile_20pct, XFeat, gaussian_rbf: reference and corrected target](tensile_20pct/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 16.00/82.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.613 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9063 / 0.9051 / 0.9192 |
| geometric_verification_s | 0.0170 / 0.0162 / 0.0201 |
| matching_pipeline_s | 0.8873 / 0.8832 / 0.9001 |
| field_fit_s | 0.0182 / 0.0169 / 0.0216 |
| patch_projection_s | 0.0009 / 0.0009 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000078.

Projection capacity selection: {'input': 1471, 'selected': 1471, 'applied': False}.

![tensile_20pct, EfficientLoFTR, affine: reference and corrected target](tensile_20pct/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 8.00/36.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/efficientloftr/affine/image1_ref.png) · [Target PNG](tensile_20pct/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/efficientloftr/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.598 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9063 / 0.9051 / 0.9192 |
| geometric_verification_s | 0.0170 / 0.0162 / 0.0201 |
| matching_pipeline_s | 0.8873 / 0.8832 / 0.9001 |
| field_fit_s | 0.0182 / 0.0169 / 0.0216 |
| patch_projection_s | 0.0009 / 0.0009 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5913212560117245, 'patch_projection_s': 0.0022172349272295833, 'fit_and_patch_s': 0.593538686982356}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.002002.

Projection capacity selection: {'input': 1471, 'selected': 1471, 'applied': False}.

![tensile_20pct, EfficientLoFTR, gaussian_rbf: reference and corrected target](tensile_20pct/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 8.00/37.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.170 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1466 / 1.1258 / 1.1477 |
| geometric_verification_s | 0.0144 / 0.0114 / 0.0155 |
| matching_pipeline_s | 1.1253 / 1.1073 / 1.1268 |
| field_fit_s | 0.0176 / 0.0159 / 0.0194 |
| patch_projection_s | 0.0015 / 0.0009 / 0.0053 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000045.

Projection capacity selection: {'input': 366, 'selected': 366, 'applied': False}.

![tensile_20pct, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](tensile_20pct/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/raco/affine/image1_ref.png) · [Target PNG](tensile_20pct/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/raco/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.170 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1466 / 1.1258 / 1.1477 |
| geometric_verification_s | 0.0144 / 0.0114 / 0.0155 |
| matching_pipeline_s | 1.1253 / 1.1073 / 1.1268 |
| field_fit_s | 0.0176 / 0.0159 / 0.0194 |
| patch_projection_s | 0.0015 / 0.0009 / 0.0053 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.23470678902231157, 'patch_projection_s': 0.0012287059798836708, 'fit_and_patch_s': 0.235935584991239}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000045.

Projection capacity selection: {'input': 366, 'selected': 366, 'applied': False}.

![tensile_20pct, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](tensile_20pct/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/raco/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.132 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.3050 / 8.0568 / 8.5459 |
| description_pair_s | 7.4094 / 7.2017 / 7.5632 |
| detection_pair_s | 0.4446 / 0.4271 / 0.5275 |
| geometric_verification_s | 0.0472 / 0.0462 / 0.0486 |
| matching_pipeline_s | 8.2496 / 8.0204 / 8.4914 |
| mutual_matching_s | 0.3471 / 0.3440 / 0.3508 |
| field_fit_s | 0.0426 / 0.0314 / 0.0469 |
| patch_projection_s | 0.0017 / 0.0015 / 0.0019 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000275.

Projection capacity selection: {'input': 2383, 'selected': 2383, 'applied': False}.

![tensile_20pct, DaD + DeDoDe-B, affine: reference and corrected target](tensile_20pct/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dad_b/affine/image1_ref.png) · [Target PNG](tensile_20pct/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dad_b/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.132 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.3050 / 8.0568 / 8.5459 |
| description_pair_s | 7.4094 / 7.2017 / 7.5632 |
| detection_pair_s | 0.4446 / 0.4271 / 0.5275 |
| geometric_verification_s | 0.0472 / 0.0462 / 0.0486 |
| matching_pipeline_s | 8.2496 / 8.0204 / 8.4914 |
| mutual_matching_s | 0.3471 / 0.3440 / 0.3508 |
| field_fit_s | 0.0426 / 0.0314 / 0.0469 |
| patch_projection_s | 0.0017 / 0.0015 / 0.0019 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.7714353819610551, 'patch_projection_s': 0.0018403540598228574, 'fit_and_patch_s': 0.7732758449856192}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000275.

Projection capacity selection: {'input': 2383, 'selected': 2383, 'applied': False}.

![tensile_20pct, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_20pct/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.095 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.5516 / 8.4192 / 8.6413 |
| description_pair_s | 7.1550 / 7.0164 / 7.2282 |
| detection_pair_s | 0.9985 / 0.9877 / 1.0051 |
| geometric_verification_s | 0.0407 / 0.0399 / 0.0412 |
| matching_pipeline_s | 8.5181 / 8.3915 / 8.6168 |
| mutual_matching_s | 0.3332 / 0.3290 / 0.3484 |
| field_fit_s | 0.0213 / 0.0195 / 0.0250 |
| patch_projection_s | 0.0017 / 0.0016 / 0.0019 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000059.

Projection capacity selection: {'input': 1287, 'selected': 1287, 'applied': False}.

![tensile_20pct, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](tensile_20pct/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dedode_b/affine/image1_ref.png) · [Target PNG](tensile_20pct/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dedode_b/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.095 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.5516 / 8.4192 / 8.6413 |
| description_pair_s | 7.1550 / 7.0164 / 7.2282 |
| detection_pair_s | 0.9985 / 0.9877 / 1.0051 |
| geometric_verification_s | 0.0407 / 0.0399 / 0.0412 |
| matching_pipeline_s | 8.5181 / 8.3915 / 8.6168 |
| mutual_matching_s | 0.3332 / 0.3290 / 0.3484 |
| field_fit_s | 0.0213 / 0.0195 / 0.0250 |
| patch_projection_s | 0.0017 / 0.0016 / 0.0019 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.4800009180326015, 'patch_projection_s': 0.0012636090395972133, 'fit_and_patch_s': 0.4812646210193634}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000059.

Projection capacity selection: {'input': 1287, 'selected': 1287, 'applied': False}.

![tensile_20pct, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_20pct/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.176 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.6274 / 7.4346 / 7.9298 |
| description_pair_s | 7.0953 / 6.7993 / 7.3739 |
| detection_pair_s | 0.1432 / 0.1280 / 0.2330 |
| geometric_verification_s | 0.0402 / 0.0382 / 0.0441 |
| matching_pipeline_s | 7.6030 / 7.4112 / 7.9004 |
| mutual_matching_s | 0.3385 / 0.3382 / 0.3397 |
| field_fit_s | 0.0200 / 0.0186 / 0.0249 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0025 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000092.

Projection capacity selection: {'input': 1281, 'selected': 1281, 'applied': False}.

![tensile_20pct, DIC tensor + DeDoDe-B, affine: reference and corrected target](tensile_20pct/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dic_tensor_b/affine/image1_ref.png) · [Target PNG](tensile_20pct/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.312 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.6274 / 7.4346 / 7.9298 |
| description_pair_s | 7.0953 / 6.7993 / 7.3739 |
| detection_pair_s | 0.1432 / 0.1280 / 0.2330 |
| geometric_verification_s | 0.0402 / 0.0382 / 0.0441 |
| matching_pipeline_s | 7.6030 / 7.4112 / 7.9004 |
| mutual_matching_s | 0.3385 / 0.3382 / 0.3397 |
| field_fit_s | 0.0200 / 0.0186 / 0.0249 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0025 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5769224009709433, 'patch_projection_s': 0.0024305180413648486, 'fit_and_patch_s': 0.5793530669761822}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.001619.

Projection capacity selection: {'input': 1281, 'selected': 1281, 'applied': False}.

![tensile_20pct, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_20pct/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 8.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.081 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7755 / 7.5720 / 7.8096 |
| description_pair_s | 7.0950 / 6.9991 / 7.2232 |
| detection_pair_s | 0.1381 / 0.1380 / 0.2543 |
| geometric_verification_s | 0.0464 / 0.0458 / 0.0471 |
| matching_pipeline_s | 7.7336 / 7.5244 / 7.7657 |
| mutual_matching_s | 0.3396 / 0.3372 / 0.3562 |
| field_fit_s | 0.0388 / 0.0371 / 0.0397 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000258.

Projection capacity selection: {'input': 2891, 'selected': 2891, 'applied': False}.

![tensile_20pct, Learned DIC + DeDoDe-B, affine: reference and corrected target](tensile_20pct/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dic_learned_b/affine/image1_ref.png) · [Target PNG](tensile_20pct/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.081 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7755 / 7.5720 / 7.8096 |
| description_pair_s | 7.0950 / 6.9991 / 7.2232 |
| detection_pair_s | 0.1381 / 0.1380 / 0.2543 |
| geometric_verification_s | 0.0464 / 0.0458 / 0.0471 |
| matching_pipeline_s | 7.7336 / 7.5244 / 7.7657 |
| mutual_matching_s | 0.3396 / 0.3372 / 0.3562 |
| field_fit_s | 0.0388 / 0.0371 / 0.0397 |
| patch_projection_s | 0.0016 / 0.0014 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8921861150301993, 'patch_projection_s': 0.0017714499263092875, 'fit_and_patch_s': 0.8939576910343021}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: False. Final weight change: 0.000258.

Projection capacity selection: {'input': 2891, 'selected': 2891, 'applied': False}.

![tensile_20pct, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_20pct/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.347 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.0144 / 6.8610 / 7.2484 |
| description_pair_s | 6.8641 / 6.7136 / 7.1013 |
| detection_pair_s | 0.1163 / 0.1152 / 0.1224 |
| geometric_verification_s | 0.0098 / 0.0093 / 0.0100 |
| matching_pipeline_s | 6.9996 / 6.8438 / 7.2317 |
| mutual_matching_s | 0.0037 / 0.0037 / 0.0052 |
| field_fit_s | 0.0145 / 0.0123 / 0.0149 |
| patch_projection_s | 0.0024 / 0.0018 / 0.0027 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000059.

Projection capacity selection: {'input': 160, 'selected': 160, 'applied': False}.

![tensile_20pct, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](tensile_20pct/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/horg_b/affine/image1_ref.png) · [Target PNG](tensile_20pct/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/horg_b/affine/result.json)
- [Full fitted field and provenance](tensile_20pct/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 24 of 24 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.347 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.0144 / 6.8610 / 7.2484 |
| description_pair_s | 6.8641 / 6.7136 / 7.1013 |
| detection_pair_s | 0.1163 / 0.1152 / 0.1224 |
| geometric_verification_s | 0.0098 / 0.0093 / 0.0100 |
| matching_pipeline_s | 6.9996 / 6.8438 / 7.2317 |
| mutual_matching_s | 0.0037 / 0.0037 / 0.0052 |
| field_fit_s | 0.0145 / 0.0123 / 0.0149 |
| patch_projection_s | 0.0024 / 0.0018 / 0.0027 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.1464427460450679, 'patch_projection_s': 0.0010586929274722934, 'fit_and_patch_s': 0.14750153000932187}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000059.

Projection capacity selection: {'input': 160, 'selected': 160, 'applied': False}.

![tensile_20pct, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](tensile_20pct/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.833. Preview absolute intensity difference median/P95: 9.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](tensile_20pct/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](tensile_20pct/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](tensile_20pct/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](tensile_20pct/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](tensile_20pct/horg_b/gaussian_rbf/model.json.gz)
