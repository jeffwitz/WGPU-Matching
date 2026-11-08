# Live-DIC frame 100: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `progressive_100`.

Prescribed stretch factors: x = 1.0384330045576058, y = 0.9817227188104968. Axial engineering strain is 3.843300 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.14 / 0.23 / 0.26 | 14389 / 29111 / 30720 | True |
| DeepMatching INRIA hierarchy | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.14 / 0.23 / 0.26 | 14389 / 29111 / 30720 | True |
| HORG | affine → affine | Pass | 28 / 28 | 11 | 28 | 2.68 / 4.35 / 4.89 | 8 / 155 / 693 | False |
| HORG | gaussian_rbf → affine | Pass | 28 / 28 | 11 | 28 | 2.68 / 4.35 / 4.89 | 8 / 155 / 693 | False |
| CPM-HOG | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.27 / 0.43 / 0.45 | 1722 / 2849 / 32026 | True |
| CPM-HOG | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.27 / 0.43 / 0.45 | 1722 / 2849 / 32026 | True |
| XFeat | affine → affine | Pass | 28 / 28 | 20 | 28 | 1.87 / 2.47 / 2.83 | 835 / 1945 / 1945 | True |
| XFeat | gaussian_rbf → affine | Pass | 28 / 28 | 20 | 28 | 1.87 / 2.47 / 2.83 | 835 / 1945 / 1945 | True |
| EfficientLoFTR | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.31 / 0.43 / 0.44 | 1850 / 3151 / 3188 | True |
| EfficientLoFTR | gaussian_rbf → gaussian_rbf | Pass | 28 / 28 | 28 | 28 | 0.41 / 0.68 / 0.77 | 1850 / 3151 / 3188 | False |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.16 / 0.23 / 0.24 | 460 / 728 / 728 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.16 / 0.23 / 0.24 | 460 / 728 / 728 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.11 / 0.17 / 0.17 | 3046 / 6369 / 6369 | True |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.11 / 0.17 / 0.17 | 3046 / 6369 / 6369 | True |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.27 / 0.40 / 0.43 | 1663 / 4834 / 4834 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.27 / 0.40 / 0.43 | 1663 / 4834 / 4834 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.08 / 0.13 / 0.14 | 2399 / 5222 / 5222 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.08 / 0.13 / 0.14 | 2399 / 5222 / 5222 | True |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.09 / 0.16 / 0.18 | 3268 / 6077 / 6077 | True |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.09 / 0.16 / 0.18 | 3268 / 6077 / 6077 | True |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 28 / 28 | 28 | 28 | 0.10 / 0.14 / 0.14 | 259 / 489 / 489 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 28 / 28 | 28 | 28 | 0.10 / 0.14 / 0.14 | 259 / 489 / 489 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.126 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.7854 / 17.0848 / 17.8744 |
| geometric_verification_s | 0.1246 / 0.1030 / 0.1282 |
| matching_pipeline_s | 17.7290 / 17.0198 / 17.8114 |
| field_fit_s | 0.0584 / 0.0523 / 0.0602 |
| patch_projection_s | 0.0017 / 0.0014 / 0.0020 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000029.

Projection capacity selection: {'input': 14389, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_100, DeepMatching INRIA hierarchy, affine: reference and corrected target](progressive_100/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/deepmatching/affine/image1_ref.png) · [Target PNG](progressive_100/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/deepmatching/affine/result.json)
- [Full fitted field and provenance](progressive_100/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.126 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 17.7854 / 17.0848 / 17.8744 |
| geometric_verification_s | 0.1246 / 0.1030 / 0.1282 |
| matching_pipeline_s | 17.7290 / 17.0198 / 17.8114 |
| field_fit_s | 0.0584 / 0.0523 / 0.0602 |
| patch_projection_s | 0.0017 / 0.0014 / 0.0020 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.3979567289352417, 'patch_projection_s': 0.0035205460153520107, 'fit_and_patch_s': 2.4014773769304156}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000029.

Projection capacity selection: {'input': 14389, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_100, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](progressive_100/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.334 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.1285 / 0.1191 / 0.1288 |
| geometric_verification_s | 0.0072 / 0.0070 / 0.0072 |
| matching_pipeline_s | 0.1161 / 0.1043 / 0.1163 |
| field_fit_s | 0.0112 / 0.0110 / 0.0133 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: False. Final weight change: 0.000295.

Projection capacity selection: {'input': 8, 'selected': 8, 'applied': False}.

![progressive_100, HORG, affine: reference and corrected target](progressive_100/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.962. Preview absolute intensity difference median/P95: 16.00/97.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/horg/affine/image1_ref.png) · [Target PNG](progressive_100/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/horg/affine/result.json)
- [Full fitted field and provenance](progressive_100/horg/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.334 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.1285 / 0.1191 / 0.1288 |
| geometric_verification_s | 0.0072 / 0.0070 / 0.0072 |
| matching_pipeline_s | 0.1161 / 0.1043 / 0.1163 |
| field_fit_s | 0.0112 / 0.0110 / 0.0133 |
| patch_projection_s | 0.0012 / 0.0012 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.009823214029893279, 'patch_projection_s': 0.0013315390096977353, 'fit_and_patch_s': 0.011154904030263424}.

Selection reason: insufficient_correspondences_affine_fallback. IRLS converged: False. Final weight change: 0.000295.

Projection capacity selection: {'input': 8, 'selected': 8, 'applied': False}.

![progressive_100, HORG, gaussian_rbf: reference and corrected target](progressive_100/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.962. Preview absolute intensity difference median/P95: 16.00/97.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/horg/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/horg/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/horg/gaussian_rbf/model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.228 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.4169 / 0.4106 / 0.4837 |
| geometric_verification_s | 0.0242 / 0.0239 / 0.0242 |
| matching_pipeline_s | 0.3861 / 0.3816 / 0.4523 |
| field_fit_s | 0.0275 / 0.0255 / 0.0282 |
| patch_projection_s | 0.0013 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000086.

Projection capacity selection: {'input': 1722, 'selected': 1722, 'applied': False}.

![progressive_100, CPM-HOG, affine: reference and corrected target](progressive_100/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 8.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/cpm/affine/image1_ref.png) · [Target PNG](progressive_100/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/cpm/affine/result.json)
- [Full fitted field and provenance](progressive_100/cpm/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.228 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.4169 / 0.4106 / 0.4837 |
| geometric_verification_s | 0.0242 / 0.0239 / 0.0242 |
| matching_pipeline_s | 0.3861 / 0.3816 / 0.4523 |
| field_fit_s | 0.0275 / 0.0255 / 0.0282 |
| patch_projection_s | 0.0013 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.7711299600778148, 'patch_projection_s': 0.0025347369955852628, 'fit_and_patch_s': 0.7736648260615766}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000086.

Projection capacity selection: {'input': 1722, 'selected': 1722, 'applied': False}.

![progressive_100, CPM-HOG, gaussian_rbf: reference and corrected target](progressive_100/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 8.00/29.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/cpm/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/cpm/gaussian_rbf/model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.699 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6316 / 0.6235 / 0.6386 |
| geometric_verification_s | 0.0158 / 0.0156 / 0.0214 |
| matching_pipeline_s | 0.6147 / 0.6004 / 0.6198 |
| field_fit_s | 0.0135 / 0.0130 / 0.0178 |
| patch_projection_s | 0.0010 / 0.0010 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000078.

Projection capacity selection: {'input': 835, 'selected': 835, 'applied': False}.

![progressive_100, XFeat, affine: reference and corrected target](progressive_100/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.962. Preview absolute intensity difference median/P95: 17.00/83.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/xfeat/affine/image1_ref.png) · [Target PNG](progressive_100/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/xfeat/affine/result.json)
- [Full fitted field and provenance](progressive_100/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.699 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6316 / 0.6235 / 0.6386 |
| geometric_verification_s | 0.0158 / 0.0156 / 0.0214 |
| matching_pipeline_s | 0.6147 / 0.6004 / 0.6198 |
| field_fit_s | 0.0135 / 0.0130 / 0.0178 |
| patch_projection_s | 0.0010 / 0.0010 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.39947764202952385, 'patch_projection_s': 0.0011740599293261766, 'fit_and_patch_s': 0.4006517579546198}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000078.

Projection capacity selection: {'input': 835, 'selected': 835, 'applied': False}.

![progressive_100, XFeat, gaussian_rbf: reference and corrected target](progressive_100/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.962. Preview absolute intensity difference median/P95: 17.00/83.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.295 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9667 / 0.9653 / 0.9736 |
| geometric_verification_s | 0.0219 / 0.0191 / 0.0228 |
| matching_pipeline_s | 0.9444 / 0.9439 / 0.9560 |
| field_fit_s | 0.0200 / 0.0162 / 0.0210 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000034.

Projection capacity selection: {'input': 1850, 'selected': 1850, 'applied': False}.

![progressive_100, EfficientLoFTR, affine: reference and corrected target](progressive_100/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/efficientloftr/affine/image1_ref.png) · [Target PNG](progressive_100/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/efficientloftr/affine/result.json)
- [Full fitted field and provenance](progressive_100/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.284 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9667 / 0.9653 / 0.9736 |
| geometric_verification_s | 0.0219 / 0.0191 / 0.0228 |
| matching_pipeline_s | 0.9444 / 0.9439 / 0.9560 |
| field_fit_s | 0.0200 / 0.0162 / 0.0210 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8854585870867595, 'patch_projection_s': 0.002103657927364111, 'fit_and_patch_s': 0.8875623180065304}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: False. Final weight change: 0.000389.

Projection capacity selection: {'input': 1850, 'selected': 1850, 'applied': False}.

![progressive_100, EfficientLoFTR, gaussian_rbf: reference and corrected target](progressive_100/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/34.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.151 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1264 / 1.1225 / 1.1410 |
| geometric_verification_s | 0.0134 / 0.0089 / 0.0179 |
| matching_pipeline_s | 1.1101 / 1.1037 / 1.1215 |
| field_fit_s | 0.0152 / 0.0143 / 0.0176 |
| patch_projection_s | 0.0018 / 0.0011 / 0.0045 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000067.

Projection capacity selection: {'input': 460, 'selected': 460, 'applied': False}.

![progressive_100, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](progressive_100/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/raco/affine/image1_ref.png) · [Target PNG](progressive_100/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/raco/affine/result.json)
- [Full fitted field and provenance](progressive_100/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.151 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.1264 / 1.1225 / 1.1410 |
| geometric_verification_s | 0.0134 / 0.0089 / 0.0179 |
| matching_pipeline_s | 1.1101 / 1.1037 / 1.1215 |
| field_fit_s | 0.0152 / 0.0143 / 0.0176 |
| patch_projection_s | 0.0018 / 0.0011 / 0.0045 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.2532005050452426, 'patch_projection_s': 0.0013151660095900297, 'fit_and_patch_s': 0.2545158840948716}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000067.

Projection capacity selection: {'input': 460, 'selected': 460, 'applied': False}.

![progressive_100, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](progressive_100/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/raco/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.094 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.2920 / 8.2806 / 8.3063 |
| description_pair_s | 7.4011 / 7.3112 / 7.4056 |
| detection_pair_s | 0.4391 / 0.4316 / 0.5018 |
| geometric_verification_s | 0.0494 / 0.0470 / 0.0498 |
| matching_pipeline_s | 8.2501 / 8.2458 / 8.2715 |
| mutual_matching_s | 0.3784 / 0.3662 / 0.3819 |
| field_fit_s | 0.0305 / 0.0304 / 0.0335 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000046.

Projection capacity selection: {'input': 3046, 'selected': 3046, 'applied': False}.

![progressive_100, DaD + DeDoDe-B, affine: reference and corrected target](progressive_100/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dad_b/affine/image1_ref.png) · [Target PNG](progressive_100/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dad_b/affine/result.json)
- [Full fitted field and provenance](progressive_100/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.094 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.2920 / 8.2806 / 8.3063 |
| description_pair_s | 7.4011 / 7.3112 / 7.4056 |
| detection_pair_s | 0.4391 / 0.4316 / 0.5018 |
| geometric_verification_s | 0.0494 / 0.0470 / 0.0498 |
| matching_pipeline_s | 8.2501 / 8.2458 / 8.2715 |
| mutual_matching_s | 0.3784 / 0.3662 / 0.3819 |
| field_fit_s | 0.0305 / 0.0304 / 0.0335 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0018 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8772223880514503, 'patch_projection_s': 0.0017769649857655168, 'fit_and_patch_s': 0.8789995330153033}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000046.

Projection capacity selection: {'input': 3046, 'selected': 3046, 'applied': False}.

![progressive_100, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_100/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.238 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.7628 / 8.7328 / 8.7963 |
| description_pair_s | 7.2653 / 7.2404 / 7.3467 |
| detection_pair_s | 0.9800 / 0.9785 / 0.9949 |
| geometric_verification_s | 0.0446 / 0.0432 / 0.0452 |
| matching_pipeline_s | 8.7339 / 8.6956 / 8.7720 |
| mutual_matching_s | 0.4056 / 0.3628 / 0.4909 |
| field_fit_s | 0.0245 / 0.0199 / 0.0338 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0020 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000087.

Projection capacity selection: {'input': 1663, 'selected': 1663, 'applied': False}.

![progressive_100, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](progressive_100/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dedode_b/affine/image1_ref.png) · [Target PNG](progressive_100/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dedode_b/affine/result.json)
- [Full fitted field and provenance](progressive_100/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.238 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.7628 / 8.7328 / 8.7963 |
| description_pair_s | 7.2653 / 7.2404 / 7.3467 |
| detection_pair_s | 0.9800 / 0.9785 / 0.9949 |
| geometric_verification_s | 0.0446 / 0.0432 / 0.0452 |
| matching_pipeline_s | 8.7339 / 8.6956 / 8.7720 |
| mutual_matching_s | 0.4056 / 0.3628 / 0.4909 |
| field_fit_s | 0.0245 / 0.0199 / 0.0338 |
| patch_projection_s | 0.0016 / 0.0015 / 0.0020 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.539513707975857, 'patch_projection_s': 0.003961556940339506, 'fit_and_patch_s': 0.5434753689914942}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000087.

Projection capacity selection: {'input': 1663, 'selected': 1663, 'applied': False}.

![progressive_100, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_100/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 8.00/30.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.060 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7295 / 7.6717 / 7.8431 |
| description_pair_s | 7.0982 / 7.0791 / 7.2180 |
| detection_pair_s | 0.1904 / 0.1293 / 0.2069 |
| geometric_verification_s | 0.0432 / 0.0418 / 0.0435 |
| matching_pipeline_s | 7.7048 / 7.6421 / 7.8125 |
| mutual_matching_s | 0.3676 / 0.3598 / 0.3758 |
| field_fit_s | 0.0251 / 0.0210 / 0.0258 |
| patch_projection_s | 0.0014 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000097.

Projection capacity selection: {'input': 2399, 'selected': 2399, 'applied': False}.

![progressive_100, DIC tensor + DeDoDe-B, affine: reference and corrected target](progressive_100/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dic_tensor_b/affine/image1_ref.png) · [Target PNG](progressive_100/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](progressive_100/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.060 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7295 / 7.6717 / 7.8431 |
| description_pair_s | 7.0982 / 7.0791 / 7.2180 |
| detection_pair_s | 0.1904 / 0.1293 / 0.2069 |
| geometric_verification_s | 0.0432 / 0.0418 / 0.0435 |
| matching_pipeline_s | 7.7048 / 7.6421 / 7.8125 |
| mutual_matching_s | 0.3676 / 0.3598 / 0.3758 |
| field_fit_s | 0.0251 / 0.0210 / 0.0258 |
| patch_projection_s | 0.0014 / 0.0014 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.6546613710233942, 'patch_projection_s': 0.0012919269502162933, 'fit_and_patch_s': 0.6559534620027989}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000097.

Projection capacity selection: {'input': 2399, 'selected': 2399, 'applied': False}.

![progressive_100, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_100/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.071 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8148 / 7.6449 / 7.8219 |
| description_pair_s | 7.1270 / 7.0120 / 7.2053 |
| detection_pair_s | 0.1591 / 0.1384 / 0.2384 |
| geometric_verification_s | 0.0504 / 0.0454 / 0.0512 |
| matching_pipeline_s | 7.7746 / 7.6099 / 7.7890 |
| mutual_matching_s | 0.3717 / 0.3574 / 0.4123 |
| field_fit_s | 0.0309 / 0.0271 / 0.0346 |
| patch_projection_s | 0.0015 / 0.0015 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000062.

Projection capacity selection: {'input': 3268, 'selected': 3268, 'applied': False}.

![progressive_100, Learned DIC + DeDoDe-B, affine: reference and corrected target](progressive_100/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dic_learned_b/affine/image1_ref.png) · [Target PNG](progressive_100/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](progressive_100/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.071 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.8148 / 7.6449 / 7.8219 |
| description_pair_s | 7.1270 / 7.0120 / 7.2053 |
| detection_pair_s | 0.1591 / 0.1384 / 0.2384 |
| geometric_verification_s | 0.0504 / 0.0454 / 0.0512 |
| matching_pipeline_s | 7.7746 / 7.6099 / 7.7890 |
| mutual_matching_s | 0.3717 / 0.3574 / 0.4123 |
| field_fit_s | 0.0309 / 0.0271 / 0.0346 |
| patch_projection_s | 0.0015 / 0.0015 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.8954125499585643, 'patch_projection_s': 0.002710852073505521, 'fit_and_patch_s': 0.8981235560495406}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000062.

Projection capacity selection: {'input': 3268, 'selected': 3268, 'applied': False}.

![progressive_100, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_100/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.092 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.2970 / 7.1252 / 7.6085 |
| description_pair_s | 7.1524 / 6.9801 / 7.4543 |
| detection_pair_s | 0.1162 / 0.1159 / 0.1202 |
| geometric_verification_s | 0.0095 / 0.0094 / 0.0102 |
| matching_pipeline_s | 7.2812 / 7.1094 / 7.5883 |
| mutual_matching_s | 0.0033 / 0.0033 / 0.0034 |
| field_fit_s | 0.0144 / 0.0143 / 0.0181 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0021 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000040.

Projection capacity selection: {'input': 259, 'selected': 259, 'applied': False}.

![progressive_100, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](progressive_100/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/horg_b/affine/image1_ref.png) · [Target PNG](progressive_100/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/horg_b/affine/result.json)
- [Full fitted field and provenance](progressive_100/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 28 of 28 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.092 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.2970 / 7.1252 / 7.6085 |
| description_pair_s | 7.1524 / 6.9801 / 7.4543 |
| detection_pair_s | 0.1162 / 0.1159 / 0.1202 |
| geometric_verification_s | 0.0095 / 0.0094 / 0.0102 |
| matching_pipeline_s | 7.2812 / 7.1094 / 7.5883 |
| mutual_matching_s | 0.0033 / 0.0033 / 0.0034 |
| field_fit_s | 0.0144 / 0.0143 / 0.0181 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0021 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.17960015195421875, 'patch_projection_s': 0.0013772749807685614, 'fit_and_patch_s': 0.1809775149449706}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000040.

Projection capacity selection: {'input': 259, 'selected': 259, 'applied': False}.

![progressive_100, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_100/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.963. Preview absolute intensity difference median/P95: 9.00/26.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_100/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_100/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_100/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_100/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_100/horg_b/gaussian_rbf/model.json.gz)
