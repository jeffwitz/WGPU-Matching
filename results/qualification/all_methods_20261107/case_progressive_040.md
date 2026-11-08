# Live-DIC frame 040: all methods and both projection requests

[Return to the complete qualification report](index.md).

The GIF alternates between the unchanged reference and the target sampled through the estimated forward displacement. No analytic warp is used to create the corrected image. The exact same viewport and shared palette are used for both frames; out-of-image samples remain black. In tensile corrections, side bands can be physically outside the target field of view even for a correct estimate; black alone is not a failure verdict. The numeric population includes only analytically visible requested centres. Unlabelled PNG previews are retained for external image viewers.

Native reference and target: 3840 × 2160 pixels. Case identifier: `progressive_040`.

Prescribed stretch factors: x = 1.0011196739864576, y = 0.999664342060021. Axial engineering strain is 0.111967 percent. The analytical map uses centre (1920, 1080) and target = centre + diag(stretch_x, stretch_y) × (source − centre).

Error metrics use original-image pixels: RMSE, median error, P95 and maximum error are conditional on estimated centres. Correct coverage retains every visible request in its denominator. IRLS convergence is reported independently.

| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| DeepMatching INRIA hierarchy | affine → affine | Pass | 32 / 32 | 28 | 32 | 1.32 / 2.07 / 2.09 | 21170 / 30720 / 30720 | True |
| DeepMatching INRIA hierarchy | gaussian_rbf → gaussian_rbf | Pass | 32 / 32 | 28 | 32 | 1.32 / 2.05 / 2.06 | 21170 / 30720 / 30720 | True |
| HORG | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.03 / 0.05 / 0.06 | 190 / 387 / 693 | True |
| HORG | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.03 / 0.05 / 0.06 | 190 / 387 / 693 | True |
| CPM-HOG | affine → affine | Pass | 32 / 32 | 28 | 32 | 1.32 / 2.07 / 2.09 | 21082 / 32026 / 32026 | True |
| CPM-HOG | gaussian_rbf → gaussian_rbf | Pass | 32 / 32 | 28 | 32 | 1.32 / 2.05 / 2.06 | 21082 / 32026 / 32026 | True |
| XFeat | affine → affine | Pass | 32 / 32 | 14 | 32 | 2.28 / 3.22 / 3.44 | 2100 / 3087 / 3087 | True |
| XFeat | gaussian_rbf → affine | Pass | 32 / 32 | 14 | 32 | 2.28 / 3.22 / 3.44 | 2100 / 3087 / 3087 | True |
| EfficientLoFTR | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.60 / 0.94 / 0.96 | 2210 / 3344 / 3344 | True |
| EfficientLoFTR | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.60 / 0.94 / 0.96 | 2210 / 3344 / 3344 | True |
| RaCo-ALIKED-LightGlue+ | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.12 / 0.20 / 0.22 | 550 / 883 / 883 | True |
| RaCo-ALIKED-LightGlue+ | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.12 / 0.20 / 0.22 | 550 / 883 / 883 | True |
| DaD + DeDoDe-B | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.75 / 1.16 / 1.18 | 4518 / 7289 / 7289 | True |
| DaD + DeDoDe-B | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.75 / 1.16 / 1.18 | 4518 / 7289 / 7289 | True |
| DeDoDe v2-L + DeDoDe-B | affine → affine | Pass | 32 / 32 | 28 | 32 | 1.33 / 2.09 / 2.11 | 3708 / 6103 / 6103 | True |
| DeDoDe v2-L + DeDoDe-B | gaussian_rbf → affine | Pass | 32 / 32 | 28 | 32 | 1.33 / 2.09 / 2.11 | 3708 / 6103 / 6103 | True |
| DIC tensor + DeDoDe-B | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.29 / 0.42 / 0.48 | 3473 / 5984 / 5984 | True |
| DIC tensor + DeDoDe-B | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.29 / 0.42 / 0.48 | 3473 / 5984 / 5984 | True |
| Learned DIC + DeDoDe-B | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.08 / 0.12 / 0.12 | 4024 / 6633 / 6633 | True |
| Learned DIC + DeDoDe-B | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.08 / 0.12 / 0.12 | 4024 / 6633 / 6633 | True |
| HORG/SIFT detector + DeDoDe-B | affine → affine | Pass | 32 / 32 | 32 | 32 | 0.03 / 0.04 / 0.05 | 310 / 547 / 547 | True |
| HORG/SIFT detector + DeDoDe-B | gaussian_rbf → affine | Pass | 32 / 32 | 32 | 32 | 0.03 / 0.04 / 0.05 | 310 / 547 / 547 | True |

## DeepMatching INRIA hierarchy

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.168 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 16.2772 / 16.1706 / 16.3001 |
| geometric_verification_s | 0.0930 / 0.0912 / 0.0995 |
| matching_pipeline_s | 16.2374 / 16.1306 / 16.2583 |
| field_fit_s | 0.0358 / 0.0356 / 0.0378 |
| patch_projection_s | 0.0013 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000052.

Projection capacity selection: {'input': 21170, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_040, DeepMatching INRIA hierarchy, affine: reference and corrected target](progressive_040/deepmatching/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 13.00/69.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/deepmatching/affine/image1_ref.png) · [Target PNG](progressive_040/deepmatching/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/deepmatching/affine/result.json)
- [Full fitted field and provenance](progressive_040/deepmatching/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.173 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 16.2772 / 16.1706 / 16.3001 |
| geometric_verification_s | 0.0930 / 0.0912 / 0.0995 |
| matching_pipeline_s | 16.2374 / 16.1306 / 16.2583 |
| field_fit_s | 0.0358 / 0.0356 / 0.0378 |
| patch_projection_s | 0.0013 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.1037219119025394, 'patch_projection_s': 0.0020921750692650676, 'fit_and_patch_s': 2.105814168928191}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: True. Final weight change: 0.000013.

Projection capacity selection: {'input': 21170, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_040, DeepMatching INRIA hierarchy, gaussian_rbf: reference and corrected target](progressive_040/deepmatching/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 13.00/69.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/deepmatching/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/deepmatching/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/deepmatching/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/deepmatching/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/deepmatching/gaussian_rbf/model.json.gz)


## HORG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.027 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.1294 / 0.1210 / 0.1329 |
| geometric_verification_s | 0.0081 / 0.0080 / 0.0081 |
| matching_pipeline_s | 0.1162 / 0.1084 / 0.1205 |
| field_fit_s | 0.0112 / 0.0112 / 0.0115 |
| patch_projection_s | 0.0013 / 0.0011 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000045.

Projection capacity selection: {'input': 190, 'selected': 190, 'applied': False}.

![progressive_040, HORG, affine: reference and corrected target](progressive_040/horg/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 10.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/horg/affine/image1_ref.png) · [Target PNG](progressive_040/horg/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/horg/affine/result.json)
- [Full fitted field and provenance](progressive_040/horg/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.027 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.1294 / 0.1210 / 0.1329 |
| geometric_verification_s | 0.0081 / 0.0080 / 0.0081 |
| matching_pipeline_s | 0.1162 / 0.1084 / 0.1205 |
| field_fit_s | 0.0112 / 0.0112 / 0.0115 |
| patch_projection_s | 0.0013 / 0.0011 / 0.0016 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.14498305995948613, 'patch_projection_s': 0.0012290499871596694, 'fit_and_patch_s': 0.14621218398679048}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000045.

Projection capacity selection: {'input': 190, 'selected': 190, 'applied': False}.

![progressive_040, HORG, gaussian_rbf: reference and corrected target](progressive_040/horg/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 10.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/horg/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/horg/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/horg/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/horg/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/horg/gaussian_rbf/model.json.gz)


## CPM-HOG

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.168 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.5192 / 0.5151 / 0.5959 |
| geometric_verification_s | 0.0871 / 0.0864 / 0.0968 |
| matching_pipeline_s | 0.4736 / 0.4704 / 0.5545 |
| field_fit_s | 0.0387 / 0.0386 / 0.0460 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000041.

Projection capacity selection: {'input': 21082, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_040, CPM-HOG, affine: reference and corrected target](progressive_040/cpm/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 13.00/69.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/cpm/affine/image1_ref.png) · [Target PNG](progressive_040/cpm/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/cpm/affine/result.json)
- [Full fitted field and provenance](progressive_040/cpm/affine/model.json.gz)

### Requested gaussian_rbf; effective gaussian_rbf

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.173 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.5192 / 0.5151 / 0.5959 |
| geometric_verification_s | 0.0871 / 0.0864 / 0.0968 |
| matching_pipeline_s | 0.4736 / 0.4704 / 0.5545 |
| field_fit_s | 0.0387 / 0.0386 / 0.0460 |
| patch_projection_s | 0.0014 / 0.0013 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 2.039821709971875, 'patch_projection_s': 0.00182560202665627, 'fit_and_patch_s': 2.041647396981716}.

Selection reason: gaussian_rbf_lowest_spatial_cv_error. IRLS converged: True. Final weight change: 0.000007.

Projection capacity selection: {'input': 21082, 'selected': 8192, 'applied': True, 'tiles': [16, 9], 'criterion': 'balanced source-space quotas; uniform within each cell'}.

![progressive_040, CPM-HOG, gaussian_rbf: reference and corrected target](progressive_040/cpm/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 13.00/69.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/cpm/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/cpm/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/cpm/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/cpm/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/cpm/gaussian_rbf/model.json.gz)


## XFeat

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.069 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6848 / 0.6347 / 0.7042 |
| geometric_verification_s | 0.0227 / 0.0196 / 0.0242 |
| matching_pipeline_s | 0.6543 / 0.6096 / 0.6748 |
| field_fit_s | 0.0256 / 0.0207 / 0.0259 |
| patch_projection_s | 0.0019 / 0.0013 / 0.0037 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 2100, 'selected': 2100, 'applied': False}.

![progressive_040, XFeat, affine: reference and corrected target](progressive_040/xfeat/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.999. Preview absolute intensity difference median/P95: 19.00/96.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/xfeat/affine/image1_ref.png) · [Target PNG](progressive_040/xfeat/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/xfeat/affine/result.json)
- [Full fitted field and provenance](progressive_040/xfeat/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 2.069 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.6848 / 0.6347 / 0.7042 |
| geometric_verification_s | 0.0227 / 0.0196 / 0.0242 |
| matching_pipeline_s | 0.6543 / 0.6096 / 0.6748 |
| field_fit_s | 0.0256 / 0.0207 / 0.0259 |
| patch_projection_s | 0.0019 / 0.0013 / 0.0037 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.670353619963862, 'patch_projection_s': 0.0031721419654786587, 'fit_and_patch_s': 0.6735258529661223}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000069.

Projection capacity selection: {'input': 2100, 'selected': 2100, 'applied': False}.

![progressive_040, XFeat, gaussian_rbf: reference and corrected target](progressive_040/xfeat/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.999. Preview absolute intensity difference median/P95: 19.00/96.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/xfeat/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/xfeat/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/xfeat/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/xfeat/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/xfeat/gaussian_rbf/model.json.gz)


## EfficientLoFTR

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.521 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9693 / 0.9631 / 0.9698 |
| geometric_verification_s | 0.0212 / 0.0208 / 0.0215 |
| matching_pipeline_s | 0.9458 / 0.9399 / 0.9498 |
| field_fit_s | 0.0219 / 0.0186 / 0.0225 |
| patch_projection_s | 0.0014 / 0.0009 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000075.

Projection capacity selection: {'input': 2210, 'selected': 2210, 'applied': False}.

![progressive_040, EfficientLoFTR, affine: reference and corrected target](progressive_040/efficientloftr/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 9.00/39.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/efficientloftr/affine/image1_ref.png) · [Target PNG](progressive_040/efficientloftr/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/efficientloftr/affine/result.json)
- [Full fitted field and provenance](progressive_040/efficientloftr/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.521 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 0.9693 / 0.9631 / 0.9698 |
| geometric_verification_s | 0.0212 / 0.0208 / 0.0215 |
| matching_pipeline_s | 0.9458 / 0.9399 / 0.9498 |
| field_fit_s | 0.0219 / 0.0186 / 0.0225 |
| patch_projection_s | 0.0014 / 0.0009 / 0.0014 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.668798458063975, 'patch_projection_s': 0.0022643239935860038, 'fit_and_patch_s': 0.6710630250163376}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000075.

Projection capacity selection: {'input': 2210, 'selected': 2210, 'applied': False}.

![progressive_040, EfficientLoFTR, gaussian_rbf: reference and corrected target](progressive_040/efficientloftr/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 9.00/39.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/efficientloftr/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/efficientloftr/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/efficientloftr/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/efficientloftr/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/efficientloftr/gaussian_rbf/model.json.gz)


## RaCo-ALIKED-LightGlue+

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.102 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.2008 / 1.1259 / 1.2121 |
| geometric_verification_s | 0.0110 / 0.0079 / 0.0140 |
| matching_pipeline_s | 1.1792 / 1.1077 / 1.1990 |
| field_fit_s | 0.0159 / 0.0121 / 0.0200 |
| patch_projection_s | 0.0016 / 0.0010 / 0.0022 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000048.

Projection capacity selection: {'input': 550, 'selected': 550, 'applied': False}.

![progressive_040, RaCo-ALIKED-LightGlue+, affine: reference and corrected target](progressive_040/raco/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/raco/affine/image1_ref.png) · [Target PNG](progressive_040/raco/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/raco/affine/result.json)
- [Full fitted field and provenance](progressive_040/raco/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.102 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 1.2008 / 1.1259 / 1.2121 |
| geometric_verification_s | 0.0110 / 0.0079 / 0.0140 |
| matching_pipeline_s | 1.1792 / 1.1077 / 1.1990 |
| field_fit_s | 0.0159 / 0.0121 / 0.0200 |
| patch_projection_s | 0.0016 / 0.0010 / 0.0022 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.2928430059691891, 'patch_projection_s': 0.001499683945439756, 'fit_and_patch_s': 0.2943427989957854}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000048.

Projection capacity selection: {'input': 550, 'selected': 550, 'applied': False}.

![progressive_040, RaCo-ALIKED-LightGlue+, gaussian_rbf: reference and corrected target](progressive_040/raco/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 9.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/raco/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/raco/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/raco/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/raco/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/raco/gaussian_rbf/model.json.gz)


## DaD + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.664 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.3174 / 8.2224 / 8.3807 |
| description_pair_s | 7.2972 / 7.1974 / 7.3094 |
| detection_pair_s | 0.4325 / 0.4266 / 0.5046 |
| geometric_verification_s | 0.0532 / 0.0528 / 0.0570 |
| matching_pipeline_s | 8.2768 / 8.1705 / 8.3370 |
| mutual_matching_s | 0.4820 / 0.4808 / 0.4860 |
| field_fit_s | 0.0407 / 0.0375 / 0.0475 |
| patch_projection_s | 0.0015 / 0.0015 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000085.

Projection capacity selection: {'input': 4518, 'selected': 4518, 'applied': False}.

![progressive_040, DaD + DeDoDe-B, affine: reference and corrected target](progressive_040/dad_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 10.00/44.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dad_b/affine/image1_ref.png) · [Target PNG](progressive_040/dad_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dad_b/affine/result.json)
- [Full fitted field and provenance](progressive_040/dad_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.664 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.3174 / 8.2224 / 8.3807 |
| description_pair_s | 7.2972 / 7.1974 / 7.3094 |
| detection_pair_s | 0.4325 / 0.4266 / 0.5046 |
| geometric_verification_s | 0.0532 / 0.0528 / 0.0570 |
| matching_pipeline_s | 8.2768 / 8.1705 / 8.3370 |
| mutual_matching_s | 0.4820 / 0.4808 / 0.4860 |
| field_fit_s | 0.0407 / 0.0375 / 0.0475 |
| patch_projection_s | 0.0015 / 0.0015 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 1.4114422720158473, 'patch_projection_s': 0.0022556509356945753, 'fit_and_patch_s': 1.413698059041053}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000085.

Projection capacity selection: {'input': 4518, 'selected': 4518, 'applied': False}.

![progressive_040, DaD + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_040/dad_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 10.00/44.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dad_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/dad_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dad_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dad_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/dad_b/gaussian_rbf/model.json.gz)


## DeDoDe v2-L + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.177 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.6513 / 8.2766 / 8.7554 |
| description_pair_s | 7.0468 / 6.8781 / 7.4246 |
| detection_pair_s | 1.0052 / 1.0029 / 1.0829 |
| geometric_verification_s | 0.0493 / 0.0492 / 0.0506 |
| matching_pipeline_s | 8.6257 / 8.2468 / 8.7217 |
| mutual_matching_s | 0.2399 / 0.2351 / 0.5254 |
| field_fit_s | 0.0222 / 0.0222 / 0.0248 |
| patch_projection_s | 0.0016 / 0.0012 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000000.

Projection capacity selection: {'input': 3708, 'selected': 3708, 'applied': False}.

![progressive_040, DeDoDe v2-L + DeDoDe-B, affine: reference and corrected target](progressive_040/dedode_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 13.00/69.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dedode_b/affine/image1_ref.png) · [Target PNG](progressive_040/dedode_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dedode_b/affine/result.json)
- [Full fitted field and provenance](progressive_040/dedode_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 1.177 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 8.6513 / 8.2766 / 8.7554 |
| description_pair_s | 7.0468 / 6.8781 / 7.4246 |
| detection_pair_s | 1.0052 / 1.0029 / 1.0829 |
| geometric_verification_s | 0.0493 / 0.0492 / 0.0506 |
| matching_pipeline_s | 8.6257 / 8.2468 / 8.7217 |
| mutual_matching_s | 0.2399 / 0.2351 / 0.5254 |
| field_fit_s | 0.0222 / 0.0222 / 0.0248 |
| patch_projection_s | 0.0016 / 0.0012 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.5463324590818956, 'patch_projection_s': 0.0013824310153722763, 'fit_and_patch_s': 0.5477149930084124}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000000.

Projection capacity selection: {'input': 3708, 'selected': 3708, 'applied': False}.

![progressive_040, DeDoDe v2-L + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_040/dedode_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 1.000. Preview absolute intensity difference median/P95: 13.00/69.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dedode_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/dedode_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dedode_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dedode_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/dedode_b/gaussian_rbf/model.json.gz)


## DIC tensor + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.265 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7600 / 7.3346 / 7.8191 |
| description_pair_s | 7.0340 / 6.8203 / 7.1037 |
| detection_pair_s | 0.1717 / 0.1273 / 0.1807 |
| geometric_verification_s | 0.0509 / 0.0455 / 0.0522 |
| matching_pipeline_s | 7.7269 / 7.2899 / 7.7762 |
| mutual_matching_s | 0.4653 / 0.2445 / 0.4930 |
| field_fit_s | 0.0382 / 0.0306 / 0.0400 |
| patch_projection_s | 0.0016 / 0.0013 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000054.

Projection capacity selection: {'input': 3473, 'selected': 3473, 'applied': False}.

![progressive_040, DIC tensor + DeDoDe-B, affine: reference and corrected target](progressive_040/dic_tensor_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 9.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dic_tensor_b/affine/image1_ref.png) · [Target PNG](progressive_040/dic_tensor_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dic_tensor_b/affine/result.json)
- [Full fitted field and provenance](progressive_040/dic_tensor_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.265 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.7600 / 7.3346 / 7.8191 |
| description_pair_s | 7.0340 / 6.8203 / 7.1037 |
| detection_pair_s | 0.1717 / 0.1273 / 0.1807 |
| geometric_verification_s | 0.0509 / 0.0455 / 0.0522 |
| matching_pipeline_s | 7.7269 / 7.2899 / 7.7762 |
| mutual_matching_s | 0.4653 / 0.2445 / 0.4930 |
| field_fit_s | 0.0382 / 0.0306 / 0.0400 |
| patch_projection_s | 0.0016 / 0.0013 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 1.0784710459411144, 'patch_projection_s': 0.0016551520675420761, 'fit_and_patch_s': 1.0801263329340145}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000054.

Projection capacity selection: {'input': 3473, 'selected': 3473, 'applied': False}.

![progressive_040, DIC tensor + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_040/dic_tensor_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 9.00/31.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dic_tensor_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/dic_tensor_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dic_tensor_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dic_tensor_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/dic_tensor_b/gaussian_rbf/model.json.gz)


## Learned DIC + DeDoDe-B

60% synthetic status — affine: Pass; gaussian_rbf: Pass.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.077 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5577 / 7.4212 / 7.6903 |
| description_pair_s | 6.9732 / 6.9351 / 7.0764 |
| detection_pair_s | 0.1387 / 0.1375 / 0.1506 |
| geometric_verification_s | 0.0505 / 0.0498 / 0.0523 |
| matching_pipeline_s | 7.5090 / 7.3743 / 7.6424 |
| mutual_matching_s | 0.2485 / 0.2439 / 0.4649 |
| field_fit_s | 0.0401 / 0.0382 / 0.0449 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000037.

Projection capacity selection: {'input': 4024, 'selected': 4024, 'applied': False}.

![progressive_040, Learned DIC + DeDoDe-B, affine: reference and corrected target](progressive_040/dic_learned_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 10.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dic_learned_b/affine/image1_ref.png) · [Target PNG](progressive_040/dic_learned_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dic_learned_b/affine/result.json)
- [Full fitted field and provenance](progressive_040/dic_learned_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.077 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.5577 / 7.4212 / 7.6903 |
| description_pair_s | 6.9732 / 6.9351 / 7.0764 |
| detection_pair_s | 0.1387 / 0.1375 / 0.1506 |
| geometric_verification_s | 0.0505 / 0.0498 / 0.0523 |
| matching_pipeline_s | 7.5090 / 7.3743 / 7.6424 |
| mutual_matching_s | 0.2485 / 0.2439 / 0.4649 |
| field_fit_s | 0.0401 / 0.0382 / 0.0449 |
| patch_projection_s | 0.0016 / 0.0016 / 0.0017 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 1.2208047240274027, 'patch_projection_s': 0.0015635719755664468, 'fit_and_patch_s': 1.2223684260388836}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000037.

Projection capacity selection: {'input': 4024, 'selected': 4024, 'applied': False}.

![progressive_040, Learned DIC + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_040/dic_learned_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 10.00/28.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/dic_learned_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/dic_learned_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/dic_learned_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/dic_learned_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/dic_learned_b/gaussian_rbf/model.json.gz)


## HORG/SIFT detector + DeDoDe-B

60% synthetic status — affine: Fail; gaussian_rbf: Fail.

### Requested affine; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.024 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.1966 / 7.1208 / 7.2033 |
| description_pair_s | 7.0540 / 6.9788 / 7.0624 |
| detection_pair_s | 0.1117 / 0.1117 / 0.1135 |
| geometric_verification_s | 0.0102 / 0.0098 / 0.0119 |
| matching_pipeline_s | 7.1814 / 7.1041 / 7.1873 |
| mutual_matching_s | 0.0031 / 0.0018 / 0.0031 |
| field_fit_s | 0.0145 / 0.0136 / 0.0152 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0015 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Selection reason: affine_requested. IRLS converged: True. Final weight change: 0.000096.

Projection capacity selection: {'input': 310, 'selected': 310, 'applied': False}.

![progressive_040, HORG/SIFT detector + DeDoDe-B, affine: reference and corrected target](progressive_040/horg_b/affine/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 10.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/horg_b/affine/image1_ref.png) · [Target PNG](progressive_040/horg_b/affine/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/horg_b/affine/result.json)
- [Full fitted field and provenance](progressive_040/horg_b/affine/model.json.gz)

### Requested gaussian_rbf; effective affine

**Pass.** 32 of 32 requested centres have an estimated field.

Correct coverage (<5 px / all requested): 1.000. Conditional precision (<5 px / estimated): 1.000. False estimates at ≥5 px: 0. Median analytic error: 0.024 px.

| Timed component | Median / min / max (s) |
|---|---:|
| Complete affine call | 7.1966 / 7.1208 / 7.2033 |
| description_pair_s | 7.0540 / 6.9788 / 7.0624 |
| detection_pair_s | 0.1117 / 0.1117 / 0.1135 |
| geometric_verification_s | 0.0102 / 0.0098 / 0.0119 |
| matching_pipeline_s | 7.1814 / 7.1041 / 7.1873 |
| mutual_matching_s | 0.0031 / 0.0018 / 0.0031 |
| field_fit_s | 0.0145 / 0.0136 / 0.0152 |
| patch_projection_s | 0.0015 / 0.0014 / 0.0015 |

Components are nested: matching_pipeline_s includes extraction sub-stages and geometric verification. Do not sum this table. These repeated complete-call times measure the affine path; the automatic-RBF request reuses its last pair set.

Execution/input details are preserved verbatim in the downloadable per-seed result under extraction: detector dimensions and point counts, descriptor backend, selected ONNX graph, preprocessing and stage times. The shared run manifest records native image dimensions and model hashes.

Automatic-basis single fit/patch times (seconds): {'field_fit_s': 0.2269972120411694, 'patch_projection_s': 0.0013736789114773273, 'fit_and_patch_s': 0.22837097896263003}.

Selection reason: affine_within_one_standard_error_of_best_rbf. IRLS converged: True. Final weight change: 0.000096.

Projection capacity selection: {'input': 310, 'selected': 310, 'applied': False}.

![progressive_040, HORG/SIFT detector + DeDoDe-B, gaussian_rbf: reference and corrected target](progressive_040/horg_b/gaussian_rbf/comparison.gif)

Valid target-sampling fraction of the reference ROI: 0.998. Preview absolute intensity difference median/P95: 10.00/27.00. These are photometric diagnostics, not displacement error.

[Reference PNG](progressive_040/horg_b/gaussian_rbf/image1_ref.png) · [Target PNG](progressive_040/horg_b/gaussian_rbf/image1_def_cor.png)

- [Every observed pair and rejection](progressive_040/horg_b/correspondences.csv.gz)
- [Per-seed errors, estimates and timings](progressive_040/horg_b/gaussian_rbf/result.json)
- [Full fitted field and provenance](progressive_040/horg_b/gaussian_rbf/model.json.gz)
