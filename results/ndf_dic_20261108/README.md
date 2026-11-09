# Neural Deformation Field DIC: implementation and qualification log

**Status as of 2026-11-09: research prototype, not a production initializer.** The analytic representation, a pair-conditioned global matcher, and IC-GN qualification are implemented. Pair-model inference and field projection remain in CPU PyTorch. Local normalized cross-correlation (NCC) now has an optional three-pass WebGPU Shading Language (WGSL) implementation; images and fields are uploaded from CPU memory for that stage and compact outputs are read back. The existing inverse-compositional Gauss-Newton (IC-GN) solver runs on WebGPU. No neural deformation field inference shader exists. Independent 4K synthetic tests show useful translation performance but limited heterogeneous coverage. A held-out experiment now combines sparse global-rotation search with image-derived affine preconditioning and improves heterogeneous IC-GN coverage from 15.47% to 35.88% on ten independent pairs, but translations receive no benefit and the pipeline remains CPU-heavy and far from the analytic ceiling. This is neither a GPU-resident pipeline nor a real-image 4K qualification.

## Scope and definitions

Digital image correlation (DIC) estimates material displacement by matching image texture. Inverse-compositional Gauss-Newton (IC-GN) is the local iterative photometric solver used downstream in this project. Neural Deformation Field DIC (NDF-DIC) is the proposed predictor of a spatially varying map from a reference image to a deformed image.

This first milestone checks three prerequisites before network training. A graphics processing unit (GPU) is the intended inference target; a central processing unit (CPU) is used here as a deterministic reference. WebGPU is the portable graphics and compute interface planned for inference, and WebGPU Shading Language (WGSL) is its shader language.

The checks are:

1. Bilinear image sampling has the expected coordinate derivative.
2. The cubic B-spline displacement representation has a correct analytical spatial Jacobian and supports differentiable physical losses.
3. The chosen representation can fit continuous, spatially varying fields more accurately than one global affine map and a comparable direct bilinear grid.

The fitting experiments are **oracle capacity tests**: the fitting solver receives analytic displacement samples. They establish what a representation can express; they do not establish that a model can infer that field from two images.

## Mathematical representation

Coordinates are pixel-centre coordinates `(x, y)`: array index `(0, 0)` is the centre of the upper-left pixel. The map is

\[
\boldsymbol\varphi(\mathbf{x})=\mathbf{x}+\mathbf{u}(\mathbf{x}),\qquad
\mathbf{F}(\mathbf{x})=\nabla\boldsymbol\varphi(\mathbf{x})=\mathbf{I}+\nabla\mathbf{u}(\mathbf{x}).
\]

The prototype represents each displacement component as a sum of cubic cardinal B-spline levels. The basis derivatives are evaluated analytically, so the field Jacobian does not require finite differences. A small bending penalty is applied to control-net second differences while leaving affine control nets unpenalized. The comparison parameterizations are a six-parameter global affine displacement and a sum of direct bilinear displacement grids at the same nominal 128, 64, and 32 pixel spacings.

For an estimated map, the physical residual prototype samples the deformed image at \(\boldsymbol\varphi(\mathbf{x})\). It compares photometrically normalized intensities and transports the deformed-image gradient into the reference frame:

\[
\mathbf{r}_g(\mathbf{x})=\mathbf{F}(\mathbf{x})^\mathsf{T}\nabla I_1(\boldsymbol\varphi(\mathbf{x}))-a\nabla I_0(\mathbf{x}).
\]

Here \(a\) is a global gain correction. The gradient residual is not a direct comparison of the two image gradients; it includes both spatial advection and the transpose Jacobian. The implementation uses differentiable bilinear sampling and robust Charbonnier/Huber penalties.

## P0: derivative and physical-residual checks

Run the reproducible CPU reference with:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python \
  python/benchmarks/validate_ndf_field.py \
  --output results/ndf_dic_20261108/p0_p1_capacity_512x384.json
```

On the recorded run (PyTorch 2.15.0 CPU, double precision):

| Check | Maximum absolute discrepancy |
|---|---:|
| Bilinear sampler coordinate derivative vs. finite differences | 3.46e-11 |
| Analytical spline Jacobian vs. automatic differentiation | 1.11e-16 |
| Analytical spline Jacobian vs. finite differences | 6.25e-12 |

The physical objective back-propagates to spline coefficients (gradient norm 0.0312 in the deterministic check). On a synthetic affine pair, the oracle map gives an intensity residual of 0.0671 and an advected-gradient residual of 0.1917. Adding a translation error of `(4, -3)` pixels increases these to 0.6751 and 0.7515. Perturbing only the Jacobian leaves the intensity residual unchanged, as expected, and increases the gradient residual to 0.2420.

The oracle residuals are not zero. Image synthesis, discrete differentiation, filtering, and bilinear resampling make the continuous transport identity approximate on sampled images. Loss weights therefore require calibration against these discretization floors; a non-zero oracle loss is not by itself evidence of a wrong map.

## P1: representation capacity

Fields were fit on a 512 x 384 pixel domain from analytic displacement samples spaced every 8 pixels and evaluated on a denser 3 pixel grid. The stored JavaScript Object Notation (JSON) data contains displacement error, Jacobian error, Green-Lagrange strain error, determinant checks, coefficient counts, and fit time. Green-Lagrange strain is the finite-deformation measure \(\mathbf{E}=\tfrac12(\mathbf{F}^\mathsf{T}\mathbf{F}-\mathbf{I})\); it is not the infinitesimal linear strain tensor.

| Analytic field | Global affine displacement root mean square error (RMSE) | Cubic spline displacement RMSE | Direct bilinear grid RMSE |
|---|---:|---:|---:|
| Translation, 230 px / -125 px | 0.000 px | 0.000 px | 0.000 px |
| 120 degree rotation plus translation | 0.000 px | 0.000 px | 0.000 px |
| Strong global affine map | 0.000 px | 0.000 px | 0.000 px |
| Smooth heterogeneous, 256 px wavelength | 15.408 px | 0.022 px | 0.545 px |
| Smooth heterogeneous, 128 px wavelength | 7.502 px | 0.163 px | 1.271 px |
| Localized shear band, 24 px width | 13.008 px | 0.273 px | 0.650 px |
| 36 px displacement jump | 9.010 px | 3.423 px | 3.759 px |

All three representations reproduce affine fields, as expected. The affine model cannot represent the smooth heterogeneous or localized fields. With 852 spline coefficients, the cubic multi-resolution field gives substantially lower capacity error than the 608-coefficient direct bilinear multi-grid on the tested continuous fields. This is an oracle fitting comparison, not a learned inference result.

The same experiment was repeated at 1024 x 576 pixels, fitting samples every 16 pixels and evaluating every 8 pixels. The cubic spline displacement root mean square errors were 0.049 px (256 pixel wavelength), 0.222 px (128 pixel wavelength), and 0.128 px (24 pixel shear band), versus 0.668 px, 1.513 px, and 0.530 px for direct bilinear grids. The corresponding spline Jacobian errors were 0.0090, 0.0332, and 0.0155 in Frobenius norm. The global affine displacement errors on those fields were 15.368, 7.500, and 14.339 px. This confirms that the capacity result is not specific to the smaller fitting grid.

The discontinuity test exposes a structural limit: globally continuous spline and bilinear bases smear a displacement jump. Their finite errors do not mean either basis has resolved the interface. A future discontinuous model needs region-aware branches, a boundary/validity model, or a representation defined piecewise. Do not apply a smoothness penalty across a physical separation.

### Numerical lesson from the first fit

An initial unregularized multi-level solve could have very low displacement error while having an incorrect Jacobian. The levels overlap strongly and their coefficients are not uniquely identified by displacement samples alone. The current solve uses a mild second-difference penalty and all model comparisons report displacement, Jacobian, and finite-strain errors. A low displacement root-mean-square error alone is not accepted as a P1 pass.

## Current decision

The P0 implementation checks pass at double-precision numerical tolerance. The cubic multi-resolution representation is a viable candidate for continuous fields with the tested wavelengths and localization width. It does not solve discontinuities, and a 24-pixel band is only a capacity test, not a claim about image-pair recovery.

The pair-model training and inference were run with CPU PyTorch. The repository also has a software WebGPU adapter for the existing IC-GN solver. Therefore the reported timing is a mixed CPU/WebGPU pipeline, not GPU inference.

## P2: first learned pair-conditioned global search

The next prototype shares a small convolutional feature encoder between the two images, evaluates an all-pairs cosine correlation on the stride-four feature grid, and derives a coarse displacement at each reference cell. A small spatial refinement head uses the reference feature, correlation-weighted target feature, coarse displacement, and maximum matching probability. A multi-resolution cubic spline is fitted to this predicted field for the field/Jacobian diagnostic. During inference, the model receives only the image pair. The analytic displacement is used only to train the model and to score held-out pairs.

The reproducible training command for the larger small-image experiment was:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python \
  python/benchmarks/train_ndf_pair_matcher.py \
  --output-dir results/ndf_dic_20261108/p2_extended_20261108 \
  --train-pairs 256 --validation-pairs 64 --epochs 80 --batch-size 8
```

Training and validation use different texture-seed ranges. The validation set contains 32 translation and 32 heterogeneous pairs, totalling 16,049 valid coarse reference cells. Images are 96 x 64 pixels; each query searches 384 candidate cells. On the unseen textures, translation displacement median error falls from a zero-displacement baseline of 20.67 px to 1.20 px, with 92.5% of valid coarse cells within 5 px. For the heterogeneous family, the median falls from 19.76 px to 2.80 px and 68.8% of cells fall within 5 px. The 90th percentile is still 12.1 px. After projection of the raw predicted field onto the multi-resolution spline basis, displacement error is 7.26 px root mean square and Jacobian error is 1.19 in Frobenius norm. This is measurable pair-to-field learning on held-out textures, but it is not sufficient evidence of a useful DIC initializer.

### Large-displacement scale stress test

To test larger physical movements without increasing the coarse correlation matrix, 128 training pairs and 32 independent validation pairs were generated at 768 x 512 pixels, then area-downsampled to 96 x 64. The speckle diameter was scaled with the images so the downsampled pattern remained resolved. The model therefore searched the whole 768 x 512 domain with the same 384 target candidates; adjacent candidate centres are 32 pixels apart in the generated native coordinate system. Training lasted 60 epochs on the CPU in 40.6 seconds.

| Held-out family | Zero-displacement median error | Predicted median error | Predicted 90th percentile | Within 5 px | Correct coarse target cell |
|---|---:|---:|---:|---:|---:|
| Translation | 144.3 px | 13.0 px | 41.2 px | 11.5% | 70.2% |
| Heterogeneous affine plus smooth variation | 150.2 px | 27.2 px | 114.9 px | 4.8% | 54.9% |

The model learns substantial large-motion information from the image pair. It does not yet predict a sufficiently accurate continuous field: spline-projected root mean square error is 31.5 px for translation and 68.3 px for heterogeneous motion; Jacobian errors are 0.90 and 1.38. A diagnostic local affine consensus accepts 99.8% of translation nodes and 90.4% of heterogeneous nodes, yet has displacement errors of 40.3 px and 47.1 px respectively. This is a concrete false-consensus warning: high acceptance is not equivalent to correct initialization. The discrete matcher is coarse by construction, and hard matches are quantized at 32 native pixels, so the small within-5-pixel fractions are expected; the more serious issue is that field projection and local consensus do not yet turn those coarse matches into a reliable continuous map.

This stress test covers several hundred pixels of motion in a 768 x 512 synthetic domain, not a 4K image. It is still a controlled downsampled experiment, not native-resolution processing. The correlation memory is quadratic in the number of feature cells; applying this exact all-pairs implementation to 4K is infeasible. It must be replaced by a hierarchical or tiled search before any 4K inference claim.

### P2 decision

The pair-conditioned model demonstrates coarse large-displacement matching on held-out speckles, but this early 96 x 64 model does not meet the scientific objective. Heterogeneous fields retain large tail errors, the spline Jacobian is poor, and local consensus can accept false fields. Later IC-GN and 4K tests are recorded below.

## P3: first physical-loss ablation

The matching and field-refinement network was trained twice from the same initialization and the same 48 training / 16 validation texture pairs for 24 epochs. The control used correspondence cross-entropy and displacement supervision. The second run added a weight 0.02 on a single-scale photometric residual plus the physically advected image-gradient residual defined above. The synthetic validity mask was used only inside the training loss. Validation used image pairs and analytic truth only for scoring.

| Family | Physical loss | Coarse displacement median error | Spline field RMSE | Spline Jacobian RMSE | Local affine projected field RMSE |
|---|---:|---:|---:|---:|---:|
| Translation | off | 3.43 px | 7.71 px | 1.738 | 3.22 px |
| Translation | weight 0.02 | 3.62 px | 7.88 px | 1.767 | 3.44 px |
| Heterogeneous | off | 8.65 px | 13.65 px | 2.404 | 9.49 px |
| Heterogeneous | weight 0.02 | 8.84 px | 13.75 px | 2.409 | 9.66 px |

The single-scale physical term did not improve this validation set; the small differences are slight regressions across both families. This is an informative negative result. The residual as currently weighted and sampled does not solve the coarse field errors. It should not be promoted as a benefit or expanded to a multi-scale loss until the cause is diagnosed. Candidate causes include interpolation floors, large initial warping errors, and the difference between a local physical residual and the global correspondence objective.

## P4: first IC-GN integration

`python/benchmarks/qualify_ndf_pair_icgn.py` feeds the NDF field into the existing WebGPU inverse-compositional Gauss-Newton (IC-GN) solver. The model sees only an area-downsampled image pair; the predicted low-resolution displacement and its spline Jacobian are converted to the six local affine initialization parameters at the native-image subset centres. The qualification activates every requested centre without using analytic visibility. Ground truth is used only to score visibility and accuracy. Zero initialization and analytic local-affine initialization are executed as controls.

The first controlled campaign used eight held-out 768 x 512 image pairs per family, each containing a native image pair and an area-downsampled 96 x 64 model input. It produced:

| Family | NDF-DIC correct after IC-GN | Zero initialization | Analytic local-affine control | NDF coverage |
|---|---:|---:|---:|---:|
| Translation | 228 / 584 | 6 / 584 | 498 / 584 | 39.0% |
| Heterogeneous | 50 / 466 | 0 / 466 | 373 / 466 | 10.7% |

The neural initialization improves over zero initialization in both families, but remains far below the analytic local-affine control. Median NDF initial displacement root mean square error across cases is 114.5 px for translation and 206.6 px for heterogeneous motion. Despite this large error, some translation windows converge correctly, confirming that IC-GN can recover individual subsets from a poor field when local texture and basin of attraction permit it. This does not mean the predicted field is accurate. The heterogeneous result is not yet useful enough for the intended high-coverage initialization.

The median number of correct windows per second is 167.6 for translation and 42.7 for heterogeneous motion. These times include CPU model inference on the 96 x 64 pair, CPU spline fitting, and the WebGPU IC-GN solve, but exclude native-image generation and acquisition. The model does not run on WebGPU; therefore these figures are not end-to-end GPU throughput and must not be compared as a qualified production speed. Median CPU model inference is 0.157 s and 0.108 s respectively.

The same 16 pairs were then evaluated with three ways to turn the predicted coarse map into an initialization:

| Projection | Translation correct / visible | Heterogeneous correct / visible | Median correct windows/s, translation / heterogeneous |
|---|---:|---:|---:|
| Three-level spline fit | 228 / 584 | 50 / 466 | 167.6 / 42.7 |
| One-level spline fit | 228 / 584 | 49 / 466 | 233.9 / 35.8 |
| Direct bilinear interpolation of the predicted grid | 230 / 584 | 67 / 466 | 295.0 / 78.2 |

Direct interpolation is currently the best simple projection, especially for heterogeneous motion, and avoids an ill-conditioned fit. The absolute improvement remains modest; only 14.4% of visible heterogeneous windows are correctly converged.

The complete per-pair measurements, including accepted windows, iteration counts, and solver times, are in `p4_icgn_20261108.json`. The eight-pair campaign is the first meaningful solver qualification, but remains a small synthetic validation set; textures are independent of training, while motion families come from the same synthetic generator. It is not a final independent scientific test.

### Native ultra-high-definition (UHD, 3840 x 2160) stress tests

The first 96 x 54 model input failed completely on a 40-fold reduction. Retraining for the 16:9 aspect ratio did not fix this. Investigation found a scale-dependent synthetic texture edge-softness setting and a large amplification of coarse-grid errors when converted back to native pixels. These findings led to independent scale-aware texture parameters and a 192 x 108 model input, but they did not solve heterogeneous matching.

The current 192 x 108 checkpoint was trained on 128 independent 192 x 108 pairs with seven-pixel characteristic speckles, 32 validation pairs, and 60 epochs. It achieves a validation median flow error of 1.15 px for translation and 3.05 px for heterogeneous motion; 94.4% and 64.1% of the soft field samples, respectively, are within 5 px. These are model-input coordinates. The heterogeneous field root mean square error after direct bilinear projection is still 12.84 px, and its Jacobian error is 1.91, so its gradients are not sufficiently accurate for direct IC-GN initialization at 4K scale.

To test a coarse-to-fine path, a zero-mean normalized cross-correlation (NCC) search was added around each predicted displacement. The convolutional neural network (CNN) predicts the search centre; NCC scores local intensity patches and uses a parabolic peak fit for subpixel output. The original path runs in CPU PyTorch on the 192 x 108 downsampled images. An optional WGSL implementation evaluates the same bounded search in three GPU passes: prepare reference patches, score candidate offsets in parallel, then reduce each point to its best and second-best score. The 4K images are still downsampled before this local search; IC-GN alone operates on the original 3840 x 2160 arrays. Inputs are copied from NumPy to WebGPU and results are read back, so this stage is not GPU-resident end to end.

The first stress pair, whose texture and motion seeds overlap the validation partition, yielded 867/1,560 correct translation windows (55.6%) and 0/871 heterogeneous windows. This pair is diagnostic and is not counted in the independent result.

The independent UHD run used four new texture/motion pairs per family (seed offset 200000), with all requested subset centres activated and analytic visibility used only for scoring. Training, validation, and test texture seeds are disjoint. The synthetic generator scales the characteristic speckle to 140 native pixels (seven pixels at model input), so this is a scale-consistency stress test, not a realistic claim about typical physical speckle sizes at 4K.

| Family | Correct IC-GN windows | Visible windows | Coverage | Median correct windows/s |
|---|---:|---:|---:|---:|
| Translation | 2,933 | 6,549 | 44.8% | 1,593 |
| Heterogeneous | 26 | 4,540 | 0.6% | 12.5 |

The four translation pairs individually yield 544/1,224, 737/1,829, 829/1,736, and 823/1,760 correct windows. Heterogeneous pairs yield 0/1,086, 0/1,091, 11/1,173, and 15/1,190. The matching score for some failed heterogeneous nodes remains high while their median native displacement error is hundreds of pixels; score alone is not a reliable acceptance criterion. The analytic local-affine solver control converges on most visible windows, so these failures are due to the initializer, not a general IC-GN inability.

The original timing includes CPU model inference, CPU NCC, CPU field projection, and WebGPU IC-GN. It excludes synthetic image generation. A paired single-case comparison on the same 4K translation pair found 0.802 s versus 0.092 s for local NCC (8.7 times lower measured wall time), with 559 versus 558 correctly converged windows out of 1,224 visible. This is a single-pair comparison, not a stable performance guarantee. On four new pairs per family, the parallel WGSL path took a median 8.9 ms for local refinement. The neural model and field projection remain on CPU, and inputs are uploaded for every pair. This is not a GPU-resident or real-camera qualification.

### Current technical conclusions

1. The compact multi-resolution spline can represent the tested continuous displacement fields, but this capacity result does not imply that the image encoder can infer them.
2. Pair-conditioned global matching learns translation on held-out speckles. Its present heterogeneous predictions are not accurate enough at the native scale.
3. Local NCC can convert a useful coarse translation hypothesis into a strong IC-GN initialization on some 4K synthetic pairs. The same procedure fails on almost all independent heterogeneous windows.
4. The physical intensity and advected-gradient term did not improve the first small validation ablation. No full multi-scale physical-loss training has been qualified.
5. Pair-model inference and field conversion remain CPU-side. Local NCC and IC-GN can run on WebGPU, but image/field uploads and compact output reads still cross the CPU/GPU boundary.
6. A full 4K neural field, multi-hypothesis heterogeneous search, real-image evaluation, inference WebGPU port, and native-pixel refinement remain unimplemented.

### P5: parallel WebGPU local refinement and independent 4K qualification

The three-pass WGSL NCC implementation was compared against the CPU PyTorch reference on deterministic translation data. The test checks the selected displacement and score; all seven NDF-focused tests pass on the available software adapter. One work item per reference node prepares source patches, one per node/candidate scores offsets in parallel, and one per node selects the best and runner-up scores and fits a subpixel peak. Each call still reallocates buffers and uploads both downsampled images, the initial field, and parameters. Persistent GPU-resident model outputs and buffer reuse remain future work.

A dedicated warm benchmark then used the Intel Lunar Lake adapter, 1,296 query points on a 48 x 27 grid, 192 x 108 random grayscale images with a known (3, 2) pixel translation, an eight-pixel search radius, and a five-pixel patch radius. It discarded one warm-up and measured seven calls, including image/field uploads and compact result reads:

| Implementation | Median time | 90th percentile | Agreement with CPU result |
|---|---:|---:|---:|
| CPU PyTorch reference | 0.937 s | 1.035 s | Reference |
| Serial WGSL | 19.24 ms | 20.83 ms | 98.8% within 0.1 px on valid points |
| Parallel WGSL | 7.36 ms | 8.33 ms | 98.8% within 0.1 px on valid points |

The parallel shader is 127 times faster than this CPU reference in the measured microbenchmark and 2.61 times faster than the serial WGSL shader. Validity masks agreed exactly; the 99th-percentile endpoint difference was 0.94 px, mostly from peak ambiguities and parabolic subpixel interpolation. This is a local-kernel benchmark, not an end-to-end neural initializer measurement.

On the same 4K synthetic translation pair (seed offset 200000), local NCC took 0.802 s on CPU and 0.092 s on the Intel Lunar Lake (LNL) integrated graphics adapter through Vulkan WebGPU, while IC-GN correctness differed by one window (559/1,224 versus 558/1,224). The first GPU call includes setup costs. Repeated cases usually took 8–10 ms for NCC, so the single-pair timing is not directly comparable to warmed repeated timings.

An independent four-pair-per-family campaign used seed offset 300000 and the 256-pair checkpoint trained on 192 x 108 images at native scale 1. All 5,670 visible translation windows and 4,173 visible heterogeneous windows were scored with analytic truth only after prediction and solver execution. Every requested window was activated; inference did not receive analytic visibility.

| Coarse estimate | Motion family | Correct IC-GN / visible | Coverage | Median correct windows/s |
|---|---|---:|---:|---:|
| Soft correspondence expectation | Translation | 2,641 / 5,670 | 46.6% | 4,972 |
| Soft correspondence expectation | Heterogeneous | 9 / 4,173 | 0.22% | 0 |
| Hard top-one correspondence | Translation | 1,479 / 5,670 | 26.1% | 4,455 |
| Hard top-one correspondence | Heterogeneous | 10 / 4,173 | 0.24% | 0 |

The hard top-one variant is worse on translation and does not help heterogeneous motion, so the soft field remains the preferred baseline. The heterogeneous initializer is a scientific failure: a high NCC score is sometimes assigned to a location hundreds of native pixels from the physical match, and the analytic IC-GN control converges on most of these windows. The model confidence is not calibrated for rejection.

The pair model is still inferred on CPU. Its coarse matcher searches features on a stride-four grid, and NCC searches only within an eight-pixel radius at the downsampled model resolution. Native 4K pixels are used by IC-GN, not by the NCC refinement. The synthetic texture uses seven-pixel speckles at model resolution (140 pixels on the 4K raster), so this is a scale-consistency test rather than a realistic speckle-size qualification.

The extended 256-training / 64-validation run at native scale 1 is recorded at `results/ndf_dic_20261109/p2_native192_speckle7_256pairs/training.json`. Best validation has 98.1% hard top-one matches within 5 model pixels for translation and 77.8% for heterogeneous motion; the heterogeneous 90th-percentile flow error remains 17.5 model pixels. This validation gain has not translated into heterogeneous 4K IC-GN convergence.

### P6: rasterization-aware training and feature-guided refinement

The rasterization-aware run synthesized 256 training and 64 validation pairs at eight times the 192 x 108 model dimensions, then area-downsampled them before input. It completed 60 epochs on the CPU in about 16 minutes. The best validation heterogeneous median flow error was 19.18 pixels in the eight-times native coordinate system (about 2.40 model-input pixels); the 90th percentile was 126.65 native pixels (about 15.83 model-input pixels). The raw bilinear field root mean square error was 85.44 native pixels (10.68 model-input pixels), and its Jacobian error was 1.72. Training and validation textures do not overlap the test seeds. This tests rasterization effects, but is below the 20-fold reduction used for the 3840 x 2160 to 192 x 108 qualification.

The four-pair-per-family independent test at seed offset 400000 yielded 2,830/6,129 (46.2%) correctly converged translation windows and 0/5,012 heterogeneous windows. Rasterization-aware training did not improve the target failure.

I also measured a learned-feature local search before the WebGPU NCC refinement, using a search radius of eight cells on the same 48 x 27 feature grid. On a paired four-pair test at seed offset 500000, adding this step changed translation convergence from 2,464/5,819 (42.3%) to 3,155/5,819 (54.2%), a gain of 691 windows (11.9 percentage points). Heterogeneous convergence remained 0/5,193 in both runs. The learned-feature step is CPU-side and adds compute; its raw feature scores are not yet reliable enough to serve as confidence. This is a useful translation ablation, but it does not address the main heterogeneous failure.

### P7: robust global-affine and local-consensus controls

To isolate the effect of projecting the pair model's predicted field, the paired four-case-per-family test at seed offset 500000 was repeated with two geometric controls. The global-affine path uses deterministic random sample consensus (RANSAC) followed by Huber iteratively reweighted least squares. It fits one global affine map to the predicted node positions; it is therefore a global-motion control, not evidence that the non-linear Neural Deformation Field (NDF) has been recovered.

| Projection of the same pair-model prediction | Family | Correctly converged / visible | Coverage | Median correct windows/s |
|---|---|---:|---:|---:|
| Robust global affine | Translation | 5,426 / 5,819 | 93.25% | 27,170 |
| Robust global affine | Heterogeneous | 0 / 5,193 | 0% | 0 |
| Local consensus (20-cell radius) | Translation | 2,022 / 5,819 | 34.75% | 713 |
| Local consensus (20-cell radius) | Heterogeneous | 0 / 5,193 | 0% | 0 |

The global affine control recovers most translation windows, showing that the coarse predictions contain useful global-motion information for this family. It fails on every heterogeneous window. Local consensus does not rescue heterogeneous motion and reduces translation coverage while costing roughly half a second per pair in this run. Neither projection replaces a reliable non-linear field estimator.

The RANSAC result is a post-hoc geometric control, not an inference oracle: analytic truth is used only to score the returned field and IC-GN outcomes. All 5,193 visible heterogeneous windows remain in the denominator. Raw outputs are `results/ndf_dic_20261109/p4_uhd_global_affine_ransac_4pairs_seed500000.json` and `results/ndf_dic_20261109/p4_uhd_local_consensus_projection_4pairs_seed500000.json`.

### P8: training at the exact 4K-to-model raster scale

The earlier rasterization-aware training generated pairs at eight times the model dimensions, while the 3840 x 2160 qualification downsamples by a factor of twenty. To test this domain mismatch, a separate model was trained on 128 texture-disjoint pairs generated at 3840 x 2160 and area-downsampled to 192 x 108, with 32 independent validation pairs and 60 epochs. Each coarse query compares 1,296 target cells. The measured training phase took 274 seconds on the central processing unit (CPU); this excludes the expensive image-pair generation performed before the timer starts.

The best validation checkpoint reached a heterogeneous median flow error of 85.4 native pixels, a 90th percentile of 464.8 pixels, 57.7% correct coarse target cells, and raw-grid flow root mean square error of 286.3 pixels. It therefore remained too inaccurate for local DIC refinement.

The checkpoint was then evaluated on four new translation and four new heterogeneous 4K pairs (seed offset 600000), with the same 8-pixel model-grid local normalized cross-correlation (NCC), projection, and WebGPU IC-GN settings as the previous qualification:

| Training scale | Family | Correct IC-GN / visible | Coverage | Median correct windows/s |
|---|---|---:|---:|---:|
| 8x, earlier checkpoint | Translation | 2,830 / 6,129 | 46.2% | 5,719 |
| 20x, exact-raster checkpoint | Translation | 3,204 / 6,811 | 47.0% | 12,761 |
| 8x, earlier checkpoint | Heterogeneous | 0 / 5,012 | 0% | 0 |
| 20x, exact-raster checkpoint | Heterogeneous | 0 / 5,665 | 0% | 0 |

The exact-raster training does not improve heterogeneous convergence. The analytic IC-GN controls converged correctly on 5,431 of the 5,665 visible heterogeneous windows, so the zero result is attributable to the initializer in this test rather than a lack of solver capacity. Translation coverage is essentially unchanged. This closes the specific rasterization-mismatch hypothesis; it does not close the broader data-diversity or matching-architecture questions.

On the same exact-raster checkpoint and seed set, switching from the soft correspondence expectation to the hard top-one target cell reduced translation convergence to 2,352/6,811 (34.5%) and produced only 1/5,665 heterogeneous windows (0.02%). The lone success is not a robust improvement. This rules out soft-mode averaging as the sole cause of the heterogeneous failure; the soft version remains the better translation baseline.

A top-eight probability-weighted estimate, which discards low-probability global matches but keeps several candidates, reached 2,697/6,811 translation windows (39.6%) and 1/5,665 heterogeneous windows (0.02%). It also underperformed the soft estimate for translation and did not change the conclusion for heterogeneous fields.

### P9: contextual feature support

To test whether the coarse pair matcher lacked enough local context, an optional residual feature branch increased its theoretical receptive field from about 17 to 41 model-input pixels. A model trained on 128 texture-disjoint scale-eight pairs and 32 validation pairs improved held-out heterogeneous validation median flow error to 11.03 native pixels (90th percentile 73.73 pixels; top-one target-cell accuracy 64.06%). However, the independent 4K qualification (four pairs per family, seed offset 700000) still produced 0/3,686 correctly converged heterogeneous windows. Translation reached 3,135/6,398 (49.0%). The analytic IC-GN control converged on 3,460/3,686 heterogeneous windows, indicating that the initializer remains the limiting component. The context branch alone is not a solution to the heterogeneous-field failure.

The encoder, matching, and field projection remain CPU PyTorch; bounded local normalized cross-correlation (NCC) and inverse-compositional Gauss-Newton (IC-GN) execute through WebGPU. This is not a GPU-resident neural pipeline. Raw results are `results/ndf_dic_20261109/p2_context41_scale8_128pairs/training.json` and `results/ndf_dic_20261109/p4_uhd_context41_4pairs_seed700000.json`.

The same contextual architecture trained at the exact twenty-fold raster scale (128 training pairs, 32 validation pairs, 60 epochs) achieved 64.7% heterogeneous top-one cell accuracy, but only 3.3% of validation cells were within 5 pixels and the median heterogeneous flow error was 26.9 native pixels. On four independent 4K cases per family (seed offset 800000), it converged 2,649/6,080 translation windows (43.6%) and 0/4,547 heterogeneous windows. The analytic IC-GN control converged on 4,305/4,547 heterogeneous windows. This is a negative qualification of the exact-scale contextual variant; its translation coverage is below the earlier scale-eight context result and the runs use different seeds. Full results: `results/ndf_dic_20261109/p2_context41_scale20_128pairs/training.json` and `results/ndf_dic_20261109/p4_uhd_context41_scale20_4pairs_seed800000.json`.

The scale-consistent synthetic image generator now switches to blocked fast Fourier transform convolution for Gaussian kernels with radii of at least 32 pixels. On a 3840 × 2160 random image, the 56-pixel-standard-deviation blur ran 14.7 times faster than the prior direct method, with maximum absolute output error 2.98×10⁻⁸. The training script can cache deterministic low-resolution train and validation tensors separately for matched ablations. The exact-scale contextual training above predates these changes and used the slower generator. A parity test and cache smoke check pass.

An exact-scale ablation then added the single-scale photometric and advected-gradient training loss at weight 0.02. It used the same initialization, cached training and validation pairs, and 24 epochs as the weight-zero control. On heterogeneous validation, median flow error changed from 47.38 to 48.20 native pixels, the within-5-pixel fraction from 1.58% to 1.45%, and raw-grid flow root mean square error from 279.78 to 283.32 pixels. Translation median error changed from 19.08 to 19.71 pixels. This physical term did not improve the validation gate; no 4K IC-GN qualification was run for the rejected variant. Raw files: `results/ndf_dic_20261109/p2_context41_scale20_128pairs/training.json`, `p2_context41_scale20_physics002_128pairs/training.json`, and the shared deterministic pair cache in `p2_context41_scale20_shared_cache/`.

A local learned-feature search of radius 8 feature cells was also evaluated on the same independent 4K pairs as the exact-scale contextual checkpoint. Translation convergence fell from 2,649/6,080 to 2,181/6,080; heterogeneous convergence remained 0/4,547. Feature matching reduced the median heterogeneous candidate-field error, but the following local NCC returned it to 144.48 pixels. The sequential feature-plus-NCC combination is therefore not useful in this setting. Raw paired result: `results/ndf_dic_20261109/p4_uhd_context41_scale20_feature8_4pairs_seed800000.json`.

An affine-compensated local NCC ablation then warped target patches by the model-predicted Jacobian, with no analytical truth available to inference. Full compensation changed the paired outcome to 1,331/6,080 translation windows and 7/4,547 heterogeneous windows; a hybrid that chose between fixed and warped patches obtained 2,383/6,080 and 7/4,547. Adding the radius-eight feature search produced 2,181/6,080 and 13/4,547, with all thirteen heterogeneous successes concentrated in one of four pairs. A paired 50%-damped Jacobian plus feature search produced 2,124/6,080 translation successes and 0/4,547 heterogeneous successes. The patch geometry prior can reduce errors in an individual heterogeneous case, but it is not yet a repeatable initializer and can reduce translation coverage. Full data and interpretation: `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`; raw outputs are the `affine_ncc`, `affine_d025`, `affine_d05`, `hybrid_ncc`, and `feature8_hybrid` files under `results/ndf_dic_20261109/`.

The P1 field-capacity gate was also rerun independently of matching: fitting exact displacement samples to a 3-scale cubic B-spline field (852 coefficients) reproduced smooth heterogeneous waves and a 24-pixel localized band with 0.022–0.273 pixel displacement RMSE, and corresponding Jacobian RMSE below 0.032. A 36-pixel displacement jump remained smoothed (3.42 pixel RMSE), as expected for a globally continuous basis. This supports the continuous representation but is an oracle fit, not evidence that the image-pair network can recover the field. Gradient/autodiff checks were near machine precision. Results: `results/ndf_dic_20261109/p1_field_capacity_512x384.json` and the full analysis in `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`.

A texture-scale diagnostic on the experimental reference image found a radial half-correlation length of 6.44 pixels natively and 0.56 pixels after the current 20× reduction to 192 × 108. The synthetic training texture retained 4.48 pixels after the same resize because its characteristic native speckle size was 140 pixels. The input therefore preserves much more texture in training than in the real image, and its feature grid has an 80-native-pixel stride. This is a major experimental-domain mismatch; a useful next model must preserve high-resolution texture while avoiding an unbounded global all-pairs search. Diagnostic and per-scale profiles: `results/ndf_dic_20261109/texture_scale_adrien_reference.json` and `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`.

A paired 4K-scale synthetic test then compared the model trained on 140-pixel speckles with a 48-pair, 20-epoch pilot trained on 7-pixel speckles, both reduced to the same 192 × 108 input. On identical held-out seeds, heterogeneous IC-GN successes increased from 42/5,465 to 820/5,465, while translation successes increased from 190/5,010 to 2,802/5,010. The gain is concentrated in two of four heterogeneous pairs; two remain complete failures, and the training budgets differ. This is strong evidence that texture-scale mismatch materially harms the current predictor, but the pilot is not robustly qualified. Raw paired outputs and caveats are in `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`.

A controlled 128-pair/60-epoch ablation then fixed the model initialization seed, generated-pair seeds, architecture, and solver, and changed only the synthetic speckle scale from 140 to 7 native pixels. On the same held-out four-pair test set, the physical-scale checkpoint raised correctly converged heterogeneous windows from 42/5,465 to 1,264/5,465 (23.1%); translation rose from 190/5,010 to 3,766/5,010 (75.2%). Two heterogeneous pairs still had zero successes. The improvement demonstrates the importance of matching texture scale during training, while the limited test size and continued pair failures prevent a robustness claim. Full paired data: `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`.

The subsequent ten-pair-per-family test, again paired on identical seeds, is the more reliable current result: physical-scale training reached 897/10,643 heterogeneous windows (8.4%) and 11,316/14,891 translation windows (76.0%), versus 14/10,643 and 411/14,891 for the 140-pixel training texture. Only two of ten heterogeneous fields had any correctly converged windows, with one field contributing 877 of the 897 successes. The texture-scale correction is therefore a major improvement over the mismatched baseline, but the NDF-DIC prototype remains unreliable under heterogeneous deformation. Raw test files: `results/ndf_dic_20261109/p4_uhd_physical_texture_128pairs_seed20261112_10pairs_seed1000000.json` and `results/ndf_dic_20261109/p4_uhd_training_texture_on_physical_10pairs_seed1000000.json`.

An oracle-only rotation diagnostic on the same ten pairs removed the known global synthetic rotation from each target before matching, then mapped the predicted field back before IC-GN. Heterogeneous convergence rose from 897/10,643 (8.4%) to 1,447/10,643 (13.6%); all ten pairs had at least one success, although one pair lost 77 windows. Translation counts were unchanged. Because this uses the known synthetic angle, it is evidence that orientation is one significant nuisance, not an image-only solution. The remaining heterogeneous coverage is still poor. The corrected diagnostic and its caveats are documented in `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`.

A no-oracle sparse rotation-bank experiment then selected a global rotation from the images alone: 36 orientations were ranked using 256 spatially distributed reciprocal feature queries and global affine consensus. On ten new heterogeneous pairs, it raised correctly converged windows from 1,089/12,100 (9.00%) to 1,751/12,100 (14.47%), while leaving the paired translation count effectively unchanged (10,129 versus 10,130). Six heterogeneous pairs improved, three were unchanged, and one lost 161 windows. A coarser 18-angle bank recovered 1,594/12,100 (13.17%), 76% of the 36-angle bank's net gain, at lower cost. A corrected reference run uses the same 40-pixel consensus radius as both banks. Across the heterogeneous set the full timed rate fell from 1,831 correct windows per second to 1,238 with 18 angles and 908 with 36 angles; summed pipeline time rose from 0.595 seconds to 1.288 and 1.929 seconds, respectively. The experiment therefore demonstrates a useful quality-throughput trade-off, not a faster or generally reliable solution. Feature inference and hypothesis scoring remain CPU PyTorch; local correlation and IC-GN use WebGPU, while inputs to the learned model are reduced to 192 x 108. Details, per-pair deltas, limitations, and raw JSON files are documented in `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`.

A paired search-radius ablation on the same eight test pairs increased the local NCC radius from 8 to 24 model pixels. Translation changed only from 3,204 to 3,239 correct windows (47.0% to 47.6%); heterogeneous coverage remained 0/5,665. Heterogeneous median local endpoint errors worsened in all four cases. Per-case NCC refinement time rose from roughly 8–10 ms to 37–41 ms. The larger search window admits more ambiguous speckle matches and is not a useful setting.

Two independent frozen-gate validations (20 pairs per family total) found that image-derived global affine preconditioning plus a residual field raised heterogeneous IC-GN coverage to 8,056/24,396 (33.02%), versus 2,484/24,396 (10.18%) for the matched no-search baseline. Correct results appeared in 19/20 pairs rather than 7/20, and aggregate throughput rose from 1,187 to 2,927 correctly converged windows per timed second. The result generalized across both held-out seed sets, but one of 20 heterogeneous pairs regressed and absolute coverage remains far below the analytic IC-GN ceiling. Translation coverage was essentially unchanged (21,462/28,649 versus 21,560/28,649) while throughput fell from 21,866 to 8,302 because the rotation search and affine gate add cost; one false affine prefit also cost 97 correct windows. This is evidence for a useful heterogeneous preconditioner, not a general NDF qualification: matching still runs on reduced 192 x 108 CPU tensors. An initial four-pair multi-scale advection-loss pilot did not improve heterogeneous validation or IC-GN coverage (191/5,717 versus 218/5,717); the larger matched follow-up below supersedes that preliminary estimate. Local affine-to-spline preconditioning was rejected by its Jacobian-folding guard on the smoke cases. Full statistics and limits, plus paired-results figures, are in `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md`, `affine_prefit_heldout_comparison.png`, and `multiscale_physical_loss_heldout_comparison.png`.

A controlled 128-pair/60-epoch comparison then trained the same pair matcher with and without a three-level intensity and advected-gradient loss, using identical initialization, cached training and validation pairs, and frozen settings. In the first 20 independent held-out pairs per family, heterogeneous convergence increased from 893/22,682 (3.94%) to 1,241/22,682 (5.47%); 10 cases gained, 10 tied, none lost. A third seed group extended the comparison to 30 pairs per family: heterogeneous convergence rose from 1,376/35,893 (3.83%) to 1,953/35,893 (5.44%), with 14 gains, 16 ties, and no losses. The total gain is concentrated: the median per-case change is zero, and one case contributes +139. Translation coverage over 30 pairs rose from 33,248/42,683 (77.90%) to 35,679/42,683 (83.59%), with 29 gains and one loss of 3 windows. Aggregate throughput changed from 777 to 1,115 correctly converged heterogeneous windows per timed second, and 22,687 to 25,337 for translation. This reverses the small pilot's direction, but absolute heterogeneous coverage remains poor. Inputs are still reduced to 192 x 108 for the CPU PyTorch model; only local correlation and IC-GN use WebGPU. The result supports continuing the physical-loss line, not calling the model a robust or full-GPU DIC initializer. See `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md` for validation metrics, raw runs, and limitations.

A fourth paired seed group compared direct evaluation of the learned displacement grid with one- and three-level cubic B-spline projections and local affine consensus. Every smoothed projection reduced IC-GN coverage; direct grid evaluation remains the best current choice. The consensus variant also increased the measured pipeline time substantially. The per-case results and raw data are in the detailed qualification report.

The strongest current result combines the multi-scale checkpoint with an image-only 18-angle rotation probe, a gated robust affine prefit, and one learned nonlinear residual prediction. Across 20 additional heterogeneous pairs, correct IC-GN coverage rose from 1,874/23,752 (7.89%) to 8,480/23,752 (35.70%); 19 cases improved and one tied. Correct windows per timed second rose from 1,360 to 2,714. Translation coverage stayed at about 82.7%, but throughput fell because the rotation probe costs extra time. No analytic transformation entered inference. This supports coarse geometric preconditioning followed by a learned nonlinear residual, while leaving GPU-resident inference and broad heterogeneity coverage unresolved. The paired raw comparison is `results/ndf_dic_20261109/p3_multiscale_rotation18_affine_prefit_paired_20pairs_seed11000000_12000000.json`.

The rotation-search option inserts identity when the requested grid does not contain zero. Thus `rotation-bank-steps=18` evaluates 18 orientations, while `rotation-bank-steps=9` evaluates 10 (the nine-point uniform grid plus identity); the historical “9-angle” label was imprecise. That 10-orientation search preserved translation results exactly and reduced heterogeneous coverage by only 29/24,595 windows versus 18 orientations, while improving correct-window throughput by 19–24% across the two families. A new two-seed comparison evaluated an adaptive eight-probe strategy: six coarse orientations followed by two half-step probes around the highest-scoring orientation. Across 20 paired cases per family, it exactly matched translation coverage (24,053/28,514) and lost 40 heterogeneous windows versus the 10-orientation bank (8,766/22,416 versus 8,806/22,416, a 0.18-point difference), while increasing correct-window throughput by 7.4% and 4.8%, respectively. This is a modest speed-quality trade, not a coverage improvement. The adaptive run was not compared directly with a uniform six-orientation bank on the same seeds; the 10-orientation bank remains the simpler reference. The full results and figure are in `results/ndf_dic_20261109/webgpu_local_ncc_and_field_qualification.md` and `results/ndf_dic_20261109/rotation_adaptive_heldout_comparison.png`.

A further held-out check refined the top two coarse peaks instead of only the best peak. It tested nine or ten unique orientations rather than eight, but added zero correct IC-GN windows across ten pairs per family and reduced throughput by 1.7–3.0%. The extra peak is not retained in the default search. Details are in the qualification report and `results/ndf_dic_20261109/p3_rotation_refine_top2_vs_top1_seed18000000.json`.

## Reproducibility files

- `python/benchmarks/ndf_field.py`: differentiable sampler, cubic spline basis and Jacobian, direct bilinear grid, physical losses, and oracle fitting.
- `python/benchmarks/validate_ndf_field.py`: deterministic P0/P1 run.
- `python/benchmarks/benchmark_ndf_local_ncc.py`: warmed CPU/serial WGSL/parallel WGSL comparison, including input transfers and compact output readback.
- `python/benchmarks/ndf_pair_model.py`: shared learned features and global coarse pair correlation.
- `python/benchmarks/train_ndf_pair_matcher.py`: synthetic training, held-out evaluation, spline projection, and local-consensus diagnostic.
- `python/benchmarks/qualify_ndf_pair_icgn.py`: native-image qualification through the existing WebGPU IC-GN solver, including zero and analytic controls.
- `python/benchmarks/summarize_ndf_physics_ablation.py`: paired aggregation and figure for the matched multi-scale physical-loss comparison.
- `python/benchmarks/ndf_pair_model.py`: pair matcher, local learned-feature search, and local NCC refinement.
- `results/ndf_dic_20261108/p0_p1_capacity_512x384.json`: raw measurements.
- `results/ndf_dic_20261108/p0_p1_capacity_1024x576.json`: higher-resolution run, when present.
- `results/ndf_dic_20261108/p2_extended_20261108/training.json`: held-out coarse-grid training measurements.
- `results/ndf_dic_20261108/p2_native_scale8_20261108/training.json`: large-motion scaled-image measurements.
- `results/ndf_dic_20261108/p3_pair_baseline_20261108/training.json` and `p3_pair_physics_002_20261108/training.json`: matched physical-loss ablation.
- `results/ndf_dic_20261108/p4_icgn_smoke.json` and `p4_icgn_controls_smoke.json`: initial single-pair smoke tests.
- `results/ndf_dic_20261108/p4_icgn_20261108.json`: eight-pair-per-family IC-GN qualification with zero and analytic controls.
- `results/ndf_dic_20261108/p4_icgn_raw_bilinear_20261108.json` and `p4_icgn_single_spline_20261108.json`: paired field-projection comparison.
- `results/ndf_dic_20261108/p4_uhd_smoke.json`: failed 3840 x 2160 stress test at forty-fold downsampling.
- `results/ndf_dic_20261108/p4_uhd_independent_localncc_radius8.json`: independent four-pair-per-family 4K synthetic qualification with local NCC and WebGPU IC-GN.
- `results/ndf_dic_20261109/p2_native192_speckle7_256pairs/training.json`: extended 256-pair training and validation.
- `results/ndf_dic_20261109/p2_native_scale8_256pairs/training.json`: rasterization-aware training and validation at native scale 8.
- `results/ndf_dic_20261109/p4_cpu_refinement_smoke.json`: CPU NCC comparison case.
- `results/ndf_dic_20261109/p4_uhd_wgpu_parallel_smoke.json`: first parallel WGSL NCC smoke test.
- `results/ndf_dic_20261109/p4_uhd_independent_wgpu_parallel_4pairs.json`: independent qualification with soft coarse flow.
- `results/ndf_dic_20261109/p4_uhd_independent_wgpu_parallel_hard_4pairs.json`: paired hard top-one coarse-flow ablation.
- `results/ndf_dic_20261109/p4_uhd_rasteraware_wgpu_parallel_4pairs.json`: independent 4K evaluation of the rasterization-aware checkpoint.
- `results/ndf_dic_20261109/p4_uhd_rasteraware_wgpu_parallel_4pairs_seed500000.json`: baseline of the paired local-feature search experiment.
- `results/ndf_dic_20261109/p4_uhd_rasteraware_feature_local_ncc_4pairs.json`: same seeds with local learned-feature search before WGSL NCC.
- `results/ndf_dic_20261109/p4_uhd_global_affine_ransac_4pairs_seed500000.json`: robust global affine projection control on the paired seed set.
- `results/ndf_dic_20261109/p4_uhd_local_consensus_projection_4pairs_seed500000.json`: local-consensus projection control on the paired seed set.
- `results/ndf_dic_20261109/p2_native_scale20_128pairs/training.json`: exact-raster 128-pair training and independent validation.
- `results/ndf_dic_20261109/p4_uhd_scale20_train_4pairs_seed600000.json`: independent 4K IC-GN qualification of that exact-raster checkpoint.
- `results/ndf_dic_20261109/p4_uhd_scale20_train_hard_4pairs_seed600000.json`: paired hard top-one ablation on the same checkpoint and test seeds.
- `results/ndf_dic_20261109/p4_uhd_scale20_train_topk8_4pairs_seed600000.json`: paired top-eight probability-weighted estimate ablation.
- `results/ndf_dic_20261109/p4_uhd_scale20_train_radius24_4pairs_seed600000.json`: paired 24-pixel local-search-radius ablation on the same checkpoint and test seeds.
- `results/ndf_dic_20261109/local_ncc_warm_benchmark.json`: seven-repeat warmed local NCC microbenchmark on Intel LNL.
- `results/ndf_dic_20261109/p4_physical_texture_baseline_consensus40_10pairs_seed6000000.json`, `p4_rotation_bank18_global_probe_10pairs_seed6000000.json`, and `p4_rotation18_affine_prefit_gated003_10pairs_seed6000000.json`: frozen, held-out comparison of the no-search, rotation-bank, and image-derived global affine prefit pipelines.
- `results/ndf_dic_20261109/p4_physical_texture_baseline_consensus40_10pairs_seed7000000.json` and `p4_rotation18_affine_prefit_gated003_10pairs_seed7000000.json`: independent replication with the same frozen checkpoint and affine gates.
- `results/ndf_dic_20261109/p3_matched_single_scale_control_seed20261113/training.json` and `p3_multiscale_physics_weight002_levels3_seed20261113/training.json`: matched single-scale and three-level physical-loss training runs.
- `results/ndf_dic_20261109/p3_matched_single_scale_control_icgn_4pairs_seed7000000.json` and `p3_multiscale_physics_icgn_4pairs_seed7000000.json`: paired four-pair-per-family IC-GN pilot on new texture seeds.
- `results/ndf_dic_20261109/p4_local_bspline_prefit_smoke_seed5000000.json`: local affine consensus to B-spline prefit folding diagnostics; guard rejected the fit.
