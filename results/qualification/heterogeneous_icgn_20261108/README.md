# Heterogeneous matching and seeded IC-GN qualification — 8 November 2026

## Scope and method

This first end-to-end study connects the learned DIC keypoint detector, the official DeDoDe-B descriptor exported as a fixed-shape ONNX graph, reciprocal cosine matching in WebGPU Shading Language, local affine consensus, affine or Gaussian radial-basis-function (RBF) field projection, and the native WebGPU inverse-compositional Gauss–Newton (IC-GN) solver from the separate `tests_gpuio/DIC/LucasKanadeICGN` project. That project was read and used as an installed dependency; its files were not changed.

Each 1,024 × 576 pair is generated from the same deterministic speckle texture with analytic motion. Six fixed seed pairs cover one affine control, two smooth heterogeneous fields, two discontinuous fields with a gap, and one rigid rotation. The matcher sees only image pixels; analytic flow and deformation are used by the evaluator after matching. There are 144 IC-GN windows per case, each with a 64 × 64 pixel support and 64 pixel spacing. We report the subset of windows whose reference-to-target centres remain visible.

The `zero_initialization` row is a zero-warp IC-GN control. `analytic_local_affine_oracle` seeds the solver from the exact local affine approximation of the generator's analytic flow and provides a solver ceiling. It is not a matcher. Final success means the solver accepted a window and its final centre translation is within 2 pixels of analytic truth. The affine-gradient parameter error is reported separately from displacement error.

The initialization adapter calls the solver's existing private seeded-frame method, which is also used internally by its pyramid path. The solver's public `register()` API does not yet accept a caller-provided six-parameter initial field. This experiment therefore validates the actual existing WGSL IC-GN kernel, while recording the private adapter as an integration limitation. No external solver source was modified.

## Results

`Direct correct <5 px` counts measured source-target pairs against analytic flow before consensus. `Consensus-kept` counts pairs retained by the same local affine filter. The final column gives correctly converged windows after IC-GN using the Gaussian RBF projected initialization, out of all visible windows. A failed field fit contributes zero. Full per-case affine and RBF outcomes are in `summary.json` and each case JSON.

| Motion and seeds | Visible windows | Method | Direct matches | Direct correct <5 px | Consensus-kept | Correct RBF + IC-GN |
|---|---:|---|---:|---:|---:|---:|
| Affine 8101/9101 | 99 | DeepMatching | 2,048 | 0 | 0 | 0/99 |
|  |  | HORG native | 14 | 3 | 0 | 0/99 |
|  |  | EfficientLoFTR | 108 | 5 | 0 | 0/99 |
|  |  | DIC-Detector + DeDoDe-B | 549 | 33 | 0 | 0/99 |
|  |  | DIC-Detector + HORG-64 | 719 | 3 | 0 | 0/99 |
| Heterogeneous 8102/9102 | 65 | DeepMatching | 2,048 | 1 | 0 | 0/65 |
|  |  | HORG native | 96 | 1 | 0 | 0/65 |
|  |  | EfficientLoFTR | 434 | 4 | 0 | 0/65 |
|  |  | DIC-Detector + DeDoDe-B | 408 | 6 | 0 | 0/65 |
|  |  | DIC-Detector + HORG-64 | 418 | 0 | 0 | 0/65 |
| Heterogeneous 8103/9103 | 77 | DeepMatching | 2,048 | 1,174 | 1,098 | 64/77 |
|  |  | HORG native | 32 | 0 | 0 | 0/77 |
|  |  | EfficientLoFTR | 1,444 | 1,427 | 1,424 | 64/77 |
|  |  | DIC-Detector + DeDoDe-B | 850 | 733 | 735 | 65/77 |
|  |  | DIC-Detector + HORG-64 | 625 | 0 | 0 | 0/77 |
| Discontinuous 8104/9104 | 84 | DeepMatching | 2,048 | 1,104 | 1,104 | 29/84 |
|  |  | HORG native | 15 | 2 | 0 | 0/84 |
|  |  | EfficientLoFTR | 17 | 13 | 13 | 24/84 |
|  |  | DIC-Detector + DeDoDe-B | 804 | 683 | 671 | 32/84 |
|  |  | DIC-Detector + HORG-64 | 550 | 4 | 0 | 0/84 |
| Discontinuous 8105/9105 | 91 | DeepMatching | 2,048 | 1,104 | 1,105 | 48/91 |
|  |  | HORG native | 87 | 31 | 28 | 41/91 |
|  |  | EfficientLoFTR | 1,621 | 1,596 | 1,596 | 46/91 |
|  |  | DIC-Detector + DeDoDe-B | 1,116 | 1,080 | 1,073 | 46/91 |
|  |  | DIC-Detector + HORG-64 | 644 | 456 | 456 | 57/91 |
| Rigid 8106/9106 | 106 | DeepMatching | 2,048 | 1,408 | 1,391 | 95/106 |
|  |  | HORG native | 51 | 21 | 13 | 95/106 |
|  |  | EfficientLoFTR | 2,220 | 2,209 | 2,209 | 95/106 |
|  |  | DIC-Detector + DeDoDe-B | 1,179 | 1,102 | 1,075 | 95/106 |
|  |  | DIC-Detector + HORG-64 | 720 | 35 | 0 | 0/106 |

The HORG-64 row is a descriptor ablation on the exact same DIC-Detector keypoints, not native HORG. It is a 64-component, fixed-scale variant computed with WGSL. Its paired descriptor calculation took 8–11 ms, versus 2.38–2.47 s for the 256-component DeDoDe-B ONNX/WebGPU descriptor. But HORG-64 produced too few reliable matches in five of six cases to fit a field; it only improved the discontinuous 8105/9105 case, where Gaussian RBF yielded 57/91 correct convergences versus 46/91 for DeDoDe-B. This is a strong speed-quality trade-off in one case, not a general replacement.

The Gaussian RBF field yielded 64–65 correct windows for the successful heterogeneous case and 46–57 for the stronger discontinuous case. Neither basis can rescue the affine or heterogeneous 8102/9102 cases where consensus found no usable correspondences. A broad smooth projection can also bridge true discontinuities; these six generated examples are not enough to qualify that behavior for experimental cracks.

The zero-warp control converged correctly on none of the six cases. The analytic local-affine oracle reached 45/65 to 95/106 depending on the case, so matcher results must be interpreted with solver conditioning and local affine approximation error. This six-case pilot is not a distributional estimate; one of the two smooth heterogeneous fields failed completely.

## Timings and execution placement

The run used Intel Graphics LNL. The graph profile (`../features/dedode_onnx_wgpu_20261108/descriptor_profile_summary.json`) measured a median 1.136 s per DeDoDe-B ONNX/WebGPU invocation at a fixed 1,024 × 576 input and 8,192-point capacity. Across one warmup and three measured calls, all 484 ONNX Runtime operator events were assigned to WebGPU. The summed event duration is a runtime profile metric, not hardware GPU timestamp data. The descriptor outputs are copied to NumPy and then uploaded to a separate `wgpu-py` allocation; the pipeline is not zero-copy or fully GPU-resident.

In the warmed rerun, paired DIC-Detector inference took 8–13 ms. Paired description took 2.38–2.47 s for DeDoDe-B and 8–11 ms for HORG-64. Mutual matching and local consensus took 78–109 ms for DeDoDe-B and 33–45 ms for HORG-64. DeepMatching full-pipeline medians across successful cases were 0.389 s for affine (four cases) and 1.160 s for Gaussian RBF (four cases); EfficientLoFTR medians were 1.257 s affine (four cases) and 2.542 s RBF (four cases). HORG-64 took 0.120 s affine and 0.546 s RBF in its one field-fit-success case. These are one run per case, not repeated timing medians. The total includes solver reference preparation and seeded IC-GN. The compact descriptor clearly reduces descriptor latency, but its low useful-match coverage makes the end-to-end trade-off poor in most cases. The compressed `.npz` files contain pairs, analytic errors, initialization parameters, solver parameters, status and iteration counts.

## Reproduction

The fixed-shape graph and trained detector weights are external files. Export the official DeDoDe-B graph using `python/benchmarks/export_dedode_descriptor_onnx.py`, then run:

```bash
python python/benchmarks/qualify_heterogeneous_icgn.py \
  --output results/qualification/heterogeneous_icgn_20261108 \
  --descriptor-onnx /path/to/dedode_b_descriptor_1024x576_k8192.onnx \
  --weights results/features/dic_detector_training_discontinuous_20261107/dic_detector_weights.npz
```

The default IC-GN source path points to the local `tests_gpuio` checkout and can be changed with `--live-dic-source`. The qualification script uses a private solver entry point; a future stable public initialization API is needed before this adapter can become a supported package integration.

## Next experiments

1. Add at least ten independent textures and warp seeds per motion family; keep this six-case set as a fixed regression suite.
2. Improve HORG-64 or train a compact descriptor specifically on the DIC detector's keypoints; the current generic descriptor is fast but unreliable on most fields.
3. Vary field wavelength and amplitude independently, and evaluate cracks, gap-adjacent windows, and partial visibility separately.
4. Replace the private IC-GN adapter with a versioned public seeded-registration API in the solver project, under that project's ownership and tests.
