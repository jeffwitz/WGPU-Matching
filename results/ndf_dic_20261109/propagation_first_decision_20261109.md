# Propagation-first qualification and implementation decision

## Scope

This campaign started from commit `4dce5d0` and asks whether status-only propagation fills the initialized field reliably enough to defer a learned multiscale decoder. Digital image correlation (DIC) windows use the inverse-compositional Gauss-Newton (IC-GN) solver on the native 3840 × 2160 synthetic images. The learned pair model and local normalized cross-correlation (NCC) operate on the 192 × 108 raster. NCC here means zero-mean normalized cross-correlation (ZNCC) over local patches. The propagation loop uses only the solver's accepted status and image-derived parameters. Analytic flow, visibility, and oracle solves are used after prediction for scoring.

No model, loss, projection, solver threshold, or solve setting was changed. A new analytic shear-band generator supports the held-out qualification. The implementation uses the existing WebGPU shader pipeline; there is no claim of end-to-end GPU residency for the learned pair model.

## Archived propagation baseline

The archived 80 per-window runs were replayed with frontier propagation. All four propagated-correct totals match the frozen references exactly, so gate G1 passes at zero relative error.

| Variant | One pass correct | Correct after propagation | Oracle correct | Pairs with seeds | G1 |
|---|---:|---:|---:|---:|---|
| Image-derived affine | 2,070 | 19,915 | 21,367 | 18/20 | Pass |
| Affine + local NCC | 16,266 | 20,651 | 21,367 | 19/20 | Pass |
| Affine + neural deformation field residual | 1,748 | 19,070 | 21,367 | 17/20 | Pass |
| Affine + neural deformation field residual + local NCC | 13,172 | 19,917 | 21,367 | 18/20 | Pass |

Gate G2 permits no more than 0.1% accepted visible windows with at least 2 px error. It passes in every variant. One false accepted seed occurs for image-derived affine only: pair `16051009_16091009`, window 1587 at `(1759.5, 1695.5)` px, error 9.466 px. It did not propagate. No accepted invisible windows were observed.

The implementation is in `qualify_ndf_pair_icgn.py`; the report contains ring totals, per-pair timing, accepted-error coordinates, and reproduction instructions in `causal_ablation_frontier_corrected_20261109/README.md`.

## Zero-seed rotation hypotheses

For pairs 17051005 and 17051007, the three highest image-only rotation scores were tested on the same evenly spaced subset of 128 windows. Every tested hypothesis produced 0 accepted IC-GN seeds. The 60° candidate ranked second on both pairs; status-only selection therefore retained the first-ranked 20° candidate by stable tie order. This experiment does not establish that either angle is usable. It shows that rotation score alone and this sparse solver check do not recover a seed for these cases.

The full scores and individual solver reports are in `zero_seed_rotation_top3_20261109/`.

## Coarse-matcher source ablation

All methods use the same local NCC, affine-fit threshold, native-pixel solver, and propagation. HORG means histogram of oriented and rotated gradients. Its keypoint detector returns too few points on the 192 × 108 matching raster to fit an affine hypothesis in all 20 pairs. DeepMatching obtains an affine hypothesis in 10 of 20 pairs.

| Source | Affine hypotheses | Pairs with seeds | Correct after propagation | Oracle | Median pipeline time |
|---|---:|---:|---:|---:|---:|
| Learned coarse matcher | 20/20 | 19/20 | 20,651 | 21,367 | 0.102 s |
| HORG | 0/20 | 1/20 | 1,099 | 21,367 | 0.083 s |
| DeepMatching | 10/20 | 10/20 | 11,989 | 21,367 | 0.101 s |

On pairs where a method finds a seed, propagation reaches 99.97–100% of the oracle-correct count. The low HORG time is not a useful speed result: 20/20 affine attempts failed at this raster, and only one pair had any accepted seed. Per-pair failure reasons, fit support, and timing are in `matcher_source_frontier_20261109/summary.json`.

## Held-out motion-family generalization

Two new seed groups were run with five pairs in each requested family and shear-band gradient. The region test uses 8-connected components of oracle-correct windows and retains components with at least 16 windows. A region passes the seed check only when it contains a first-pass solver-accepted window.

Rigid, affine, and discontinuous seeded pairs pass the 97% oracle gate. Seedless pairs remain in the results and are not credited as propagated. Both groups of shear bands with peak displacement gradient 2.0 fail gate G3: all five seeded pairs in the first group and all four seeded pairs in the second recover less than 97% of the oracle-correct windows. Four or five distinct oracle regions in each group contain a seed, so the deficit occurs after propagation starts, not only from missing initial seeds. In one second-group pair, propagation reaches 47 rings and still remains below the gate.

Gate G2 also fails for these extreme shear bands. The first group has 59 accepted visible windows with at least 2 px error among 6,142 accepted windows (0.96%); the second has 71/5,908 (1.20%). The report lists every occurrence with its window coordinate, error, and ring. All other family/gradient groups pass G2.

Seedless pairs with oracle support include:

- Discontinuous: texture seeds 18051010 (912 oracle windows) and 19051008 (1,344).
- Shear gradient 0.33: 18051007 (1,752), 18051013 (1,287), 19051007 (1,289), and 19051009 (1,656).
- Shear gradient 2.0: 19051007 (1,192 oracle windows in two retained regions).

These pairs and all per-ring attempts and acceptances are in `frontier_generalization_20261109/summary.json`. The no-seed cases are plausible targets for a predictor that can produce a field without a good local matcher seed; the results do not prove a decoder will solve them.

## Masked experimental pair

The Adrien reference/target pair was processed with HORG, an image-derived affine fit, WebGPU local NCC, then masked IC-GN and propagation. The region-of-interest mask was used only for solver reference validity, not for matching or affine selection. The run had 4,096 raw HORG correspondences, 207 accepted matcher correspondences, and 49 affine inliers. IC-GN accepted 6,593 windows in one pass and 7,103 after propagation, adding 510. Maps show accepted states and propagation rings only. There is no analytical truth, so these are counts, not accuracy measurements. See `adrien_masked_horg_propagation_20261109/`.

## Translation coordinate anomaly

Re-running the same ten raw-bilinear translation pairs before and after the half-pixel coordinate correction gives 11,033/13,444 correct visible windows before and 10,485/13,444 after. Gross losses are 553 windows and gains are 5, for a net loss of 548. Every pair regresses in net correct windows. Initial displacement root mean square error also increases in every pair, placing the change before IC-GN. The cause is the corrected sampling location on the feature grid; the old sample location was geometrically wrong but happened to give better outcomes with this checkpoint. This result does not justify reverting the coordinate convention or changing thresholds. Per-window maps and exports are in `translation_coordinate_anomaly_20261109/`.

## Decision

Propagation is highly effective after a trustworthy seed exists: the archived baseline reaches at least 99.97% of the oracle on seeded pairs, and held-out rigid/affine/moderate-shear cases usually saturate the oracle. It does not solve the high-gradient shear cases, and it cannot recover the seedless discontinuous and shear cases. Those failures make a learned field decoder a plausible next research target, but the present evidence does not justify implementing one as a general replacement. Any next decoder experiment should target post-affine residual fields and the gradient-2.0 shear/seedless cases, while preserving the current affine + NCC + propagation path as the measured baseline. No new training or decoder was part of this qualification.

## Reproduction scripts

- `benchmark_ndf_causal_ablation.py --summarize-only --propagation frontier` reproduces the archived gate summary from saved per-window outputs.
- `benchmark_ndf_zero_seed_rotation.py` verifies the six fixed-subset rotation hypotheses.
- `benchmark_ndf_matcher_source.py --summarize-only` summarizes the learned/HORG/DeepMatching comparison; omit `--summarize-only` to rerun it.
- `benchmark_ndf_frontier_generalization.py` runs the two held-out seed groups and recomputes the 8-connected-region and gate summaries.
- `benchmark_ndf_adrien_masked_counts.py` regenerates the masked experimental accepted-count maps.
- `benchmark_ndf_translation_coordinate_anomaly.py` reruns the pre-/post-coordinate per-window comparison; its report documents how to archive the pre-fix qualifier from commit `4b575de`.
