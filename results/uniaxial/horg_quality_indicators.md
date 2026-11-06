# HORG confidence indicators: exploratory study

## Question and data

Analytic displacement is used to evaluate the indicators, not to calculate them at runtime. An initialization is counted as correct when its vector error relative to analytic truth is ≤2 px.

- Initial sample: 96 POIs on archived frames 040, 100, and 199.
- Holdout sample: 18 other frames (10, 20, 30, 50, 60, 80, 90, 110, 120, 140, 150, 160, 170, 180, 185, 190, 195, and 198), for 576 POIs. Images were regenerated under `/tmp` using OpenCV remapping and stretches from `machine_truth.csv`. Regeneration reproduces archived frames 040, 100, and 199 bit-for-bit. No files in `tests_gpuio` were modified.
- All 21 frames use the same synthetic texture, homogeneous affine field, and image-generation method. This is not equivalent to validation on heterogeneous DIC, substantial rotation, or experimental images.

## Indicator tested: leave-one-out affine consistency

For each seed, an affine transform is fit from source to target coordinates using the eight nearest accepted POIs in the same frame, excluding the POI being evaluated. Its target coordinate is then predicted and compared in pixels with its own HORG correspondence. The calculation does not use analytic error; it only asks whether the local displacement agrees with already accepted neighboring POIs.

The exploratory threshold of 1.5 px was selected on the three initial frames. It gives these results:

| Sample | Current HORG filter: kept / correct ≤2 px / false >2 px | After local consistency ≤1.5 px |
|---|---:|---:|
| Initial, 96 POIs | 87 / 85 / 2 | 85 / 85 / 0 |
| Holdout, 576 POIs | 555 / 538 / 17 | 539 / 532 / 7 |
| **Total, 672 POIs** | **642 / 623 / 19** | **624 / 617 / 7** |

On the three initial frames, the two false positives have local residuals of 1.985 and 2.815 px. The largest local residual among correct POIs is 1.464 px. This appeared to provide adequate separation, but the independent holdout shows that seven false positives remain below 1.5 px. The filter identifies 10 of the 17 holdout false positives while rejecting six correct POIs. It is a useful **warning signal**, but not a reliable rejection rule on its own.

The 21 POIs that HORG had already rejected in the 18 holdout frames were all more than 14 px wrong. Current HORG selection therefore removes gross errors in this sample; remaining errors are locally plausible correspondences that differ from analytic truth by a few pixels.

## Other indicators observed

| Rule applied to already accepted POIs | Holdout retained | Correct ≤2 px | False >2 px |
|---|---:|---:|---:|
| Current HORG rule | 555 | 538 | 17 |
| Inlier count ≥7 | 499 | 497 | 2 |
| Inlier count ≥10 | 445 | 445 | 0 |
| Candidate count ≥16 | 438 | 438 | 0 |
| Inlier fraction ≥0.8 | 322 | 322 | 0 |

Absolute inlier count is the strongest simple internal indicator in this holdout. A threshold of ≥10 inliers passes no false positives among 445 POIs, but rejects 93 correct POIs compared with the current rule. It may suit a conservative mode, but should not be the default when coverage matters.

RANSAC residual and inlier fraction are insufficient on their own. For the two initial false positives, residuals are 0.889 and 0.438 px, and inlier fractions are 0.545 and 0.625; both pass current thresholds. False positives in the holdout also have relatively high inlier fractions.

## Practical recommendation

Keep the current HORG rule as the primary result, add leave-one-out affine consistency and inlier count as confidence diagnostics, and flag suspicious cases for independent review. Leave-one-out consistency alone should not remove a point: it misses seven false positives in the holdout frames. A threshold of ≥10 inliers can define a “high confidence” subset when avoiding errors matters more than coverage.

Before turning these diagnostics into automatic filters, validate them on heterogeneous/rotated fields and experimental images. A future high-confidence decision should combine geometric support with an independent patch measure or temporal consistency, and preserve an “uncertain” state when there are too few neighbors or the neighbors are themselves inconsistent.

## Per-POI data

- [Initial frames, 96 seeds](horg_quality_indicators.csv): analytic error, HORG support, local/global affine residual, and simulated decisions.
- [Holdout frames, 576 seeds](horg_quality_holdout_analytic_error.csv): same fields, including the eight-neighbor leave-one-out indicator.
