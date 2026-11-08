Coverage, error, and confidence
===============================

The number of accepted matches alone does not summarize initializer quality. To compare methods on images with known truth, use the same visible seeds and report coverage, precision, and error separately.

Let :math:`S` be the visible requested seeds and :math:`A\subseteq S` the
accepted estimates. Their counts are :math:`N_v=|S|` and :math:`N_a=|A|`.
For an accepted centre, define the scalar Euclidean displacement error
:math:`e_i=\|\widehat{\mathbf u}_i-\mathbf u_i^{\rm true}\|_2` in pixels.

Useful measurements are:

* **Accepted coverage:** :math:`N_a/N_v`.
* **Precision within 2 px among accepted:**
  :math:`\#\{i\in A:e_i\le2\}/N_a`.
* **Accurate coverage:** :math:`\#\{i\in A:e_i\le2\}/N_v`.
  This penalizes both incorrect matches and missing estimates.
* **Conditional RMSE:** :math:`\sqrt{N_a^{-1}\sum_{i\in A}e_i^2}`.
  A low value does not imply high coverage. Precision and error are undefined
  when no estimates are accepted; they must not be reported as zero.

The repairable benchmark threshold is strictly :math:`e_i<5`; the tighter
comparison uses :math:`e_i\le2`. Neither is a universal confidence threshold
or a measured IC-GN convergence guarantee. Experimental images usually lack
analytic truth, so internal diagnostics must be distinguished from independent
error measurements.

HORG leave-one-out diagnostic
-----------------------------

The current diagnostic fits an affine transform to the eight accepted POIs nearest to a point, excluding that point, and measures the distance between the prediction and the point's proposed displacement. It measures agreement with neighbors, not physical truth. A group of neighbors that is jointly wrong can still appear consistent.

In an exploratory holdout of 576 synthetic seeds, a 1.5 px leave-one-out threshold lets 7 false positives pass and rejects 6 correct POIs. Absolute RANSAC support is more discriminative in that sample, but a threshold of 10 inliers also rejects 93 correct results. These observations do not yet justify automatic rejection.

The :doc:`benchmark guide <../benchmarks>` explains the protocols. The web
edition retains the :doc:`confidence-indicator study
<../_generated_results/uniaxial/horg_quality_indicators>` and its
:doc:`source data <../_generated_results/uniaxial/horg_quality_indicators.csv>`.
