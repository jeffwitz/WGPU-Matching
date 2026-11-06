Coverage, error, and confidence
===============================

The number of accepted matches alone does not summarize initializer quality. To compare methods on images with known truth, use the same visible seeds and report coverage, precision, and error separately.

Let:

* `N_visible` be the number of seeds whose analytically transformed location lies inside the target;
* `N_accepted` be the number of those seeds marked `usable`;
* `e_i` be the vector error between an estimated displacement and analytic truth.

Useful measurements are:

* **Accepted coverage:** `N_accepted / N_visible`.
* **Precision within 2 px among accepted:** `count(e_i <= 2) / N_accepted`.
* **Accurate coverage:** `count(e_i <= 2) / N_visible`. This penalizes both incorrect matches and rejected seeds.
* **Conditional RMSE:** `sqrt(mean(e_i**2))` over accepted correspondences only. A low RMSE does not imply high coverage.

The 2 px threshold is used to compare methods in the Live-DIC reports; it is not a universal confidence threshold. Real applications usually lack analytic truth, so internal diagnostics must be distinguished from error measured against an independent reference.

HORG leave-one-out diagnostic
-----------------------------

The current diagnostic fits an affine transform to the eight accepted POIs nearest to a point, excluding that point, and measures the distance between the prediction and the point's proposed displacement. It measures agreement with neighbors, not physical truth. A group of neighbors that is jointly wrong can still appear consistent.

In an exploratory holdout of 576 synthetic seeds, a 1.5 px leave-one-out threshold lets 7 false positives pass and rejects 6 correct POIs. Absolute RANSAC support is more discriminative in that sample, but a threshold of 10 inliers also rejects 93 correct results. These observations do not yet justify automatic rejection.

.. seealso::

   * :doc:`../benchmarks` for protocols, denominators, and CSV data.
   * :doc:`Detailed confidence-indicator report <../_generated_results/uniaxial/horg_quality_indicators>`.
   * :doc:`Confidence data preview <../_generated_results/uniaxial/horg_quality_indicators.csv>`.
