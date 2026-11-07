Benchmark metric definitions
============================

.. list-table::
   :header-rows: 1

   * - Metric
     - Definition
   * - Visible seeds
     - Requested source seeds whose analytically transformed locations are inside the target.
   * - Accepted
     - Visible seeds for which the method returns `usable=True`.
   * - Within 2 px
     - Accepted visible results with Euclidean displacement error no greater than 2 px.
   * - Accurate coverage
     - Within-2-px count divided by visible seeds.
   * - Conditional RMSE
     - Square root of the mean squared vector error among accepted visible results only.
   * - First-call time
     - One matcher call after image loading and device creation; includes first-use work such as pipeline creation.
   * - Repeated median
     - Median over the stated repeated calls, reported separately from the first call.

Benchmark reports may use different timing boundaries. Read each protocol before comparing numbers across browser, native Python, and CPU INRIA measurements.

Detector and initializer evidence
----------------------------------

* Repeatability requires the transformation, visible support, pairing rule
  (one-way nearest, mutual nearest, or one-to-one assignment) and pixel units.
  These definitions need not give the same count.
* A nearest-set export parity check can match several source points to one
  candidate. It is not a bijective parity proof or a matching benchmark.
* Report correct coverage over all visible requested seeds as well as
  accepted-set precision. Empty-set RMSE and precision are undefined; display
  an empty value rather than zero.
* The repairable criterion is strictly below 5 px. The stricter criterion is
  at most 2 px. Neither establishes downstream IC-GN convergence.
* A detection on a resized UHD source is reduced-resolution detection.
  Record both source and model dimensions and the resize coordinate mapping.
* Reusing an already prepared reference is a different timing protocol from
  processing two new images. Report warm-up, repetition count and timing scope.
* A sum of per-stage medians is not the median of complete end-to-end calls.
  The common-detector script currently reports the former. Its historical
  ``total_pair_s_median`` field must be interpreted with this qualification.
* Summed ONNX operator durations, wall-clock session time and GPU timestamps
  measure different quantities. CPU event counts are not CPU time fractions.
* A single-repeat high-budget result is a quality observation and an indicative
  latency measurement; it cannot establish timing dispersion or significance.

The model score, detector repeatability, raw-pair precision, accepted affine
coverage and IC-GN convergence are distinct validation levels. Cross-method
context tables combine historical protocols; they do not establish isolated
backend speedups when input resolutions or point budgets differ.
