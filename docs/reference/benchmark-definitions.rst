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
