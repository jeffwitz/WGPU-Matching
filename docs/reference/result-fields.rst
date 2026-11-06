Result fields
=============

`WGPUMatching.match_horg` and `WGPUMatching.match_deepmatching` return a `MatchField`. The `matches` list contains one entry for every requested seed, including rejected estimates. `accepted` returns only entries whose matcher-specific `usable` flag is true.

Common fields
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Meaning
   * - `x`, `y`
     - float
     - Source-image coordinate where displacement is reported.
   * - `dx`, `dy`
     - float
     - Target minus source displacement in pixels.
   * - `usable`
     - bool
     - The method-specific acceptance decision; not an analytic accuracy guarantee.
   * - `score`
     - float
     - Primary method-specific match score. Scales are not comparable between algorithms.
   * - `seed_x`, `seed_y`
     - float or None
     - Requested source center, when available. A hierarchical matcher may snap the reported source coordinate to its atomic grid.
   * - `target_x`, `target_y`
     - float or None
     - Target location when the algorithm reports it explicitly. Otherwise compute `(x + dx, y + dy)`.

Algorithm-specific diagnostics
------------------------------

DeepMatching may populate `reciprocal`, `second_score`, `margin`, and root coordinates. In hierarchical mode, a reciprocal match passed the two-direction consistency check. This does not certify its error against physical or analytic truth.

HORG may populate `inliers`, `candidates`, `inlier_fraction`, and `residual` from the local affine RANSAC fit. It may also populate `local_consistency_px` and `local_consistency_neighbors` from the leave-one-out diagnostic. That diagnostic is exposed for analysis and does not currently reject a point.

The metadata dictionary holds method-level information such as keypoint counts, descriptor counts, parameter values, or timing fields. Its exact keys depend on the selected method and mode. Use `field.as_dict()` for a nested Python dictionary that includes both matches and metadata.
