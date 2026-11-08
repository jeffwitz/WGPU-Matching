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


Direct observations and local affine coefficients
--------------------------------------------------

``include_correspondences=True`` exports the underlying observations for
HORG, hierarchical DeepMatching and CPM. The metadata preserves measured
source and target coordinates, matcher scores and native acceptance masks.
DeepMatching observations are source-best atomic hypotheses before reciprocal
rejection; repeated root votes are not independent rows. CPM observations are work-grid
matches, and HORG observations are descriptor pairs. The ordinary HORG and
CPM seed translations are fitted quantities and must not be expanded into
independent observed pairs.

``fit_local_affine(..., include_affine_models=True)`` adds an array of shape
``(seed_count, 8)`` to ``LocalAffineField.affine_models``. Each row stores
``[a1, a2, a0, 1, b1, b2, b0, 1]`` for the target-coordinate model

.. math::

   x'=a_0+a_1(x-x_c)+a_2(y-y_c),\qquad
   y'=b_0+b_1(x-x_c)+b_2(y-y_c).

Only consume coefficients whose corresponding match is ``usable``. These
are target-coordinate coefficients: the identity contributes one to
:math:`a_1` and :math:`b_2`. They are not the displacement-gradient parameters
``[u, v, du/dx, du/dy, dv/dx, dv/dy]`` returned by field patch projection.
The optional export preserves the default local-fit output and is used by
the common verification filter to test original observed pairs.
