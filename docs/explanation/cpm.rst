CPM-HOG: sparse-grid PatchMatch
===============================

CPM-HOG is an experimental coarse-to-fine matcher for proposing DIC
initializations. It is inspired by Coarse-to-Fine PatchMatch, but it is not a
numerically equivalent port of the original CPM implementation.

Sparse work grid
----------------

The requested DIC seeds and the internal matching grid are separate. The DIC
window determines where users want initial values and whether a requested seed
has a full window in the reference image. The matching support determines the
local patch compared at each internal grid point. A configurable work-grid
stride controls how many points are evaluated. Requested seeds are associated
with nearby measured points; they are never labelled as directly measured.

Descriptor cost
---------------

The matcher reuses the repository's INRIA-style eight-channel HOG descriptors.
It builds them once on the reduced image, then forms coarser descriptor levels
by averaging each 2 by 2 block. At a candidate displacement, WGSL computes the
cosine similarity of the two HOG patches. Cost is ``1 - similarity``. The
support is a rigid translation at each scale; it does not deform internally.

PatchMatch passes
-----------------

At the coarsest level, each work-grid point tests a deterministic stratified
set of displacements inside the configured search radius. Every propagation
pass reads a complete previous state buffer and writes a separate next buffer.
The point tests its own state and the four spatial neighbors, then searches
random offsets around its current best state. This ping-pong design avoids
unsynchronized in-place updates. At finer descriptor levels, the previous
displacements double in level coordinates and are refined with smaller random
search radii.

The search runs in both image directions. A forward match is reciprocal when
the inverse displacement, interpolated from valid neighbors at the reached
target position, closes within the configured threshold. A score threshold can
reject low-similarity points as well.

Seed initialization
-------------------

For each requested seed, the matcher selects nearby usable work-grid matches
and fits a distance-weighted affine displacement model. Three robust
reweighting rounds remove large residuals. At least six non-degenerate
neighbors are needed; otherwise the nearest measured displacement is returned
as a rejected fallback. The :class:`wgpu_matching.CPMOptions` parameter
``affine_radius_px`` sets
the local support radius; ``None`` uses twice the work-grid stride. The six
affine parameters and measured grid point are retained in
``MatchField.metadata``.

Differences from CPM
--------------------

The original CPM paper uses dense SIFT descriptors and an L1 patch cost. This
implementation uses the existing HOG descriptors and a normalized cosine
cost, runs on a sparse grid, and pools descriptors instead of recomputing
features at every image scale. The local affine fit adapts the measured sparse
field to requested DIC seeds. The matcher does not implement IC-GN refinement
or impose a mechanical displacement regularizer.

Read the :doc:`Python API reference <../reference/api>` and the
:doc:`benchmark definitions <../reference/benchmark-definitions>` before interpreting
coverage or run time. The simple translation parity fixture validates the
Python/browser implementations; it does not qualify CPM on large or
heterogeneous deformation. The published 60% CPM measurements predate a
work-grid geometry correction and must be rerun before describing current
large-strain performance.


The :doc:`complete initializer qualification
<../_generated_results/qualification/all_methods_20261107/index>` re-runs
the current pipelines on the same progressive and 10–60 percent tensile
cases, with both projection requests and a visual comparison for every
outcome. It evaluates inferred field initializations and preserves the
older local-initializer experiments separately.
