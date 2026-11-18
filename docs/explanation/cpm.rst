CPM-HOG: method, implementation, and scientific scope
=====================================================

CPM-HOG is an experimental, sparse correspondence method in this project. It
uses a coarse-to-fine PatchMatch search over local histograms of oriented
gradients, implemented with WebGPU compute shaders. Its purpose is to propose
displacements and local affine initial values for digital image correlation
(DIC). It is not a reproduction of the published CPM program, a dense optical
flow estimator, or a replacement for the inverse compositional Gauss–Newton
(IC-GN) solver.

This page first explains the published algorithm and the DIC problem that
motivated it, then describes the implementation precisely, including each
departure from the paper and why that departure was made. The final sections
state what the available experiments establish and what they do not.

Problem and notation
--------------------

Let ``I0`` be a reference image and ``I1`` a deformed image. For a material
point ``x = (x, y)`` in the reference, the unknown displacement ``u(x)`` maps
it to ``x + u(x)`` in the deformed image. A correspondence is therefore a
pair ``(x, x + u(x))``. In DIC, one ultimately wants a useful initial
displacement and, often, a local affine approximation at each requested
point of interest (POI), so that a photometric optimizer such as IC-GN can
converge to a subpixel solution.

The difficult part addressed by CPM-HOG is finding plausible correspondences
when the displacement is too large for a local optimizer to capture from a
zero or near-zero initial guess. A descriptor gives each image location a
feature vector; a patch cost compares a reference descriptor patch with a
candidate target patch translated by a proposed displacement. Searching for a
low-cost displacement is a nearest-neighbour-field problem. The desired field
is spatially coherent in many ordinary regions, so a good displacement found
at one location can be a useful candidate at a nearby location. Random search
is needed as well, because spatial propagation alone cannot discover an
arbitrarily large motion or recover after a bad local hypothesis.

This is only a correspondence prior. It does not establish that every local
patch has a unique match, nor that one smooth field is mechanically valid
everywhere. Speckle is stochastic and can contain locally similar patterns;
large strain can change the shape of a patch; cracks, masks, and out-of-view
regions can invalidate a support. CPM-HOG therefore needs geometric
verification, and its output still needs validation by the downstream DIC
solver.

The published coarse-to-fine PatchMatch method
----------------------------------------------

The method called CPM in this project is inspired by Hu, Song, and Li,
“Efficient Coarse-to-Fine PatchMatch for Large Displacement Optical Flow,”
published at the 2016 Institute of Electrical and Electronics Engineers
Conference on Computer Vision and Pattern Recognition (CVPR) [Hu2016]_. That
paper uses PatchMatch as a sparse correspondence
search inside a larger optical-flow pipeline. It is useful to separate its
core search from the complete flow method when comparing it with this DIC
implementation.

Nearest-neighbour field and matching cost
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each sampled reference location ``x_i``, the method seeks a target
location ``x_i + d_i`` whose descriptor patch is similar. The collection of
displacements ``d_i`` is a sparse nearest-neighbour field (NNF). At a pyramid
level ``l``, write ``F0_l`` and ``F1_l`` for the descriptor maps and
``P_l`` for the patch support. The paper's matching cost is the sum of
absolute descriptor differences over the patch:

.. math::

   C_l(x_i,d) = \sum_{q \in P_l}
       \left\|F_{0,l}(x_i+q)-F_{1,l}(x_i+d+q)\right\|_1.

The paper uses dense 128-component SIFT Flow features, computed at every
pixel and pyramid level. The descriptor is contextual: it is not merely a
gradient histogram at the centre. Summing the componentwise absolute
differences over a patch makes the match depend on both the local descriptor
and its spatial arrangement.

Coarse-to-fine search
~~~~~~~~~~~~~~~~~~~~~

The images and their feature maps are represented at multiple resolutions.
At the coarsest level, a regular sparse grid is initialized with candidate
displacements. Each candidate is scored by the patch cost. The algorithm
then alternates two sources of proposals:

* **Spatial propagation.** A displacement already good at a neighbouring
  grid location is tested at the current location. Nearby locations often
  share similar motion, so this can move a useful hypothesis through a large
  part of the image.
* **Random search.** New offsets around the current candidate are tested
  with a search radius that shrinks geometrically. Large early proposals
  discover long displacements; smaller later proposals refine them.

The paper uses alternating scan directions and in-place propagation. The
current displacement field at a coarse level is scaled when passed to the
next finer level, where it is refined. Thus, the pyramid reduces the search
range in pixel units while retaining large motions in the original image.
The paper specifically warns that an excessively small random-search radius
at fine levels can prevent recovery of small structures undergoing large
motion.

In the paper's optical-flow application, the sparse matches are checked in
both image directions. A forward match is compared with the reverse field at
its target location; inconsistent matches are removed. EpicFlow, an
edge-preserving interpolation method, then interpolates the remaining sparse
matches into a dense optical-flow field. EpicFlow is part of that full
application, not part of the PatchMatch proposal mechanism itself [EpicFlow]_.

Relation to point-wise DIC initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The motivation is also related to a DIC-specific line of work. Li, Li, and
Liang's 2019 paper, “Enhanced feature-based path-independent initial value
estimation for robust point-wise digital image correlation,” describes a
path-independent initialization approach using dense DAISY descriptors and
a PatchMatch variant, followed by point-wise IC-GN. The published abstract
describes estimating initial values independently for points of interest,
rather than relying on a prior loading sequence [Li2019]_. This is a more direct
application context than generic optical flow.

CPM-HOG is not a port of that DIC method either. The publicly available
description motivates the use of dense local texture descriptors and a
PatchMatch-style search, but this project makes its own choices about the
descriptor, work grid, validity, projection, GPU execution, and returned
data. The paper's exact implementation details should not be inferred from
this project, and this documentation does not claim that the current code
reproduces its reported performance.

Our CPM-HOG algorithm
---------------------

The implementation has six stages: image validation, descriptor construction,
sparse-grid construction, bidirectional coarse-to-fine search, reciprocal
filtering, and projection from the internal grid to requested DIC seeds.
Large image and descriptor computations are performed by WebGPU kernels.
Python orchestrates resources and collects compact match results.

1. Inputs and coordinate conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend accepts two same-sized, two-dimensional, monochrome ``uint8``
images. Both directions of matching are run on the same pair. Coordinates
and reported displacements are expressed in pixels of the original input
images. Internally, the images are represented at a reduced descriptor
scale, and each coarser pyramid level introduces another factor of two.

There are three distinct spatial concepts which must not be conflated:

* The **DIC window size** controls the default requested seed lattice and
  whether a requested seed has its full reference window inside the image.
* The **matching support** controls the rigid patch compared by the matcher,
  its border-validity test, and the maximum useful pyramid depth.
* The **work-grid stride** controls the spacing between locations at which
  matching is actually performed.

Requested POIs need not lie on the work grid. The matcher computes sparse
work-grid matches first and later estimates a local affine model at each
requested POI. The returned metadata records the supporting measured grid
points so that an estimated value is not mistaken for a direct match.

2. HOG descriptor construction and pyramid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPM-HOG reuses the repository's existing INRIA-style, eight-channel HOG
descriptor builder, also used by DeepMatching. The source grayscale image is
presmoothed, and the existing WGSL feature builder produces eight
non-negative orientation-response channels. The channels are stored as two
four-channel floating-point textures. The base descriptor map is formed at
the configured reduced scale by averaging the corresponding base-image HOG
responses; it does not rerun the HOG extractor independently on a resized
image. Further descriptor levels are built by averaging each 2-by-2 block
of the previous level.

This produces a compact, GPU-friendly descriptor pyramid. It is not SIFT
Flow: the channels are local orientation responses rather than the paper's
dense 128-dimensional contextual descriptors. It is not the HORG descriptor
used by another project matcher either; CPM-HOG uses the eight-channel HOG
texture produced by the shared feature builder.

Let ``s0 = 2**downscale_levels`` be the original-image pixel spacing of the
base descriptor map. At level ``l``, a descriptor cell corresponds to
``s_l = s0 * 2**l`` original pixels. The matching patch width is
approximately ``matching_support_px / s_l`` cells, with a minimum of two
cells. The requested level count is clipped so that the coarsest patch
remains about four or more descriptor cells wide. This avoids building
levels where the matching support collapses to too few samples.

The resulting support is a square, rigid support in descriptor coordinates.
The patch does not rotate, stretch, or shear as part of its cost. That
simplification is computationally useful but is a real limitation under
large local affine deformation: even the correct centre displacement can
receive a poor score when the contents of the two rigid supports no longer
cover equivalent material regions.

3. Sparse work grid
~~~~~~~~~~~~~~~~~~~

The internal grid covers locations whose matching support fits in the
reference image. Its first centre is at ``(support - 1) / 2`` in each axis;
later centres are separated by ``work_grid_stride_px``. The grid is built
from the image dimensions and matching support, not from the requested DIC
window lattice.

At each work-grid point the state stores an integer displacement ``(dx,dy)``
in the current descriptor level, a scalar cost, and a validity flag. A
candidate is invalid if it exceeds the configured search radius or if the
source or translated target patch crosses the descriptor-map boundary. The
search radius is specified in original-image pixels and converted to the
current level. A radius of zero means that the full current descriptor-map
extent is eligible.

The sparse grid is a deliberate computational trade-off. The paper also
uses sparse grid locations for its PatchMatch search, but the present grid
is tailored to DIC initialization and is not the same parameterization as
the paper's experiments. Grid spacing controls both computation and how
much local motion variation can be sampled. A coarse grid can miss a narrow
localization band; a fine grid increases patch evaluation cost and does not
by itself make the patch objective affine invariant.

4. Cosine patch cost
~~~~~~~~~~~~~~~~~~~~

For a source descriptor vector ``a_q`` and target vector ``b_q`` at each
patch cell ``q``, the WGSL kernel accumulates a single cosine similarity over
all cells and channels:

.. math::

   S(x,d) =
   \frac{\sum_{q\in P} a_q^\mathsf{T} b_q}
        {\sqrt{\left(\sum_{q\in P}\|a_q\|_2^2\right)
                      \left(\sum_{q\in P}\|b_q\|_2^2\right)}+\epsilon},
   \qquad C(x,d)=1-\operatorname{clip}(S(x,d),0,1).

The similarity is normalized across the whole patch, not computed as a mean
of individually normalized cell scores. Similarity is clipped to the range
``[0,1]`` because these HOG response channels are non-negative. Lower cost is
better. This cost is intentionally different from the componentwise L1 cost
of the paper. It is simple to evaluate in a WGSL kernel and uses the
descriptor already available in the repository, but its comparative merits
must be established by benchmark rather than assumed from the paper.

5. Coarse-to-fine PatchMatch search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the coarsest level, each work-grid point tests the zero displacement and
a deterministic, stratified set of candidate offsets within the radius.
Stratification spreads the initial trials over the search square instead of
letting a small number of random trials cluster by chance. The hash used for
candidate generation is a deterministic function of point index, user seed,
iteration, and trial.

Each subsequent pass tests:

* the point's previous displacement;
* the four axial neighbours' previous displacements, re-evaluated at the
  current point;
* the configured number of hash-generated offsets around the current best
  displacement.

The random radius shrinks between passes. When moving from a coarser level
to a finer one, the displacement in level coordinates is doubled, which
preserves the represented displacement in original-image pixels. The
inherited candidate is then refined at the finer level.

All points in a pass read the same complete previous-state buffer and write
to a separate next-state buffer; the two buffers are swapped after dispatch.
This synchronous “ping-pong” update is important for WebGPU correctness:
invocations in one dispatch cannot safely perform arbitrary in-place
neighbour updates without synchronization. It is deterministic and
race-free, but it is not numerically identical to the paper's
scan-order-dependent in-place propagation. A neighbour's new result becomes
available only on the next pass, so the number and ordering of passes affect
how quickly a good hypothesis spreads.

Both reference-to-target and target-to-reference searches use this same
procedure. The current default values are available through
:class:`wgpu_matching.CPMOptions`; in particular the defaults include a
128-pixel DIC window, 32-pixel matching support, 16-pixel work-grid stride,
four descriptor levels, six propagation iterations per level, four random
trials per pass, and a 256-pixel search radius. These are configuration
defaults, not paper parameters or generally optimal settings.

6. Reciprocal consistency and score filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a forward displacement ``d_f(x)``, the matcher evaluates the reverse
field at ``x + d_f(x)`` using bilinear interpolation on the reverse work
grid. Interpolation is rejected if any corner with non-zero interpolation
weight has an invalid reverse match. The forward-backward discrepancy is

.. math::

   e_{FB}(x)=\left\|d_f(x)+
       \widetilde d_r(x+d_f(x))\right\|_2,

where ``d_r`` is the reverse displacement and the tilde denotes strict
bilinear interpolation. A match is reciprocal when ``e_FB`` is no greater
than ``reciprocal_threshold_px``. It must also satisfy the configured minimum
similarity. The default minimum similarity is zero, so by default a match
is not rejected for low score alone; it still must have a valid support and
pass reciprocal consistency.

Reciprocity suppresses many one-way ambiguities, but it is not proof of
physical correctness. A mutually consistent false match can survive, and
two physically distinct regions with similar texture can form a coherent
wrong field. The current consistency test also does not use a material mask
or a crack-boundary graph. Users should therefore treat the accepted flag as
an algorithmic filter, not as a calibrated probability of successful
IC-GN convergence.

7. Projection onto requested DIC points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every requested seed, the implementation selects usable work-grid
matches within ``affine_radius_px`` (by default, twice the work-grid stride)
and fits two weighted local planes to their measured displacements:

.. math::

   u(\Delta x,\Delta y)=a_0+a_x\Delta x+a_y\Delta y,
   \qquad
   v(\Delta x,\Delta y)=b_0+b_x\Delta x+b_y\Delta y.

Here ``(Delta x, Delta y)`` is relative to the requested seed. The fit starts
with radial weights ``1 / (1 + (r/R)^2)`` and uses three reweighting rounds.
After each fit, observations whose residual exceeds
``max(2 px, 2.5 * median residual)`` are removed. At least six non-degenerate
observations are required. This is a local robust regression, not the
PatchMatch search itself and not a global smoothness penalty.

When successful, the reported six values encode the local position mapping
in the implementation's order:

.. math::

   [x+a_0,\;1+a_x,\;a_y,\;y+b_0,\;b_x,\;1+b_y].

The direct work-grid match coordinates, support count, fit error, score, and
reciprocity metadata are retained. If the fit fails, a nearest measured
displacement may be returned as a rejected fallback; it must not be treated
as a usable affine seed. CPM-HOG does not execute IC-GN after this step.

Why this implementation differs from the paper
----------------------------------------------

The changes below are intentional. They target a portable DIC initializer
that fits this repository, rather than reproducing the paper's optical-flow
experiment. That distinction also means that published CPM accuracy and
runtime numbers cannot be transferred to CPM-HOG.

.. list-table:: Main methodological differences
   :header-rows: 1
   :widths: 22 34 44

   * - Topic
     - Published CPM paper
     - This implementation and reason
   * - Target problem
     - Sparse matches are an intermediate for dense optical flow; EpicFlow
       interpolates them afterwards.
     - Sparse matches are projected to requested DIC seeds as local affine
       initial values. We do not generate dense optical flow because the
       downstream consumer asks for DIC initial guesses at selected points.
   * - Descriptor
     - Dense 128-component SIFT Flow feature at each pixel and pyramid level.
     - Existing eight-channel INRIA-style HOG response maps. This reuses a
       validated GPU feature builder and avoids adding a separate SIFT Flow
       implementation and its memory/computation cost.
   * - Patch metric
     - Sum of absolute differences over descriptor components and patch
       locations.
     - Patch-wide normalized cosine cost. It is compact and directly
       expressible in WGSL over the existing feature textures. It is a
       scientific design change whose quality is judged by our DIC results.
   * - Feature pyramid
     - Feature maps are part of the paper's multi-scale extraction pipeline.
     - HOG is built once at reduced resolution and subsequent levels are
       mean-pooled 2-by-2. Reuse avoids recomputing feature extraction at
       every level and keeps the representation and buffers manageable on
       integrated GPUs.
   * - Parallel update
     - Alternating scan-order propagation with in-place updates.
     - Synchronous ping-pong state buffers. WGSL dispatches have no global
       barrier that would make arbitrary in-place neighbour reads safe.
       Separate read/write states prevent data races and give deterministic
       pass semantics.
   * - Initial hypotheses
     - The paper's evaluation uses its own random initialization and
       parameter settings.
     - Deterministic stratified candidates at the coarsest level. This makes
       browser and Python runs reproducible and reduces random clustering;
       it does not guarantee that the true motion is sampled.
   * - Geometric verification
     - Forward-backward validation is part of the full optical-flow method.
     - Strict bilinear reverse-field validation is retained, with explicit
       rejection if an interpolation support includes an invalid match.
       This prevents fabricating inverse evidence across holes in the sparse
       grid.
   * - Output and solver
     - Sparse flow matches are interpolated to dense flow by EpicFlow.
     - A local robust affine fit maps sparse matches to each requested seed.
       This format is directly useful to initialize a DIC subset, but it is
       not a mechanically validated field and can mix distinct motion
       regions if its support crosses a discontinuity.
   * - Software and license
     - The paper describes an algorithm; its associated reference
       implementation is distributed under GPL-3.0.
     - This project implements the method independently in shared WGSL and
       Python orchestration rather than copying that code. The separation
       supports browser/Python parity, WebGPU portability, and the project's
       licensing boundary. The paper itself should not be confused with the
       licence of a particular code repository.

Why HOG instead of SIFT Flow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project already had a WebGPU implementation that builds eight-channel
HOG feature maps for DeepMatching. Reusing it gave CPM-HOG a working GPU
descriptor path and let the first implementation focus on search, reciprocal
filtering, and DIC seed projection. Implementing SIFT Flow would have added
a separate dense feature algorithm, different storage requirements, and a
new cross-runtime parity surface before the matching hypothesis itself had
been validated on speckle images.

This was a pragmatic first experiment, not a claim that HOG is scientifically
equivalent to SIFT Flow or better suited to every speckle deformation. The
HOG support represents local orientations and is relatively inexpensive,
but it does not provide the same contextual descriptor as SIFT Flow. The
cosine cost and support size can make the objective ambiguous or poorly
conditioned under strong strain.

Why sparse work-grid matching and local affine projection?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DIC caller supplies a finite set of POIs, and the main use is to seed a
local optimizer rather than export optical flow for every pixel. Matching
only at a regular subset of locations bounds the number of GPU invocations.
Local affine regression then converts nearby measured translations into the
six local parameters expected by a DIC initializer.

This is also a limitation. It assumes that nearby accepted measurements can
support one local plane. A narrow localization band may be undersampled; a
support crossing a displacement jump may mix two motions. Current CPM-HOG
does not prepare the ROI connectivity constraints used by some other
project components, and it does not perform crack-aware segmentation. The
local fit's robust reweighting removes large residuals but does not guarantee
that it will discover every physical boundary.

Why ping-pong propagation instead of in-place scans?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The published CPU/GPU algorithms can impose a scan order and consume
updates made earlier in that order. A single WebGPU compute dispatch has no
device-wide synchronization point between arbitrary workgroups. Letting one
invocation overwrite a state that another invocation may read would create
an execution-order-dependent data race. The implementation therefore reads
one immutable state array and writes another, swapping them at pass
boundaries. The cost is extra state bandwidth and potentially slower
propagation per iteration; the benefit is defined, portable, deterministic
parallel behavior.

Why not run IC-GN inside CPM-HOG?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPM-HOG is an initializer experiment. The repository already contains
separate DIC solvers and qualification paths, so combining them in the
matcher would conflate correspondence quality with photometric convergence
and duplicate solver responsibilities. The matcher returns displacement
and local affine proposals; an application may then pass those proposals to
IC-GN and independently check convergence, residuals, and final error.

An accepted CPM-HOG match is therefore not equivalent to a converged DIC
window. In particular, a descriptor match can be wrong yet reciprocal, and
the rigid descriptor cost can prefer the wrong region under non-affine
distortion. Solver-side residual checks and clear failure states remain
necessary.

Configuration and reproducibility
---------------------------------

Options are exposed through :class:`wgpu_matching.CPMOptions`. The most
important controls are:

.. list-table:: CPM-HOG search controls
   :header-rows: 1
   :widths: 28 18 54

   * - Option
     - Current default
     - Effect
   * - ``patch_size_px``
     - 128 px
     - Default DIC window and requested seed-lattice spacing; not the rigid
       descriptor support.
   * - ``matching_support_px``
     - 32 px
     - Original-image width of the descriptor patch and border-validity
       support.
   * - ``work_grid_stride_px``
     - 16 px
     - Spacing of internal sparse match locations.
   * - ``downscale_levels``
     - 2
     - Base descriptor reduction factor is ``2**value``.
   * - ``levels``
     - 4
     - Requested descriptor-pyramid depth; clipped according to patch size.
   * - ``search_radius_px``
     - 256 px
     - Maximum displacement radius in original-image pixels; zero uses the
       current level's full extent.
   * - ``propagation_iterations``
     - 6
     - Number of synchronous search passes per level.
   * - ``random_trials``
     - 4
     - Number of local random candidates per non-initialization pass.
   * - ``initial_hypotheses``
     - 16
     - Number of stratified candidates per point during coarsest
       initialization.
   * - ``local_search_radius_px``
     - 16 px
     - Fine-scale random-search radius control.
   * - ``reciprocal_threshold_px``
     - 8 px
     - Maximum forward-backward discrepancy.
   * - ``minimum_similarity``
     - 0.0
     - Minimum descriptor similarity; zero disables score-based rejection
       for the non-negative similarity used here.
   * - ``affine_radius_px``
     - ``None``
     - Radius for local affine regression; ``None`` uses twice the work-grid
       stride.
   * - ``seed``
     - 1
     - Deterministic candidate-generation seed.

See the :doc:`Python API reference <../reference/api>` for API details and
the :doc:`benchmark definitions <../reference/benchmark-definitions>` for
the meaning of accepted, correct, and visible windows. The browser/Python
translation fixture checks implementation parity on a simple known motion;
it is not evidence of robustness to high strain.

Evidence, failure modes, and interpretation
-------------------------------------------

The smoke fixture reports 3/3 accepted points with zero displacement error
for a deterministic synthetic translation of ``(8, -4)`` pixels. This
establishes that the basic path runs on the tested fixture; it does not
establish general accuracy.

The historical progressive tensile experiments reported 32/32 accepted
seeds at 0.11% strain, 2/28 at 3.84%, and 0/28 at 10%. At 10%, 13/28 raw
proposals were within 5 pixels but all were rejected by the configured
acceptance checks. The historical 60% experiment accepted 0/16 seeds and
reported an all-proposal displacement RMSE of 757.235 pixels. These runs
predate a work-grid geometry correction; they must be treated as historical
measurements and not as current qualification of the corrected code.

The available oracle-cost studies point to two distinct failure mechanisms.
At some configurations and seeds, the known displacement has a better HOG
cost than the candidate found by the search, which indicates a search or
propagation failure. At larger supports, the known displacement itself can
have a poor rigid-patch cost under affine deformation, which indicates an
objective mismatch: the search may be doing what its cost asks while the
cost no longer represents the correct material correspondence. More random
trials cannot reliably fix the second problem.

The :doc:`current all-method initializer qualification
<../_generated_results/qualification/all_methods_20261107/index>` should be
used for comparisons on its shared cases. It tests inferred fields and
includes progressive and large-strain tensile cases; consult the per-case
exports and the accompanying interpretation before generalizing to
heterogeneous fields. Historical local-initializer studies remain separate.

For a DIC application, the relevant endpoint is not the fraction of raw
descriptor matches that look plausible. It is the number and spatial
distribution of windows that receive a usable initialization and converge
correctly under IC-GN, while false accepted initializations remain rare.
CPM-HOG has not been qualified as a robust high-deformation initializer on
the cited historical tests. Its output should be checked by the downstream
solver, and failure or unsupported seeds should remain explicit rather than
being silently filled by interpolation.

Reproduction links
------------------

* `CPM-HOG results and validation notes <https://github.com/jeffwitz/WGPU-Matching/blob/main/results/cpm/README.md>`_
* `Progressive tensile comparison <https://github.com/jeffwitz/WGPU-Matching/blob/main/results/cpm/progressive_tensile_040_100_199.md>`_
* `Oracle cost diagnostics <https://github.com/jeffwitz/WGPU-Matching/blob/main/results/cpm/oracle_diagnostics_60.md>`_
* `Oracle support study <https://github.com/jeffwitz/WGPU-Matching/blob/main/results/cpm/oracle_support_study_60.md>`_
* :doc:`Current initializer qualification <../_generated_results/qualification/all_methods_20261107/index>`

References
----------

.. [Hu2016] Y. Hu, R. Song, and Y. Li, “Efficient Coarse-to-Fine PatchMatch
   for Large Displacement Optical Flow,” *Proceedings of the IEEE Conference
   on Computer Vision and Pattern Recognition*, 2016, pp. 5704–5712.
   `DOI: 10.1109/CVPR.2016.615 <https://doi.org/10.1109/CVPR.2016.615>`_.
   `Open-access paper <https://openaccess.thecvf.com/content_cvpr_2016/papers/Hu_Efficient_Coarse-To-Fine_PatchMatch_CVPR_2016_paper.pdf>`_.

.. [Li2019] J. Li, X. Li, and Z. Liang, “Enhanced feature-based
   path-independent initial value estimation for robust point-wise digital
   image correlation,” *Optics and Lasers in Engineering*, vol. 121,
   pp. 189–202, 2019.
   `DOI: 10.1016/j.optlaseng.2019.04.016 <https://doi.org/10.1016/j.optlaseng.2019.04.016>`_.
   `Publisher abstract <https://www.sciencedirect.com/science/article/pii/S0143816618317093>`_.

.. [EpicFlow] J. Revaud, P. Weinzaepfel, Z. Harchaoui, and C. Schmid,
   “EpicFlow: Edge-Preserving Interpolation of Correspondences for Optical
   Flow,” *Proceedings of the IEEE Conference on Computer Vision and Pattern
   Recognition*, 2015. EpicFlow is cited here as the interpolation component
   in the original CPM optical-flow pipeline, not as a component of CPM-HOG.
