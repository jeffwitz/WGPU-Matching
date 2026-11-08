From interest points to DIC initializations
===========================================

A detector selects locations; a descriptor encodes neighbourhoods; a matcher
proposes pairs; geometric fitting estimates motion at requested DIC centres.
DeepMatching and EfficientLoFTR establish correspondences without first
requiring an independently detected sparse point set.

The classical DoG detector is explained in :doc:`horg`; the hierarchical and
grid matchers in :doc:`deepmatching` and :doc:`cpm`.

Method roles in this repository
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Component
     - Produces
     - Current experimental path
   * - DoG / HORG
     - Scale-space points / handcrafted descriptors
     - WGSL detector, descriptor, matching and affine RANSAC
   * - DaD
     - Learned points
     - PyTorch oracle or ONNX/WebGPU detector; common B descriptor on CPU
   * - DeDoDe v2-L / B
     - Separate learned detector / descriptor
     - ONNX/WebGPU detector; common B descriptor on CPU
   * - DIC tensor / learned DIC
     - Local-information / learned-saliency points
     - WGSL detector; common B descriptor on CPU
   * - XFeat
     - Learned features and sparse or semi-dense pairs
     - Optional ONNX benchmark export
   * - RaCo / ALIKED / LightGlue+
     - Points / descriptors / pairs
     - Optional combined ONNX benchmark export
   * - DeepMatching / CPM-HOG / EfficientLoFTR
     - Correspondences without an independent sparse detector
     - See individual method explanations and execution reports

What makes a useful point?
--------------------------

Let :math:`T(\mathbf x)=\mathbf x+\mathbf u(\mathbf x)` map a material point
to its target location. Three properties matter independently:

* **Repeatability:** a target detection lies near :math:`T(\mathbf x)`.
* **Discrimination:** the descriptor distinguishes that point from other speckles.
* **Support:** correct pairs surround each requested centre with enough
  geometric diversity to estimate the chosen motion model.

Strong corners can be indistinguishable from hundreds of other corners.
Correct pairs concentrated in one region leave other windows unsupported.
Increasing the budget can improve coverage while introducing weaker points
and greater matching cost. Spatial quotas cannot create reliable texture.

Local deformation and invariance
--------------------------------

For a smooth field near :math:`\mathbf x_c`,

.. math::

   T(\mathbf x_c+\boldsymbol\delta)
   \simeq T(\mathbf x_c)+F_c\boldsymbol\delta,
   \qquad F_c=I+\nabla\mathbf u(\mathbf x_c).

Translation changes the centre; :math:`F_c` changes local appearance.
Rotation augmentation does not establish invariance to anisotropic stretch
or shear. Affine fitting after matching cannot recover correct pairs that
the descriptors never found.

Scores and uncertainty
----------------------

A saliency logit, a tensor eigenvalue, descriptor similarity, RANSAC inlier
fraction and photometric residual measure different things. None is
automatically a probability of IC-GN convergence. Thresholds across unrelated
detectors are not interchangeable. Spatial localization covariance is also
different from descriptor ambiguity or uncertainty in the final displacement.

Evaluate the chain
------------------

With visible seed set :math:`S`, usable estimates :math:`A\subseteq S` and
analytic error :math:`e_i`, report

.. math::

   C_\tau=\frac{\#\{i\in A:e_i<\tau\}}{|S|},\qquad
   P_\tau=\frac{\#\{i\in A:e_i<\tau\}}{|A|}.

These are correct coverage and accepted-set precision; precision is undefined
for an empty accepted set. The repository separately uses inclusive
:math:`e_i\le2` and strict :math:`e_i<5` criteria. Detector repeatability has
its own visible-point denominator and pairing rule.

Undoing a resize restores coordinate units, not spatial detail. Read
:doc:`../reference/benchmark-definitions` before comparing errors or FPS.
Network input resolution, point budget, descriptor, acceptance rule, reference
reuse and readback scope must accompany each result.


Continue with :doc:`dad`, :doc:`dedode`, :doc:`dic-detector` and
:doc:`learned-feature-pipelines` for the individual mechanisms, published
sources and repository adaptations.
