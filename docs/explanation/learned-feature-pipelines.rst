XFeat, RaCo/ALIKED/LightGlue and EfficientLoFTR
===============================================

These models spend computation in different stages. They are not
interchangeable keypoint detectors.

XFeat: sparse and semi-dense features
-------------------------------------

`XFeat (Potje et al., CVPR 2024) <https://arxiv.org/abs/2404.19174>`_
uses a lightweight convolutional architecture supporting sparse and semi-dense
matching, with refinement of coarse local matches. Its design limits channel
count while retaining spatial resolution.

For DIC, many inexpensive candidate pairs can improve support. That does
not establish unique recognition of strongly stretched speckles. The
community ONNX export used here represents a specific pipeline rather than
every mode in the paper. The :doc:`XFeat/DeepMatching study
<../_generated_results/features/deepmatching_vs_xfeat>` evaluates initializers
at common DIC centres; raw feature count alone would not measure that outcome.

RaCo: rank and localization uncertainty
---------------------------------------

`RaCo (Shenoi et al., 2026) <https://arxiv.org/abs/2602.15755>`_ learns
repeatable points, a ranking under limited point budgets, and localization
covariance. It trains with perspective crops and augmentation.
Covariance describes spatial uncertainty, not descriptor-match correctness.

For independent localization errors, propagation through a local transform
would give the approximate residual covariance

.. math::

   \Sigma_r\simeq F\Sigma_A F^\mathsf T+\Sigma_B.

This is a possible uncertainty-weighting model, not a claim that current
RANSAC uses exported RaCo covariances. Model outputs actually consumed must
be established from the selected export.

ALIKED describes; LightGlue matches
-----------------------------------

`ALIKED (Zhao et al., 2023) <https://arxiv.org/abs/2304.03608>`_ learns
supporting sampling locations for its sparse deformable descriptor head.
The sampling deformation is a network mechanism; it is not a measured
material deformation gradient.

`LightGlue (Lindenberger et al., ICCV 2023)
<https://arxiv.org/abs/2306.13643>`_ learns interactions between feature sets
and adapts matching computation to pair difficulty. Such contextual decisions
differ from independent nearest-descriptor matching. The precise LightGlue+
export and revision matter: original LightGlue results do not automatically
describe every later variant.

In a RaCo-ALIKED-LightGlue+ pipeline, detection, description and matching
therefore have separate roles. A pipeline gain cannot be assigned to RaCo
alone. Graph rewrites in this repository concern execution placement;
numerical parity and scientific matching quality require separate evidence.

EfficientLoFTR: matching without a sparse detector
--------------------------------------------------

`EfficientLoFTR (Wang et al., CVPR 2024)
<https://arxiv.org/abs/2403.04765>`_ uses aggregated attention to reduce
redundant coarse-feature processing and two-stage fine correlation to improve
localization. It produces semi-dense pairs without requiring independently
repeatable sparse peaks in both images.

Its learned subpixel refinement is distinct from downstream photometric
IC-GN on original UHD pixels. Mapping reduced-network coordinates back to
UHD does not restore the omitted image detail.

Explicit global coarse similarity has quadratic memory cost. With a
stride-eight grid of :math:`N=(W/8)(H/8)` points per image, an FP32 matrix
requires :math:`4N^2` bytes. At 3840x2160 this is about 67.2 GB (62.6 GiB),
before activations. This describes the dense formulation, not a lower bound
on hierarchical or tiled alternatives.

Interpreting evidence
---------------------

The :doc:`60% context table
<../_generated_results/features/detector_modern_matchers_60pct_20261107>`
combines methods with different input resolutions and timing scopes.
It supports observed coverage at common seeds. Isolated speed rankings
require controlled protocols. GPU provider selection alone does not establish
GPU residency: CPU fallback, dense tensors, inter-runtime readback and
reference reuse determine cost per correctly initialized window.
