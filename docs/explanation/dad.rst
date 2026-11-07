DaD: descriptor-independent interest points
===========================================

Published mechanism
-------------------

`DaD (Edstedt et al., 2025) <https://arxiv.org/html/2503.07347v1>`_
learns a spatial distribution from image logits:

.. math::

   p_\theta(\mathbf x\mid I)
   =\frac{\exp s_\theta(\mathbf x,I)}
          {\sum_{\mathbf z}\exp s_\theta(\mathbf z,I)}.

Training uses a descriptor-free reinforcement-learning objective with
geometric rewards and balanced top-K sampling. Sampling prevents degenerate
concentration. Two detector types emerge, preferentially selecting bright
and dark structures. Their pointwise maximum is distilled into one detector
through a KL-divergence objective. Distillation is a training operation.

The policy-gradient identity illustrates learning through discrete selections:

.. math::

   \nabla_\theta\mathbb E_{\mathcal K\sim p_\theta}[R(\mathcal K)]
   =\mathbb E[R(\mathcal K)\nabla_\theta\log p_\theta(\mathcal K)].

This identity is not the paper's complete sampling/reward implementation.
Inference evaluates the released detector; it does not run reinforcement
learning. DaD produces points, not descriptors or image-to-image matches.
The `official code and weights <https://github.com/Parskatt/dad>`_ provide the
reference implementation.

Repository adaptation
---------------------

The optional adapter uses official weights. Our combination samples DeDoDe-B
at DaD points, matches mutual cosine neighbours and fits local affine models
in WGSL. This combination is an experimental protocol, not an official
end-to-end DaD model.

The ONNX export fuses eligible convolution/batch-normalization pairs and
expresses selection through pooling, top-K and index arithmetic. It returns
coordinates and scores rather than a dense response map. K is fixed in the
graph; the public adapter maps floating coordinates to source pixel centres.
Dynamic spatial axes do not establish arbitrary-resolution validation.

Relevance and limitations for DIC
---------------------------------

Bright/dark diversity motivates testing on dark speckles and light gaps.
It does not imply affine invariance. Even repeatable detections can yield
poor descriptor matches after strong stretch.

Near-tied scores can exchange top-K points under numerical perturbations.
Small score error therefore does not establish point-set equality.
Nearest-set distance can count several points near the same candidate and
does not prove a bijection. Seed-level matching results are needed in addition
to score and coordinate checks.

The :doc:`high-budget experiment
<../_generated_results/features/detector_onnx_high_budget_20261107>` measures
one globally affine synthetic stress pair. It does not establish robustness
to localization bands or experimental damage. Commands are in
:doc:`../how-to/run-learned-detectors`; contracts in
:doc:`../reference/features`.
