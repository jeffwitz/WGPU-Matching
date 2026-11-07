DIC-Detector: conditioning and learned saliency
===============================================

The repository has two score models: a structure-tensor baseline and a small
learned network. Both select image locations. Neither directly predicts
displacement or evaluates the full affine IC-GN normal matrix.

The minimum-eigenvalue criterion belongs to the classical good-features-to-track
family; see `Shi and Tomasi (CVPR 1994)
<https://doi.org/10.1109/CVPR.1994.323794>`_. The derivation below describes
translation conditioning and the implementation-specific approximations.

Deriving the baseline
---------------------

For a small translation increment, brightness constancy linearizes as
:math:`I(\mathbf x+\boldsymbol\delta)\simeq I(\mathbf x)+
\nabla I^\mathsf T\boldsymbol\delta`. The quadratic least-squares term is

.. math::

   H_t=\sum_{\mathbf x\in\Omega}w_{\mathbf x}
   \begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}.

The smaller eigenvalue measures the weakest observed translation direction.
Flat regions have two weak eigenvalues; edges constrain mostly one direction;
two-dimensional texture constrains both. The condition ratio

.. math::

   \kappa=\lambda_{\max}/\max(\lambda_{\min},\epsilon)

measures anisotropy, but a good ratio alone does not establish adequate signal.
For affine motion, the Jacobian also contains coordinate-weighted gradients,
and the normal matrix has six parameters. A good 2x2 translation tensor does
not guarantee a good 6x6 affine system. Neither establishes global uniqueness
among similar speckles.

The WGSL baseline averages central-difference gradient products on a
subsampled support. It ranks minimum eigenvalues, suppresses neighbouring
grid maxima, applies tile quotas and estimates a bounded parabolic peak
offset. This is a grid approximation, not a continuous dense corner search.
The complete finite-difference support must remain within valid image/ROI
pixels. A quota can remain unfilled.

The actual learned objective
----------------------------

The network uses a 1-to-8 3x3 convolution, an eight-channel depthwise 3x3
convolution and an 8-to-1 pointwise head, with SiLU hidden activations:
169 parameters and a nominal 5x5 raw-score receptive field. Tensor diagnostics,
NMS and ROI selection use their own, larger supports.

The teacher averages Sobel-gradient products over 9x9 neighbourhoods and marks
the upper 28% of minimum-eigenvalue values on a stride-four grid. Training uses

.. math::

   L=L_{\mathrm{BCE}}+\tfrac12 L_{\mathrm{equiv}},\qquad
   L_{\mathrm{equiv}}=
   \mathrm{SmoothL1}\bigl(\sigma(s_A(\mathbf x)),
                  \sigma(s_B(T(\mathbf x)))\bigr).

The target score is bilinearly sampled using analytic flow; invalid
correspondences are excluded. Held-out splits use separate texture and
deformation seeds.

This is a saliency proxy. It does not implement the second DIC-reliability
head proposed in the specification. Descriptor ambiguity, successful IC-GN
convergence and iteration count are not current training labels. Sigmoid
outputs therefore must not be interpreted as calibrated success probabilities.

Scientific implications
-----------------------

Local gradient energy changes under deformation. Reproducing a tensor teacher
while encouraging corresponding scores to agree is an empirical compromise,
not exact affine equivariance. Good repeatability still needs descriptor and
seed-level validation.

WGSL evaluates exported weights; PyTorch supplies training and numerical
oracle checks. Score parity verifies implementation consistency, not scientific
usefulness. The :doc:`held-out training study
<../_generated_results/features/dic_detector_discontinuous_training_20261107>`
and :doc:`matching budget sweep
<../_generated_results/features/detector_budget_sweep_60pct_20261107>` evaluate
different stages. Reproduction is in :doc:`../how-to/train-dic-detector`.
