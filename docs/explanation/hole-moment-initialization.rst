Hole moments as local initialization hypotheses
===============================================

This experiment uses the shape of an internal void to propose initial states
for digital image correlation (DIC). Texture in the surrounding material must
validate those states before inverse-compositional Gauss–Newton (IC-GN)
propagation. It is an optional research experiment, not a qualified replacement
for the existing initializer.

The :doc:`measured experiment and remapping animations
</_generated_results/hudge_hole_moments_20261117/README>` include unsuccessful
variants, two synthetic controls and every falsely accepted synthetic window.

What shape moments identify
--------------------------

For a filled hole domain :math:`H`, define its area, centroid and covariance:

.. math::

   A=\int_H 1\,d\mathbf x,\qquad
   \mathbf c=A^{-1}\int_H\mathbf x\,d\mathbf x,\qquad
   \mathbf S=A^{-1}\int_H(\mathbf x-\mathbf c)(\mathbf x-\mathbf c)^T\,d\mathbf x.

If one affine map transports the entire domain, its constant determinant
cancels between the transformed area integral and its normalization:

.. math::

   \mathbf S_1=\mathbf F\mathbf S_0\mathbf F^T,\qquad
   \mathbf c_1=\mathbf F\mathbf c_0+\mathbf t.

For positive-definite covariances and orientation-preserving maps, the complete
solution family is

.. math::

   \boldsymbol\varphi_\theta(\mathbf x)=\mathbf c_1+
   \mathbf S_1^{1/2}\mathbf R(\theta)\mathbf S_0^{-1/2}(\mathbf x-\mathbf c_0).

The remaining parameter is an angle in whitened coordinates, not generally the
physical polar rotation. An ellipse still has a continuous affine stabilizer:
:math:`\mathbf S^{1/2}\mathbf R(\theta)\mathbf S^{-1/2}` leaves its domain unchanged.
Its only rigid rotational symmetries may be discrete, but that is a different
transformation group. Texture must resolve the affine ambiguity.

For nonlinear motion the moment-derived map is an effective shape hypothesis.
It is not the deformation gradient at every surrounding material point.
An elliptical target contour does not prove affine material transport:
tangential redistribution remains unobserved. There is no material point at
the centre of a hole against which to validate a material gradient.

Texture resolves the remaining ambiguity
---------------------------------------

For a fixed material sector :math:`\Omega`, candidate samples are compared by
zero-mean normalized cross-correlation (ZNCC):

.. math::

   C(\theta)=\frac{\sum_{\mathbf x\in\Omega}
   [I_0(\mathbf x)-\overline I_0]
   [I_1(\boldsymbol\varphi_\theta(\mathbf x))-\overline I_{1,\theta}]}
   {\sqrt{\sum_{\mathbf x\in\Omega}[I_0(\mathbf x)-\overline I_0]^2}
    \sqrt{\sum_{\mathbf x\in\Omega}[I_1(\boldsymbol\varphi_\theta(\mathbf x))-
    \overline I_{1,\theta}]^2}}.

Every angle uses the same reference samples. Invalid target support is reported
and may reject a hypothesis; it never silently removes difficult samples from
the denominator. The experiment retains several angular peaks and refines
their geometric parameters photometrically. Contours do not enter that loss.

An exploratory safeguard splits the sector pixels into two checkerboards.
Only one part participates in angular ranking and refinement. The other
checks the result, including after the local solver changes its parameters.
Neighbouring image samples remain correlated; this is a consistency check,
not an independent accuracy measurement.

Curvature and the solver boundary
---------------------------------

A sector may require a quadratic map:

.. math::

   \varphi_a(\mathbf x)=\mathbf f_a^T\mathbf x+t_a+
   \tfrac12(\mathbf x-\mathbf o)^T\mathbf H_a(\mathbf x-\mathbf o),\qquad
   \nabla\varphi_a=\mathbf f_a+\mathbf H_a(\mathbf x-\mathbf o),
   \quad a\in\{1,2\}.

Each symmetric Hessian contributes three coefficients, for twelve map
parameters in total. Spatial derivatives are checked by finite differences.
The current prototype fits these sectors on the central processing unit
(CPU), then sends their local affine tangents to the graphics processing unit
(GPU) solver. The external solver currently rejects material masks for its
quadratic model. This experiment therefore does not test a complete masked
quadratic IC-GN solve.

At a requested centre :math:`\mathbf x_i`, the six transmitted coefficients
are ordered as :math:`[u,u_x,u_y,v,v_x,v_y]`, with
:math:`\mathbf u=\boldsymbol\varphi(\mathbf x_i)-\mathbf x_i` and
:math:`\nabla\mathbf u=\nabla\boldsymbol\varphi(\mathbf x_i)-\mathbf I`.
The implementation does not set the four gradients to zero.

Interpretation of the present evidence
-------------------------------------

The moments identify useful hypotheses on the synthetic controls, but small
partial windows can accept false correspondences. More accepted windows are
not sufficient evidence of better initialization. Averages of mutually
inconsistent local affines can also fold despite positive local determinants.

On the supplied plate at level 191, controlled affine and quadratic proposals
produce no additional validated anchor. Existing accepted states are preserved.
No claim of improved near-hole initialization or full GPU residency is made.
The quantitative report distinguishes local accepted states, direct synthetic
errors, diagnostic remapped fields and unobserved pixels.
