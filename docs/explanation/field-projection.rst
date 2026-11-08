From measured pairs to a smooth initialization field
=====================================================

This stage is shared by the feature pipelines. It estimates a displacement
field from measured point pairs, then supplies six affine parameters at each
requested DIC centre. It changes the scientific question: success of a global
projection is not success of each raw match or of a matcher's original local
initializer. Read :doc:`../benchmarks` with that distinction in mind.

Observations, verification and inference
-----------------------------------------

An observation is a pair :math:`(\mathbf x_i,\mathbf y_i)` produced by image
matching, with displacement :math:`\mathbf d_i=\mathbf y_i-\mathbf x_i`.
An affine prediction at a seed is an inferred quantity. Reusing predictions
as independent observations would duplicate evidence and hide the true
support. The adapters preserve coordinates, acceptance masks and identifiers.

The current complete qualification first applies the existing local affine
RANSAC to all pipelines, with the same support and residual thresholds. An
observation survives when an accepted local model near its source predicts
its measured target within the threshold. The filter runs in WGSL and retains
the original pair, not the model prediction. The full observed set, upstream
acceptance and geometric rejection remain in the downloadable records.

This consensus is a geometric assumption, not an oracle. A coherent wrong
cluster can survive. A true region lacking enough support can be rejected.
Global projection cannot recover information that neither matching nor
geometric verification supplied.

Affine model and Gaussian extension
------------------------------------

For numerical conditioning, define :math:`\mathbf z=(\mathbf x-\mathbf o)/s`,
where :math:`\mathbf o` is the mean source location and :math:`s` the larger
image dimension. Fit the normalized displacement

.. math::

   \widehat{\mathbf u}(\mathbf z)
   =\mathbf a_0+\mathbf a_x z_x+\mathbf a_y z_y
    +\sum_{k=1}^{K}\mathbf c_k
       \exp\!\left[-\frac{\|\mathbf z-\boldsymbol\mu_k\|^2}{2\ell^2}\right].

The physical displacement is :math:`\mathbf u(\mathbf x)=s\widehat{\mathbf u}(\mathbf z)`.
Setting :math:`K=0` gives the affine displacement model: two translations and
four displacement-gradient components. The Gaussian RBF extension adds smooth
spatial variation while retaining the affine component. It is a bounded
candidate family, not a proof that Gaussians are optimal for every material.

Centres are chosen deterministically by farthest-point sampling of source
locations. A ridge penalty applies to Gaussian coefficients, leaving the
affine coefficients unpenalized. Density weights compensate for unequal
numbers of pairs in source-image cells. These are sampling weights; they are
not calibrated measurement uncertainties.

Robust fitting and its limits
-------------------------------

At each IRLS iteration, the normal equations use density weights multiplied
by Huber weights. For vector residual magnitude :math:`r_i` and cutoff
:math:`\delta\sigma`, the robust factor is

.. math::

   w_i^{\rm robust}=\min\!\left(1,\frac{\delta\sigma}{\max(r_i,\epsilon)}\right).

A CPU pilot estimates a robust residual scale with a positive floor. The final
normal equations, solution, residuals and weight updates execute in WGSL using
FP32. The code records iteration count, final weight change and convergence.
An apparently stable solve can still fit the wrong correspondence cluster.
Conversely, analytic accuracy and the chosen weight-convergence tolerance are
independent checks; both must be shown.

IRLS is not a replacement for geometric outlier rejection under a dominant
false cluster. The refreshed benchmark therefore compares its verified-pair
path with an explicitly separate raw-pair ablation.

Choosing the basis without analytic truth
------------------------------------------

An explicit affine request fits the simpler model. A Gaussian-RBF request
compares affine and a finite set of centre counts, widths and ridge penalties
using deterministic spatial folds. Held-out pair residuals are measured in
original-image pixels. The candidate with the smallest median held-out error
is considered; affine is retained when its score is within one estimated
standard error of the best Gaussian candidate. The standard-error estimate
uses fold medians; it is a model-selection heuristic, not a calibrated
confidence interval for the physical displacement.

Candidate selection and pilot fitting run in CPU float64. The selected final
fit and patch evaluation run on the active WebGPU device. Solver failure can
try the next scored candidate and eventually affine; the effective basis and
selection reason are always exported. The request name alone does not tell
which model produced the figure.

In the current bounded GPU solver there are at most 64 Gaussian centres and
8192 observations. The benchmark reduces larger sets with deterministic,
spatially balanced quotas, archives the chosen observation identifiers and
retains the complete unreduced set. This capacity limit is not a detector
failure and is included in timing and provenance.

Projection onto an affine DIC window
-------------------------------------

At a requested centre :math:`\mathbf x_c`, the initializer returns

.. math::

   \mathbf p_c=[u_c,v_c,u_{,x},u_{,y},v_{,x},v_{,y}].

The translation is the field value at the exact centre. In ``support_fit``
mode the displacement gradients minimize sampled support error with that
translation fixed. A symmetric grid of offsets :math:`\mathbf r_j` gives

.. math::

   \underset{G}{\operatorname{minimize}}\;
   \sum_j\|\mathbf u(\mathbf x_c+\mathbf r_j)
                -\mathbf u(\mathbf x_c)-G\mathbf r_j\|^2.

This is not an unconstrained six-parameter refit that can move the centre.
The alternative ``taylor`` mode uses the analytic centre gradient. The support
RMS and maximum error describe approximation of the fitted smooth field by
one affine patch. Neither is a photometric error or an IC-GN convergence test.

What the field does not establish
----------------------------------

A global affine field is exactly the right functional form for homogeneous
synthetic traction. Its successful extrapolation to every visible seed can
therefore exceed the number of directly measured local initializers. This is
a useful initialization result under that prior, but not evidence that every
region was independently matched.

A Gaussian field remains smooth. Neither basis resolves cracks, disconnected
regions, occlusions or arbitrary localization. Extrapolation outside measured
support is especially uncertain. The qualification includes the experimental
Adrien pair as a visual check, without analytic-error claims. The downstream
image-based solver must still validate its own convergence.

See :doc:`../how-to/project-correspondences` for calls and coordinate mapping,
and :doc:`../reference/benchmark-definitions` for coverage and timing units.
