:orphan:

Using this handbook
====================

This handbook describes a Python library for finding image correspondences
and constructing initial values for digital image correlation (DIC).
It follows the Diataxis framework: the route depends on what you need to do,
not on an implementation-file hierarchy.

.. list-table:: Choose a reading route
   :header-rows: 1
   :widths: 36 64

   * - Reader need
     - Start here
   * - Obtain a first working result
     - Tutorials: install the package, generate a bounded test pair, then detect or match.
   * - Complete a specific task
     - How-to guides: adapter selection, explicit centres, field projection, training or benchmarks.
   * - Understand the mathematics and limits
     - Explanations: observation versus inference, detector theory, descriptors, search and robust fitting.
   * - Check exact behaviour
     - Reference: interfaces, units, defaults, result fields, metrics and glossary.

For practical work, begin with the first-matches tutorial and run its small
synthetic case before loading experimental images. The first-detector tutorial
separates point selection from matching. Continue to the projection guide when
you need a six-parameter affine initializer at a window centre.

For scientific assessment, read the observation/inference explanation before
the qualification tables. The current campaign compares eleven pipelines on
ten cases, with two field requests per case. Every method is checked at
60 percent homogeneous synthetic traction. Passing means all visible requested
centres have displacement error strictly below 5 pixels; the stricter
2-pixel count and the numerical solver's convergence flag remain separate.
These are centre-initialization errors, not measured convergence of inverse
compositional Gauss–Newton (IC-GN).

The experimental Adrien pair has no analytic field. Its figures show the
reference and the target sampled through an estimated affine or radial basis
function (RBF) field. Appearance, photometric residual and geometric agreement
are diagnostics, not a physical-accuracy guarantee. Homogeneous traction has
the exact functional form of a global affine field; its successful coverage
must not be generalized to localization or cracks.

Read resolution and timing boundaries with every table. Native source-image
size can differ substantially from detector or network input size. The
current common learned-descriptor path still includes processor-side stages.
A selected graphics backend does not prove that every intermediate tensor
stays on that device.

The print edition retains all theory, interface reference, per-case benchmark
synthesis and fixed native-pixel comparison crops. The web edition additionally
contains the complete alternating animations, rejected-match records and
compressed fitted-field archives. Historical measurements remain available
under their original protocols and are not substituted into the new campaign.

Technical abbreviations are defined on first use within each topic and listed
in the glossary. Literal code, field names and paths preserve their executable
spelling. The book's sources and build procedure remain in the repository so
the printed evidence can be regenerated from the measured records.
