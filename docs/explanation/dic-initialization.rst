A matcher initializes DIC; it does not replace it
=================================================

Digital image correlation (DIC) associates windows or points in a reference image with their locations in a deformed image. A local method such as inverse-compositional Gauss–Newton (IC-GN) then refines these displacements by optimizing photometric similarity under a local deformation model. Such an optimizer is most likely to converge when its initial estimate lies inside the subset's convergence basin.

The library works before that refinement:

1. Build an image representation that remains discriminative under displacement or deformation.
2. Propose an initial displacement for each requested point of interest.
3. Expose scores and diagnostics that downstream software can use when deciding what to refine or reject.

DeepMatching computes dense hierarchical correspondences by correlation. HORG-DIC detects keypoints, compares their descriptors, and estimates robust local motion. Either can help when a subset is too far from its target position for direct IC-GN initialization, but the methods have different mechanisms and acceptance rules.

The native matcher output is a list of displacements at POIs. It is not a regularized dense field, it does not resolve occlusions, and it does not calculate strain or stress. IC-GN, spatial interpolation, and uncertainty propagation are separate stages.

An optional Python projection layer accepts direct correspondences from these
or external matchers and fits either a robust affine displacement model or a
bounded Gaussian-RBF model with an affine component. It evaluates that model
over each requested DIC support and projects the result to six affine IC-GN
parameters. This creates a smooth initializer only; it does not replace the
independent image-based IC-GN solve. Gaussian-RBF settings are selected within
a finite cross-validation search, and a simpler affine model is preferred when
its score lies within one estimated standard error of the best RBF candidate.
The projection diagnostics distinguish measured observations from inferred
patch parameters and expose when robust iterations did not converge.

Continue with the :doc:`correlation hierarchy <deepmatching>`,
:doc:`keypoint theory <horg>` or :doc:`smooth-field explanation <field-projection>`.
The :doc:`projection recipe <../how-to/project-correspondences>` gives calls;
the :doc:`benchmark guide <../benchmarks>` gives measured cases and limitations.
