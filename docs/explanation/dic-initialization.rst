A matcher initializes DIC; it does not replace it
=================================================

Digital image correlation (DIC) associates windows or points in a reference image with their locations in a deformed image. A local method such as inverse-compositional Gauss–Newton (IC-GN) then refines these displacements by optimizing photometric similarity under a local deformation model. Such an optimizer is most likely to converge when its initial estimate lies inside the subset's convergence basin.

WGPU-Matching works before that refinement:

1. Build an image representation that remains discriminative under displacement or deformation.
2. Propose an initial displacement for each requested point of interest.
3. Expose scores and diagnostics that downstream software can use when deciding what to refine or reject.

DeepMatching computes dense hierarchical correspondences by correlation. HORG-DIC detects keypoints, compares their descriptors, and estimates robust local motion. Either can help when a subset is too far from its target position for direct IC-GN initialization, but the methods have different mechanisms and acceptance rules.

The current output is a list of displacements at POIs. It is not a regularized dense field, it does not resolve occlusions, and it does not calculate strain or stress. IC-GN, spatial interpolation, and uncertainty propagation are separate stages.

.. seealso::

   * :doc:`deepmatching` explains the correlation hierarchy.
   * :doc:`horg` explains the keypoint, descriptor, and RANSAC path.
   * :doc:`../benchmarks` describes measured cases and limits on generalization.
