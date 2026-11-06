HORG-DIC: keypoints, chirality, and local fitting
=================================================

HORG-DIC is an initializer for DIC under large rotations and heterogeneous deformation. The paper, `Chirality-aware initialization for reliable digital image correlation under large rotation and heterogeneous deformation <https://doi.org/10.1016/j.measurement.2026.123133>`_, describes HORG construction, keypoint matching, affine initialization, and subsequent IC-GN refinement.

The Python package implements this sequence:

1. Reduce and smooth the image at multiple scales.
2. Build a difference-of-Gaussians (DoG) pyramid and localize extrema, with quadratic interpolation of position and scale.
3. Build a signed, 64-component HORG descriptor around each extremum, retaining orientation/chirality information.
4. Compare descriptors, find the two nearest neighbors, and apply Lowe's ratio test.
5. Build a local correspondence list for each requested seed.
6. Estimate a local affine transform with RANSAC, then expose support, residual, and a leave-one-out consistency diagnostic.

The pyramid, detection, descriptor, neighbor search, filtering, and RANSAC stages run as WGSL kernels orchestrated by `wgpu-py`. The leave-one-out result is diagnostic: current versions do not use it to discard an otherwise accepted match.

Why preserve chirality?
-----------------------

A patch under rotation or reflection can retain a strong local appearance while reversing the direction of its orientations. A descriptor that retains orientation signs helps distinguish these cases from a match with compatible geometry. The local affine fit then checks whether multiple correspondences support similar motion around a POI.

Robustness is not a guarantee. A patch with too few keypoints, repetitive texture, or occlusion may not provide a strong consensus. A rejected match does not prove that no motion occurred; an accepted match does not certify that its error is below a chosen tolerance. See :doc:`confidence indicators <quality-and-confidence>` and their evaluation.

This package provides the initialization stage. It does not implement the IC-GN refinement described in the HORG-DIC paper.
