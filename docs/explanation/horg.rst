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

What the detector finds
-----------------------

The scale-space construction follows the classical
`SIFT treatment by Lowe (2004) <https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf>`_.
The scale-space image is :math:`L(\mathbf x,\sigma)=G_\sigma*I`.
DoG responses :math:`D=L(\mathbf x,k\sigma)-L(\mathbf x,\sigma)`
approximate a scale-normalized Laplacian up to a factor. Extrema are sought
in position and scale, then refined by a local quadratic expansion. Formally,
the stationary offset is :math:`\Delta=-H_D^{-1}\nabla D` when the local
Hessian is nonsingular. Contrast and edge-like configurations need filtering.
These extrema identify blob-like texture structures; they do not establish
that a corresponding blob remains distinguishable after deformation.

Scale selection handles approximately isotropic changes in support. An
anisotropic stretch changes a circular neighbourhood into an ellipse, so
scale and orientation normalization alone do not provide full affine
invariance.

Why preserve chirality?
-----------------------

Proper rotations preserve handedness; reflections reverse it. For two vectors,
:math:`\det[Av,Aw]=\det(A)\det[v,w]`, which explains this distinction.
Image-gradient vectors transform by the inverse transpose under a geometric
warp, so support resampling and orientation assignment still matter.

The current shader accumulates three branch-gradient magnitudes in each of
16 cells, sorts them and appends a chirality component computed from signed
2D cross products of the ordered vectors. This gives four entries per cell,
or 64 components, followed by normalization/clipping. That discrete feature
can change near ties or weak gradients. It is not a general guarantee of
affine invariance. Local RANSAC then tests whether several pairs support one
affine motion in a seed neighbourhood.

Robustness is not a guarantee. A patch with too few keypoints, repetitive texture, or occlusion may not provide a strong consensus. A rejected match does not prove that no motion occurred; an accepted match does not certify that its error is below a chosen tolerance. See :doc:`confidence indicators <quality-and-confidence>` and their evaluation.

This package provides the initialization stage. It does not implement the IC-GN refinement described in the HORG-DIC paper.


The :doc:`complete initializer qualification
<../_generated_results/qualification/all_methods_20261107/index>` re-runs
the current pipelines on the same progressive and 10–60 percent tensile
cases, with both projection requests and a visual comparison for every
outcome. It evaluates inferred field initializations and preserves the
older local-initializer experiments separately.
