DeepMatching: dense correspondences through a deformable hierarchy
==================================================================

DeepMatching searches for correspondences with a multiscale correlation hierarchy. Revaud, Weinzaepfel, Harchaoui, and Schmid describe a hierarchical multilayer correlation architecture for dense matching, inspired by convolutional networks, that handles non-rigid deformation and repetitive texture: `article <https://arxiv.org/abs/1506.07656>`_. The name does not mean that this reimplementation uses a trained neural network.

Dense descriptors
-----------------

The Python hierarchical path computes pixelwise histograms of gradient orientations. It smooths and normalizes the maps, projects them into orientation components, and averages them into a descriptor pyramid. This representation is more robust than raw intensity while retaining response maps that can be evaluated by correlation.

Bottom-up aggregation
---------------------

Neighboring atomic cells are grouped into larger patches. Each level combines response maps from four children with sparse convolution and local max-pooling. A parent can represent a configuration whose parts have moved by a few cells, instead of requiring the entire patch to undergo one rigid translation.

Top-down localization and reciprocity
-------------------------------------

The search begins with coarse patches, then follows local maxima downward to recover a fine-scale leaf and its location. Correspondences are assembled over the root grid. A reciprocal intersection keeps a pair when it is compatible in both search directions.

In the INRIA-style implementation here, requested centers are projected onto the matcher's atomic grid. The hierarchy is evaluated over the full root grid and then sampled at the requested seeds. `mode="direct"` is a separate path: it computes dense patch correlation for the requested seeds without the hierarchy.

Practical consequences
----------------------

* The hierarchy explores large displacements without comparing every large patch at every full-resolution position.
* Atomic-cell discretization limits subpixel precision in the hierarchical displacement.
* Reciprocity is a consistency check between two searches, not proof of analytic accuracy.
* Root size, image reduction, search radius, and boundary handling affect cost or coverage.
* Rotation/scale-invariant variants and INRIA's rectification of incomplete border patches are not all implemented.

Sparse, seed-local evaluation
-----------------------------

The default INRIA-style mode evaluates the complete root grid even when `seed_points` requests only a few outputs. `mode="inria_sparse"` is an experimental alternative: it maps explicit seeds to their containing roots and evaluates only those roots. It is useful when a previous stage has already identified a small set of POIs that need a fallback initialization.

This saves work in proportion to the number of distinct seed roots, but reciprocal candidate competition is then limited to the selected roots. The result can differ from the global reciprocal field if an unselected root contains a competing match. On the tested Live-DIC subsets, sparse and global modes returned identical decisions and displacements; this is measured evidence for those inputs, not a general parity guarantee. See :doc:`the sparse study <../benchmarks>` and the :doc:`usage example <../how-to/seeds-and-fields>`.

The Live-DIC parity experiment shows the distinction between implementation parity and accuracy: WebGPU and INRIA vectors are identical for shared matches in the measured setup, yet analytic truth shows errors above 2 px. See :doc:`the benchmark report <../benchmarks>` and the `original DeepMatching paper <https://arxiv.org/abs/1506.07656>`_.
