Train and run DIC-Detector
==========================

This guide trains a small keypoint-score network from generated speckle pairs,
exports a compact NumPy weight bundle, and evaluates its WGSL implementation.
Training is optional; ordinary ``pip install wgpu-matching`` does not install
PyTorch.

Install the research dependencies
---------------------------------

From the repository root, install the optional extra in an isolated environment:

.. code-block:: console

   python -m pip install -e ".[research]"

This trainer does not require the official DeDoDe source package. That package
is needed only for the separate DeDoDe v2 and DeDoDe-B reference experiments.

Train reproducibly
------------------

The default run uses 640 training pairs over ten epochs at 256×256, then
evaluates disjoint validation and test textures and deformation seeds. The
generator applies translations, large rotations, affine stretch/shear, smooth
heterogeneous motion, piecewise-discontinuous motion with an invalid gap,
photometric change, blur, and camera noise to the same underlying speckle
texture. It writes the model, split seeds, motion-family counts, losses,
per-case metrics, and SHA-256 under ``results/features/dic_detector_training_20261107/``.

.. code-block:: console

   python python/benchmarks/train_dic_detector.py \
       --epochs 10 --samples-per-epoch 64 --batch-size 8 \
       --image-size 256 --validation-pairs 24 --seed 73191

The separately evaluated discontinuity-aware checkpoint was trained with
``--seed 73192`` and written under
``results/features/dic_detector_training_discontinuous_20261107/``. Reproduce
that run with:

.. code-block:: console

   python python/benchmarks/train_dic_detector.py \
       --epochs 10 --samples-per-epoch 64 --batch-size 8 \
       --image-size 256 --validation-pairs 24 --seed 73192 \
       --output-dir results/features/dic_detector_training_discontinuous_20261107

Its training set contained 111 discontinuous pairs among 640 total pairs. Its
held-out report compares it with the earlier checkpoint on the same generated
textures and warp seeds; see :doc:`the discontinuity study
<../_generated_results/features/dic_detector_discontinuous_training_20261107>`.

The score target is based on local structure-tensor information, with an
additional consistency loss at analytically corresponding coordinates. This is
an initial detector objective, not supervision of descriptor uniqueness or
IC-GN convergence. The validation split is diagnostic; the test split uses
separate texture and warp seeds.

Validate WGSL against PyTorch
------------------------------

The parity utility compares WGSL scores to the same network evaluated in
PyTorch at the exact integer grid candidates. A first warm-up compiles/caches
the pipelines; the reported latency is then a median over repeated calls.

.. code-block:: console

   python python/benchmarks/validate_dic_detector_wgsl.py \
       results/features/dic_detector_training_20261107/dic_detector_weights.npz \
       --width 3840 --height 2160 --keypoints 8192 --repeats 3 \
       --output results/features/dic_detector_wgsl_parity_4k_20261107.json

For application use, pass the weight path to ``WGPUMatching.detect_dic``. The
network, tensor diagnostics, NMS, spatial quotas, and selected-point readback
run on WebGPU. The current Python API uploads the image and weights for each
call; it does not yet cache reference-image scores across a sequence.

Scientific scope
----------------

The structure-tensor baseline and the learned score model select candidate
locations. They do not produce image-to-image matches by themselves. To measure
initialization quality, use a common descriptor and reciprocal matcher, then
apply the same WGSL affine RANSAC criteria used for DaD and DeDoDe v2. Report
keypoint repeatability, raw match precision, affine coverage, and analytic seed
error separately. The current repository does not contain the target IC-GN
solver, so convergence after initialization is not yet measured.

Compare detector choices with a common descriptor
-------------------------------------------------

After installing the optional research dependencies and official DeDoDe
source, train or provide a DIC-Detector weight bundle. This reproduces the
reference protocol on the progressive frames and the 60% synthetic stress
case. DaD/DeDoDe inference and DeDoDe-B description run on the selected
PyTorch device; local-affine RANSAC uses the repository's WGSL backend.

.. code-block:: console

   python python/benchmarks/experiment_learned_detectors.py \
       --cases progressive_040 progressive_100 progressive_199 synthetic_60pct \
       --budgets 512 1024 --candidate-multiplier 2 --max-side 1024 \
       --device cpu --torch-threads 8 --repeats 2 \
       --dic-weights results/features/dic_detector_training_20261107/dic_detector_weights.npz \
       --output results/features/detector_common_descriptor_20261107.json

The script writes a completed-case checkpoint between cases. Then summarize
the common-descriptor results beside the historical full-pipeline studies:

.. code-block:: console

   python python/benchmarks/summarize_detector_comparison.py

The comparison page reports candidate counts, repeatability, raw match
precision, accepted affine coverage, analytic error, and timing by stage. Its
global top-K is scoped to the returned candidate pool; the DIC detector's
WGSL response-map selection already uses spatial tiles. It is not an exact
global top-K over every NMS peak in the UHD response map.
