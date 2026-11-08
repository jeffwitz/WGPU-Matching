Train and evaluate the first DIC descriptor
===========================================

This guide reproduces the first learning stage: transform HORG-64 descriptors
using same-texture positive correspondences and hard negatives. It does not
train the later convolutional descriptor. PyTorch is an optional research
dependency; inference weights are NumPy arrays consumed by WebGPU shaders.

Install the research dependencies
---------------------------------

From the repository root, install the optional training packages:

.. code-block:: console

   python -m pip install -e ".[research]"

The primary inference package does not require PyTorch.

Train reproducibly
------------------

The training utility generates a speckle texture, applies a known motion, and
extracts HORG descriptors at source points and their exact floating-point
target positions. It uses separate texture seeds for training and validation.
Other locations in the same texture, at least 48 pixels away in the source
image, form hard negatives. The output directory contains the generated
descriptor pairs, model weights, hashes, split metadata, and validation
history.

.. code-block:: console

   python python/benchmarks/train_dic_descriptor.py \
       --output results/descriptors/dic_descriptor_horg_training_20261108 \
       --train-pairs 64 --validation-pairs 16 --points-per-pair 96 \
       --epochs 40 --batch-size 4 --seed 20261108

The supplied result was trained on the CPU with PyTorch 2.15.0 and completed
in about 18 seconds after the descriptor pairs had been generated. This is a
small feasibility run, not a statistically powered training study. The held-
out texture seeds and weight hashes are recorded in the result directory.

Run the WGSL descriptor
-----------------------

The public API extracts the configured HORG scales and applies the trained
projection in WGSL:

.. code-block:: python

   with WGPUMatching(device_type="gpu") as matcher:
       features = matcher.detect_dic(image, learned_weights_path=detector_weights).features
       described = matcher.describe_dic_keypoints(
           image, features, weights_path=descriptor_weights
       )

The returned feature coordinates remain floating point. The current API reads
the intermediate HORG descriptors to Python before projecting them on the
GPU, then reads the compact result back before matching. It does not provide
zero-copy GPU residency.

The projection shader implements linear, residual MLP, and multiscale MLP
variants. Automated parity tests compare each shader result with a NumPy
reference. The residual connection and 96-unit hidden layer have explicit
tests because an earlier shader revision left part of that hidden layer
unwritten.

Train the patch convolutional descriptor
----------------------------------------

The first HORG projection did not improve the fixed descriptor in the
six-field convergence study. The next experimental model samples a native
64-pixel patch and applies a small convolution, depthwise convolution, spatial
pooling, and a 64-value projection. The feature map is held between two WGSL
passes; the intermediate GPU buffer is not copied back to the host.

.. code-block:: console

   python python/benchmarks/train_dic_cnn_descriptor.py \
       --output results/descriptors/dic_descriptor_cnn_training_20261108 \
       --train-pairs 24 --validation-pairs 8 --points-per-pair 48 \
       --epochs 15 --batch-size 2 --seed 20261109 --no-affine-augmentation

The trainer selects the epoch with the highest validation retrieval score;
the fixed test fields are not consulted during selection. This first run uses
few textures and is only a feasibility experiment. Its held-out six-field
IC-GN comparison and 4K timings are in :doc:`the patch-CNN report
<../_generated_results/descriptors/dic_descriptor_cnn_training_20261108/README>`.

An expanded run used 80 training textures and 20 disjoint validation
textures, 48 points per pair, 20 epochs, and seed ``20261110``:

.. code-block:: console

   python python/benchmarks/train_dic_cnn_descriptor.py \
       --output results/descriptors/dic_descriptor_cnn_training_20261108_expanded \
       --train-pairs 80 --validation-pairs 20 --points-per-pair 48 \
       --epochs 20 --batch-size 4 --seed 20261110 --no-affine-augmentation

The selected epoch was 7, with 6.04% validation top-1 retrieval. The weight
archive SHA-256 is
``861491fb6045678e1583622977afe909cb144d358770b68ce1e7a0593a3cdfff``.
The expanded independent qualification used ten new fields in each of five
motion families, plus the six frozen historical cases. Results and counts of
failed field fits are documented in :doc:`the expanded qualification
<../_generated_results/qualification/dic_descriptor_expanded_icgn_20261108/README>`.

For inference, pass the exported NumPy weight file to
``WGPUMatching.describe_dic_cnn_keypoints``. Points without a fully in-image
64-pixel support are marked invalid. The matching pipeline reads the final
descriptors back to Python and uploads them for reciprocal matching, so the
entire pipeline is not GPU-resident.

Qualify through IC-GN
---------------------

The qualification script reuses the six fixed fields, common detector,
projection rules, and IC-GN parameters. The two fields where all previous
methods failed remain held-out regressions and are not part of training or
validation.

.. code-block:: console

   python python/benchmarks/qualify_heterogeneous_icgn.py \
       --output results/qualification/dic_descriptor_expanded_icgn_20261108 \
       --descriptor-onnx /path/to/dedode_b_descriptor_1024x576_k8192.onnx \
       --weights results/features/dic_detector_training_discontinuous_20261107/dic_detector_weights.npz \
       --compact-descriptor results/descriptors/dic_descriptor_horg_training_20261108/dic_descriptor_multiscale_mlp.npz \
       --cnn-descriptor results/descriptors/dic_descriptor_cnn_training_20261108_expanded/dic_patch_cnn_descriptor.npz \
       --independent-eval-per-family 10 --seed 970000000

The command also runs DeepMatching, native HORG, and EfficientLoFTR baselines.
It writes raw direct matches, descriptor ranking diagnostics, projected
initial parameters, and per-window IC-GN results. The report
:doc:`summarizes the observed limitations
<../../_generated_results/descriptors/baseline_failure_analysis>`: in this
pilot, learning a transform of HORG did not improve the number of correctly
converged windows and was slower than fixed HORG-64.

Train with detector points and image-wide hard negatives
--------------------------------------------------------

The follow-up experiment uses points returned by the WebGPU DIC-Detector and
analytically mapped, subpixel positive locations. Same-texture detector points
form the negative pool; each training step selects the 128 most similar valid
negatives in both directions. Supports must remain physically valid across a
17 by 17 sample lattice on the 64 by 64 patch. Train and validation texture
identities are disjoint, and checkpoint selection macro-averages top-1
retrieval across motion families.

.. code-block:: console

   python python/benchmarks/train_dic_cnn_descriptor.py \
     --output results/descriptors/dic_descriptor_hardneg_training_20261108 \
     --train-pairs 24 --validation-pairs 10 --points-per-pair 128 \
     --negative-points-per-image 1024 --epochs 12 --seed 20261108 \
     --no-affine-augmentation

To evaluate full-image retrieval among increasing numbers of native 4K
candidates:

.. code-block:: console

   python python/benchmarks/benchmark_dic_descriptor_retrieval.py \
     --weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \
     --output results/descriptors/dic_descriptor_hardneg_training_20261108/retrieval_full_pool.json \
     --families rigid affine heterogeneous discontinuous --queries 512 --seed 20261108

The retrieval benchmark counts only queries with a detector point within five
pixels of the analytic target position. It is not an IC-GN success rate. The
hard-negative run improves affine and discontinuous retrieval on the measured
4K pairs, but leaves rotation and heterogeneous retrieval essentially
unresolved. Independent random affine patch augmentation did not help. See
:doc:`the complete descriptor report
<../_generated_results/descriptors/dic_descriptor_hardneg_training_20261108/README>`.

Requalify the saved checkpoint through the external WebGPU IC-GN solver:

.. code-block:: console

   PYTHONPATH=python/src:python/benchmarks python python/benchmarks/qualify_dic_descriptor_checkpoint.py \
     --output results/qualification/dic_descriptor_hardneg_icgn_20261108 \
     --scratch /tmp/dic_descriptor_hardneg_icgn_20261108 \
     --cnn-descriptor results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \
     --detector-weights results/features/dic_detector_training_discontinuous_20261107/dic_detector_weights.npz \
     --independent-eval-per-family 10

This tests ten independent fields per family and preserves all six fixed
regressions. Failed field projections are reported separately from scored
windows: a failed fit is not a measured zero-success IC-GN run. The results
are in :doc:`the hard-negative requalification
<../_generated_results/qualification/dic_descriptor_hardneg_icgn_20261108/README>`.
