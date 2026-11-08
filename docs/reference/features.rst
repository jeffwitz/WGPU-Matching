Feature detector contracts and API
===================================

For concepts and method assumptions, start with
:doc:`../explanation/detector-principles`. For installation, exporting and
benchmark commands, use :doc:`../how-to/run-learned-detectors`. The
:doc:`../tutorials/first-detector` lesson needs no downloaded model weights.

Coordinates and preprocessing
-----------------------------

``FeatureSet.xy`` contains floating-point ``(x, y)`` positions in the source
image. Integer coordinates identify pixel centres, matching the convention in
``matcher-contracts.json``. The adapters map the upstream normalized
``grid_sample(align_corners=False)`` coordinates to source pixel centres and
undo a common aspect-preserving resize. They replicate grayscale to RGB, use
bicubic resizing, and apply ImageNet normalization. Each result records the
input and model resolution so reduced-resolution detection is explicit.

The optional ``roi_mask`` is applied after upstream top-K detection, so a mask
can leave fewer than the requested number of points. The output metadata makes
that behavior explicit.

Execution and score contracts
-----------------------------

* DaD/DeDoDe v2 reference adapters use optional PyTorch packages.
* ``OnnxWGPUKeypointDetector`` uses an external keypoint-mode ONNX model and
  ONNX Runtime WebGPU. It returns CPU arrays, with no shared wgpu-py allocation.
* Its graph has a fixed K capacity. A smaller requested budget slices that
  output after inference; a larger request raises an error.
* At K=1024 coordinates and scores occupy 12 KiB; at K=8192, 96 KiB.
* DIC tensor scores are minimum gradient-tensor eigenvalues. Learned DIC
  scores are sigmoid outputs, not calibrated correspondence probabilities.
* DIC ROI support checks run before selection; DaD/DeDoDe ROI filtering is
  applied after top-K and can leave fewer points than requested.
* ``DeDoDeBDescriptor`` is the PyTorch reference implementation. The optional
  ``DeDoDeBOnnxWGPUDescriptor`` samples the same official 256-component
  descriptor through a fixed-shape ONNX graph whose operators ran on ONNX
  Runtime's WebGPU execution provider in the measured export. The output still
  crosses to NumPy and is uploaded by ``wgpu-py``; there is no shared allocation
  between these runtimes.
* ``WGPUMatching.describe_horg_keypoints`` computes a compact 64-value HORG
  descriptor for caller-supplied points with WGSL. This controlled ablation
  uses a fixed SIFT support scale and estimates orientation from local
  gradients; it is not the native multi-octave HORG pipeline. In the
  six-field IC-GN pilot, it reduced paired descriptor time from about 2.4 s
  to about 10 ms, but yielded enough reliable matches for field fitting in
  only one of six fields. See the :doc:`heterogeneous seeded IC-GN report
  <../_generated_results/qualification/heterogeneous_icgn_20261108/README>`.
* ``WGPUMatching.describe_dic_keypoints`` applies an optional trained
  projection to one or three HORG scales using WGSL. Linear, residual MLP, and
  multiscale MLP weight bundles are supported. The projection is tested
  against a NumPy oracle for all three architectures. In the dated six-field
  study, the trained multiscale projection did not improve HORG-64: it reached
  53/91 correct IC-GN windows on the one field where HORG-64 reached 57/91,
  and it did not produce a usable field in the other five. Its paired
  descriptor time was 83.9 ms versus 22.7 ms on native 4K images with 4096
  points. Read the :doc:`scientific explanation
  <../explanation/dic-descriptor>` and :doc:`failure analysis
  <../_generated_results/descriptors/baseline_failure_analysis>` before using
  these experimental weights.
* ``WGPUMatching.describe_dic_cnn_keypoints`` evaluates a trained 64-pixel
  patch convolutional neural network on the supplied floating-point points.
  It runs the first convolution and feature-map pooling/projection in two
  WGSL passes, keeps the intermediate feature map in a WebGPU storage buffer,
  and returns a 64-component unit-normalized descriptor. Points whose support
  crosses the image boundary are marked invalid. PyTorch is only needed for
  optional training; the exported NumPy weight archive is sufficient for
  inference. The expanded independent qualification obtained 684/1013
  correct converged windows on discontinuous fields, but failed to produce a
  field on held-out rigid, affine, and heterogeneous families. It did not
  recover either historical regression failure. See :doc:`the method explanation
  <../explanation/dic-descriptor>` and :doc:`training and qualification guide
  <../how-to/train-dic-descriptor>`.
  The complete expanded comparison is available as the :doc:`qualification
  report <../_generated_results/qualification/dic_descriptor_expanded_icgn_20261108/README>`.
  The subsequent same-texture hard-negative training, 4K retrieval sweep and
  affine-augmentation control are documented in the :doc:`dated descriptor
  report <../_generated_results/descriptors/dic_descriptor_hardneg_training_20261108/README>`;
  the corresponding independent IC-GN run is in the :doc:`requalification
  report <../_generated_results/qualification/dic_descriptor_hardneg_icgn_20261108/README>`.
* Mutual cosine nearest neighbours can execute in WGSL; descriptor arrays are
  uploaded per call and only the best index and cosine score per query return
  to the host. The measured full 60% workload preserves the exact direct-pair
  digest and projected-field error, but WGSL matching was slower than NumPy
  for 256-component descriptors. See the :doc:`dated DeDoDe ONNX/WebGPU report
  <../_generated_results/features/dedode_onnx_wgpu_20261108/README>` for timings,
  parity, graph placement, and the fixed-shape limitation.

API
---

.. autoclass:: wgpu_matching.FeatureSet
   :members:

.. autoclass:: wgpu_matching.DescribedFeatures
   :members:

.. autoclass:: wgpu_matching.DaDDetector
   :members:

.. autoclass:: wgpu_matching.DeDoDeV2Detector
   :members:

.. autoclass:: wgpu_matching.OnnxWGPUKeypointDetector
   :members:

.. autoclass:: wgpu_matching.DeDoDeBDescriptor
   :members:

.. autoclass:: wgpu_matching.DeDoDeBOnnxWGPUDescriptor
   :members:

.. autofunction:: wgpu_matching.mutual_nearest_cosine
.. autofunction:: wgpu_matching.mutual_nearest_cosine_wgpu

.. autoclass:: wgpu_matching.FeatureMatchField
   :members:

Measured evidence
-----------------

Training and WebGPU score-parity commands are in :doc:`../how-to/train-dic-detector`.
The native-resolution tensor baseline measurements are stored in
``results/features/dic_detector_tensor_baseline_20261107.md`` and the learned
model training and score-parity runs in the adjacent dated result directory.

The measured studies are also rendered through the documentation's result
pages: :doc:`native 4K detector latency and repeatability
<../_generated_results/features/dic_detector_native_4k_20261107>`,
:doc:`DaD and DeDoDe v2 ONNX/WebGPU export audit
<../_generated_results/features/detector_onnx_webgpu_20261107>`,
:doc:`high-budget ONNX/WebGPU validation
<../_generated_results/features/detector_onnx_high_budget_20261107>`,
:doc:`60% tensile context across feature methods
<../_generated_results/features/detector_modern_matchers_60pct_20261107>`,
:doc:`ONNX/WebGPU versus PyTorch common-descriptor test at 60% traction
<../_generated_results/features/detector_onnx_common_descriptor_60pct_20261107>`,
:doc:`progressive tensile frames 040/100/199
<../_generated_results/features/detector_onnx_progressive_20261107>`,
:doc:`held-out synthetic repeatability
<../_generated_results/features/dic_detector_heldout_20261107>`, and the
:doc:`common-descriptor DaD/DeDoDe/HORG/DIC comparison
<../_generated_results/features/detector_common_descriptor_20261107>`. A
separate 60% stress report is linked from :doc:`the benchmark index
<../benchmarks>`.

To reproduce the ONNX descriptor profile, export the static model with
``python/benchmarks/export_dedode_descriptor_onnx.py`` and run
``python/benchmarks/profile_dedode_descriptor_onnx.py MODEL.onnx
--output profile-summary.json``. The profiler reports per-operator provider
counts and summed ONNX Runtime event durations; those event durations are not
hardware GPU timestamps.

The follow-up training set adds piecewise-discontinuous motion with an invalid
gap. On 24 held-out generated pairs, the report separates repeatability by
motion family and compares the checkpoint trained with discontinuities against
the earlier checkpoint. This remains a detector-only experiment; it does not
measure descriptor matching or IC-GN convergence:
:doc:`discontinuous-motion training study
<../_generated_results/features/dic_detector_discontinuous_training_20261107>`.
The corresponding 4K WGSL/PyTorch score check is available as
:doc:`raw parity data
<../_generated_results/features/dic_detector_discontinuous_wgsl_parity_4k_20261107.json>`.
