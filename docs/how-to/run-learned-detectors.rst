Run and export learned feature detectors
=========================================

This guide uses optional official weights for DaD and DeDoDe v2. For the
scientific rationale, read :doc:`../explanation/detector-principles`.
Run commands from the repository root in an isolated Python environment.

Install the optional research dependencies and the official DeDoDe source
package in an isolated environment:

.. code-block:: console

   python -m pip install -e ".[research]"
   python -m pip install "git+https://github.com/Parskatt/DeDoDe.git"

The DaD pip package is maintained by its authors and supplies the official
detector implementation. DeDoDe's upstream repository describes its detector,
descriptor-B code, and their weights as MIT-licensed; the DINOv2-based
descriptor-G has a separate Apache-2.0 license and is not used here. See the
upstream `DaD repository <https://github.com/Parskatt/dad>`_ and
`DeDoDe repository <https://github.com/Parskatt/DeDoDe>`_ for current notices.
Model weights are downloaded to PyTorch's external cache by default and are
never included in the WGPU-Matching package. Benchmark records include their
checkpoint SHA-256 when available.

Optional ONNX Runtime/WebGPU inference
--------------------------------------

DaD and DeDoDe v2 can also be exported as optional ONNX keypoint graphs and
run through ONNX Runtime's WebGPU execution provider. This path does not import
PyTorch at inference time. For inference, install the optional runtime. To
export models, install the research and ONNX export extras, then add the
upstream DeDoDe package:

.. code-block:: console

   python -m pip install -e ".[onnx-webgpu]"
   python -m pip install -e ".[research,onnx-export,onnx-webgpu]"
   python -m pip install "git+https://github.com/Parskatt/DeDoDe.git"

Export uses the official detector weights and the standalone benchmark tool:

.. code-block:: console

   python python/benchmarks/export_learned_detector_onnx.py \
     --method dad --output /path/to/dad-keypoints.onnx --output-mode keypoints
   python python/benchmarks/export_learned_detector_onnx.py \
     --method dedodev2 --output /path/to/dedode-v2-keypoints.onnx --output-mode keypoints

The graph returns a fixed top-K point capacity (1024 by default; larger
capacities can be requested with ``--max-keypoints``) and accepts dynamic
spatial dimensions. ``OnnxWGPUKeypointDetector`` converts source
coordinates back to the original image and records preprocessing, ONNX
``session.run``, postprocessing, output-byte and total-call measurements. The
adapter's ``max_side`` controls the longest model-input dimension; an UHD source
is resized for detection, so this is not native-resolution 4K inference. It
returns only point coordinates and scores from ONNX, not the dense detector
map. ONNX Runtime returns these outputs as CPU arrays; this is a small readback
(12 KiB for K=1024), and no allocation is shared with ``wgpu-py``.

For a common DeDoDe-B descriptor and WGSL affine-RANSAC comparison, run the
existing experiment with explicitly exported graphs and a candidate budget
within each graph's fixed capacity:

.. code-block:: console

   python python/benchmarks/experiment_learned_detectors.py \
     --dad-onnx /path/to/dad-keypoints.onnx \
     --dedode-onnx /path/to/dedode-v2-keypoints.onnx \
     --detectors DaD-ONNX-WebGPU DeDoDe-v2-ONNX-WebGPU \
     --max-side 640 --budgets 512 --candidate-multiplier 2 \
     --cases synthetic_60pct --repeats 3

The ONNX detector and WGSL RANSAC do not make the complete pipeline
GPU-resident: DeDoDe-B extraction and mutual descriptor matching in this
comparison remain Python/NumPy/PyTorch reference stages. The timing report
separates the detector's input preparation, ONNX call, coordinate processing
and total call from descriptor, matching and geometry stages.
