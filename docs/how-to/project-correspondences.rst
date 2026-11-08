Project matches to affine or Gaussian-RBF patch initializers
============================================================

The projection stage accepts direct point pairs from any detector/matcher and
builds a shared displacement model. It can turn that model into the six
translation-plus-gradient parameters expected by an affine IC-GN window.

Collect direct HORG correspondences
-----------------------------------

HORG normally returns local affine estimates at DIC seeds. To use its underlying
descriptor matches as observations, request them explicitly. They include the
Lowe-ratio rejections and their ratio values for later review.

.. code-block:: python

   from wgpu_matching import CorrespondenceSet, HORGOptions, WGPUMatching

   matcher = WGPUMatching(device_type="gpu")
   horg = matcher.match_horg(
       reference,
       target,
       options=HORGOptions(max_features=1024),
       include_correspondences=True,
   )
   observations = CorrespondenceSet.from_match_field(horg)

DeepMatching and CPM fields can also be adapted when they provide direct target
coordinates. A field that only contains an already fitted local displacement
is not expanded into synthetic point pairs. This avoids treating an inferred
field as independent measurements.

For these two matchers, explicitly export the underlying measured coordinates.
DeepMatching exports atomic pairs; CPM exports work-grid observations, preserving
each method's native acceptance flags:

.. code-block:: python

   from wgpu_matching import CPMOptions, DeepMatchingOptions

   deep = matcher.match_deepmatching(
       reference, target, options=DeepMatchingOptions(), include_correspondences=True
   )
   deep_observations = CorrespondenceSet.from_match_field(deep)

   cpm = matcher.match_cpm(
       reference, target, options=CPMOptions(), include_correspondences=True
   )
   cpm_observations = CorrespondenceSet.from_match_field(cpm)

HORG's normal output is already a local affine estimate, so request its raw
descriptor pairs as shown above. The same projection API accepts point pairs
from learned matchers through ``from_preprocessed_pairs``.

Adapt external matcher coordinates
----------------------------------

For XFeat, ALIKED/LightGlue, EfficientLoFTR, or another matcher, pass its direct
source/target keypoints through the explicit resize, padding, and crop mapping.
Source and target scales are separate and use ``(x, y)`` order.
The adapter implements ``p = scale * (native - crop) + padding``. For a
pixel-centred raster resize with scale ``s`` and no leading padding, supply
``padding = (s - 1) / 2`` on each axis. This yields the inverse mapping
``native = (p + 0.5) / s - 0.5``. Bottom/right padding adds no leading offset.

.. code-block:: python

   observations = CorrespondenceSet.from_preprocessed_pairs(
       source_keypoints,
       target_keypoints,
       image_size=(native_width, native_height),
       matcher_id="EfficientLoFTR",
       source_scale_xy=(model_width / native_width, model_height / native_height),
       target_scale_xy=(model_width / native_width, model_height / native_height),
       source_padding_xy=((model_width / native_width - 1) / 2,
                          (model_height / native_height - 1) / 2),
       target_padding_xy=((model_width / native_width - 1) / 2,
                          (model_height / native_height - 1) / 2),
   )

`CorrespondenceSet` coordinates are converted to original image pixels before
fitting. Scores are retained as provenance; uncalibrated scores from different
matchers are not compared or used as probabilities. Only direct, finite,
in-image rows marked accepted enter the fit. Rejected, inferred, invalid, and
duplicate rows remain listed in the model diagnostics.

Fit and export affine IC-GN parameters
--------------------------------------

.. code-block:: python

   import numpy as np
   from wgpu_matching import ProjectionOptions

   seeds = np.asarray([[300.5, 220.0], [620.0, 220.0]])
   model = matcher.fit_initialization_field(
       observations,
       (native_width, native_height),
       options=ProjectionOptions(basis="gaussian_rbf"),
   )
   patches = model.initialize_patches(
       seeds,
       patch_size=(128, 96),
       projection="support_fit",
       seed_ids=(12, 13),
   )
   matcher.close()

Each result is marked ``estimated_from_field`` and returns
``[u, v, du/dx, du/dy, dv/dx, dv/dy]`` at the exact requested center. The
support RMS and maximum errors quantify how well the smooth field is represented
by one affine model across that DIC window. They do not predict IC-GN convergence.
Use ``projection="taylor"`` to use the analytic center gradient instead.

The affine fit and the final affine/RBF IRLS solve run in WGSL on the active
WebGPU device, as does patch evaluation. Automatic RBF candidate selection is a
bounded, deterministic float64 CPU cross-validation step. The selected fit uses
FP32 GPU normal equations and a robust scale estimated by a float64 CPU pilot;
the model records these backends, timings, residuals, weights, candidate scores,
and any non-convergence. ``irls_damping`` can relax robust-weight updates; its
default is zero (standard IRLS). This prototype uploads host correspondence
arrays to wgpu-py, so it does not claim zero-copy interoperability with ONNX
Runtime.

Serialize coefficients and diagnostics with ``model.to_dict()``. Keep the
``WGPUMatching`` context open while evaluating a GPU-bound model.
