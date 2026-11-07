Your first interest-point detector
===================================

This lesson detects locations on one generated speckle image. You will inspect
what a detector returns before attempting to describe or match points. It
requires the core package and a working WebGPU adapter, with no model weights.

Create an image
----------------

Install from the repository root with ``python -m pip install .``. In Python:

.. code-block:: python

   import numpy as np
   from PIL import Image, ImageFilter
   from wgpu_matching import DICDetectorOptions, WGPUMatching

   rng = np.random.default_rng(27)
   binary = (rng.random((256, 256)) > 0.75).astype(np.uint8) * 255
   image = np.asarray(
       Image.fromarray(binary).filter(ImageFilter.GaussianBlur(1.2))
   ).copy()

Blurring gives the spots a finite spatial size. This small teaching texture
is not a scientific validation dataset.

Detect and inspect
------------------

.. code-block:: python

   with WGPUMatching(device_type="gpu") as matcher:
       result = matcher.detect_dic(
           image, options=DICDetectorOptions(max_keypoints=128)
       )
   features = result.features
   print(features.xy.shape)
   print(features.xy[:5])
   print(features.scores[:5])

The coordinate array has two columns, x then y, in source pixel units.
There can be fewer than 128 points: finite support, suppression and tile
quotas limit the selected set. Scores rank the tensor response. Exact
floating values may vary with the adapter.

Display the detections
-----------------------

.. code-block:: python

   from PIL import ImageDraw

   preview = Image.fromarray(image).convert("RGB")
   draw = ImageDraw.Draw(preview)
   for x, y in features.xy:
       draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline="red")
   preview.save("first-detector.png")

Open the PNG and inspect how selected locations cover the texture. You have
computed locations in a single image; no displacement was estimated. Continue
with :doc:`../explanation/detector-principles` to understand the next stages,
or :doc:`../how-to/run-learned-detectors` to run the optional learned models.
