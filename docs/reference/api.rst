Python API
==========

The public package exports DeepMatching, HORG-DIC, and CPM-HOG through
``WGPUMatching`` and their corresponding option records:

The shared matcher defaults and coordinate conventions are recorded in
``python/src/wgpu_matching/matcher-contracts.json``. Python's option records
load their defaults from this file; the browser UI and browser matcher
functions use the same file. The parity check compares both option mappings
against the manifest before comparing GPU results.

The manifest also declares supported numeric ranges, DeepMatching modes, and
the maximum manual-seed count. The browser and Python APIs reject invalid
options before allocating GPU resources; seed coordinates must be finite and
inside the source image.

Matcher and device lifecycle
----------------------------

.. autoclass:: wgpu_matching.WGPUMatching
   :members:
   :special-members: __init__, __enter__, __exit__

Option records
--------------

.. autoclass:: wgpu_matching.HORGOptions
   :members:

.. autoclass:: wgpu_matching.DeepMatchingOptions
   :members:

.. autoclass:: wgpu_matching.CPMOptions
   :members:

Results
-------

.. autoclass:: wgpu_matching.MatchField
   :members:

.. autoclass:: wgpu_matching.Match
   :members:

Image utility
-------------

.. autofunction:: wgpu_matching.load_gray
