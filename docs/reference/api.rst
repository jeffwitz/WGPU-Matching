Python API
==========

The public package exports the following API from `wgpu_matching`:

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

Results
-------

.. autoclass:: wgpu_matching.MatchField
   :members:

.. autoclass:: wgpu_matching.Match
   :members:

Image utility
-------------

.. autofunction:: wgpu_matching.load_gray
