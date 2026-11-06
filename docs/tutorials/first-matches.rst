Find your first displacements
=============================

This tutorial creates a deterministic speckle image, shifts it 7 px right and 5 px down, then computes correspondences with direct DeepMatching, hierarchical DeepMatching, and HORG-DIC.

Prerequisites
-------------

* Python 3.11 or later.
* A WebGPU adapter (a hardware GPU is recommended).
* The package installed from the repository root: `python -m pip install .`

Create a reference and target pair
----------------------------------

The pattern comes from a fixed random seed. Repeating random 4 × 4 blocks creates local speckle structure for both descriptor families. The target has a black border where shifted source pixels have no support.

.. literalinclude:: ../../python/examples/quickstart.py
   :language: python
   :lines: 1-24

Compute all three fields
------------------------

Save the complete listing as `quickstart.py` and run it:

.. code-block:: console

   python quickstart.py

The script sends the same three source centers to all three methods. Its patch sizes are smaller than the UHD defaults so the example runs quickly. It prints the active adapter, each displacement, and the matcher's own `usable` decision.

.. literalinclude:: ../../python/examples/quickstart.py
   :language: python
   :lines: 25-

The estimates should be close to `(+7, +5)` px. The INRIA hierarchy works on an atomic grid and returns quantized displacements; direct matching and HORG can return subpixel values. `usable` comes from method-specific acceptance rules and is not a score that can be compared between methods.

Next
----

* To select an adapter or pass measured seeds, see :doc:`the how-to guides <../how-to/index>`.
* To understand why the outputs differ, read :doc:`the matcher explanations <../explanation/index>`.
* For full Live-DIC metrics, see :doc:`the benchmark report <../benchmarks>`.
