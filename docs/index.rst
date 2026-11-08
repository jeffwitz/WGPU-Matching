WGPU-Matching
=============

**WebGPU initializers for digital image correlation (DIC).**

WGPU-Matching provides three Python methods for finding correspondences between a reference image and a deformed image: DeepMatching, HORG-DIC, and the experimental CPM-HOG initializer. Their compute kernels are written in WGSL and run through ``wgpu-py`` on a native WebGPU adapter. Installing the package does not compile a project-specific extension.

An optional Python projection layer accepts direct matches from these or
external methods, fits an affine or Gaussian-RBF displacement field, and
produces six-parameter affine initializers for requested DIC windows.

These methods provide **initial displacement estimates** at points of interest. They do not compute a full DIC field or perform IC-GN refinement.

.. image:: _static/pipeline.svg
   :alt: Native and learned matchers produce observed pairs, followed by verification and optional smooth field projection to window initializers before an external image-based solver.
   :width: 100%

Choose a path
-------------

* :doc:`Tutorial <tutorials/first-matches>` — install the package, create a reproducible speckle pair, and compute your first displacements.
* :doc:`How-to guides <how-to/index>` — select an adapter, pass explicit seeds, run a benchmark, or inspect uncertain initializations.
* :doc:`Explanations <explanation/index>` — learn how detection, description, learned feature models, the correlation hierarchy, HORG and CPM-HOG work, how they fit into DIC, and what the measurements show.
* :doc:`Reference <reference/index>` — look up the Python API, options, result fields, metrics, and raw reports.
* :doc:`Reports and raw data <_generated_results/index>` — browse rendered experiment reports, CSV previews, and complete data downloads.

Current benchmark route
-----------------------

The :doc:`complete initializer qualification
<_generated_results/qualification/all_methods_20261107/index>` compares eleven
current pipelines on progressive tensile frames, synthetic traction through
60 percent and the masked Adrien experimental pair. It includes the latest
DaD, DeDoDe v2 and learned DIC detector paths, three repeated complete affine
calls, both projection requests, raw observations and an animation for every
success or failure.

A pass at 60 percent means all visible requested centres have an estimated
error strictly below 5 pixels. A smooth field reconstructed from matches is
not an independent measurement at every centre. Adrien has no analytic truth.
The :doc:`benchmark guide <benchmarks>` separates this current protocol from
the historical local-initializer studies.

Install
-------

Python 3.11 or later and a working WebGPU backend are required. From the repository root:

.. code-block:: console

   python -m pip install .

Then follow the :doc:`getting-started tutorial <tutorials/first-matches>`. For documentation development and quality checks, see :doc:`contributing <contributing>`.

.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/index
   how-to/index
   explanation/index
   benchmarks
   _generated_results/index
   reference/index
   contributing
