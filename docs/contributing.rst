Contributing and quality checks
===============================

WGSL source and frontend parity
-------------------------------

The canonical WGSL files live in ``python/src/wgpu_matching/shaders/``. Python loads them as package resources, and the browser modules fetch those same checked-in files by relative URL. Do not add a second shader copy at the repository root. ``python/tests/test_shader_source_layout.py`` checks the single-source layout and the package-data rule.

The shared ``matcher-contracts.json`` records Python/browser option names, defaults, constraints, and coordinate conventions for DeepMatching, HORG, and CPM. Python option dataclasses and browser engine defaults load from that file, and the browser form is populated from it at startup. The common DIC defaults are a 128 px window and 512 px stride; DeepMatching also defaults to a 512 px root, two downscale levels, and a 256 px search radius, while HORG defaults to reduction 4 and 1024 features. CPM defaults to a 32 px matching support on a 16 px work grid.

The same manifest declares scalar ranges, enum values, and the 512-point manual-seed limit. The browser and Python APIs validate options against those constraints before starting GPU work. Manual coordinates are also checked for finite values and image bounds. Keep the constraints in the manifest and update both validation tests when adding an option.

The browser/Python parity fixture runs DeepMatching and HORG on a known ``(8, -4)`` px translation, a mild affine warp, and three irregular fractional manual seeds. CPM runs those translation seeds with a 64 px DIC window and a separate 32 px matching support; parity includes work-grid counts, local affine results, reciprocity, and oracle costs. For DeepMatching and HORG, the harness compares mapped result fields and requires at least six accepted estimates per grid case to be within 2 px of analytic truth. CPM must accept all three seeds within 2 px for this small translation fixture. Run it on a machine with Python WebGPU and Chromium WebGPU:

.. code-block:: console

   python -m pip install -e .
   node tests/run_matcher_parity.mjs

Set ``PYTHON`` or ``CHROMIUM_BIN`` if the executables are not named ``python3`` and ``google-chrome``. The parity runner is part of the software-Vulkan CI job and checks visible defaults for all three methods against the shared manifest. This deterministic control is not a substitute for the full Live-DIC analytic benchmarks; broader deformation parity remains to be measured.

Continuous integration checks the Python package, pure-Python contract tests, lint, wheel resources, and a strict Sphinx HTML build. Those checks do not require a GPU; hardware matcher benchmarks remain explicit experiments.

During each Sphinx build, the report-page extension scans `results/`. Markdown reports become themed pages; CSV and JSON files get data pages with a complete-file download. CSV pages render a 50-row preview so wide or long raw files remain usable in the browser. Generated source pages live under the ignored `docs/_generated_results/` directory and are recreated automatically.

Set up a development environment
---------------------------------

From the repository root:

.. code-block:: console

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e ".[dev,test,docs]"

Run the checks locally
----------------------

.. code-block:: console

   ruff check python/src python/benchmarks python/examples python/tests docs/_ext
   pytest -q
   sphinx-build -b html -W --keep-going docs docs/_build/html
   python -m build

The tutorial example requires a WebGPU adapter. Run it on a machine with a working adapter:

.. code-block:: console

   python python/examples/quickstart.py

Documentation structure
------------------------

The documentation is organized around user needs:

* **Tutorials** guide a first successful run.
* **How-to guides** solve specific operational tasks.
* **Explanations** discuss algorithm behavior, DIC context, and evidence.
* **Reference** records API signatures, option values, output fields, and metric definitions.

Tie benchmark claims to a report and raw data. State image size, seed population, acceptance denominator, adapter, and timing boundary. Do not present synthetic extrapolation as experimental validation, or method-specific scores as interchangeable.
