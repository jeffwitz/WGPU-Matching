Contributing and quality checks
===============================

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
