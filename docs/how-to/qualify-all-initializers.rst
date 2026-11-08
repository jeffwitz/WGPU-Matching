Re-run the complete initializer qualification
==============================================

Use this procedure to rebuild the current comparison and its complete visual
gallery. It evaluates every current pipeline on progressive traction, synthetic
traction from 10 to 60 percent, and the masked experimental Adrien pair.
This is a field-initialization benchmark, not a detector-only repeatability or
an IC-GN convergence benchmark.

Prepare inputs and optional models
------------------------------------

Use an environment containing the package, the optional ONNX/WebGPU providers,
the official DeDoDe-B reference descriptor and the external model files.
The main package does not require PyTorch. This benchmark uses it for the
common descriptor and therefore needs the research environment.

The source paths and default model locations are explicit in
``python/benchmarks/qualify_all_initializers.py``. Supply a JSON object through
``--models`` to override model paths without editing the script. Keys are
``dad_b``, ``dedode_b``, ``xfeat``, ``efficientloftr``,
``efficientloftr_dynamic``, ``raco`` and ``dic_learned_b``. The manifest stores
all model hashes; fixed-budget detector exports must contain 8192 outputs.
Use the weight/export procedures in :doc:`run-learned-detectors`.

The fixture repositories are read-only. The benchmark writes into its own
output directory and leaves all historical result files intact. Check free
space: all comparisons include images, animations, raw pairs and fitted-model
archives, so a complete run is substantially larger than a summary table.

Run one frozen campaign
------------------------

From the repository root, with the research environment active:

.. code-block:: bash

   PYTHONPATH=python/src OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
     python python/benchmarks/qualify_all_initializers.py \
     --output results/qualification/all_methods_20261107 \
     --repeats 3 --render-width 1280

The command prepares models once, warms each case/method, then measures three
complete affine calls. Device/model initialization, image decoding and figure
rendering are outside the timed region. Each call processes both images;
reference preparation is not cached. Do not run other GPU benchmarks during
this measurement.

To resume the exact same campaign, add ``--resume``. Changed images, model
hashes, protocol or implementation fingerprints require a new output directory;
the script refuses to mix those results. ``--methods`` and ``--cases`` allow
small smoke runs in separate directories. Never replace the full campaign
with an undisclosed subset.

Generate and inspect the report
--------------------------------

.. code-block:: bash

   python python/benchmarks/plot_qualification.py \
     results/qualification/all_methods_20261107
   PYTHONPATH=python/src python python/benchmarks/report_qualification.py \
     results/qualification/all_methods_20261107

The plot script requires the optional Matplotlib package and only reads recorded
results; it does not run inference.

Run the controlled ablation after the main timing campaign, then audit all
records and regenerate the report:

.. code-block:: bash

   PYTHONPATH=python/src python python/benchmarks/ablate_qualification_projection.py \
     results/qualification/all_methods_20261107 \
     --output results/qualification/raw_projection_ablation_20261107
   python python/benchmarks/validate_qualification.py \
     results/qualification/all_methods_20261107

The ablation uses the exact cached direct pairs with native acceptance, without
the additional common local filter. It tests affine fields for every case and
both basis requests at 60 percent and for Adrien. It measures projection only,
without rerunning a detector or claiming a new full-pipeline time.

The report command creates the index, every per-case gallery and contact sheets under
``visual_review``. Each method includes both an explicit affine request and
a Gaussian-RBF request with recorded effective basis. An automatic request may
select affine. Every failure has a visual comparison too. A fit from rejected
proposals is labelled diagnostic and contributes no accepted initialization.

The whole-image figures are bounded previews sampled from native target
pixels. For the 60-percent and Adrien cases, fixed central crops also show
native pixel detail. These images are visualizations of the initialization
field, not a second image-based refinement. Plain reference and corrected PNG
files are kept beside each GIF for external viewers.

Interpret the verdict correctly
---------------------------------

Passing 60 percent requires all visible requested centres to have finite
estimated displacement error strictly below 5 pixels. Exactly 5 pixels fails.
The table separately reports at most 2 pixels, correct coverage over every
visible request, conditional error, false estimates and actual full-call times.

No analytic truth selects observations or a fitted model. The same local
RANSAC verifies observed pairs for every method before field projection.
Raw and rejected observations remain downloadable. The current GPU field
solver's 8192-pair limit uses a documented spatial capacity reduction.

Adrien has no analytic field: visual agreement, consensus residual and
photometric difference do not establish physical accuracy. Its gallery links
the method's separate synthetic 60-percent verdict. See
:doc:`../explanation/field-projection` for the smooth-field assumptions.
