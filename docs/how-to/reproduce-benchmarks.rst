Reproduce the Python benchmarks
===============================

The Live-DIC images are not included in this repository. You need a `uhd_tensile` directory containing `reference.png`, `frame_040.png`, `frame_100.png`, and `frame_199.png`. Pass its path explicitly:

.. code-block:: console

   python python/benchmarks/benchmark_live_dic.py \
       --live-dic /path/to/uhd_tensile \
       --repeats 5 \
       --output /tmp/python_runtime_comparison.csv

The script reads analytic truth archived under `results/uniaxial`; it does not pass truth to either matcher. Seeds outside the target are excluded from quality metrics. Timing covers the detector call after image loading and device creation. The first call is reported separately from repeated calls.

To compare full-grid and sparse DeepMatching on selected seeds:

.. code-block:: console

   python python/benchmarks/benchmark_sparse_deepmatching.py \
       --frame 040 \
       --seed-counts 1,4,8,16,32 \
       --repeats 2 \
       --output /tmp/sparse_deepmatching.csv

Use seed counts no larger than the number of visible grid seeds in the chosen frame. The full/sparse parity result in the CSV applies only to the evaluated seeds; sparse mode restricts reciprocal competition to selected roots.

To rerun the large-strain comparison on the generated images in this repository:

.. code-block:: console

   python python/benchmarks/benchmark_tensile_60pct.py \
       --dataset results/uniaxial/tensile_60pct \
       --output /tmp/tensile_comparison.csv

This series is synthetic. Above 10% strain, it extrapolates a material model whose experimental validity has not been established. Analytic truth exactly describes the image remapping used by the generator; it does not validate the constitutive model.

Run the affine RANSAC/ECC preconditioning experiment with:

.. code-block:: console

   python python/benchmarks/experiment_affine_deepmatching.py --ecc --seed 42

This command writes JSON results under `results/uniaxial/tensile_60pct`. The series has an exactly global affine image field, so it is an upper-bound experiment. Before using this approach on heterogeneous deformation, define a valid-support mask and evaluate local residuals.
