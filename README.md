# WGPU-Matching

WebGPU feature matchers for initializing digital image correlation (DIC). The Python package provides DeepMatching, histogram-of-oriented-and-rotated-gradients DIC (HORG-DIC), and coarse-to-fine PatchMatch with histograms of oriented gradients (CPM-HOG) through `wgpu-py`; compute kernels run as WebGPU Shading Language (WGSL) shaders, with no project-specific native extension to compile.

An experimental speckle-specific keypoint detector is also available in the Python application programming interface (API). Its structure-tensor baseline and trained compact score model run in WGSL; DaD (*Distilled Reinforcement Learning for Diverse Keypoint Detection*) and DeDoDe v2 (*Detect, Don’t Describe — Describe, Don’t Detect*, second detector version) offer optional PyTorch reference and Open Neural Network Exchange (ONNX)/WebGPU detector adapters. The common DeDoDe-B descriptor still uses a processor-side reference implementation. These detector experiments are separate from the three established pair matchers and do not yet measure downstream inverse compositional Gauss–Newton (IC-GN) convergence.

These methods estimate initial correspondences at points of interest. They do not compute a complete DIC displacement field or perform IC-GN refinement.

DeepMatching, HORG, and CPM have browser and native Python frontends. They load the same canonical WGSL sources from `python/src/wgpu_matching/shaders/`; the Python package embeds them as resources, while the browser fetches them from the repository's static tree. Their shared option defaults, constraints, and coordinate conventions live in `python/src/wgpu_matching/matcher-contracts.json`. A deterministic Python/browser parity check covers translation, a mild affine warp, irregular manual seeds, and invalid-option rejection: `node tests/run_matcher_parity.mjs`. This is a small integration control; Live-DIC analytic benchmarks remain the scientific comparison.

## Python quick start

Requirements: Python 3.11 or later and a working WebGPU backend.

```console
python -m pip install .
python python/examples/quickstart.py
```

The example creates a deterministic speckle image pair with a known `(7, 5)` pixel translation, then runs the direct DeepMatching, DeepMatching compatible with INRIA (Institut national de recherche en informatique et en automatique), and HORG-DIC matchers. See the [first-matches tutorial](docs/tutorials/first-matches.rst) for a walkthrough and the [API reference](docs/reference/api.rst) for options and result fields.

## Documentation

The English Sphinx documentation follows the four Diátaxis paths: [tutorials](docs/tutorials/index.rst), [how-to guides](docs/how-to/index.rst), [explanations](docs/explanation/index.rst), and [reference](docs/reference/index.rst). The [benchmark report](docs/benchmarks.rst) describes the datasets, error metrics, timing boundaries, and limitations, with links to the raw results.

Build the documentation locally with:

```console
python -m pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```

## Current complete qualification

The [eleven-pipeline qualification](results/qualification/all_methods_20261107/index.md) re-runs all current native and learned pipelines on progressive tensile frames, synthetic traction from 10 to 60 percent and the masked Adrien pair. It includes the latest detectors, three measured complete affine calls, per-seed errors, raw observations and a reference/corrected animation for **every success and failure**. Both affine and automatically selected affine/Gaussian radial-basis-function (RBF) fields are shown.

Passing 60 percent requires all visible requested centres to have a finite displacement error **strictly below 5 pixels**. Correct coverage, conditional precision and false estimates are separate metrics. These are reconstructed field initializers, not independent image matches at each seed. The experimental Adrien images have no analytic accuracy verdict. The [raw-pair ablation](results/qualification/raw_projection_ablation_20261107/index.md) tests the same observations without the additional common local geometric verification.

For a reproducible campaign, follow the [qualification guide](docs/how-to/qualify-all-initializers.rst). The [print handbook build](docs/contributing.rst) uses the same scientific explanations and a generated chapter from the current measured results.

## Historical local-initializer benchmark snapshot

On three ultra-high-definition (UHD, 3840 × 2160 pixels) Live-DIC frames and a shared grid of 32 points per frame, HORG accepted 87/96 points and DeepMatching accepted 88/96. Among accepted points, 85/87 HORG matches and 79/88 DeepMatching matches were within 2 px of the analytic displacement. Conditional root mean square error (RMSE) was 0.557 px for HORG and 1.378 px for DeepMatching. Mean repeated Python call times were approximately 0.151 s and 1.319 s, respectively, on the recorded adapter.

This is one workload and one adapter. The benchmark page reports results per frame and documents the timing protocol. Its 60% tensile test also includes a global affine preconditioning experiment; because that synthetic series has an exact global affine field, its result should not be generalized to heterogeneous deformation.

## Browser demo

The repository also contains a standalone WebGPU browser demo. Serve the repository root over Hypertext Transfer Protocol (HTTP) and open `http://localhost:8000/` in a WebGPU-capable browser:

```console
python -m http.server 8000
```

The browser and Python APIs include **experimental CPM-HOG**. They build the validated INRIA-style HOG descriptors on WebGPU, then run deterministic sparse-grid PatchMatch passes with spatial propagation, random search, and forward/backward consistency. Requested DIC seeds are associated with measured work-grid points and receive a local affine fit when enough accepted neighbors are available; metadata preserves measured positions, support, fit residuals, and affine parameters.

This is a CPM-inspired adaptation, not a numerically equivalent port of the original. The paper scores dense 128-component scale-invariant-feature-transform (SIFT) patches with a sum-of-absolute-differences cost; this implementation pools the repository's INRIA-style HOG descriptors into a factor-two pyramid and uses normalized cosine patch cost. HOG is computed once at base resolution, and displacements are quantized to that descriptor scale. The local affine fit prepares seed initializations but does not refine them with IC-GN. The original CPM repository uses the GNU General Public License version 3.0 (GPL-3.0; GNU means “GNU’s Not Unix”); no source code from it is copied. The known-translation browser smoke passes at 0 px error. The old published 60% CPM row is historical. The complete qualification above re-runs the current implementation and separately scores its projected field; a successful small translation is not evidence of large-strain robustness. See the [three-method 60% comparison](results/uniaxial/tensile_60pct/three_method_browser_60.md) and the [validation harness](tests/cpm-browser-smoke.html).

For development checks and continuous-integration (CI) details, see [Contributing](docs/contributing.rst) and the [quality workflow](.github/workflows/quality.yml).
