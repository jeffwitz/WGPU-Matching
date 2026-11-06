# WGPU-Matching

WebGPU feature matchers for initializing digital image correlation (DIC). The Python package provides DeepMatching, HORG-DIC, and CPM-HOG through `wgpu-py`; compute kernels run as WGSL shaders, with no project-specific native extension to compile.

These methods estimate initial correspondences at points of interest. They do not compute a complete DIC displacement field or perform IC-GN refinement.

DeepMatching, HORG, and CPM have browser and native Python frontends. They load the same canonical WGSL sources from `python/src/wgpu_matching/shaders/`; the Python package embeds them as resources, while the browser fetches them from the repository's static tree. Their shared option defaults, constraints, and coordinate conventions live in `python/src/wgpu_matching/matcher-contracts.json`. A deterministic Python/browser parity check covers translation, a mild affine warp, irregular manual seeds, and invalid-option rejection: `node tests/run_matcher_parity.mjs`. This is a small integration control; Live-DIC analytic benchmarks remain the scientific comparison.

## Python quick start

Requirements: Python 3.11 or later and a working WebGPU backend.

```console
python -m pip install .
python python/examples/quickstart.py
```

The example creates a deterministic speckle image pair with a known `(7, 5)` pixel translation, then runs the direct DeepMatching, INRIA-compatible DeepMatching, and HORG-DIC matchers. See the [first-matches tutorial](docs/tutorials/first-matches.rst) for a walkthrough and the [API reference](docs/reference/api.rst) for options and result fields.

## Documentation

The English Sphinx documentation follows the four Diátaxis paths: [tutorials](docs/tutorials/index.rst), [how-to guides](docs/how-to/index.rst), [explanations](docs/explanation/index.rst), and [reference](docs/reference/index.rst). The [benchmark report](docs/benchmarks.rst) describes the datasets, error metrics, timing boundaries, and limitations, with links to the raw results.

Build the documentation locally with:

```console
python -m pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```

## Current benchmark snapshot

On three UHD Live-DIC frames and a shared grid of 32 points per frame, HORG accepted 87/96 points and DeepMatching accepted 88/96. Among accepted points, 85/87 HORG matches and 79/88 DeepMatching matches were within 2 px of the analytic displacement. Conditional RMSE was 0.557 px for HORG and 1.378 px for DeepMatching. Mean repeated Python call times were approximately 0.151 s and 1.319 s, respectively, on the recorded adapter.

This is one workload and one adapter. The benchmark page reports results per frame and documents the timing protocol. Its 60% tensile test also includes a global affine preconditioning experiment; because that synthetic series has an exact global affine field, its result should not be generalized to heterogeneous deformation.

## Browser demo

The repository also contains a standalone WebGPU browser demo. Serve the repository root over HTTP and open `http://localhost:8000/` in a WebGPU-capable browser:

```console
python -m http.server 8000
```

The browser and Python APIs include **experimental CPM-HOG**. They build the validated INRIA-style HOG descriptors on WebGPU, then run deterministic sparse-grid PatchMatch passes with spatial propagation, random search, and forward/backward consistency. Requested DIC seeds are associated with measured work-grid points and receive a local affine fit when enough accepted neighbors are available; metadata preserves measured positions, support, fit residuals, and affine parameters.

This is a CPM-inspired adaptation, not a numerically equivalent port of the original. The paper scores dense 128D SIFT patches with an L1 cost; this implementation pools the repository's INRIA-style HOG descriptors into a factor-two pyramid and uses normalized cosine patch cost. HOG is computed once at base resolution, and displacements are quantized to that descriptor scale. The local affine fit prepares seed initializations but does not refine them with IC-GN. The original CPM repository is GPL-3.0; no source code from it is copied. The known-translation browser smoke passes at 0 px error. The published 60% CPM row is a pre-correction historical measurement and needs a rerun; current large-strain performance is unqualified. See the [three-method 60% comparison](results/uniaxial/tensile_60pct/three_method_browser_60.md) and the [validation harness](tests/cpm-browser-smoke.html).

For development checks and CI details, see [Contributing](docs/contributing.rst) and the [quality workflow](.github/workflows/quality.yml).
