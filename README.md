# WGPU-Matching

WebGPU feature matchers for initializing digital image correlation (DIC). The Python package provides DeepMatching and HORG-DIC through `wgpu-py`; compute kernels run as WGSL shaders, with no project-specific native extension to compile.

These methods estimate initial correspondences at points of interest. They do not compute a complete DIC displacement field or perform IC-GN refinement.

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

The browser UI now includes an **experimental CPM-WGPU** path. It builds the validated INRIA-style HOG descriptors on WebGPU, then runs deterministic sparse-grid PatchMatch passes with spatial propagation, random search, and forward/backward consistency. Requested DIC seeds are associated with the nearest directly evaluated work-grid center; the CSV preserves both coordinates and the association distance.

This first implementation is a CPM-inspired prototype, not a complete CPM port. The paper uses factor-two image pyramids, transfers flow between levels, performs interleaved propagation and exponentially shrinking random search, and scores dense 128D SIFT patches with an L1 cost. This prototype pools the repository's INRIA-style HOG descriptors into a factor-two pyramid, uses a normalized cosine patch cost and synchronous four-neighbor ping-pong GPU passes. HOG is computed once at the base resolution rather than recomputed at every pyramid level. Displacements are quantized to that descriptor scale; local affine initializations are not implemented yet. The original CPM repository is GPL-3.0; no source code from it is copied. The synthetic smoke run passes at 0 px error for a known translation, but CPM currently accepts no matches on the 60% tensile frame. See the [three-method 60% comparison](results/uniaxial/tensile_60pct/three_method_browser_60.md) and the [validation harness](tests/cpm-browser-smoke.html).

For development checks and CI details, see [Contributing](docs/contributing.rst) and the [quality workflow](.github/workflows/quality.yml).
