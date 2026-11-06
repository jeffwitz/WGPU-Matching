# CPM-WGPU validation status

The browser CPM-HOG implementation remains experimental and is not qualified as a reliable initializer for large deformation. Current progressive results and diagnostics are linked below.

## Current prototype

- The INRIA-style HOG base descriptors are built on WebGPU and average-pooled into a factor-two descriptor pyramid.
- At each level, deterministic GPU passes use four-neighbor spatial propagation and hash-seeded random search. The best displacement is doubled when moving to the next finer level.
- The cost is normalized cosine similarity over the HOG patch. Forward/backward vectors are compared by strict bilinear interpolation of the inverse field, rejecting interpolation across invalid grid points.
- Work-grid locations are distinct from requested DIC seed centers. Requested centers receive local affine projection when enough valid reciprocal measurements support the fit; the output retains the direct measurement coordinates, support counts, and fit error. Unsupported fits remain unusable.
- A deterministic stratified initialization is used at the coarsest level. An analytic affine oracle can be enabled in diagnostics to compare found and true-displacement costs by scale; it is not used to produce matches.

The DIC window size is now independent of the rigid patch support used by CPM-HOG. A 128 px DIC window can, for example, use a 32 px matching support. The latter controls patch cost, border validity, and effective pyramid depth.

This differs from the paper's per-level dense SIFT descriptors and L1 cost. The paper's implementation uses GPL-3.0; this repository does not copy its code. See the implementation notes in the repository's `AGENTS.md` and the user-facing summary in the root `README.md`.

## Validation status

The deterministic synthetic harness in `tests/cpm-browser-smoke.html` returns 3/3 accepted matches with 0 px error on the tested `(8,-4)` translation when run with Chromium Vulkan flags. On the original progressive tensile frames, CPM accepts 32/32 seeds at 0.11%, 2/28 at 3.84%, and 0/28 at 10%; at 10%, 13/28 raw proposals are within 5 px but all are rejected. The progressive comparison and cost diagnosis are in [`progressive_tensile_040_100_199.md`](progressive_tensile_040_100_199.md) and [`oracle_diagnostics_60.md`](oracle_diagnostics_60.md). At 60%, CPM accepts 0/16 and all-proposal error RMSE is 757.235 px; see the [three-method report](../uniaxial/tensile_60pct/three_method_browser_60.md).

The oracle experiment finds both search and objective limitations: at 32 px support the known displacement has lower cost than the current candidate for many seeds, while at 64 px support the rigid patch often scores the affine truth poorly. Stratified initialization and a larger local-affine radius alone have not qualified CPM. It should not be used for IC-GN initialization on these strains without further validation.

## Automated checks

The browser translation smoke harness is `tests/cpm-browser-smoke.html`; the `cpm-webgpu-smoke` CI job runs it under headless Chrome with Mesa software Vulkan. The larger scientific comparisons are reproducible browser harnesses in `tests/cpm-progressive-browser.html` and `tests/cpm-tensile-60-browser.html`; they need the read-only neighboring `tests_gpuio` assets for the original progressive frames and remain separate experiments, not CI tests.
