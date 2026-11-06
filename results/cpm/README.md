# CPM-WGPU validation status

The browser CPM-HOG implementation is experimental. Its first accuracy, coverage, and runtime measurements are now published below; CPM did not produce any accepted matches in that trial.

An initial 60% tensile comparison is now available at [`three_method_browser_60.md`](../uniaxial/tensile_60pct/three_method_browser_60.md). It reports CPM's actual zero-accepted result alongside DeepMatching and HORG. This first CPM measurement is a diagnostic result, not validation of an accepted CPM initializer.

## Current prototype

- The INRIA-style HOG base descriptors are built on WebGPU and average-pooled into a factor-two descriptor pyramid.
- At each level, deterministic GPU passes use four-neighbor spatial propagation and hash-seeded random search. The best displacement is doubled when moving to the next finer level.
- The cost is normalized cosine similarity over the HOG patch. Forward and backward fields are checked at the nearest measured work-grid location.
- Work-grid locations are distinct from requested DIC seed centers. CSV output retains both positions and their distance.

This differs from the paper's per-level dense SIFT descriptors and L1 cost. The paper's implementation uses GPL-3.0; this repository does not copy its code. See the implementation notes in the repository's `AGENTS.md` and the user-facing summary in the root `README.md`.

## Validation status

The deterministic synthetic harness in `tests/cpm-browser-smoke.html` returns 3/3 accepted matches with 0 px error on the tested `(8,-4)` translation when run with the Chromium Vulkan flags from the read-only `tests_gpuio` launcher. The three-method 60% run is described in [`the comparison report`](../uniaxial/tensile_60pct/three_method_browser_60.md). That real workload exposed a major failure: CPM returns zero accepted matches, with 691.960 px RMSE among its rejected displacement proposals. The next development step must address the coarse-level large-displacement search before the matcher is considered useful for IC-GN initialization.

## Planned benchmark record

Once execution is available, compare CPM-WGPU, DeepMatching WebGPU, and HORG-DIC on the same requested Live-DIC seeds and analytic truth for frames 040, 100, and 199. Report accepted count, accurate coverage at 2 px, conditional RMSE/median/P95/maximum, and repeated median runtime. Keep all method parameters and raw per-seed records beside the summary.
