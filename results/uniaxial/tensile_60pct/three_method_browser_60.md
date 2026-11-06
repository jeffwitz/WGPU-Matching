# Three-method comparison at 60% tensile strain

This is a synthetic stress test, not an experimental validation. Frame 199 is an extrapolation of the Live-DIC tensile image to 60% engineering strain. Sixteen evaluation seeds were used; all have analytically visible target locations. The source images and analytic displacement field are documented in the adjacent series report.

**Historical CPM result:** this browser comparison predates the correction that made CPM's propagation-grid dimensions follow matching-support size independently of the DIC window. The DeepMatching and HORG rows are unaffected, but the CPM row, timings, and proposal diagnostics below are not measurements of the current CPM implementation. Rerun this case before using CPM's 60% figures as current results.

The browser ran three full calls per method on the same decoded images and WebGPU device. Timings include per-call preprocessing, GPU work, pipeline creation, synchronization, and result readback; image decoding is excluded. Chromium was launched with Vulkan/WebGPU flags. Its adapter information was unavailable, so the selected adapter cannot be confirmed. See the raw JSON and CSV beside this report.

| Method | Accepted | Accepted with error <5 px | Repairable coverage | Accepted ≤2 px | Accepted RMSE (px) | Median time (s) |
|---|---:|---:|---:|---:|---:|---:|
| DeepMatching-INRIA | 6/16 | 5/16 | 31.25% | 2/16 | 147.423 | 1.307 |
| HORG | 0/16 | 0/16 | 0% | 0/16 | — | 0.146 |
| CPM-WGPU | 0/16 | 0/16 | 0% | 0/16 | — | 0.517 |

“Repairable coverage” is the number of requested seeds with an accepted initializer whose vector error is below 5 px, divided by all 16 visible seeds. The five DeepMatching vectors under 5 px do not imply that the other accepted vectors are useful: the accepted-set RMSE is 147.423 px. CPM has no accepted seed after strict reciprocal validation and local-affine support requirements. Its 16 rejected proposals have 757.235 px RMSE; this is reported as a diagnostic, not as accepted accuracy. The work grid has zero reciprocal/usable locations in this run.

For this case, CPM is slower than HORG and faster than full DeepMatching, but it does not provide a usable initialization. The timing therefore does not constitute a performance win for DIC. CPM used a 32 px rigid matching support, separate from the 128 px DIC window, a 32 px work-grid step, a 1024 px search radius, 16 stratified initial hypotheses, and local affine radius 64 px. These settings and every per-seed result are in the [raw JSON](three_method_browser_60.json); a compact table is in the [CSV](three_method_browser_60.csv).

The matching-support sweep and analytic-oracle cost analysis are reported in [`results/cpm/oracle_diagnostics_60.md`](../../cpm/oracle_diagnostics_60.md). They show that support size alone does not cure this failure: the best candidate can have a worse descriptor cost than the known true displacement, while the larger support can make the true affine deformation a poor fit to the rigid-patch cost.
