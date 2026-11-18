@/home/jeff/.codex/RTK.md

# Contract: propagation-first baseline after causal ablation (4dce5d0)

Repository: WGPU-Matching. Work from commit `4dce5d0` or its descendant. The purpose is to establish whether propagation and better image-derived seeds remove the remaining need for a learned multiscale decoder. Do not implement that decoder in this contract.

## Terms and evidence rules

- **Digital image correlation (DIC):** image-based measurement of material displacement.
- **Inverse-compositional Gauss-Newton (IC-GN):** the local iterative solver used to refine DIC windows.
- **Normalized cross-correlation (NCC):** the local photometric search used here.
- **Neural deformation field (NDF):** the learned residual field used in one ablation variant.
- **WebGPU Shading Language (WGSL):** the shader language used by the GPU solve and match kernels.
- **Continuous integration (CI):** automated repository checks.
- In every matching or propagation loop, do not read analytical truth, visibility, or masks. Use them only after the predictions are fixed, for scoring.
- Preserve archived measurements. Write new dated output folders. Do not change thresholds or solve settings to pass a gate.
- Define every acronym at first use in user-facing documentation and result reports.

## Task 0 — concise, current instructions

Move the prior root `AGENTS.md` content to `docs/development-log.md` without altering that archived text. Keep this file below 32 KiB, put this contract first, and retain concise repository-wide development instructions after it. Update the stale status at the top of `results/ndf_dic_20261108/README.md`, which still reports the old 15.47% to 35.88% comparison.

## Task 1 — status-only frontier propagation

Add `--propagation frontier` to `python/benchmarks/qualify_ndf_pair_icgn.py` and propagation results to `python/benchmarks/benchmark_ndf_causal_ablation.py`, following the user's `propagation_probe.py` if available; otherwise document the implementation directly.

- Initial seeds are exactly windows accepted by the first IC-GN status.
- Visit the active frontier only. A candidate is the componentwise median of first-order predictions from accepted 8-connected neighbors. Predict displacement at the new center with each neighbor's affine gradient, and retain the neighbor gradients in the prediction.
- A window may be attempted at most three times. Stop when no eligible frontier remains.
- Do not alter the subset size, stride, interpolation, iteration count, tolerance, or minimum normalized cross-correlation.
- Per pair, report seed count, one-pass correct count, propagated correct count, oracle count, accepted errors of at least 2 px, accepted invisible windows, ring count, additional accepted windows, one-pass solve time, and propagation time. Store per-window status, parameters, attempts, and ring for audit.

Reference totals from the archived 80 per-window exports (20 pairs, 22,416 visible windows, 21,367 oracle-correct windows): affine-only 19,915; affine + NCC 20,651; affine + NDF 19,070; affine + NDF + NCC 19,917. Seeded pairs: 18/20, 19/20, 17/20, and 18/20 respectively. Gate G1: reproduce each total within 1%.

Gate G2: accepted windows with displacement error at least 2 px must be no more than 0.1% of accepted windows; enumerate every such occurrence. Report accepted invisible windows separately.

## Task 2 — baseline policy

All new campaigns must use `--field-residual none` with local NCC. Keep the qualifier's existing command-line default unchanged so archived commands remain reproducible.

## Task 3 — zero-seed rotation hypotheses

For pairs 17051005 and 17051007, rank the top three existing rotation hypotheses by the existing image-derived ranking score. Initialize each and solve the same fixed sparse subset of at most 128 windows. Select by accepted solver-status count only. Report seeds for every hypothesis. Analytical truth may score the selected result after selection, never choose it.

## Task 4 — matcher-source ablation

On the same 20 pairs, same NCC, and same propagation, compare the current learned coarse orientation/affine matcher with WebGPU HORG and DeepMatching. Use common scoring and unchanged IC-GN settings. Report pairs with at least one accepted seed, post-propagation coverage, and complete pipeline time. If an existing matcher cannot supply the required affine hypothesis without changing its algorithm, document that limitation and implement the closest existing supported path without claiming equivalence.

## Task 5 — independent generalization

Evaluate new deterministic seed groups for rigid, affine, and discontinuous fields; analytic shear bands with peak displacement gradient 0.33, 0.67, and 2.0; and the masked Adrien pair. For Adrien, report accepted counts and spatial maps only, without synthetic-truth claims.

Define regions as 8-connected components of oracle-correct windows containing at least 16 windows. A pair passes the seed-coverage check only if every such region contains at least one accepted seed. Gate G3: on a new seed group, every pair that has a seed must recover at least 97% of its oracle-correct windows after propagation. If a gate fails, report the frontier/ring where it stopped. Do not tune thresholds.

## Task 6 — translation regression anomaly

Investigate commit `18be5d9`, which lost 548 of 13,444 translation windows on raw bilinear projection, with all ten pairs regressing. Use per-window exports to identify the cause. Do not change any algorithm or threshold without a demonstrated cause.

## Completion decision

Do not train a new model or alter losses/projections. Report whether any independent family has a region whose seeded propagation remains below 97% of oracle, and whether any no-seed pair plausibly needs a learned field decoder. This evidence determines whether a decoder merits implementation. Make no real-image accuracy claim from synthetic results.

## Repository-wide instructions

- Prefer the codebase knowledge-graph tools for code discovery when they are available; otherwise use `rg` and focused source reads.
- Keep model weights outside Git unless explicitly approved. Preserve historical measurements and external read-only source trees.
- Keep Python and JavaScript on the canonical shared WGSL sources. Run relevant tests, lint, documentation, and benchmark checks before reporting completion.
- Do not report a gate as passed unless its output is present in the new report and can be reproduced from the saved command and data.

## 2026-11-11 — anchor availability audit

The P0 archive audit reproduces all totals in section 2.2 of `CdC_Codex.md`; the exact median for affine + NDF is 84.5. See `results/seed_propagation_20261111/p0_anchor_availability.md`.

The P1 oracle-seeded experiment is recorded in `results/seed_propagation_20261111/p1_oracle/README.md`. G1 passes for all five draws of four oracle seeds: 21,366–21,367 correct out of 21,367 oracle windows, with zero false accepted visible windows. These are oracle seeds, not a no-oracle qualification. P2 and P3 remain open.

P2 independent qualification is recorded in `results/seed_propagation_20261111/p2_independent/evaluation/README.md`. Heterogeneous coverage is 23,920/24,994 with no visible false acceptances. G2 is incomplete: the translation baseline rejects all near-identity prefits, local affine fallback was not calibrated across the full development set, and end-to-end timing is missing. See `results/seed_propagation_20261111/p2_status.md`.

Weighted estimates from accepted neighbours and their solver retries are documented in `results/seed_propagation_20261111/p2_weighted_estimates/README.md` and `results/seed_propagation_20261111/p2_weighted_retry/README.md`. They are post-hoc estimates, never solver-accepted measurements. The opt-in translation-affine correction and post-hoc comparison are in `results/seed_propagation_20261111/p2_translation_affine_fix/README.md`; G2 remains incomplete.

The later independent P2 confirmation in `results/seed_propagation_20261111/p2_confirmatory/evaluation/README.md` passes the numerical G2 criteria: 23,570/24,641 heterogeneous and 28,641/29,459 translation windows are correct, with no visible false accepted windows. Integrated end-to-end execution time is still unqualified. See `p2_status.md` for the baseline correction and timing limitations.

P3 results and remaining limits are indexed in `results/seed_propagation_20261111/p3_status.md`. Periodic textures yield no accepted windows; fine supports can cross an unknown crack. An analytical-boundary experiment is explicitly oracle-only. The experimental Adrien image has acceptance counts but no available truth for accuracy claims. P4 and P5 of `CdC_Codex.md` require an explicit decision before modifying the external solver or implementing the resident GPU pipeline.

The user clarified that periodic textures are outside the intended speckle-initialization scope; their result remains documented but is not a blocker. The original G2 acceptance conditions do not include execution time: those numerical conditions pass on the fresh confirmation groups. Timing is still missing under the additional condition recorded in the confirmation protocol and is required for the later performance qualification. Do not conflate those two judgments.

Local-affine escalation can now resume a partially accepted field instead of discarding it. The archived development regression is `results/seed_propagation_20261111/p2_resume_escalation_17051007_v2/README.md`; preserving a partially accepted field is unit-tested but has not yet been validated on an independent image pair. Global rotation inputs are still produced eagerly and P4/P5 remain open.

Local affine fits are limited to unresolved valid centres. Experimental `--escalate-unresolved` additionally retries unresolved components inside regions that already have anchors. On five previously inspected pairs, accepted states were exactly preserved but no further correct windows were recovered; see `results/seed_propagation_20261111/p2_unresolved_affine_five_pairs/README.md`. This option remains disabled by default and is not independently qualified.

Experimental deferred generation of forced rotations is documented in `results/seed_propagation_20261111/p2_lazy_rotation_five_pairs/README.md`. It avoids three additional exports on each of five inspected pairs and falls back successfully on the development pair 17051007. The initial affine/NCC generation already includes a global image-derived ranking. This is not a production end-to-end timing or an independent non-regression qualification; the experimental mode remains opt-in.

The deferred path was also compared on all 40 already inspected confirmation pairs: `results/seed_propagation_20261111/p2_lazy_rotation_all_forty/README.md`. It retained the same 52,211 correct windows and requested no forced rotation exports on those pairs. This retrospective comparison does not establish performance on new data. The campaign runner exposes the path as `--lazy-rotations`; default behavior remains eager.

Experimental `--shared-prefit` exports affine-only and affine-plus-local-correlation initializations from one global inference; the pilot accepts the shared archive. `p2_shared_prefit_paired/README.md` reports exact state and solver parity on a synthetic pair and median scripted preparation 14.70 s separate versus 7.55 s shared. Two further previously inspected translation/heterogeneous pairs also have exact parity. `p2_shared_lazy_pipeline_paired/README.md` reports a 14.47 s full scripted median on one pair against the earlier 45.37 s eager campaign, with exact accepted states; the campaigns were not measured simultaneously and include process startup, synthesis and diagnostics. This is not production latency or GPU residency.

The fixed-protocol new-pair check `results/seed_propagation_20261111/p2_shared_lazy_new_pairs/README.md` compares two new translations and two new heterogeneous fields. Shared and separate prefit arrays agree exactly; eager and deferred paths both recover 5,011/5,242 visible windows with zero visible false acceptances and identical accepted masks. This four-pair sample is not general qualification. New rigid, affine and opening tests were separately predeclared in `p2_shared_lazy_rotations_protocol.md`.

The separately fixed six-pair rigid/affine/opening experiment is reported in `results/seed_propagation_20261111/p2_shared_lazy_rotations/README.md`: exact prefit parity, 7,096/7,390 correct for either path, no visible false acceptance and no extra rotation requested. The two rigid angles were about −70° and +69°; openings align with the grid. Six pairs do not establish general robustness.

An opt-in cache reuses the exact quantized solver inputs and validates their hashes, archive and external source files. `p2_cached_pipeline_paired_isolated/README.md` reports one synthetic-pair median 16.45 s regenerated versus 11.54 s cached, including cache creation, with exact output and provenance parity; the dominant saved cost is synthetic image regeneration, not GPU inference. On Adrien, both paths give the same 7,097 accepted windows, but cache reading was slower in the single diagnostic and accuracy is unknown; see `p2_solver_images_adrien/README.md`. The cache is not GPU-resident and remains opt-in.

Experimental `--batch-anchor-confirmation` batches independent neighbour solves and replays the original confirmation order. On Adrien, three measured repetitions gave median verification 15.30 s serial versus 0.83 s batched, 1,538 versus 27 solver calls and exactly equal saved states/provenance; complete verification-process medians were 23.32 versus 8.63 s, excluding initial image hypotheses. No ground truth exists for Adrien. Four newly generated synthetic pairs also give exact parity and 4,695/4,900 correct with zero visible false acceptance; see `p2_anchor_batch_adrien/README.md` and `p2_anchor_batch_new_pairs/README.md`. The mode remains opt-in, requires a deterministic window-independent solver, and uses more speculative active windows. With at most eight candidate anchors it retains serial confirmation. P4/P5 and production GPU residency remain open.

Experimental GPU anchor confirmation and frontier propagation are documented in `results/seed_propagation_20261111/p2_gpu_anchor_frontier_four/README.md` and `p2_gpu_anchor_frontier_adrien/README.md`. The paths preserve accepted masks and provenance on four synthetic cases and the Adrien pair, with small observed floating-point differences. Global matching and sparse hypothesis selection still use host-mediated data; this is not a fully GPU-resident pipeline. The adapter uses private live-dic solver APIs and has explicit solver-configuration limits.

GPU ranking of sparse solver hypotheses is an additional opt-in stage (`--gpu-hypothesis-selection`). See `results/seed_propagation_20261111/p2_gpu_selection_four/README.md` and `p2_gpu_selection_adrien/README.md`. This stage still reads the selected field back to the host before GPU anchor confirmation; global matching remains host-mediated. The paired Adrien measurements do not establish an overall speedup.

Experimental `--gpu-resident-selection` keeps the selected sparse hypothesis and target on the solver device through anchor confirmation and propagation; hypothesis counts and frontier counters still synchronize with the host. See `results/seed_propagation_20261111/p2_resident_selection_four/README.md` and `p2_resident_selection_adrien/README.md`. The outputs match the preceding GPU path bitwise on the tested cases. Initial image matching and hypothesis creation remain host-mediated, so the whole pipeline is not GPU-resident.

The experimental `--ndf-onnx-webgpu-cache` path exports the coarse pair network and encoder to fixed-shape ONNX Runtime WebGPU graphs. See `results/seed_propagation_20261111/p2_ndf_onnx_webgpu_adrien/README.md` and `p2_ndf_onnx_end_to_end/README.md`: matched results on the tested pairs, but the tested WebGPU path is slower on Adrien and passes through CPU arrays between runtimes. It is opt-in and requires the surrounding PyTorch research scripts; no full GPU-residency or production speed claim.

The frozen P3 image-only qualification now includes new localized shear bands at gradients 0.33, 0.67 and 2.0. G3 fails for gradient 2.0: 123/4,080 visible accepted windows are wrong by at least 2 px, above the 0.5% limit. An oracle re-solve reproduces all 123 failures, identifying a solver/finite-affine qualification issue rather than establishing an initialization-only cause. See `results/seed_propagation_20261111/p3_shear_band_frozen_protocol/analysis/README.md` and `oracle_diagnostic/README.md`. No thresholds were tuned; the mechanism remains to be verified.

Correction to the preceding shear-band note: the initial 123 count used rounded-pixel truth. Exact analytical center scoring gives 57/4,080 (1.40%) at gradient 2.0; all 57 also fail the exact-parameter oracle. G3 still fails. See `p3_shear_band_frozen_protocol/exact_center_analysis/README.md`. New shear-band qualifications now use exact-center analytical truth; historical archives remain unchanged.

Follow-up shear diagnostics: `p3_shear_support_diagnostic/README.md` indicates that 32-pixel supports strongly reduce the local shear bias in oracle re-solves at the same centers; this does not establish the mechanism or production safety. On five predeclared new shear pairs, `p3_shear_32_independent/transfer/README.md` reports 5,405/7,013 correct and 5/5,410 false accepted, with 47/47 former errors recovered; two pairs still lack seeds on the opposite side. The post-hoc reseeding exploration gains coverage but introduces false acceptances and incurs high scripted hypothesis-generation cost. Original G3 remains failed.

In the predeclared 25-pair multifamily exact-center evaluation, 32-pixel same-center re-solves yielded 30,620/31,207 correct, zero visible false acceptances, and no lost 64-pixel correct windows; gradient RMSE increased on common correct windows. Five shifted-opening pairs gave no visible false accepted fine result, but 141 visible initializations used sources whose 64-pixel support crosses the boundary. See `p3_fine_transfer_multifamily_independent/fine/README.md` and `p3_fine_transfer_shifted_opening_independent/fine/README.md`. These do not overturn the original G3 failure or qualify arbitrary cracks, fine-grid density or performance. New synthetic truth is evaluated analytically at exact centers in all families; historical rounded-truth archives are unchanged.

Experimental same-centre 64→32→16 recovery is implemented in `seed_recovery.py`; GPU image-residual ZNCC is in `affine_residual.py` and the canonical WGSL shader. See `results/seed_propagation_20261111/p3_adaptive_residual_holdout_20261111/RAPPORT.md`. On 5 predeclared new severe-shear pairs, the frozen residual rule flagged 119/120 false coarse states but also 221/6979 correct coarse states. The guarded fine retry finished with 7070 correct and 1 false out of 8075 visible; 199 initially correct suspicious windows remained unresolved. On five new heterogeneous pairs the result was 6022/6092 and no visible false states. It is experimental and not the original G3 qualification. The CPU orchestrates and transfers data, and fine solver setup takes seconds. Preserve the provenance of 32/16 supports and do not claim a dense fine grid or a fully GPU-resident pipeline.

P3 decision is fixed in `results/seed_propagation_20261111/p3_decision_20261111.md`: G3 fails for extreme shear, 64-pixel propagation is baseline, 32→16 retry remains experimental. The user explicitly requested proceeding to P4. Solver changes are isolated in worktree `/home/jeff/Code/tests_gpuio-p4`, branch `codex/live-dic-seeded-api`.
P4 progress: solver branch `codex/live-dic-seeded-api`, commit `a8c3ca0`, [draft PR #4](https://github.com/jeffwitz/live-dic/pull/4), provides public affine seeded solves, optional final ZNCC for uniform full windows, and GPU affine frontier seeds. Ten previously inspected 4K pairs agree with a NumPy frontier oracle. Full propagation policy, partial/weighted correlation, and GPU residency remain open; see `results/seed_propagation_20261111/p4_public_api_parity_20261111/README.md`.
