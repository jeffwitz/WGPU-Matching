# CPM analytic-oracle diagnostics at 60% strain

## Question and protocol

The CPM cost function assumes a rigid translation over each matching support. To separate search failure from a misleading cost, the browser harness evaluates both the best displacement found by PatchMatch and the analytically known affine displacement at the same measured work-grid location and pyramid scale. The oracle is diagnostic only; it is never passed to the matcher. The experiment uses the synthetic 60% tensile frame, 16 visible seeds, DIC window 128 px, work-grid step 32 px, search radius 1024 px, deterministic seed 17, 16 initial hypotheses, and matching supports 8, 16, 32, and 64 px.

## Results

| Matching support | Effective descriptor scales (original px per level) | Reciprocal work-grid points | Accepted requested seeds | Error RMSE of all projected proposals (px) |
|---:|---|---:|---:|---:|
| 8 px | ×4 | 1 | 0/16 | 829.455 |
| 16 px | ×4 | 3 | 0/16 | 911.922 |
| 32 px | ×4, ×8 | 0 | 0/16 | 757.235 |
| 64 px | ×4, ×8, ×16 | 0 | 0/16 | 981.718 |

The coarse-to-fine implementation uses a pooled descriptor pyramid whose effective depth depends on support. In this table, ×4 is the finest descriptor scale and later entries are progressively coarser. The implementation therefore still has limited coarse search depth for small supports.

At support 32 px, the oracle has lower HOG cost than the found hypothesis for 9/16 seeds at ×4 and 11/16 at ×8. The median oracle-minus-found cost is −0.00076 and −0.00605 respectively. This points to search failure for many seeds: the cost can prefer the true displacement, but the current search does not find it. At support 64 px, the oracle cost is lower for only 0/16, 1/16, and 4/16 seeds at ×4, ×8, and ×16. For this wider rigid patch, the known affine displacement is often not favored by the cost, consistent with within-patch deformation violating the translation model. Supports 8 and 16 also fail to produce a reliable field, so reducing support alone is not a solution.

The support sweep shows no useful accepted CPM initialization at 60%. Its all-proposal RMSE is large at every support, and reciprocal counts are 0–3 out of the work grid. The local affine fit and strict reciprocal checks deliberately reject unsupported estimates rather than relabeling them as reliable.

## Interpretation and limitations

The experiment supports two distinct failure mechanisms, depending on support: poor candidate exploration at smaller supports, and a rigid-patch objective that can be inconsistent with the affine ground truth at larger supports. It does not prove that either is the only cause. The HOG cost’s oracle/found comparison samples only the measured work-grid positions and uses the current pooled HOG pyramid; it does not test alternative descriptors or a locally affine patch cost. Search and cost need separate controlled changes before performance optimization.

The complete per-seed and per-scale data are in [`oracle_support_study_60.json`](oracle_support_study_60.json). The three-method result for this run is in [`60% comparison`](../uniaxial/tensile_60pct/three_method_browser_60.md).
