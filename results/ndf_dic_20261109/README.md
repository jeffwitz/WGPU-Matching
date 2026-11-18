# Propagation-first campaign, 9 November 2026

This index summarizes the propagation-first qualification started from commit `4dce5d0`. Digital image correlation (DIC) is refined by the inverse-compositional Gauss-Newton (IC-GN) solver. Normalized cross-correlation (NCC) is used for local image search; in these scripts it is zero-mean normalized cross-correlation (ZNCC).

## Decision report

See [the full decision report](propagation_first_decision_20261109.md) for the protocol, gates, per-family findings, limitations, and reproduction commands.

## Main findings

- The four archived propagated totals reproduce exactly: 19,915/21,367 for image-derived affine, 20,651/21,367 for affine plus NCC, 19,070/21,367 for affine plus neural deformation field residual, and 19,917/21,367 for affine plus residual plus NCC. Gate G1 passes.
- Gate G2 passes for the archived campaign. One incorrect accepted seed is listed with coordinates in the report; it does not propagate.
- In held-out rigid, affine, discontinuous, and moderate shear families, propagation usually reaches at least 97% of the analytical oracle on pairs that have a seed. Extreme shear with peak displacement gradient 2.0 fails this gate and also exceeds the false-acceptance limit.
- The top-three rotation check produces no accepted seed on either historical zero-seed pair.
- On the masked Adrien experimental pair, the solver accepts 6,593 windows in one pass and 7,103 after propagation. These are status counts only; no analytical accuracy claim is possible.
- The raw-bilinear coordinate correction loses 548 net correct windows over ten translation pairs. The per-window audit attributes the change to the initial field sample coordinates; no threshold was changed.

## Detailed artifacts

- `causal_ablation_frontier_corrected_20261109/README.md`: archived reference replay and G1/G2 audit.
- `zero_seed_rotation_top3_20261109/README.md`: status-only top-three rotation selection.
- `matcher_source_frontier_20261109/README.md`: learned, HORG, and DeepMatching source comparison.
- `frontier_generalization_20261109/README.md`: independent motion families, regions, rings, and failed gates.
- `adrien_masked_horg_propagation_20261109/README.md`: experimental accepted counts and maps.
- `translation_coordinate_anomaly_20261109/README.md`: per-window translation regression analysis.

No decoder, new training run, loss change, or projection change was included in this campaign.
