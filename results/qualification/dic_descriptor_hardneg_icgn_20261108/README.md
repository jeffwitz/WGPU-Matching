# Compact DIC descriptor requalification

This report evaluates digital image correlation (DIC) initializations with the inverse compositional Gauss–Newton (IC-GN) solver, using the same detector, windows, matching consensus, affine and radial basis function (RBF) projections, and convergence limits as the preceding qualification. Six fixed fields are regression checks; independent texture and motion seeds are the test set.

Adapter: Vulkan on Intel Lunar Lake Graphics, 1024 × 576. The compact convolutional neural network (CNN) weights have SHA-256 `63c94446f979af642158ce2242a818ad90aaaefba5b6628739c43f4f5826197d`.

Counts are correct final IC-GN displacements within 2 pixels / visible windows. A field-fit failure has no visible-window denominator and is shown as a dash, not as a measured zero success. Full-support scoring is a post-run subset requiring the 64 × 64 support to remain valid in the reference and mapped target; it does not change matching or solver inputs.

## Independent test fields

| Motion | Descriptor | Affine fields fitted | Affine correct / visible | Affine full-support correct / valid | RBF fields fitted | RBF correct / visible | RBF full-support correct / valid |
|---|---|---:|---:|---:|---:|---:|---:|
| translation | HORG-64 | 6/10 | 576/628 | 493/493 | 6/10 | 576/628 | 493/493 |
| translation | DIC-CNN hard-negative | 10/10 | 931/1034 | 801/816 | 10/10 | 931/1034 | 801/816 |
| rigid | HORG-64 | 0/10 | — | — | 0/10 | — | — |
| rigid | DIC-CNN hard-negative | 0/10 | — | — | 0/10 | — | — |
| affine | HORG-64 | 0/10 | — | — | 0/10 | — | — |
| affine | DIC-CNN hard-negative | 2/10 | 11/198 | 10/140 | 2/10 | 11/198 | 10/140 |
| heterogeneous | HORG-64 | 0/10 | — | — | 0/10 | — | — |
| heterogeneous | DIC-CNN hard-negative | 2/10 | 108/165 | 100/120 | 2/10 | 120/165 | 108/120 |
| discontinuous | HORG-64 | 7/10 | 379/747 | 313/479 | 7/10 | 499/747 | 399/479 |
| discontinuous | DIC-CNN hard-negative | 10/10 | 465/1013 | 402/632 | 10/10 | 684/1013 | 559/632 |

A field is fitted only when the projection produced a usable model. Throughput is summed over the successfully fitted fields; it is not an all-field rate when any fit failed. Per-family timing and fit counts are in `aggregate.json`.

| Motion | Descriptor | Affine correct windows/s, fitted fields only | RBF correct windows/s, fitted fields only |
|---|---|---:|---:|
| translation | HORG-64 | 705.8 | 350.4 |
| translation | DIC-CNN hard-negative | 712.5 | 198.2 |
| rigid | HORG-64 | — | — |
| rigid | DIC-CNN hard-negative | — | — |
| affine | HORG-64 | — | — |
| affine | DIC-CNN hard-negative | 45.5 | 27.3 |
| heterogeneous | HORG-64 | — | — |
| heterogeneous | DIC-CNN hard-negative | 432.0 | 185.3 |
| discontinuous | HORG-64 | 436.2 | 260.4 |
| discontinuous | DIC-CNN hard-negative | 350.7 | 101.6 |

## Fixed regression fields

| Case | Descriptor | Affine projection / correct windows | RBF projection / correct windows |
|---|---|---:|---:|
| affine `8101/9101` | HORG-64 | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| affine `8101/9101` | DIC-CNN hard-negative | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| heterogeneous `8102/9102` | HORG-64 | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| heterogeneous `8102/9102` | DIC-CNN hard-negative | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| heterogeneous `8103/9103` | HORG-64 | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| heterogeneous `8103/9103` | DIC-CNN hard-negative | fit; 13/77 | fit; 61/77 |
| discontinuous `8104/9104` | HORG-64 | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| discontinuous `8104/9104` | DIC-CNN hard-negative | fit; 19/84 | fit; 31/84 |
| discontinuous `8105/9105` | HORG-64 | fit; 28/91 | fit; 57/91 |
| discontinuous `8105/9105` | DIC-CNN hard-negative | fit; 27/91 | fit; 46/91 |
| rigid `8106/9106` | HORG-64 | at least three finite in-image correspondences are required | at least three finite in-image correspondences are required |
| rigid `8106/9106` | DIC-CNN hard-negative | fit; 95/106 | fit; 95/106 |

## Interpretation and limitations

Hard-negative training improved retrieval for affine and discontinuous motions in the measured 4K retrieval pair, but did not improve rigid or heterogeneous retrieval. In independent IC-GN tests the descriptor still fails to produce a field for most affine and heterogeneous cases and for all independent rigid cases. It is not a general replacement for DeDoDe-B.

For discontinuous fields, full-support counts exclude supports touching the invalid band. This is a post-hoc scoring subset and does not show that a smooth field model is safe across a physical discontinuity. The two historical failures `affine_8101_9101` and `heterogeneous_8102_9102` remain failures.

The raw per-window output and generated image pairs remain in the scratch directory. Committed artifacts are compact summaries and this report.
