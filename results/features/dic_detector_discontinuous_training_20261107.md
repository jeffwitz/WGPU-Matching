# Discontinuity-aware DIC-Detector training study

The earlier and discontinuity-aware checkpoints were evaluated on the same 24 deterministic texture/warp pairs using the WGSL detector. The only changed factor is the learned score weights. Each run uses a 512-keypoint budget, 256×256 images, and three warmed measurements per image. The tensor baseline is also reported in each raw file.

Adapter: Vulkan (Intel(R) Graphics (LNL)). Previous weights SHA-256: `2dd4c7cdf9eabd4b2fedcedb76724f914022ac2a4f4dceaa5c695d368aeae5ba`. Discontinuity-aware weights SHA-256: `d539deef26b46ffbbe54e3d27b1395a66126ddbea7243c3221f2617f25cb9094`.
Training: 10 epochs × 64 samples, CPU PyTorch 2.15.0+cpu, 29.0 s. Training motion counts: affine 132, discontinuous 111, heterogeneous 127, rigid 139, translation 131.
WGSL/PyTorch score parity for the same checkpoint at 3840×2160 with 8192 selected points: maximum absolute score difference 2.41e-06; median WGSL dispatch/readback 64.28 ms.

## Aggregate held-out repeatability

| Checkpoint | ≤1 px | ≤2 px | ≤5 px | Pair detection median | FPS |
|---|---:|---:|---:|---:|---:|
| Previous | 1319/4816 (27.4%) | 2608/4816 (54.2%) | 3686/4816 (76.5%) | 0.0040 s | 247.2 |
| Discontinuity-aware | 1403/4756 (29.5%) | 2711/4756 (57.0%) | 3715/4756 (78.1%) | 0.0041 s | 246.9 |

## Repeatability by motion family

| Motion | Pairs | Previous ≤2 px | Discontinuity-aware ≤2 px | Previous ≤5 px | Discontinuity-aware ≤5 px |
|---|---:|---:|---:|---:|---:|
| affine | 2 | 191/365 (52.3%) | 201/359 (56.0%) | 283/365 (77.5%) | 287/359 (79.9%) |
| discontinuous | 5 | 613/954 (64.3%) | 600/924 (64.9%) | 755/954 (79.1%) | 737/924 (79.8%) |
| heterogeneous | 2 | 165/284 (58.1%) | 183/286 (64.0%) | 233/284 (82.0%) | 244/286 (85.3%) |
| rigid | 8 | 899/1751 (51.3%) | 935/1734 (53.9%) | 1327/1751 (75.8%) | 1334/1734 (76.9%) |
| translation | 7 | 740/1462 (50.6%) | 792/1453 (54.5%) | 1088/1462 (74.4%) | 1113/1453 (76.6%) |

The discontinuity-aware checkpoint improves the pooled ≤2 px rate from 54.2% to 57.0%, with almost unchanged detector time (4.05 ms per image pair). The discontinuous subset changes from 64.3% to 64.9% at ≤2 px across five pairs; this is a small exploratory difference, not evidence of a reliable crack-robust detector. The denominator varies because each checkpoint selects a different number of points that remain visible; rates and counts are both shown.

This experiment measures detector repeatability only. It uses no descriptor, mutual matching, geometric fit, or IC-GN solver. The synthetic gap is a simple piecewise translation with a masked band, not a physical crack or a localized finite-element strain field.

Raw per-pair results are in `dic_detector_discontinuous_legacy_heldout_20261107.json` and `dic_detector_discontinuous_heldout_20261107.json`; 4K WGSL/PyTorch score parity is in `dic_detector_discontinuous_wgsl_parity_4k_20261107.json`. Training seeds, motion counts, losses, and held-out PyTorch metrics are in `dic_detector_training_discontinuous_20261107/training_report.json`.
