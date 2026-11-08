# Independent affine patch augmentation control

This control asks whether independent geometric warps applied to each 64 × 64 patch can teach the compact convolutional neural network (CNN) descriptor affine robustness without changing its architecture.

The model used 60 training texture/motion pairs and 20 disjoint validation pairs, 128 query points per pair, up to 512 detector candidates per image, 12 epochs, and seed `20261109`. The training detector supplied the keypoints; positives used analytic correspondences and negatives came from the same texture. Each patch then received an independent random rotation from −180° to 180°, anisotropic scale from 0.7 to 1.45, shear from −0.25 to 0.25, and small normalized translation. Sampling was bilinear with border padding. Actual valid detector candidate counts are recorded in `training_report.json`.

The selected epoch was 2. PyTorch 2.15.0 CPU was used for training; WebGPU detection and descriptor inference used Vulkan on Intel Graphics (LNL). The weights SHA-256 is `343d2828dc9c1313440bd8efd5ae939637db21fed790d42ff2e8716583b81217`.

## Native-4K retrieval at 8,192 candidates

The held-out diagnostic uses one 3,840 × 2,160 pair per motion family, 512 source queries, and 8,192 target candidates. It counts the analytic positive only when a target detection is within five pixels. Results are top-1/top-5 ranks among those repeatable points:

| Motion | Repeatable ≤5 px | HORG-64 | Previous CNN | Hard-negative CNN | Affine-augmented CNN |
|---|---:|---:|---:|---:|---:|
| Rigid rotation | 487 | 43/74 | 0/0 | 0/0 | 0/0 |
| Affine | 435 | 3/7 | 0/1 | 5/17 | 0/1 |
| Heterogeneous | 409 | 3/9 | 1/2 | 0/1 | 1/1 |
| Discontinuous | 488 | 90/116 | 458/476 | 481/487 | 280/334 |

The augmentation does not improve affine recovery and sharply regresses the discontinuous case. It is therefore a negative result, not a recommended checkpoint. It indicates that broad invariance imposed by independent patch warps can destroy discriminative spatial details in speckle. Do not interpret the marginal heterogeneous change from 0/1 to 1/1 as a meaningful gain.

The companion JSON contains all candidate-budget rows, timing, margins and false-positive counts. Compare it with the hard-negative training report, which describes the benchmark definition and the independent IC-GN qualification.
