# Same-texture hard-negative training and descriptor retrieval

## Experimental objective

The descriptor maps a digital image correlation (DIC) detector keypoint to a normalized 64-component vector. A convolutional neural network (CNN) produces the descriptor, with inference kernels written in WebGPU Shading Language (WGSL). This experiment tests whether training with detector-selected locations and hard negatives from the same speckle improves recovery under strong image motion. It is a descriptor retrieval diagnostic; it does not itself establish that a field projection or an inverse compositional Gauss–Newton (IC-GN) solve will succeed. The graphics processing unit (GPU) is Intel Graphics (LNL, Lunar Lake); the reported 4K frames are 3,840 × 2,160 pixels.

## Training protocol

Training and validation use distinct synthetic texture identities. Positive target patches are sampled at the exact analytic floating-point correspondence, with bilinear image sampling. Queries and candidate negatives are selected from points returned by the WebGPU speckle detector. The trainer checks a 17 × 17 lattice across the full 64 × 64 descriptor support in the source and mapped target validity masks. Supports crossing image edges or the discontinuity mask are excluded.

The hard-negative model used 24 training and 10 validation texture/motion pairs, 128 query points per pair, up to 1,024 detector candidates per image, 12 epochs and seed `20261108`. Actual candidate counts vary because invalid supports are rejected; exact counts and texture-seed hashes are in `training_report.json`. The loss selects the hardest 128 same-texture negatives online in both matching directions, excluding candidates within 48 pixels to reduce false negatives. Checkpoint selection uses the macro average of top-1 retrieval across motion families, rather than a pooled score dominated by easy families.

Selected epoch: 7. PyTorch 2.15.0 CPU trained the weights; detector extraction and inference qualification used Vulkan on Intel Graphics (LNL). The model archive SHA-256 is `63c94446f979af642158ce2242a818ad90aaaefba5b6628739c43f4f5826197d`.

## Full-image retrieval at native 4K

The 3,840 × 2,160 pair uses 512 fixed source queries and candidate pools of 512, 2,048 and 8,192 target detections. The correct candidate is the nearest detected target keypoint within 5 pixels of the analytic correspondence. “Top-1/5” counts how often that positive candidate ranks in the first one/five places. It is conditional on the number of repeatable positives in the `≤5 px` column. The candidate pool is selected spatially from detector responses; invalid 64-pixel supports are discarded. The table compares fixed HORG-64, the original patch-CNN checkpoint, and the hard-negative checkpoint on the same image pair and seeds.

| Motion | Candidates | Repeatable ≤5 px | HORG top-1 / top-5 | Original CNN top-1 / top-5 | Hard-negative CNN top-1 / top-5 |
|---|---:|---:|---:|---:|---:|
| Rigid rotation | 512 | 248 | 49 / 82 | 1 / 1 | 0 / 1 |
| Rigid rotation | 2,048 | 435 | 53 / 94 | 1 / 2 | 0 / 1 |
| Rigid rotation | 8,192 | 487 | 43 / 74 | 0 / 0 | 0 / 0 |
| Affine | 512 | 119 | 4 / 7 | 0 / 8 | 11 / 29 |
| Affine | 2,048 | 273 | 4 / 6 | 0 / 3 | 11 / 38 |
| Affine | 8,192 | 435 | 3 / 7 | 0 / 1 | 5 / 17 |
| Heterogeneous | 512 | 89 | 3 / 9 | 0 / 0 | 0 / 0 |
| Heterogeneous | 2,048 | 228 | 4 / 15 | 0 / 1 | 0 / 1 |
| Heterogeneous | 8,192 | 409 | 3 / 9 | 1 / 2 | 0 / 1 |
| Discontinuous | 512 | 234 | 58 / 82 | 225 / 230 | 234 / 234 |
| Discontinuous | 2,048 | 420 | 91 / 126 | 404 / 415 | 417 / 420 |
| Discontinuous | 8,192 | 488 | 90 / 116 | 458 / 476 | 481 / 487 |

This is one 4K pair per family, not a population estimate. Hard-negative training substantially improves affine and discontinuous retrieval for this pair, but does not resolve the rotation or heterogeneous failure. It reduces neither candidate-pool size nor descriptor dimensionality. The saved JSON also contains top-20 rank, reciprocal false positives, similarity margins and stage timing.

The detector and descriptor extraction use WebGPU, but this particular retrieval diagnostic copies descriptors to NumPy and computes the global cosine ranking on the CPU. Its ranking timings are not evidence of a GPU-resident matcher. The full IC-GN qualification separately exercises WGPU-Matching's WebGPU reciprocal matcher and reports end-to-end projection and solve time.

## Affine patch augmentation control

A separate model used 60 training and 20 validation pairs, 128 queries and up to 512 detector negatives per image. During training, the two 64 × 64 patches received independent random rotations, anisotropic scales, shear and small translations. This deliberately tests whether broad affine augmentation alone solves the nuisance. It did not: at 8,192 candidates it recovered 0/487 repeatable rigid positives at top-1, 0/435 affine, 1/409 heterogeneous and 280/488 discontinuous. The hard-negative model gave 0, 5, 0 and 481 respectively. The augmented model also reduced discontinuous retrieval versus both earlier CNN checkpoints. Its checkpoint and per-family validation are archived in `../dic_descriptor_affine_aug_training_20261108/`.

The result cautions against forcing invariance: independent patch warps may remove the spatial evidence that distinguishes repeated speckle neighborhoods. More training epochs or more aggressive affine augmentation are not justified by this experiment.

## Link to IC-GN qualification

The end-to-end qualification, including field fitting and the external WebGPU IC-GN solver, is documented in [the checkpoint requalification](../../qualification/dic_descriptor_hardneg_icgn_20261108/README). That report retains the six fixed regression fields and adds ten independent fields per motion family. It reports failed field fits separately from scored windows.

## Reproduction

```console
python python/benchmarks/train_dic_cnn_descriptor.py \
  --output results/descriptors/dic_descriptor_hardneg_training_20261108 \
  --train-pairs 24 --validation-pairs 10 --points-per-pair 128 \
  --negative-points-per-image 1024 --epochs 12 --seed 20261108 \
  --no-affine-augmentation

python python/benchmarks/benchmark_dic_descriptor_retrieval.py \
  --weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \
  --output results/descriptors/dic_descriptor_hardneg_training_20261108/retrieval_full_pool.json \
  --families rigid affine heterogeneous discontinuous --queries 512 --seed 20261108
```

The saved run manifests are the authoritative record of actual parameters, seeds, candidate counts, package versions, and hashes. The optional training framework is not required for WebGPU inference.
