# Patch-CNN DIC descriptor: training and IC-GN qualification

> **Pilot scope.** This is the earlier 24-training/8-validation-texture
> feasibility run. The expanded 80/20 model and the independent five-family
> qualification supersede it for current conclusions; see the
> [expanded report](../../qualification/dic_descriptor_expanded_icgn_20261108/README.md).

Date: 8 November 2026. The inference measurements ran on Intel Graphics
(LNL), Vulkan WebGPU. PyTorch 2.15.0 CPU was used only to train the network.
The GPU implementation is a native WGSL shader pair; no PyTorch dependency is
needed for inference.

## Model and training split

The descriptor samples a 64 by 64 pixel neighbourhood around each keypoint.
It applies a stride-4 convolution with eight channels, a depthwise 3 by 3
convolution, 4 by 4 spatial average pooling, a 128-to-64 linear projection,
and L2 normalization. The representation is trained using analytically
corresponding locations from the same speckle and symmetric InfoNCE with
same-texture negatives at least 48 pixels away.

This feasibility run used 24 training textures, 8 disjoint validation
textures, 48 requested points per pair, and 15 epochs. The exported checkpoint
was epoch 13, selected by maximum validation top-1 retrieval. It scored 5.21%
for 48 candidates per source point under the configured spatial exclusion. A
uniform-random reference is 2.08%. This is a small feasibility run, not a
statistically powered model-selection study.

Training split seeds, motion families, epoch losses and the weight SHA-256 are
in `training_report.json`. The weights are
`dic_patch_cnn_descriptor.npz`, SHA-256
`942644d3267bb1f5fcf01513705cda5b1e8b70452751f0f3c76f301c09c8d252`.

## Six fixed IC-GN fields

Every pipeline used the same 2048-point DIC detector, consensus configuration,
field projection, visible-window definition and external IC-GN solve. The
new descriptor was added to the existing qualification without changing the
criteria. `--compact-descriptor` is the learned projection of HORG-64; it is
shown separately from the patch CNN. The two historical failures remain
held-out regressions.

| Field | Visible windows | DeDoDe-B | HORG-64 | Learned HORG | Patch CNN | DeepMatching | Native HORG | EfficientLoFTR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Affine regression | 99 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Heterogeneous regression A | 65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Heterogeneous B | 77 | 65 | 0 | 0 | 0 | 64 | 0 | 64 |
| Discontinuous A | 84 | 32 | 0 | 0 | **45** | 29 | 0 | 24 |
| Discontinuous B | 91 | 46 | 57 | 53 | 47 | 48 | 41 | 46 |
| Rigid rotation | 106 | 95 | 0 | 0 | 95 | 95 | 95 | 95 |

Counts are windows whose final displacement after IC-GN is within 2 pixels of
the analytical truth. A dash or zero here means no correct converged windows;
the raw JSON distinguishes an absent field estimate from a solver run that
converged incorrectly. The patch CNN is the only tested descriptor that
provided an exploitable field for discontinuous A: 45/84, compared with 32/84
for DeDoDe-B, 29/84 for DeepMatching, and 24/84 for EfficientLoFTR. It matches
DeDoDe-B, DeepMatching and EfficientLoFTR on rigid rotation (95/106), but does
not exceed the 57/91 of fixed HORG-64 on discontinuous B. Neither held-out
failure case was recovered.

## Time and direct matching evidence

At 1024 by 576, paired descriptor extraction for the patch CNN took 14 to
29 ms across the six fields. For comparison, HORG-64 took 8.0 to 17 ms,
the learned multiscale HORG projection took 29.2 to 54.0 ms, and DeDoDe-B took
1.92 to 2.06 seconds. In the only new success, discontinuous A, the full
Gaussian radial-basis pipeline took 0.429 s for 45 correct windows, or about
105 correct windows per second. The per-field JSON files contain stage timing,
initialization errors, direct matches and solver results.

The patch CNN's correct reciprocal direct matches within 5 pixels were 218 of
442 on discontinuous A, 852 of 875 on discontinuous B, and 209 of 415 on
rigid rotation. On the affine and heterogeneous regression fields it produced
only one and zero correct raw direct matches respectively; no local consensus
model survived. This agrees with the held-out failure analysis: detector
repeatability is not sufficient when same-texture descriptor ranking fails.

The convolutional shader reads the original grayscale image and samples each
point's support on the GPU. The feature map remains in a GPU storage buffer
between the two WGSL passes. Only the final 64-component descriptors return to
Python; the existing reciprocal matcher then uploads them to its own GPU
buffers. This is GPU kernel execution with a CPU boundary, not a
GPU-resident end-to-end pipeline.

The native-4K throughput run used 4096 detected points in each 3840 by 2160
image. Across three timed repetitions after warm-up, paired detection took
86.7 ms, patch-CNN description 69.1 ms, and reciprocal matching 81.6 ms. The
corresponding HORG-64 times in the same run were 86.7 ms, 19.7 ms, and 90.8
ms. Detection, description and matching therefore total about 237 ms for the
patch CNN and 197 ms for HORG-64 before geometric consensus, field projection
or IC-GN. The target image was a 37-pixel cyclic horizontal roll; these are
throughput measurements, not an analytic accuracy benchmark.

## Reproduction

Install the optional research dependencies and train:

```console
python -m pip install -e ".[research]"
python python/benchmarks/train_dic_cnn_descriptor.py \
  --output results/descriptors/dic_descriptor_cnn_training_20261108 \
  --train-pairs 24 --validation-pairs 8 --points-per-pair 48 \
  --epochs 15 --batch-size 2 --seed 20261109 --no-affine-augmentation
```

The complete scientific qualification, raw pairs and per-window solver output
are in `results/qualification/dic_descriptor_cnn_icgn_20261108/`. Reproduce it
with the matching historical DeDoDe and EfficientLoFTR model files plus:

```console
python python/benchmarks/qualify_heterogeneous_icgn.py \
  --output results/qualification/dic_descriptor_cnn_icgn_20261108 \
  --descriptor-onnx /path/to/dedode_b_descriptor_1024x576_k8192.onnx \
  --weights results/features/dic_detector_training_discontinuous_20261107/dic_detector_weights.npz \
  --compact-descriptor results/descriptors/dic_descriptor_horg_training_20261108/dic_descriptor_multiscale_mlp.npz \
  --cnn-descriptor results/descriptors/dic_descriptor_cnn_training_20261108/dic_patch_cnn_descriptor.npz
```

The shader parity test compares all convolution, pooling and projection
results against an independent NumPy implementation. The qualification uses
the external IC-GN package read-only; no external project files are changed.

## Interpretation

This small CNN is a promising next experiment because it improves one
discontinuous field over the established learned and hand-crafted descriptor
pipelines while keeping descriptor extraction in the millisecond range. The
result is not yet a general robustness claim: the affine and heterogeneous-A
regressions still fail and training and validation sets are small. The next
training run should expand texture diversity and maintain the two regression
cases as untouched tests.
