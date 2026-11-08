# HORG-64 descriptor failure analysis and learned projection

Date: 8 November 2026. The report was produced on the Intel Graphics (LNL)
integrated GPU using the Vulkan WebGPU backend. The IC-GN runner is imported
read-only from `tests_gpuio`; no files in that project were changed.

## Why the two historical fields still fail

The regression cases `affine_8101_9101` and `heterogeneous_8102_9102` were kept
out of training and validation. They remain failures for every method in the
current six-field qualification. The analytic affine oracle is also rejected
by the local field fitter on these cases because fewer than three finite,
accepted direct matches survive the configured local consensus. This identifies
the failure before IC-GN: these cases are not evidence of a solver-only limit.

| Case / descriptor | Visible source points | Target detections within 2 px | One-way descriptor top-1 matches <5 px | Reciprocal false positives >=5 px | Median top-1 error | Same-texture hard-negative cosine |
|---|---:|---:|---:|---:|---:|---:|
| Affine / DeDoDe-B | 1399 | 862 | 61 | 371 | 292.2 px | 0.592 |
| Affine / HORG-64 | 1399 | 862 | 4 | 519 | 329.6 px | 0.914 |
| Affine / learned HORG | 1399 | 862 | 5 | 479 | 332.3 px | 0.726 |
| Heterogeneous A / DeDoDe-B | 875 | 574 | 15 | 216 | 341.7 px | 0.628 |
| Heterogeneous A / HORG-64 | 875 | 574 | 0 | 227 | 367.5 px | 0.928 |
| Heterogeneous A / learned HORG | 875 | 574 | 4 | 289 | 367.2 px | 0.710 |

There are many repeatable detections in both cases, so detector repeatability is
not the sole cause. The failure is dominated by descriptor ambiguity and the
global search selecting distant regions: the median top-1 error is hundreds of
pixels. HORG's high cosine values are not confidence evidence here; incorrect
matches score almost as highly as correct ones. For example, in the affine
case, the median best-versus-second-best margin is only 0.008 for HORG-64.
Changing the geometric acceptance thresholds would conceal this evidence and
was not done.

For all three DIC-Detector descriptor variants, local consensus leaves fewer
than three usable correspondences in these cases, so no affine or
radial-basis projection can be fit. The direct matcher and the geometric
initialization therefore fail before an estimated field reaches IC-GN. An
independent analytic local-affine seed is accepted by IC-GN for 79/99 visible
windows in the affine case and 45/65 in heterogeneous A. The remaining oracle
failures show that the solver and image formation also impose a limit; the
descriptor failure does not explain every missed window. The fields are
retained as hard regressions and have not been "fixed" by fitting a global
transform.

## HORG-64 learned projection experiment

The first learning stage used HORG descriptors extracted by WGSL at the exact
analytic target coordinates. Training and validation used disjoint speckle
texture seeds. Training comprised 64 pairs, validation 16 pairs, and 96
correspondences per pair. Hard negatives came from the same texture at source
locations at least 48 pixels apart. A symmetric contrastive objective trained
linear, residual-MLP, and three-scale MLP projections. Inference for the
projection is WGSL; PyTorch is optional and was used only for training.

Validation retrieval at 96 candidates per image was weak for all variants.
The three-scale learned descriptor reached 2.08% top-1 retrieval under the
48-pixel negative mask, versus 1.95% for fixed three-scale HORG. The linear
64-to-64 projection reached 2.15%, while the residual MLP reached 1.43%.
These are small gains over a 1.04% uniform-random reference and do not establish
useful large-displacement matching.

The six-field IC-GN qualification provides the more important check:

| Field | Fixed HORG-64 | Learned three-scale HORG | DeDoDe-B |
|---|---:|---:|---:|
| Affine regression case | 0/99 | 0/99 | 0/99 |
| Heterogeneous regression case A | 0/65 | 0/65 | 0/65 |
| Heterogeneous B | 0/77 | 0/77 | 65/77 |
| Discontinuous A | 0/84 | 0/84 | 32/84 |
| Discontinuous B | 57/91 | 53/91 | 46/91 |
| Rigid rotation | 0/106 | 0/106 | 95/106 |

Only the discontinuous-B field produced accepted initializations for either
HORG variant. The trained projection reduced its correct IC-GN windows from
57 to 53 and increased paired descriptor time from 9 ms to 31 ms at 1024 x
576. It is therefore **not an improvement** and must not replace HORG-64.

## Native 4K timing

The separate native-resolution timing used 3840 x 2160 generated images,
4096 detected points per image and a target shifted by 37 pixels. One warm-up
and three timed repetitions were performed. Median paired times were:

| Stage | Fixed HORG-64 | Learned three-scale projection |
|---|---:|---:|
| DIC detector, both images | 90.7 ms | 90.7 ms |
| Descriptor, both images | 22.7 ms | 83.9 ms |
| Reciprocal cosine matching | 92.6 ms | 93.4 ms |

This is a bounded throughput smoke benchmark, not an accuracy qualification:
the roll pair is used to exercise native-resolution compute, while the six
analytic fields above determine descriptor quality. The projection is slower
than fixed HORG-64 and the descriptor extraction currently reads intermediate
HORG features to Python before uploading them for the projection. The resulting
pipeline is not GPU-resident end to end.

## Reproduction and artifacts

The training data, hashes, weights and epoch history are in
`dic_descriptor_horg_training_20261108/`. The model files are experimental
artifacts and are not distributed as a qualified model. The complete per-case
direct matches, geometric diagnostics, projections and IC-GN outputs are in
`../qualification/dic_descriptor_icgn_20261108/summary.json` and its case
directories. Native 4K raw timings are in
`dic_descriptor_horg_training_20261108/benchmark_4k.json`.

The result supports proceeding to a patch-based convolutional descriptor,
because a learned transform of HORG alone did not learn adequate discrimination.
That next model must be evaluated on held-out textures and the same six
regressions before any optimization claim.

## Patch-based convolutional follow-up

A small patch model was then trained on 24 speckle textures, selected on 8
validation textures, and run through the same six fields and IC-GN protocol.
Unlike the HORG projection, it produced a usable local field on discontinuous
A and converged 45/84 windows correctly. DeDoDe-B reached 32/84,
DeepMatching 29/84, and EfficientLoFTR 24/84 there. On discontinuous B it
reached 47/91, matching DeDoDe-B and EfficientLoFTR but below HORG-64 at
57/91. It matched the learned baselines on rigid rotation at 95/106. The two
held-out regression failures remain at 0 correct windows, as do all other
methods in those fields. This is the first encouraging DIC-specific descriptor
result, but the limited training and validation texture counts prevent a
generalization claim. Full results, hashes, timings, commands, and raw solver
outputs are in [the patch-CNN report](dic_descriptor_cnn_training_20261108/README.md) and
``results/qualification/dic_descriptor_cnn_icgn_20261108/``.

## Expanded training and independent qualification

The next model was trained on 80 textures and selected on 20 disjoint
validation textures. It achieved 6.04% validation top-1 retrieval, but this
improvement did not generalize uniformly. On ten new independent fields per
motion family, the descriptor produced correct IC-GN initialization for
931/1034 translation windows and 684/1013 discontinuous windows. It produced
no correct converged windows for rigid rotation, affine motion, or
heterogeneous smooth motion. The analytic local-affine solver oracle still
converged on 739/881, 655/800, and 740/896 windows in those three families.
The two fixed failure regressions continued to fail for every method.

For discontinuous fields, the CNN modestly exceeded DeDoDe-B (684 versus 660
windows), DeepMatching (664), and EfficientLoFTR (669), while fixed HORG-64
was faster but covered fewer windows (499). The pooled correct-window rate
was 110.8/s for the CNN. This is a narrow family-specific result, not a
general descriptor improvement. The complete by-family comparison, timing
accounting, direct-match counts, oracle ceilings, hashes, and reproduction
commands are in [the expanded qualification report](../qualification/dic_descriptor_expanded_icgn_20261108/README.md).

## Detector-point hard negatives and affine augmentation

The next training protocol uses actual points from the WebGPU digital image
correlation (DIC) detector rather than uniformly random query positions.
Positive patches are sampled at the analytic subpixel correspondence;
hard negatives are selected online from detector points in the same texture.
A 17 by 17 validity lattice checks the complete 64 by 64 descriptor support
in both images, including the mapped target support. Validation checkpoint
selection macro-averages top-1 retrieval over translation, rigid rotation,
affine, heterogeneous, and discontinuous motion.

On one independent native-4K pair per family, 512 source queries were ranked
against 512, 2,048, and 8,192 target candidates. At the largest pool,
hard-negative training changed CNN top-1/top-5 retrieval as follows: rigid
rotation 0/0 to 0/0; affine 0/1 to 5/17; heterogeneous 1/2 to 0/1; and
discontinuous 458/476 to 481/487. These measurements are conditional on a
repeatable target detection and are not population estimates. HORG-64 reached
43/74, 3/7, 3/9, and 90/116, respectively. Full retrieval tables, margins,
false positives, timings, and checkpoint hashes are in the
[hard-negative report](dic_descriptor_hardneg_training_20261108/README.md).

A separate run with independent random affine transforms applied to both
patches did not solve affine retrieval and reduced discontinuous top-1/top-5
to 280/334 at 8,192 candidates. This negative result suggests that broad
invariance augmentation can remove useful speckle discrimination. Its weights
and per-family validation are archived in
[`dic_descriptor_affine_aug_training_20261108`](dic_descriptor_affine_aug_training_20261108/README.md).

The corresponding external inverse compositional Gauss–Newton (IC-GN)
qualification tests ten independent fields per family and all six frozen
regressions. The hard-negative model still fails to produce a field for all
independent rigid cases, 8/10 affine cases, and 8/10 heterogeneous cases. It
does not recover `affine_8101_9101` or `heterogeneous_8102_9102`. Full-support
scoring is reported separately from center-visible scoring; it is a post-run
subset and does not alter matching or solving. See the
[independent requalification](../qualification/dic_descriptor_hardneg_icgn_20261108/README.md).
