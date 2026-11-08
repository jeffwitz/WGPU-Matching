DIC-Descriptor: what the first learning stage established
==========================================================

Digital image correlation (DIC) estimates motion by comparing image
neighbourhoods before and after deformation. A descriptor turns a local image
neighbourhood into a vector so that candidate points can be compared without
evaluating every possible image displacement photometrically.

This experiment asks whether training a compact transformation of the
existing HORG-64 descriptor is enough to make speckle regions recognizable
after strong motion. HORG is the 64-component histogram of oriented gradients
representation used by HORG-DIC. The learned model does **not** replace the
detector, predict motion, or reproduce DeDoDe-B. It transforms HORG vectors and
normalizes the result to unit Euclidean length.

Why matching random speckle is difficult
----------------------------------------

A speckle image contains many locally similar dark and bright blobs without
the semantic objects found in ordinary photographs. A descriptor must retain
the broad spatial arrangement of the pattern to distinguish one location from
another, while tolerating the local stretch, shear, rotation, blur, and
illumination change caused by deformation. Full affine invariance is not
necessarily desirable: if two different regions become indistinguishable,
invariance has removed the information needed to match them.

The training pairs therefore use one underlying random speckle texture and
known geometric warps. For each sampled source position :math:`x`, the
positive descriptor is evaluated at the analytically mapped target position
:math:`x' = x + u(x)`. The target sample is taken at the exact floating-point
position, rather than requiring the detector to return a point at that exact
location. Negative examples are other positions in the same texture, at least
48 source pixels away. This makes the experiment measure descriptor
discrimination separately from detector repeatability.

The objective is a symmetric contrastive loss. In this setting, the
information noise-contrastive estimation objective compares each positive
pair's cosine similarity with the similarities of candidate distractors. Hard
negatives are the same-texture candidates with the largest current similarity.
The objective can improve the ranking of sampled positive pairs, but a lower
training loss alone does not show that a reliable affine initialization or a
successful inverse compositional Gauss–Newton (IC-GN) solve will follow.

Architectures and WebGPU execution
----------------------------------

Three projections were trained: a linear 64-to-64 map, a residual multilayer
perceptron (MLP) over 64 values, and an MLP that projects three HORG scales
(192 values) to 64. Training uses optional PyTorch. At inference, the HORG
features and learned projection run in WebGPU Shading Language (WGSL) compute
shaders through ``wgpu-py``. The model weights are small NumPy arrays stored
outside the required inference dependencies.

The current implementation performs the HORG extraction for each scale,
copies those features to Python, uploads them for the learned projection,
copies the 64-component result back, then uploads it again for matching. The
numeric kernels run on the graphics processing unit (GPU), but this is not a
GPU-resident pipeline. The boundary is explicit in the API and benchmark; no
zero-copy sharing between allocations is claimed.

What the evidence says
----------------------

The validation set contained 16 speckle textures disjoint from the 64 training
textures. The three-scale projection reached 2.08 percent top-1 retrieval
under the configured spatial exclusion, compared with 1.95 percent for the
fixed three-scale HORG baseline. This is close to the 1.04 percent uniform
reference for 96 candidates. In the held-out six-field IC-GN campaign, learned
HORG produced 53 correct windows out of 91 on one discontinuous field;
fixed HORG-64 produced 57 out of 91. The learned version failed to produce
usable projections on the other five fields, including the two preserved
regression failures.

On native 3840 by 2160 images with 4096 keypoints per image, the paired
descriptor time was 83.9 milliseconds for the learned three-scale projection
and 22.7 milliseconds for fixed HORG-64. Thus this first learned transform is
neither more robust nor faster and must not replace the baseline.

The failed affine and heterogeneous regression cases show a useful separation
of causes: 862 of 1399 visible source keypoints in the affine case had a
detected target point within 2 pixels, yet HORG-64 produced only four
reciprocal matches within 5 pixels. In the heterogeneous case, 574 of 875
visible sources had such a target detection, yet HORG-64 produced no correct
top-1 match within 5 pixels. This points to descriptor ambiguity and global
retrieval, not simply a lack of detected points. It does not prove that a
convolutional descriptor will solve the problem.

The follow-up patch-based convolutional model has eight learned feature
channels, a depthwise spatial filter, coarse spatial pooling, and a compact
projection. An expanded run used 80 training and 20 validation textures,
with disjoint texture seeds and five motion families. The best validation
checkpoint reached 6.04 percent top-1 retrieval among 48 candidates. Its
SHA-256 is recorded with the model and training report. On ten independent
fields per family, the model reached the analytic oracle on translation
(931/1034 visible windows) and improved discontinuous-motion coverage to
684/1013, slightly above DeDoDe-B (660/1013), DeepMatching (664/1013), and
EfficientLoFTR (669/1013). It failed to produce a usable field on all ten
rigid-rotation samples, and on all ten affine and heterogeneous samples;
analytic local affine initializations still converged for 739/881, 655/800,
and 740/896 windows respectively. These results show that the remaining
failure is upstream of the solver for those families. They do not establish
general robustness, and the learned descriptor is not a replacement for the
current baselines.

On the same independent discontinuous fields, the expanded model's aggregate
correct-window throughput was 111 windows/s, compared with 24/s for DeDoDe-B,
60/s for DeepMatching, and 35/s for EfficientLoFTR. Fixed HORG-64 was faster
(223/s) but had lower coverage (499/1013). These are totals over ten distinct
fields, not single-pair latency quantiles; the report explains timing
accounting and per-family failures. On native 3840 by 2160 images with 4096
keypoints, descriptor extraction took 69.1 ms for the CNN and 19.7 ms for
HORG-64. The CNN does not yet beat HORG on descriptor latency, and the
complete extraction/matching/field/IC-GN chain is not GPU-resident.

The two historical regression cases remain frozen and failed for every tested
method: `affine_8101_9101` and `heterogeneous_8102_9102`. They were not used to
select training epochs or tune thresholds. The :doc:`expanded qualification
report <../_generated_results/qualification/dic_descriptor_expanded_icgn_20261108/README>`
contains the complete comparison, all five independent motion families, the
oracle ceiling, timing details, and reproduction commands.

The CNN is still a deliberately small first model. Training supervises
descriptor retrieval with analytically corresponding patches and same-texture
hard negatives; it does not yet train on observed IC-GN success labels. It
supports one 64-pixel patch scale and a 64-component output. The low validation
retrieval score and complete failures on held-out rigid, affine, and
heterogeneous samples motivate improving the training objective and sampling
before adding a larger network.

Training for 100 epochs on the same 80/20 split did not help: the best
validation retrieval fell slightly to 5.83%, compared with 6.04% at epoch 7 of
the 20-epoch run. The longer checkpoint is retained as a negative training
control and was not evaluated on the qualification fields. Thus simply
training longer is not supported by the current evidence.

Read the :doc:`failure analysis and measurements
<../../_generated_results/descriptors/baseline_failure_analysis>` for every
field, raw counts, timings, training artifacts, and reproduction paths. The
next experiments should enlarge texture diversity, evaluate more independent
fields, and test whether matching and field projection can remain GPU-resident.
Any model selection must use validation textures only, and must be judged by
correct IC-GN convergence rather than the loss value.
