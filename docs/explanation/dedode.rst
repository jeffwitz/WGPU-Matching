DeDoDe v2 and DeDoDe-B: separate detection and description
==========================================================

Learning geometric consistency
------------------------------

The original `DeDoDe paper <https://arxiv.org/abs/2308.08479>`_ decouples
detector and descriptor training. Structure-from-motion tracks provide
geometric detector supervision; a two-view objective expands initially sparse
track support. The descriptor is trained separately with a mutual-neighbour
objective. This permits changing one component without retraining both.

What v2 changes
---------------

`DeDoDe v2 <https://arxiv.org/html/2404.08928v1>`_ addresses clustered points
by suppressing maxima in the **training target distribution**. It adds
90-degree rotations and horizontal flips and shortens training. The paper
also observes that better detector repeatability can worsen downstream
relative pose, and evaluates points using a fixed dense matcher.

Training-target suppression is different from simply adding NMS at inference.
Augmentation is not a proof of exact rotational equivariance or affine
invariance. This repository uses the v2-L detector checkpoint. The v2 revision
does not mean that descriptor-B is a new v2 descriptor.

Description and mutual matching
-------------------------------

The `official implementation <https://github.com/Parskatt/DeDoDe>`_ provides
separate models. We use descriptor-B, not the DINOv2-based descriptor-G.
For unit-normalized descriptors,

.. math::

   s_{ij}=\hat{\mathbf d}_i^\mathsf T\hat{\mathbf e}_j,
   \qquad \hat{\mathbf d}=\mathbf d/\|\mathbf d\|_2.

The common matcher retains a pair if both points are each other's highest
similarity candidates and the configured similarity threshold is met.
This discrete mutual-neighbour rule is not a global assignment optimization.
It does not measure second-best ambiguity.

Sampling B at DaD, HORG or DIC-Detector points controls the description stage
when comparing detectors. The candidate pools still differ, and changing a
budget changes the descriptor competition. A detector comparison must
therefore also report raw pair precision and final seed coverage.

Repository implementation and interpretation
--------------------------------------------

PyTorch adapters are the reference path. ONNX detector exports return points
and scores and have been checked at K=1024 and K=8192. B description and
mutual matching in the common benchmark remain CPU reference stages, followed
by WGSL RANSAC.

The small-budget speckle tests show poor downstream coverage for v2-L+B.
Increasing budget recovers useful affine fits. This concerns that
detector/descriptor combination and sampling density, not all possible
DeDoDe configurations. Compare :doc:`the progressive series
<../_generated_results/features/detector_onnx_progressive_20261107>` with
:doc:`the large-budget experiment
<../_generated_results/features/detector_onnx_high_budget_20261107>`.


The :doc:`complete initializer qualification
<../_generated_results/qualification/all_methods_20261107/index>` re-runs
the current pipelines on the same progressive and 10–60 percent tensile
cases, with both projection requests and a visual comparison for every
outcome. It evaluates inferred field initializations and preserves the
older local-initializer experiments separately.
