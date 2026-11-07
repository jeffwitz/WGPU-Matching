# Detector and matcher comparison

This report combines the shared-descriptor detector experiment with separate historical full-pipeline results. The protocols are intentionally kept distinct: the first controls the descriptor and local-affine acceptance criteria, while the second reports each matcher as previously benchmarked.

## Shared descriptor and affine RANSAC

The common protocol uses official DaD and DeDoDe v2-L detector weights at a 1024 px maximum side, DeDoDe-B descriptors, mutual cosine nearest neighbours, and the same WGSL local-affine RANSAC settings. DaD/DeDoDe detector inference, DeDoDe-B description, and cosine matching ran on PyTorch/CPU; HORG/DIC point detection and affine RANSAC ran on WebGPU. These are reference results, not a GPU-resident pipeline. The DIC learned detector's internal 16×9 quota pool is followed by global top-K or a second-stage quota selection; the global top-K figure is therefore not over all NMS peaks.

Adapter: Vulkan (Intel(R) Graphics (LNL)); PyTorch CPU threads: 8; steady-state repeats: 2; candidate pool: 2× each requested budget. Error cells are accepted affine-seed RMSE/P95 in pixels. This four-case run uses the first DIC learned checkpoint (2dd4c7cdf9eabd4b2fedcedb76724f914022ac2a4f4dceaa5c695d368aeae5ba); the dedicated five-budget stress sweep uses the discontinuity-aware checkpoint and is reported separately.

| Case | Detector | Selection | Budget | Repeatable ≤2 px | Raw matches <5 px | Accepted affine | Correct <5 px | RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| progressive_040 | DaD | global_topk | 1024 | 838/1024 | 904/906 | 31/32 | 31/32 | 0.28/0.44 | 3.306 | 6.017 | 0.006 | 9.51 | 9.341 | 0.107 |
| progressive_040 | DaD | spatial_quotas | 1024 | 831/1024 | 895/898 | 31/32 | 31/32 | 0.30/0.53 | 3.306 | 6.017 | 0.004 | 7.03 | 9.339 | 0.107 |
| progressive_040 | DeDoDe-v2-L | global_topk | 1024 | 340/1024 | 349/523 | 15/32 | 15/32 | 0.75/1.37 | 9.080 | 6.325 | 0.006 | 5.76 | 15.421 | 0.065 |
| progressive_040 | DeDoDe-v2-L | spatial_quotas | 1024 | 332/1024 | 342/523 | 13/32 | 13/32 | 0.59/0.95 | 9.080 | 6.325 | 0.007 | 5.82 | 15.422 | 0.065 |
| progressive_040 | DIC-Tensor-WGSL | global_topk | 1024 | 371/1024 | 424/558 | 19/32 | 19/32 | 0.69/2.11 | 0.046 | 6.100 | 0.005 | 5.48 | 6.162 | 0.162 |
| progressive_040 | DIC-Tensor-WGSL | spatial_quotas | 1024 | 424/1024 | 486/604 | 23/32 | 23/32 | 0.69/0.99 | 0.046 | 6.100 | 0.005 | 5.92 | 6.160 | 0.162 |
| progressive_040 | HORG-SIFT-common-DeDoDe-B | global_topk | 1024 | 534/685 | 535/547 | 22/32 | 22/32 | 0.16/0.46 | 0.070 | 6.171 | 0.002 | 4.53 | 6.252 | 0.160 |
| progressive_040 | HORG-SIFT-common-DeDoDe-B | spatial_quotas | 1024 | 534/685 | 535/547 | 22/32 | 22/32 | 0.16/0.46 | 0.070 | 6.171 | 0.003 | 4.88 | 6.254 | 0.160 |
| progressive_040 | DIC-Learned-WGSL | global_topk | 1024 | 268/1024 | 280/507 | 11/32 | 11/32 | 0.30/0.62 | 0.056 | 6.092 | 0.004 | 4.47 | 6.159 | 0.162 |
| progressive_040 | DIC-Learned-WGSL | spatial_quotas | 1024 | 256/1024 | 270/501 | 11/32 | 11/32 | 0.31/0.62 | 0.056 | 6.092 | 0.004 | 5.18 | 6.161 | 0.162 |
| progressive_100 | DaD | global_topk | 1024 | 653/993 | 761/794 | 27/28 | 27/28 | 0.49/1.12 | 3.287 | 7.010 | 0.005 | 8.26 | 10.315 | 0.097 |
| progressive_100 | DaD | spatial_quotas | 1024 | 645/992 | 757/785 | 27/28 | 27/28 | 0.43/0.88 | 3.287 | 7.010 | 0.007 | 5.60 | 10.316 | 0.097 |
| progressive_100 | DeDoDe-v2-L | global_topk | 1024 | 158/999 | 173/468 | 9/28 | 8/28 | 1.97/5.59 | 13.098 | 6.952 | 0.006 | 4.93 | 20.064 | 0.050 |
| progressive_100 | DeDoDe-v2-L | spatial_quotas | 1024 | 150/998 | 167/476 | 7/28 | 6/28 | 3.00/5.82 | 13.098 | 6.952 | 0.005 | 6.28 | 20.064 | 0.050 |
| progressive_100 | DIC-Tensor-WGSL | global_topk | 1024 | 212/989 | 302/502 | 17/28 | 15/28 | 2.53/7.21 | 0.049 | 7.158 | 0.004 | 4.69 | 7.221 | 0.138 |
| progressive_100 | DIC-Tensor-WGSL | spatial_quotas | 1024 | 208/986 | 302/510 | 14/28 | 14/28 | 1.08/2.01 | 0.049 | 7.158 | 0.015 | 4.37 | 7.231 | 0.138 |
| progressive_100 | HORG-SIFT-common-DeDoDe-B | global_topk | 1024 | 448/673 | 456/489 | 20/28 | 20/28 | 0.24/0.38 | 0.074 | 7.468 | 0.002 | 6.50 | 7.556 | 0.132 |
| progressive_100 | HORG-SIFT-common-DeDoDe-B | spatial_quotas | 1024 | 448/673 | 456/489 | 20/28 | 20/28 | 0.24/0.38 | 0.074 | 7.468 | 0.003 | 5.48 | 7.555 | 0.132 |
| progressive_100 | DIC-Learned-WGSL | global_topk | 1024 | 125/980 | 169/457 | 7/28 | 7/28 | 0.84/1.37 | 0.062 | 6.940 | 0.005 | 3.61 | 7.018 | 0.142 |
| progressive_100 | DIC-Learned-WGSL | spatial_quotas | 1024 | 110/979 | 156/463 | 6/28 | 5/28 | 3.56/8.58 | 0.062 | 6.940 | 0.008 | 5.14 | 7.021 | 0.142 |
| progressive_199 | DaD | global_topk | 1024 | 595/940 | 720/756 | 26/28 | 26/28 | 0.45/0.88 | 4.171 | 7.150 | 0.005 | 6.86 | 11.336 | 0.088 |
| progressive_199 | DaD | spatial_quotas | 1024 | 596/935 | 724/757 | 27/28 | 27/28 | 0.98/1.97 | 4.171 | 7.150 | 0.013 | 4.63 | 11.342 | 0.088 |
| progressive_199 | DeDoDe-v2-L | global_topk | 1024 | 126/951 | 138/407 | 4/28 | 4/28 | 0.79/1.13 | 10.075 | 7.161 | 0.010 | 3.31 | 17.256 | 0.058 |
| progressive_199 | DeDoDe-v2-L | spatial_quotas | 1024 | 117/941 | 128/415 | 4/28 | 4/28 | 0.59/0.72 | 10.075 | 7.161 | 0.012 | 5.11 | 17.257 | 0.058 |
| progressive_199 | DIC-Tensor-WGSL | global_topk | 1024 | 166/947 | 280/477 | 11/28 | 9/28 | 3.22/8.61 | 0.050 | 7.140 | 0.005 | 4.55 | 7.204 | 0.139 |
| progressive_199 | DIC-Tensor-WGSL | spatial_quotas | 1024 | 163/935 | 274/493 | 11/28 | 9/28 | 3.39/8.61 | 0.050 | 7.140 | 0.005 | 4.41 | 7.204 | 0.139 |
| progressive_199 | HORG-SIFT-common-DeDoDe-B | global_topk | 1024 | 334/630 | 380/419 | 17/28 | 17/28 | 0.53/1.05 | 0.081 | 7.031 | 0.003 | 3.64 | 7.122 | 0.140 |
| progressive_199 | HORG-SIFT-common-DeDoDe-B | spatial_quotas | 1024 | 334/630 | 380/419 | 17/28 | 17/28 | 0.53/1.05 | 0.081 | 7.031 | 0.013 | 4.47 | 7.136 | 0.140 |
| progressive_199 | DIC-Learned-WGSL | global_topk | 1024 | 134/933 | 172/441 | 5/28 | 5/28 | 0.91/1.54 | 0.061 | 7.193 | 0.010 | 5.96 | 7.273 | 0.137 |
| progressive_199 | DIC-Learned-WGSL | spatial_quotas | 1024 | 138/929 | 177/452 | 6/28 | 6/28 | 1.07/1.92 | 0.061 | 7.193 | 0.014 | 6.82 | 7.277 | 0.137 |
| synthetic_60pct | DaD | global_topk | 1024 | 231/665 | 422/550 | 16/16 | 16/16 | 1.09/2.04 | 4.099 | 7.059 | 0.006 | 4.84 | 11.173 | 0.089 |
| synthetic_60pct | DaD | spatial_quotas | 1024 | 230/650 | 413/532 | 16/16 | 16/16 | 1.22/3.06 | 4.099 | 7.059 | 0.007 | 3.96 | 11.171 | 0.090 |
| synthetic_60pct | DeDoDe-v2-L | global_topk | 1024 | 36/636 | 51/381 | 0/16 | 0/16 | — | 10.461 | 7.247 | 0.007 | 4.78 | 17.724 | 0.056 |
| synthetic_60pct | DeDoDe-v2-L | spatial_quotas | 1024 | 41/646 | 56/377 | 0/16 | 0/16 | — | 10.461 | 7.247 | 0.008 | 5.05 | 17.725 | 0.056 |
| synthetic_60pct | DIC-Tensor-WGSL | global_topk | 1024 | 2/667 | 20/383 | 0/16 | 0/16 | — | 0.049 | 6.994 | 0.026 | 4.06 | 7.077 | 0.141 |
| synthetic_60pct | DIC-Tensor-WGSL | spatial_quotas | 1024 | 3/647 | 22/382 | 0/16 | 0/16 | — | 0.049 | 6.994 | 0.014 | 4.59 | 7.066 | 0.142 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | global_topk | 1024 | 37/443 | 91/310 | 2/16 | 2/16 | 2.98/4.16 | 0.078 | 7.152 | 0.003 | 3.77 | 7.241 | 0.138 |
| synthetic_60pct | HORG-SIFT-common-DeDoDe-B | spatial_quotas | 1024 | 37/443 | 91/310 | 2/16 | 2/16 | 2.98/4.16 | 0.078 | 7.152 | 0.003 | 3.33 | 7.241 | 0.138 |
| synthetic_60pct | DIC-Learned-WGSL | global_topk | 1024 | 63/639 | 122/418 | 5/16 | 5/16 | 1.62/2.96 | 0.055 | 6.329 | 0.006 | 4.14 | 6.397 | 0.156 |
| synthetic_60pct | DIC-Learned-WGSL | spatial_quotas | 1024 | 69/638 | 128/423 | 6/16 | 6/16 | 1.80/3.18 | 0.055 | 6.329 | 0.004 | 3.85 | 6.395 | 0.156 |

## 60% tensile stress test: complete-pipeline context

All 16 seeds are analytically visible in the deformed image. The common-descriptor entries use budget 1024 and the global top-K-from-candidate-pool selection. Historical matchers retain their own published parameter and timing scopes; use this table to compare behavior, not to claim an identical implementation path.

| Pipeline | Accepted | Accepted correct <5 px | Accepted ≤2 px | Time | Timing scope |
|---|---:|---:|---:|---:|---|
| DeepMatching INRIA | 10/16 | 8/16 | 4/16 | 16.405 s | WebGPU native-resolution full matcher |
| HORG | 0/16 | 0/16 | 0/16 | 0.146 s | Browser WebGPU full matcher |
| XFeat | 15/16 | 12/16 | 3/16 | 0.386 s | CPU ONNX inference only; affine fit excluded |
| EfficientLoFTR | 16/16 | 14/16 | 12/16 | 1.501 s | ONNX inference plus local affine; 640×360 input |
| DaD + DeDoDe-B | 16/16 | 16/16 | 15/16 | 11.173 s | CPU detector/descriptor/mutual match plus WGSL RANSAC |
| DeDoDe v2-L + DeDoDe-B | 0/16 | 0/16 | 0/16 | 17.724 s | CPU detector/descriptor/mutual match plus WGSL RANSAC |
| DIC tensor WGSL + DeDoDe-B | 0/16 | 0/16 | 0/16 | 7.077 s | WGSL detection/RANSAC; DeDoDe-B and mutual matching on CPU |
| HORG/SIFT + DeDoDe-B | 2/16 | 2/16 | 1/16 | 7.241 s | WGSL detection/RANSAC; DeDoDe-B and mutual matching on CPU |
| DIC learned WGSL + DeDoDe-B | 5/16 | 5/16 | 4/16 | 6.397 s | WGSL detection/RANSAC; DeDoDe-B and mutual matching on CPU |

The historical browser run with a smaller search bound had 6 accepted estimates, 5 below 5 px, and 147.4 px accepted-set RMSE. Those statistics belong to a different run from the Python 1024 px-search row above (10 accepted, 8 below 5 px); they must not be combined. The four-case DaD row reaches 16/16 affine seeds under 5 px in this one synthetic, globally affine stress case, but costs about 11.17 s/pair in its CPU reference path. At budget 1024, the first DIC learned checkpoint yields 5/16 correct under 5 px with global top-K (6/16 with spatial quotas), versus 0/16 for the tensor baseline; the separate budget sweep evaluates the later discontinuity-aware checkpoint. None of these figures measures actual IC-GN convergence.

## Artifacts and limitations

* The complete per-seed common-descriptor data and model hashes are in `detector_common_descriptor_20261107.json`.
* Native 4K DIC detector latency and score parity are in `dic_detector_native_4k_20261107.md` and `dic_detector_wgsl_parity_4k_20261107.json`.
* Held-out 256×256 synthetic repeatability is in `dic_detector_heldout_20261107.md`; this is detector-only and does not use descriptors.
* A discontinuity-aware checkpoint and a paired detector-only comparison are in `dic_detector_discontinuous_training_20261107.md`; this does not establish matching or IC-GN performance.
* The shared-descriptor budget sweep from 512 to 8192 on the 60% case is in `detector_budget_sweep_60pct_20261107.md`; it uses the discontinuity-aware DIC score checkpoint, not the first checkpoint above.
* The 60% tensile cases are synthetic global affine extrapolations. Smooth heterogeneous and piecewise-discontinuous motions have only small held-out synthetic samples; experimental cracks, localization bands, finite-element fields, and downstream IC-GN convergence remain unqualified.
* DaD and DeDoDe checkpoints were trained for natural-image local-feature benchmarks. DaD + DeDoDe-B is an experimental combination, not an officially released end-to-end model.


The discontinuity-aware held-out study is summarized separately because its metric is keypoint repeatability, not matched DIC-window coverage. It compares the earlier and updated score weights on identical texture/warp pairs and reports motion-family breakdowns.