# Deferred rotation hypotheses for image-only initialization

This experimental mode first verifies zero, the existing global affine and the
existing affine plus local normalized cross-correlation (NCC). Only if a region
has fewer than the frozen minimum number of confirmed inverse-compositional
Gauss–Newton (IC-GN) anchors does it generate the three additional forced
rotation hypotheses, resume from the accepted states, and, if needed, fit local
affines. The mode does not use analytical truth to decide whether to generate.

The image inputs and initial global-affine/NCC hypotheses were already
prepared. In particular, the original image-derived rotation ranking used in
that preparation has already been paid for. This experiment avoids additional
forced-hypothesis exports; it does not eliminate the initial global search or
measure total cold-start latency. The five pairs below were previously inspected
and selected for this diagnostic, not an independent acceptance test.

| Pair | Correct eager / deferred | False accepted eager / deferred | Additional rotation exports |
|---|---:|---:|---:|
| 23051001 / 23091001 | 1029 / 1029 | 0 / 0 | 0 |
| 23051003 / 23091003 | 968 / 968 | 0 / 0 | 0 |
| 23051015 / 23091015 | 1344 / 1344 | 0 / 0 | 0 |
| 24051003 / 24091003 | 901 / 901 | 0 / 0 | 0 |
| 24051011 / 24091011 | 637 / 637 | 0 / 0 | 0 |

The accepted and correct window masks are identical to the eager reference on
all five pairs. Fifteen forced-hypothesis exports were avoided. The historical
eager exports for these pairs took a median of 27.85 seconds for three exports,
including subprocess startup and diagnostic work; that historical figure is
not a current production timing or a controlled repeated end-to-end speedup.

For the previously inspected development pair 17051007 / 17091007, the initial
stage had no seeds. The deferred path generated three rotations and local-affine
hypotheses, then recovered 736 / 787 visible windows with zero false accepted,
matching the eager result. The new one-off cold generation took 22.49 seconds;
the following pilot work took 0.58 seconds. It still includes process startup,
reconstruction of synthetic inputs, model setup and diagnostic work. The
analytical-affine control converged on 735 windows and is not a mathematical
upper bound. See `../p2_lazy_rotation_fallback_17051007_verified/summary.json`.

The cache records SHA-256 fingerprints of the weights, ranking report,
generator and qualification script, and any external image inputs. A mismatched
or unverified cache is rejected; a verified cache is read without rewriting the
manifest. One earlier cache-reuse trial rewrote the manifest and cleared the
three per-rank timing fields; the original cold total above remains in the
saved summary. Cache reuse yielded the same 736 correct windows, and a later
verified reuse left the manifest checksum unchanged. The cache fingerprints
inputs, not the contents of every output buffer.

The method stays experimental: choosing not to generate alternatives can lose
coverage in cases with misleading but sufficient anchors. Five examined pairs
do not establish non-regression on independent rotations, localized deformation,
or experimental images. Global affine and NCC inputs still require their own
generation; the full pipeline is not GPU-resident and its end-to-end time is
not measured here.

Reproduce the five-pair diagnostic:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/seed_propagation_20261111/p2_confirmatory_inputs \
  --output-dir results/seed_propagation_20261111/p2_lazy_rotation_five_pairs \
  --sparse-stride 2 --lazy-rotation-weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --lazy-rotation-output results/seed_propagation_20261111/p2_lazy_rotation_five_pairs/rotations \
  --textures 23051001 23051003 23051015 24051003 24051011 --escalation \
  --local-affine-escalation --same-budget-control \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
```

Reproduce the fallback in a fresh output directory (the first run generates
three hypotheses; subsequent runs may use a verified cache):

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 \
  --output-dir results/seed_propagation_20261111/p2_lazy_rotation_fallback_17051007_verified \
  --sparse-stride 4 --lazy-rotation-weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --lazy-rotation-output results/seed_propagation_20261111/p2_lazy_rotation_fallback_17051007_verified/rotations \
  --textures 17051007 --escalation --local-affine-escalation --same-budget-control
```
