# Translation baseline: retain a valid near-identity affine

The original baseline rejected all 20 translation prefits because their linear part was close to identity. With the neural deformation field residual disabled, this discarded their translation. The new opt-in `--retain-translation-affine` retains the fitted affine in this case; it preserves the prior default for archived commands and does not alter the inverse-compositional Gauss–Newton (IC-GN) solver settings.

On the same 20 held-out translation pairs, the corrected affine plus normalized cross-correlation (NCC) baseline obtains **25,404 correct windows out of 29,706 visible**. The seed-and-propagation pipeline obtains **28,698**, with more correct windows on all 20 pairs. All 20 corrected-baseline prefits were retained.

This diagnostic was added after observing the original baseline failure. It is not the preregistered G2 comparison; a new independent evaluation is needed for a confirmatory non-regression claim. The two `seed*.json` reports and `windows_seed*` archives retain per-pair and per-window results.

From the repository root, run the following once for `SEED=21000000` and once for `SEED=22000000`:

```bash
SEED=21000000
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_ndf_pair_icgn.py \
  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --output "results/seed_propagation_20261111/p2_translation_affine_fix/seed${SEED}.json" \
  --window-diagnostics-dir "results/seed_propagation_20261111/p2_translation_affine_fix/windows_seed${SEED}" \
  --native-scale 20 --model-width 192 --model-height 108 --speckle-size .35 \
  --families translation --cases-per-family 10 --seed-offset "$SEED" --contextual-features \
  --rotation-bank-steps 9 --rotation-bank-probe-query-budget 256 --global-affine-prefit \
  --field-residual none --retain-translation-affine --local-ncc-radius 8 --local-ncc-patch-radius 4 \
  --local-ncc-backend wgpu --local-ncc-geometry translation --local-consensus-radius 40 \
  --local-consensus-threshold 5 --projection raw_bilinear
```
