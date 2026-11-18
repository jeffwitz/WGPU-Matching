# Analytical-initialization control for false windows

> **Superseded scoring:** this historical diagnostic used rounded-pixel truth. The corrected exact-center results and exact oracle control are in `../exact_center_analysis/README.md`. The 123-window count is not the corrected physical-center count.

The 123 visible windows falsely accepted by the image-only initializer at peak shear gradient 2.0 were separately re-solved from the exact local analytical displacement and displacement gradient. This is an oracle-only diagnostic: analytical truth is never provided to the tested initializer. The cached source pixels, subset (64 × 64), stride (64), interpolation, 30 iteration limit, tolerance and minimum normalized cross-correlation match the qualification.

All 123 windows are accepted again by inverse-compositional Gauss–Newton (IC-GN), and none ends within 2 pixels of the analytical center displacement. The median oracle error is 2.31, 2.27 and 2.25 pixels for the three textures; the maximum is 2.87 pixels. The final oracle and image-initialized translations differ by a median of 0.000062 pixel and at most 0.000928 pixel on those windows. This shows that those failures are not repaired by replacing the image-derived initial state with this analytical affine initial state. A finite affine subset approximating a curved displacement field is a plausible cause, but this control alone does not establish the mechanism or rule out interpolation and solver bias.

The NPZ archives contain the analytical control's accepted flags, parameters and errors; `summary.json` gives per-pair counts. Reproduce:

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/diagnose_shear_oracle.py \
  --base results/seed_propagation_20261111/p3_shear_band_frozen_protocol \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --output-dir results/seed_propagation_20261111/p3_shear_band_frozen_protocol/oracle_diagnostic_new
```
