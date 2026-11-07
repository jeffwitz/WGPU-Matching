# Projected matcher fields and visual checks

Each `visualizations/<case>/<model>/<basis>/` directory contains `image1_ref.png` and `image1_def_cor.png`. The corrected target is bilinearly sampled at `x + u(x,y), y + v(x,y)` from the fitted field; it is a visual check, not an additional matcher or a DIC solve. Preview images are capped at 1920 pixels wide. The ordered hard-linked sequence is convenient for EOG:

```bash
eog results/projection/dataset_run_adrien_visual_20261107/eog_sequence/*.png
```

The affine/RBF error metrics and exact seeds are in `summary.json` and `patch_predictions.csv`; direct extracted pairs are in `raw_correspondences.csv`.
