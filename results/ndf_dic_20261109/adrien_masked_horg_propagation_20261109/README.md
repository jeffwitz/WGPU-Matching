# Masked Adrien pair: accepted windows only

Digital image correlation (DIC) windows were initialized from histogram of oriented and rotated gradients (HORG) correspondences and an image-derived affine estimate, refined by normalized cross-correlation (NCC), then processed by the WebGPU inverse-compositional Gauss-Newton (IC-GN) solver. Zero-mean normalized cross-correlation (ZNCC) is the solver's acceptance score. The supplied region-of-interest (ROI) mask was used only by the solver's reference validity; it did not select matching points or the affine fit. This experimental pair has no analytical displacement truth, so the report gives acceptance counts and maps only, not accuracy.

- HORG correspondences: 4096 raw, 207 accepted.
- Affine fit: 49 inliers; available=True.
- WebGPU local NCC valid cells: 4376; elapsed 0.010 s.
- IC-GN accepted: 6593 after one pass; 7103 after propagation.
- Propagation added: 510; elapsed 0.145 s.

Maps: `accepted_windows.png` (green after one pass, blue added by propagation, gray rejected) and `ring_map.png` (ring zero is the seed; successive propagation rings use distinct colors; gray is rejected). Exact ring indices remain in the per-window archive.

Reproduce the count-and-map run with:

```console
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_adrien_masked_counts.py \
  --live-dic-source /path/to/Live-DIC/python/src \
  --output-dir results/ndf_dic_20261109/adrien_masked_horg_propagation_20261109
```
