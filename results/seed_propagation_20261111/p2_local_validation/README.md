# Local affine fallback validation on development pairs

This post-hoc ablation evaluates the fixed WebGPU Shading Language (WGSL) local affine source separately on all 20 heterogeneous development pairs. For each pair it fits local models to image-derived hard correspondences from the three image-ranked rotations and selects by sparse inverse-compositional Gauss–Newton (IC-GN) acceptance, with stride 2, displacement consistency 8 px and gradient consistency 0.25. Propagation then uses confirmed seeds. The analytic field is used only for scoring.

All 20 pairs have confirmed seeds; **21,367 windows are correct among 22,416 visible**, with **0 false accepted among 21,367 accepted**. The matched oracle resolves 21,367 windows. Per-pair results and per-window outputs are in `summary.json` and the NPZ archives.

The local model parameters were held fixed. This validation was added after the earlier independent experiment had been inspected, and before selecting or inspecting results on the fresh confirmatory groups 23000000 and 24000000. Input-generation diagnostics for those groups were already being computed concurrently. The validation timings are not performance measurements because another GPU workload was running concurrently.

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 \
  --output-dir results/seed_propagation_20261111/p2_local_validation \
  --rotation-manifest results/seed_propagation_20261111/p2_local_validation_hypotheses/manifest.json \
  --escalation --local-affine-validation --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3
```
