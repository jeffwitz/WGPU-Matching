# Unresolved-region local-affine escalation

The existing confirmed field is retained; local affine models are fitted only at unresolved valid centres. Ground truth is used for scoring only. These pairs were previously inspected and this is a post-hoc diagnostic, not a new independent qualification. The timings exclude image-hypothesis generation.

| Pair | Correct before / after / oracle | False before / after | States preserved | Local fits / total | Diagnostic seconds before / after |
|---|---:|---:|---|---:|---:|
| heterogeneous_texture23051001_warp23091001 | 1029 / 1029 / 1029 | 0 / 0 | True | 951/1980 | 0.960 / 1.004 |
| heterogeneous_texture23051003_warp23091003 | 968 / 968 / 968 | 0 / 0 | True | 1012/1980 | 0.756 / 0.947 |
| heterogeneous_texture23051015_warp23091015 | 1344 / 1344 / 1345 | 0 / 0 | True | 636/1980 | 1.029 / 1.145 |
| heterogeneous_texture24051003_warp24091003 | 901 / 901 / 901 | 0 / 0 | True | 1079/1980 | 0.597 / 0.712 |
| heterogeneous_texture24051011_warp24091011 | 637 / 637 / 637 | 0 / 0 | True | 1343/1980 | 0.524 / 0.690 |

Previously accepted means solver status, not proof of correctness. The compact fit may change random sampling compared with full-grid fitting. Diagnostic times are single runs and do not establish production performance. Per-window results and escalation histories are in the saved NPZ and JSON files.

Reproduce this comparison:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/seed_propagation_20261111/p2_confirmatory_inputs \
  --output-dir results/seed_propagation_20261111/p2_unresolved_affine_five_pairs --sparse-stride 2 \
  --rotation-manifest results/seed_propagation_20261111/p2_confirmatory/rotation_manifest.json \
  --textures 23051001 23051003 23051015 24051003 24051011 \
  --escalation --local-affine-escalation --escalate-unresolved --same-budget-control \
  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json
PYTHONPATH=python/src .venv/bin/python python/benchmarks/summarize_unresolved_escalation.py \
  --baseline results/seed_propagation_20261111/p2_confirmatory/evaluation --experiment results/seed_propagation_20261111/p2_unresolved_affine_five_pairs
```
