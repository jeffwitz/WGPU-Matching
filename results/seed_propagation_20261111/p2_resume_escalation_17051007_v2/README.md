# Resuming image-only local-affine escalation

The initializer now resumes from previously solver-accepted windows when local
affine hypotheses become necessary. New trials run only in unresolved connected
components; accepted states and their provenance are preserved. The reported
`selected` field identifies the first global trial, while `escalation_history`
and `source_hypothesis_names` identify subsequent sources. These are not timings
for a complete pipeline: image hypothesis generation was performed separately.

On the previously examined heterogeneous pair 17051007/17091007, all original
global hypotheses had zero seeds. The resumed local `rotation_rank2` stage
obtained 15 confirmed seeds and 736 correct visible windows out of 787, with
zero false accepted visible windows; the saved analytical affine initialization
obtained 735 correct. The previous restart-based local-affine experiment also
obtained 736 correct. A same-schedule re-solve control obtained 6 correct windows.
This pair is a development regression test, not independent evidence of a gain.
Because its first stage had no accepted windows, preservation of a partially
successful field is separately checked by a deterministic unit test; it is not
yet qualified on real-image or independent synthetic cases.

Reproduce:

```bash
PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \
  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \
  --archive results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 \
  --output-dir results/seed_propagation_20261111/p2_resume_escalation_17051007_v2 \
  --sparse-stride 4 --rotation-manifest results/seed_propagation_20261111/p2_local_affine_development/manifest.json \
  --textures 17051007 --escalation --local-affine-escalation --same-budget-control
```

All three global rotation hypotheses are still generated before this script;
on-demand generation of expensive sources remains future work. The GPU solver
is accessed through its private research adapter. No analytical truth is used
to choose hypotheses or accepted windows.
