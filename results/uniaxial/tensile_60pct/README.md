# Synthetic tensile series up to 60%

`reference.png` and the six UHD snapshots were generated with a local adaptation of `experiments/tensile_extensometer_4k/generate.py` from `DIC_FROM_H264`. The other repository was not modified. Noise, random seed, material constants, elastoplastic calculation, inverse OpenCV remapping, and border mode are preserved; only maximum axial engineering strain was increased from 10% to 60%. The generated reference has the same SHA-256 as the archived Live-DIC reference.

For the 60% snapshot, the model gives `stretch_x=1.6000`, `stretch_y=0.7910`, and `stress_true≈607 MPa`. Each row of `machine_truth.csv` contains one of the 200 loading states. The selected snapshots are the closest available to 10%, 20%, …, 60%; exact values and frame numbers are in `manifest.json`.

The follow-up comparison of **DeepMatching, HORG, and CPM-WGPU** on the same 16 visible frame-199 seeds is in [`three_method_browser_60.md`](three_method_browser_60.md), with summary CSV and per-seed JSON. The original Python DeepMatching/HORG table below is retained unchanged.

**Physical limitation:** the source generator validates the material only up to 10%. Beyond that, the images are a synthetic extrapolation of the same constitutive model and Ludwik hardening law, not a validated experimental prediction. Analytic truth remains exact for the applied image transform.

The [contact sheet](contact_sheet.png) shows all six levels. The [results CSV](matcher_comparison.csv) compares default Python matcher settings. Both detectors use the same denominator: centers projected from the reference must remain inside the target. The same seed list is passed explicitly to both; HORG preserves the original RANSAC seed index to keep deterministic sampling from the full grid. Errors are computed at the source coordinate actually reported by each matcher. The protocol is in [`benchmark_tensile_60pct.py`](../../../python/benchmarks/benchmark_tensile_60pct.py); the adapted generator is [`generate_tensile_60pct.py`](../../../python/benchmarks/generate_tensile_60pct.py).

| Axial strain | Visible POIs | HORG accepted | HORG ≤2 px | DeepMatching accepted | DeepMatching ≤2 px | HORG RMSE | DeepMatching RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9.90% | 28 | 22 | 22 | 28 | 21 | 0.610 px | 1.506 px |
| 20.07% | 24 | 0 | 0 | 22 | 19 | — | 16.772 px |
| 29.87% | 24 | 0 | 0 | 17 | 10 | — | 57.525 px |
| 40.04% | 24 | 0 | 0 | 12 | 7 | — | 295.404 px |
| 49.83% | 20 | 0 | 0 | 11 | 8 | — | 166.554 px |
| 60.00% | 16 | 0 | 0 | 6 | 2 | — | 147.423 px |

“Accepted” and “≤2 px” are counts among visible seeds. DeepMatching retains more reciprocal results here, but many become inaccurate relative to analytic truth as deformation increases. At 60%, 16 POIs are visible, six results are reciprocal, and only two POIs are within 2 px. HORG is more selective and accepts no POIs from 20% strain onward. The acceptance decisions are not interchangeable; precision within 2 px is the common test.
