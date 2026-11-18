# Weighted estimates on native-image cases

Only solver-accepted neighbour states are sources. At least two neighbours and agreement within 8 px are required. The resulting values are interpolated estimates, not inverse-compositional Gauss–Newton (IC-GN) accepted measurements.

| Case | Estimated total | Estimated visible | Correct <2 px | Error ≥2 px |
|---|---:|---:|---:|---:|
| tensile_10 | 66 | 0 | 0 | 0 |
| tensile_20 | 66 | 66 | 66 | 0 |
| tensile_30 | 66 | 0 | 0 | 0 |
| tensile_40 | 66 | 0 | 0 | 0 |
| tensile_50 | 66 | 0 | 0 | 0 |
| tensile_60 | 66 | 66 | 66 | 0 |
| frame_040 | 122 | 122 | 122 | 0 |
| frame_100 | 66 | 66 | 66 | 0 |
| frame_199 | 66 | 0 | 0 | 0 |

Analytical truth is used only to score the tensile images. No accuracy is inferred for the experimental pair. Estimates can cross unobserved material boundaries; source masks should be supplied when available.
