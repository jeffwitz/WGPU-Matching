# Native-image P3 qualification

The frozen image-only initializer and inverse-compositional Gauss–Newton (IC-GN) solver run on original pixels. For tensile cases, analytical centred affine truth comes from the archived dataset manifests. Experimental-image accuracy is not inferred.

| Case | Native pixels | Correct / visible | Baseline correct | Oracle correct | False accepted | Accepted |
|---|---|---:|---:|---:|---:|---:|
| tensile_10 | 3840 × 2160 | 1782/1782 | 1605 | 1782 | 0 | 1782 |
| tensile_20 | 3840 × 2160 | 1584/1650 | 1541 | 1584 | 0 | 1584 |
| tensile_30 | 3840 × 2160 | 1518/1518 | 1365 | 1518 | 0 | 1518 |
| tensile_40 | 3840 × 2160 | 1386/1386 | 1286 | 1386 | 0 | 1386 |
| tensile_50 | 3840 × 2160 | 1320/1320 | 1166 | 1320 | 0 | 1320 |
| tensile_60 | 3840 × 2160 | 1188/1254 | 1079 | 1188 | 0 | 1188 |
| frame_040 | 3840 × 2160 | 1856/1980 | 1856 | 1856 | 0 | 1856 |
| frame_100 | 3840 × 2160 | 1848/1914 | 1716 | 1848 | 0 | 1848 |
| frame_199 | 3840 × 2160 | 1782/1782 | 1605 | 1782 | 0 | 1782 |

The green and blue map in each case directory records accepted windows and confirmed seeds. Per-window results are in that case's qualification directory. These are scripted experimental runs, not production timing measurements.
