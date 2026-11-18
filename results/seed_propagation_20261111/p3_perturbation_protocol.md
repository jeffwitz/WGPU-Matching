# P3 fixed perturbation cohorts

Each five-pair cohort and its parameters were fixed before inspecting its outputs. The independent target-noise and periodic-pattern cohorts were added after inspecting the shifted-opening cohort. Each uses a separate texture/deformation seed group, the frozen P2 selection settings and unchanged inverse-compositional Gauss–Newton (IC-GN) solver thresholds.

| Cohort | Seed offset | Motion | Perturbation |
|---|---:|---|---|
| Shifted opening | 27000000 | discontinuous | material boundary at 0.503125 image width (1932 px for 4K) |
| Doubled texture noise | 28000000 | heterogeneous | synthetic speckle noise standard deviation 0.05 instead of 0.025 |
| Photometric variation | 29000000 | heterogeneous | target gain 1.2 and offset −0.05, clipped to image range |
| Defocus | 30000000 | heterogeneous | target Gaussian blur standard deviation 1.2 native pixels |
| Independent target noise | 31000000 | heterogeneous | additional independent target noise standard deviation 0.05 |
| Periodic texture | 32000000 | heterogeneous | 90% periodic grid pattern, 128 native-pixel period |

The shifted boundary cuts 64 px window supports, unlike the previously tested grid-aligned opening. The noise cohort doubles the generator's reference texture noise; it is not a separate independent camera-noise model. Target perturbations are applied identically during hypothesis generation and solver evaluation. Truth and physical visibility remain restricted to scoring. These experiments do not cover all traction levels or experimental images.
