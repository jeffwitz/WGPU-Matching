# Fixed P3 perturbation cohorts

The inverse-compositional Gauss–Newton (IC-GN) solver uses the frozen image-only initialization settings. Analytic truth is used only after inference for scoring. Errors refer to centre displacement, not full-support physical validity.

| Cohort | Correct / visible | Baseline correct | False / accepted | Pairs below baseline |
|---|---:|---:|---:|---:|
| shifted_opening | 5999/6391 | 4923 | 0/5999 | 0 |
| doubled_noise | 5199/5480 | 3200 | 0/5199 | 0 |
| photometric | 4723/5001 | 3727 | 0/4723 | 0 |
| defocus | 6643/6896 | 5814 | 0/6643 | 0 |
| camera_noise | 4559/4822 | 2611 | 0/4559 | 0 |
| periodic | 0/6441 | 0 | 0/0 | 0 |

Shifted opening: 132 analytically visible window centres have supports that cross the material boundary; 102 were accepted, of which 102 have centre error <2 px. Even a correct centre does not make the affine model physically valid across the discontinuity. There are 0 accepted propagation parent links whose endpoints lie on opposite sides of the boundary.

Cohorts with no accepted windows: periodic. For these cohorts the false-acceptance rate is undefined; abstention does not validate rejection.

G3 remains incomplete until the remaining prescribed cases are evaluated. See the predeclared `p3_perturbation_protocol.md` and per-window outputs.
