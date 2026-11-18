# Fine-grid transfer check on P2 confirmatory pairs

This check applies the already implemented 96 px nearest-accepted 64 px affine-state transfer to the 32 px / 16 px grid of seed groups 23000000 and 24000000. The 32 px inverse-compositional Gauss–Newton (IC-GN) solve uses the same 30 iterations, tolerance 0.001 and minimum zero-mean normalized cross-correlation (ZNCC) of 0.65 as the development experiment. Only solver-accepted fine windows count as measurements; analytical visibility is used solely for scoring.

The displacement fields of these pairs have already been examined in the coarse-grid P2 confirmatory experiment. This is an out-of-development fine-grid check, not a fully untouched final test. No transfer radius or solver threshold is tuned using its results. The cases are continuous heterogeneous fields; this does not validate transfer across material boundaries.
