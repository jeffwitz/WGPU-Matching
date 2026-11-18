# Fixed protocol: shared prefit and deferred rotations on new pairs

Recorded before generating or evaluating these pairs. Use the frozen P2
settings and seed groups 51000000 and 52000000. Generate one translation and
one heterogeneous synthetic 3840 × 2160 pair per group, giving four pairs.
These groups were not found in the existing protocol, report or manifest
index. This is a small independent regression sample, not a final statistical
qualification.

For each pair, generate the original separate affine-only and affine-plus-local
normalized cross-correlation (NCC) inputs and, independently, the shared-prefit
inputs, with the same image/model settings and retained-translation correction.
Compare all six initial and inverse-compositional Gauss–Newton (IC-GN) output
arrays, plus visibility and truth, for exact equality. Do not adjust thresholds
or exclude a pair if a comparison fails.

Run the existing eager three-rotation campaign using the separate inputs, and
the deferred campaign using the shared inputs. Score both using the same
analytical truth after inference. Report correct, lost and gained windows,
false acceptances, rotations requested, and source/propagation provenance for
each pair. Retain all negative outcomes. As in existing tests, image preparation
is performed by diagnostic scripts; these measurements do not establish
production latency or a fully GPU-resident pipeline.

If any difference is found, report it without promoting the new path as the
default. No thresholds, budgets, seeds, families or acceptance rules will be
changed after inspecting outputs.
