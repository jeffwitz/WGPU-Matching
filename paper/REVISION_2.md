# Scientific revision after author feedback

The first version is preserved in versions/v1/. The revised manuscript has 16 pages, 20 references, five original figures and 18 displayed equations.

Added mathematical_methods.tex: normalized correlation objective and relation to ZNCC; six-parameter warp; normalized reference Jacobian; inverse-compositional linearization; local conditioning versus attraction basin; reciprocal descriptor matching; gradient histogram construction; coarse-to-fine random search; contrastive descriptor objective; robust affine fitting and recentering; limits of IRLS and smooth RBF interpolation; Taylor error of first-order state transport; explicit frontier set and bounded retries. The generic formulations explain method families and are not asserted as an exact implementation of every historical benchmark.

Added causal_results.tex: fixed-data component ablation before/after propagation, windowwise gains/losses, checkpoint-specific negative evidence, oracle-compensation mechanism and its non-operational scope, exact resident-frontier parity and transfer boundary. The new causal figure is generated directly from the archived table with assertions on all four totals.

Stronger argument: affine + NCC gains 4,385 correct windows through propagation on the controlled cohort. The advantage of NCC over affine alone shrinks substantially after propagation, revealing its effect on seed availability and effort. No misleading cross-cohort subtraction is used. The learned coarse matcher remains distinct from the rejected NDF residual.

Reviewed every page through a rendered contact sheet; separately inspected the new causal figure. No undefined citations/references or overfull boxes in the final build. This revision introduces no new experimental measurements, and it does not remove the submission gaps listed in SUBMISSION.md.
