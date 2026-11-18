"""Retry unresolved DIC centers with smaller image-verified affine subsets.

This module orchestrates numerical callbacks; the supplied solver owns the GPU
execution. It never sees synthetic truth and does not infer a residual score.
"""

import numpy as np

from .seed_propagation import transfer_to_fine_grid


def retry_small_subsets(
    centres,
    shape,
    valid,
    coarse_parameters,
    coarse_accepted,
    solve,
    *,
    subset_sizes=(32, 16),
    tau_u=8.0,
    tau_g=0.25,
    suspect=None,
):
    """Preserve reliable 64-pixel states and retry only unresolved centers.

    ``solve(size, indices, initial)`` returns affine parameters and acceptance.
    Sources are exclusively the original accepted, nonsuspect coarse states;
    ``suspect``, if supplied, must be derived from images, never from truth.
    The returned ``stage`` is 0 for preserved coarse states, 32 or 16 for
    accepted re-solves, and -1 for unresolved states. This is an experimental
    initialization result, not a final measurement or a residual detector.
    """
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    count = len(centres)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    parameters = np.asarray(coarse_parameters, dtype=np.float64).reshape(-1, 6).copy()
    accepted = np.asarray(coarse_accepted, dtype=bool).reshape(-1).copy()
    suspect = np.zeros(count, bool) if suspect is None else np.asarray(suspect, bool).reshape(-1)
    if count != np.prod(shape) or any(
        len(x) != count for x in (valid, parameters, accepted, suspect)
    ):
        raise ValueError("incompatible coarse grid or masks")
    if tuple(subset_sizes) not in ((32,), (32, 16)):
        raise ValueError("supported subset schedules are (32,) and (32, 16)")
    if not np.isfinite((tau_u, tau_g)).all() or tau_u < 0 or tau_g < 0:
        raise ValueError("consistency thresholds must be finite and nonnegative")
    trusted = accepted & valid & ~suspect & np.isfinite(parameters).all(axis=1)
    accepted = trusted.copy()
    stage = np.full(count, -1, dtype=np.int16)
    stage[trusted] = 0
    sources = np.full(count, -1, dtype=np.int32)
    attempts = np.zeros(count, dtype=np.uint8)
    reasons = np.zeros(count, dtype=np.uint8)
    initial = transfer_to_fine_grid(centres, shape, parameters, trusted, centres)
    available = np.asarray(initial["available"], bool) & valid
    snapshots = {}
    for size in subset_sizes:
        indices = np.flatnonzero(available & ~accepted)
        if not len(indices):
            snapshots[size] = {"parameters": parameters.copy(), "accepted": accepted.copy()}
            continue
        guess = np.asarray(initial["parameters"][indices], dtype=np.float64)
        solved, status = solve(size, indices, guess.copy())
        solved = np.asarray(solved, dtype=np.float64).reshape(-1, 6)
        status = np.asarray(status, dtype=bool).reshape(-1)
        if len(solved) != len(indices) or len(status) != len(indices):
            raise ValueError("incompatible fine solver result")
        attempts[indices] += 1
        finite = np.isfinite(solved).all(axis=1)
        du = np.linalg.norm(solved[:, [0, 3]] - guess[:, [0, 3]], axis=1)
        dg = np.linalg.norm(solved[:, [1, 2, 4, 5]] - guess[:, [1, 2, 4, 5]], axis=1)
        good = status & finite & (du <= tau_u) & (dg <= tau_g)
        reasons[indices] = np.where(
            ~status | ~finite, 1, np.where(du > tau_u, 2, np.where(dg > tau_g, 3, 0))
        )
        winners = indices[good]
        parameters[winners] = solved[good]
        accepted[winners] = True
        stage[winners] = size
        sources[winners] = initial["source"][winners]
        snapshots[size] = {"parameters": parameters.copy(), "accepted": accepted.copy()}
    return {
        "parameters": parameters,
        "accepted": accepted,
        "stage": stage,
        "source": sources,
        "attempts": attempts,
        "rejection_reason": reasons,
        "available": available,
        "suspect": suspect,
        "trusted_coarse": trusted,
        "snapshots": snapshots,
    }
