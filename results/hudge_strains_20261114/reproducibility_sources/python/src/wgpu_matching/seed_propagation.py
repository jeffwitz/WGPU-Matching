"""Host-side, snapshot-based propagation of accepted affine DIC states.

Parameters use the solver order ``(u, du/dx, du/dy, v, dv/dx, dv/dy)``.
The caller supplies a batched solver callback and decides which anchors are valid.
"""

import numpy as np


def transport(parameters, delta, affine=True):
    """Transport a state from its centre to a neighbouring centre."""
    result = np.asarray(parameters, dtype=np.float64).copy()
    if affine:
        result[0] += result[1:3] @ delta
        result[3] += result[4:6] @ delta
    else:
        result[[1, 2, 4, 5]] = 0
    return result


def neighbours(shape, index):
    """Return direct neighbours before diagonal neighbours in stable order."""
    y, x = divmod(index, shape[1])
    for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
            yield yy * shape[1] + xx


def connected_components(mask):
    """Return indices of eight-connected components in a two-dimensional mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    unseen = set(np.flatnonzero(mask).tolist())
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbour in neighbours(mask.shape, index):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        result.append(np.asarray(sorted(component), dtype=np.int64))
    return result


def weighted_neighbour_initialization(centres, shape, valid, parameters, accepted, *,
                                      min_neighbours=2, max_disagreement=8.0, region_labels=None):
    """Estimate unresolved states from validated neighbours, without accepting them.

    Each neighbour's affine state is transported to the requested centre and
    weighted by inverse centre distance. Sources are taken from one immutable
    accepted snapshot; interpolated windows are never reused as sources.
    Conflicting predictions are withheld rather than averaged across a jump.
    Optional nonnegative integer region labels restrict sources to the same
    caller-supplied material region; negative labels disable interpolation.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    accepted = np.asarray(accepted, dtype=bool).reshape(-1)
    state = np.asarray(parameters, dtype=float).reshape(-1, 6)
    count = len(centres)
    if (count != np.prod(shape) or len(valid) != count or len(accepted) != count
            or len(state) != count or np.any(accepted & ~valid)):
        raise ValueError("incompatible grid, validity mask, or accepted states")
    if min_neighbours < 1 or max_disagreement < 0:
        raise ValueError("invalid neighbourhood settings")
    regions = np.zeros(count, dtype=np.int64) if region_labels is None else np.asarray(region_labels).reshape(-1)
    if len(regions) != count or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("region labels must be an integer per window")
    estimates = np.full((count, 6), np.nan, dtype=float)
    support = np.zeros(count, dtype=np.int32)
    spread = np.full(count, np.nan, dtype=float)
    for j in np.flatnonzero(valid & ~accepted):
        if regions[j] < 0:
            continue
        sources = [i for i in neighbours(shape, j) if accepted[i] and regions[i] == regions[j]
                   and np.isfinite(state[i]).all()]
        support[j] = len(sources)
        if len(sources) < min_neighbours:
            continue
        delta = centres[j] - centres[sources]
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 0):
            raise ValueError("neighbouring centres must be distinct")
        predictions = np.array([transport(state[i], d) for i, d in zip(sources, delta, strict=True)])
        displacement = predictions[:, [0, 3]]
        disagreement = np.linalg.norm(displacement[:, None, :] - displacement[None, :, :], axis=2).max()
        spread[j] = disagreement
        if disagreement <= max_disagreement:
            estimates[j] = np.average(predictions, axis=0, weights=1.0 / distance)
    return {"parameters": estimates, "available": np.isfinite(estimates).all(axis=1),
            "support": support, "disagreement_px": spread}


def transfer_to_fine_grid(coarse_centres, coarse_shape, coarse_parameters, coarse_accepted,
                          fine_centres, *, max_distance_px=96.0,
                          coarse_region_labels=None, fine_region_labels=None):
    """Transport the nearest accepted coarse affine state to each fine centre.

    Output states are initializations only. The returned source index is -1
    when no accepted coarse centre is within the distance limit.
    If both region arrays are supplied, only equal nonnegative labels may be
    connected; a negative label excludes a support from the transfer.
    """
    points = np.asarray(coarse_centres, dtype=float).reshape(-1, 2)
    state = np.asarray(coarse_parameters, dtype=float).reshape(-1, 6)
    accepted = np.asarray(coarse_accepted, dtype=bool).reshape(-1)
    fine = np.asarray(fine_centres, dtype=float).reshape(-1, 2)
    if len(points) != np.prod(coarse_shape) or len(state) != len(points) or len(accepted) != len(points):
        raise ValueError("incompatible coarse grid")
    if max_distance_px <= 0:
        raise ValueError("maximum distance must be positive")
    if (coarse_region_labels is None) != (fine_region_labels is None):
        raise ValueError("both region label arrays must be supplied")
    coarse_regions = np.zeros(len(points), dtype=int) if coarse_region_labels is None else np.asarray(coarse_region_labels).reshape(-1)
    fine_regions = np.zeros(len(fine), dtype=int) if fine_region_labels is None else np.asarray(fine_region_labels).reshape(-1)
    if (len(coarse_regions) != len(points) or len(fine_regions) != len(fine)
            or not np.issubdtype(coarse_regions.dtype, np.integer)
            or not np.issubdtype(fine_regions.dtype, np.integer)):
        raise ValueError("region labels must be integer arrays matching their grids")
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    if len(xs) != coarse_shape[1] or len(ys) != coarse_shape[0]:
        raise ValueError("coarse centres must form a complete grid")
    lookup = {(int(np.searchsorted(xs, x)), int(np.searchsorted(ys, y))): i
              for i, (x, y) in enumerate(points)}
    dx = np.min(np.diff(xs)) if len(xs) > 1 else max_distance_px
    dy = np.min(np.diff(ys)) if len(ys) > 1 else max_distance_px
    rx, ry = int(np.ceil(max_distance_px / dx)), int(np.ceil(max_distance_px / dy))
    out = np.full((len(fine), 6), np.nan, dtype=float)
    sources = np.full(len(fine), -1, dtype=np.int32)
    for j, (x, y) in enumerate(fine):
        if fine_regions[j] < 0:
            continue
        cx, cy = int(np.searchsorted(xs, x)), int(np.searchsorted(ys, y))
        candidates = [lookup[(ix, iy)] for iy in range(max(0, cy - ry), min(len(ys), cy + ry + 1))
                      for ix in range(max(0, cx - rx), min(len(xs), cx + rx + 1))
                      if accepted[lookup[(ix, iy)]] and coarse_regions[lookup[(ix, iy)]] == fine_regions[j]
                      and np.isfinite(state[lookup[(ix, iy)]]).all()]
        if not candidates:
            continue
        distances = np.linalg.norm(points[candidates] - fine[j], axis=1)
        best = int(np.argmin(distances))
        if distances[best] <= max_distance_px:
            source = candidates[best]
            sources[j] = source
            out[j] = transport(state[source], fine[j] - points[source])
    return {"parameters": out, "available": sources >= 0, "source": sources}


def propagate(centres, shape, valid, parameters, seeds, solve, *, tau_u=8.0,
              tau_g=0.25, consistency=True, affine=True):
    """Resolve fronts using up to two neighbours per pass and immutable snapshots.

    ``solve(indices, candidates)`` returns six-parameter states and status flags.
    A solver-accepted state becomes an anchor only after the consistency check.
    No analytical visibility or truth is used here.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    state = np.asarray(parameters, dtype=float).reshape(-1, 6).copy()
    accepted = np.asarray(seeds, dtype=bool).reshape(-1).copy()
    if len(centres) != np.prod(shape) or len(valid) != len(centres) or np.any(accepted & ~valid):
        raise ValueError("incompatible grid, validity mask, or seeds")
    parent = np.full(len(valid), -1, dtype=np.int32)
    generation = np.full(len(valid), -1, dtype=np.int32)
    origin = np.full(len(valid), -1, dtype=np.int32)
    generation[accepted] = 0
    origin[accepted] = np.flatnonzero(accepted)
    attempts = np.zeros(len(valid), dtype=np.int32)
    reason = np.zeros(len(valid), dtype=np.uint8)  # 1 solver, 2 displacement, 3 gradient
    calls = active_count = passes = 0
    while True:
        snapshot = accepted.copy()
        options = {j: [i for i in neighbours(shape, j) if snapshot[i]][:2]
                   for j in np.flatnonzero(valid & ~snapshot)}
        options = {j: sources for j, sources in options.items() if sources}
        if not options:
            break
        passes += 1
        added = np.zeros(len(valid), dtype=bool)
        for rank in range(2):
            indices = np.array([j for j, sources in options.items()
                                if len(sources) > rank and not added[j]], dtype=int)
            if not len(indices):
                continue
            sources = np.array([options[j][rank] for j in indices], dtype=int)
            candidates = np.array([transport(state[i], centres[j] - centres[i], affine)
                                   for j, i in zip(indices, sources, strict=True)])
            solved, status = solve(indices, candidates.copy())
            solved = np.asarray(solved).reshape(-1, 6)
            status = np.asarray(status, dtype=bool).reshape(-1)
            if len(status) != len(indices) or len(solved) != len(indices):
                raise ValueError("solver returned an incompatible result")
            calls += 1
            active_count += len(indices)
            attempts[indices] += 1
            du = np.linalg.norm(solved[:, [0, 3]] - candidates[:, [0, 3]], axis=1)
            dg = np.linalg.norm(solved[:, [1, 2, 4, 5]] - candidates[:, [1, 2, 4, 5]], axis=1)
            ok = status & np.isfinite(solved).all(axis=1)
            reason[indices] = np.where(~ok, 1, np.where(du > tau_u, 2, np.where(dg > tau_g, 3, 0)))
            if consistency:
                ok &= (du <= tau_u) & (dg <= tau_g)
            chosen, parents = indices[ok], sources[ok]
            state[chosen] = solved[ok]
            accepted[chosen] = True
            added[chosen] = True
            parent[chosen] = parents
            generation[chosen] = passes
            origin[chosen] = origin[parents]
            reason[chosen] = 0
        if not added.any():
            break
    return {"parameters": state, "accepted": accepted, "parent": parent,
            "origin": origin, "generation": generation, "attempts": attempts,
            "rejection_reason": reason, "passes": passes, "solver_calls": calls,
            "active_windows": active_count}
