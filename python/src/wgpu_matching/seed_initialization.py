"""Image-only sparse hypothesis verification and seed confirmation for DIC."""

import numpy as np

from .seed_propagation import connected_components, neighbours, propagate, transport


def initialize(centres, shape, valid, hypotheses, solve, *, sparse_stride=4,
               tau_u=8.0, tau_g=0.25, minimum_anchors=3, batch_confirmation=False,
               propagator=None, anchor_propagator=None, hypothesis_selector=None):
    """Select a hypothesis by solver status and expand independently verified seeds.

    ``hypotheses`` is a sequence of ``(name, N x 6 initial states)`` ordered by
    image-derived priority. No analytical field or visibility is accepted here.
    The caller supplies a solver callback taking indices and corresponding states.
    Batched confirmation speculatively evaluates neighbour proposals in eight
    non-conflicting directions, then replays the original confirmation order.
    It requires a deterministic, window-independent solver callback.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    count = len(centres)
    if count != np.prod(shape) or len(valid) != count or sparse_stride < 1:
        raise ValueError("invalid grid or sparse stride")
    grid = np.arange(count).reshape(shape)
    sparse = grid[::sparse_stride, ::sparse_stride].ravel()
    sparse = sparse[valid[sparse]]
    resident_buffer = None
    if hypothesis_selector is None:
        trials = []
        for hypothesis in hypotheses:
            if len(hypothesis) == 2:
                name, values = hypothesis
                eligible = sparse
            elif len(hypothesis) == 3:
                name, values, mask = hypothesis
                mask = np.asarray(mask, dtype=bool).reshape(count)
                eligible = sparse[mask[sparse]]
            else:
                raise ValueError("hypothesis must contain a name, states and optional validity mask")
            states = np.asarray(values, dtype=float).reshape(count, 6)
            if len(eligible):
                result, status = solve(eligible, states[eligible].copy())
            else:
                result, status = np.empty((0, 6)), np.empty(0, dtype=bool)
            result = np.asarray(result).reshape(-1, 6)
            status = np.asarray(status, dtype=bool).reshape(-1)
            if len(result) != len(eligible) or len(status) != len(eligible):
                raise ValueError("incompatible solver result")
            status &= np.isfinite(result).all(axis=1)
            trials.append({"name": name, "indices": eligible, "parameters": result, "status": status,
                           "accepted": int(status.sum())})
        if not trials:
            raise ValueError("at least one hypothesis is required")
        selected = max(range(len(trials)), key=lambda i: trials[i]["accepted"])
        chosen = trials[selected]
    else:
        selection = hypothesis_selector(sparse, hypotheses)
        selected = selection["selected_index"]
        trials = selection["trials"]
        resident_buffer = selection.get("selected_buffer")
        chosen = dict(trials[selected])
        if resident_buffer is None:
            chosen.update({key: selection[key] for key in ("indices", "parameters", "status")})
        elif anchor_propagator is None:
            resident_buffer.destroy()
            raise ValueError("resident hypothesis selection requires a GPU anchor propagator")
    if resident_buffer is None:
        candidate_indices = chosen["indices"][chosen["status"]]
        candidate_states = chosen["parameters"][chosen["status"]]
    else:
        candidate_indices = np.empty(0, dtype=int)
        candidate_states = np.empty((0, 6))
    states = np.zeros((count, 6), dtype=float)
    states[candidate_indices] = candidate_states
    if anchor_propagator is not None:
        candidates = np.zeros(count, dtype=bool)
        candidates[candidate_indices] = True
        try:
            if resident_buffer is None:
                output = anchor_propagator(centres, shape, valid, states, candidates, solve, tau_u=tau_u, tau_g=tau_g)
            else:
                output = anchor_propagator(centres, shape, valid, states, candidates, solve, tau_u=tau_u, tau_g=tau_g,
                                           selected_buffer=resident_buffer)
        finally:
            if resident_buffer is not None:
                resident_buffer.destroy()
        confirmed = output["confirmed_seeds"]
    else:
        states, confirmed = _confirm_candidates(centres, shape, valid, states, candidate_indices, solve,
                                                tau_u, tau_g, batch_confirmation)
        propagation = propagate if propagator is None else propagator
        output = propagation(centres, shape, valid, states, confirmed, solve, tau_u=tau_u, tau_g=tau_g)
    regions = connected_components(valid.reshape(shape))
    region_counts = [int(confirmed[region].sum()) for region in regions]
    output.update({"selected_hypothesis": chosen["name"], "selected_index": selected,
                   "hypothesis_acceptances": [{"name": x["name"], "accepted": x["accepted"]}
                                              for x in trials],
                   "sparse_indices": sparse, "confirmed_seeds": confirmed,
                   "region_anchor_counts": region_counts,
                   "minimum_anchors_met": all(n >= minimum_anchors for n in region_counts)})
    return output


def _confirm_candidates(centres, shape, valid, states, candidate_indices, solve, tau_u, tau_g, batch_confirmation):
    count = len(centres)
    # Confirm each isolated sparse anchor by two consistent solved neighbours.
    confirmed = np.zeros(count, dtype=bool)
    verified = np.zeros(count, dtype=bool)
    verified[candidate_indices] = True
    # Eight directional batches cannot reduce the maximum number of calls
    # when at most eight candidate parents need confirmation.
    use_batches = batch_confirmation and len(candidate_indices) > 8
    batched = {}
    if use_batches:
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            pairs = []
            for parent in candidate_indices:
                y, x = divmod(int(parent), shape[1])
                yy, xx = y + dy, x + dx
                if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                    child = yy * shape[1] + xx
                    if valid[child] and not verified[child]:
                        pairs.append((int(parent), child))
            if not pairs:
                continue
            indices = np.array([child for _, child in pairs], dtype=int)
            predictions = np.array([transport(states[parent], centres[child] - centres[parent])
                                    for parent, child in pairs])
            solved, status = solve(indices, predictions.copy())
            solved = np.asarray(solved).reshape(-1, 6)
            status = np.asarray(status, dtype=bool).reshape(-1)
            if len(solved) != len(pairs) or len(status) != len(pairs):
                raise ValueError("incompatible batched solver result")
            for pair, state, ok in zip(pairs, solved, status, strict=True):
                batched[pair] = (state.copy(), bool(ok))
    for parent in candidate_indices:
        for child in neighbours(shape, parent):
            if not verified[child]:
                continue
            prediction = transport(states[parent], centres[child] - centres[parent])
            if (np.linalg.norm(states[child, [0, 3]] - prediction[[0, 3]]) <= tau_u
                    and np.linalg.norm(states[child, [1, 2, 4, 5]]
                                       - prediction[[1, 2, 4, 5]]) <= tau_g):
                confirmed[parent] = True
                confirmed[child] = True
        proposals = [j for j in neighbours(shape, parent) if valid[j] and not verified[j]]
        if not proposals:
            continue
        proposals = np.asarray(proposals, dtype=int)
        predicted = np.array([transport(states[parent], centres[j] - centres[parent])
                              for j in proposals])
        if use_batches:
            records = [batched[(int(parent), int(child))] for child in proposals]
            solved = np.array([record[0] for record in records])
            status = np.array([record[1] for record in records])
        else:
            solved, status = solve(proposals, predicted.copy())
        solved = np.asarray(solved).reshape(-1, 6)
        status = np.asarray(status, dtype=bool).reshape(-1)
        du = np.linalg.norm(solved[:, [0, 3]] - predicted[:, [0, 3]], axis=1)
        dg = np.linalg.norm(solved[:, [1, 2, 4, 5]] - predicted[:, [1, 2, 4, 5]], axis=1)
        good = status & np.isfinite(solved).all(axis=1) & (du <= tau_u) & (dg <= tau_g)
        if good.sum() >= 2:
            confirmed[parent] = True
            accepted = proposals[good]
            states[accepted] = solved[good]
            confirmed[accepted] = True
            verified[accepted] = True
    return states, confirmed


def initialize_with_escalation(centres, shape, valid, hypotheses, solve, *, min_region=16,
                               sparse_stride=4, tau_u=8., tau_g=.25, minimum_anchors=3,
                               initial_result=None, fill_unresolved=False, batch_confirmation=False,
                               propagator=None, anchor_propagator=None, hypothesis_selector=None):
    """Retry unresolved connected regions using image-derived hypotheses.

    Each attempt creates fresh, solver-verified anchors. Already accepted windows
    are immutable; the returned stage and source index record their origin.
    An existing result can be resumed without re-solving its accepted windows.
    Hypotheses must retain their original order when extending a prior list.
    With ``fill_unresolved``, also retry large unresolved components after the
    minimum anchor count has been met. This is an experimental policy.
    """
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if initial_result is None:
        result = initialize(centres, shape, valid, hypotheses, solve, sparse_stride=sparse_stride,
                            tau_u=tau_u, tau_g=tau_g, minimum_anchors=minimum_anchors,
                            batch_confirmation=batch_confirmation, propagator=propagator,
                            anchor_propagator=anchor_propagator, hypothesis_selector=hypothesis_selector)
    else:
        result = {key: value.copy() if isinstance(value, np.ndarray) else value
                  for key, value in initial_result.items()}
        if len(result["accepted"]) != len(valid) or np.any(result["accepted"] & ~valid):
            raise ValueError("initial result is incompatible with the valid grid")
    if "source_stage" not in result:
        result["source_stage"] = np.where(result["accepted"], 0, -1).astype(np.int32)
        result["source_hypothesis"] = np.where(result["accepted"], result["selected_index"], -1).astype(np.int32)
    history = list(result.get("escalation_history", []))
    stage = max(int(result["source_stage"].max(initial=-1)), 0)
    while True:
        roi_regions = connected_components(valid.reshape(shape))
        if not fill_unresolved and all(int(result["confirmed_seeds"][region].sum()) >= minimum_anchors
                                       for region in roi_regions):
            break
        unresolved = valid & ~result["accepted"]
        regions = [r for r in connected_components(unresolved.reshape(shape)) if len(r) >= min_region]
        progress = False
        for region in regions:
            mask = np.zeros(len(valid), dtype=bool)
            mask[region] = True
            attempt = initialize(centres, shape, mask, hypotheses, solve, sparse_stride=sparse_stride,
                                 tau_u=tau_u, tau_g=tau_g, minimum_anchors=minimum_anchors,
                                 batch_confirmation=batch_confirmation, propagator=propagator,
                            anchor_propagator=anchor_propagator, hypothesis_selector=hypothesis_selector)
            new = attempt["accepted"] & ~result["accepted"]
            history.append({"region_windows": len(region), "accepted": int(new.sum()),
                            "hypothesis": attempt["selected_hypothesis"],
                            "anchors": int(attempt["confirmed_seeds"].sum())})
            if not new.any():
                continue
            stage += 1
            progress = True
            for key in ("parameters", "accepted", "parent", "origin", "generation",
                        "attempts", "rejection_reason", "confirmed_seeds"):
                result[key][new] = attempt[key][new]
            result["source_stage"][new] = stage
            result["source_hypothesis"][new] = attempt["selected_index"]
            result["solver_calls"] += attempt["solver_calls"]
            result["active_windows"] += attempt["active_windows"]
            result["passes"] += attempt["passes"]
            result["minimum_anchors_met"] = all(
                int(result["confirmed_seeds"][region].sum()) >= minimum_anchors for region in roi_regions
            )
        if not progress:
            break
    result["escalation_history"] = history
    return result
