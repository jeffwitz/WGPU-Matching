from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qualify_ndf_pair_icgn import _frontier_propagation


class _FrontierSolver:
    roi = None

    def __init__(self, accept_active: bool):
        self.accept_active = accept_active
        self.active_calls: list[np.ndarray] = []

    def _register_seeded_frame(self, _target, seeds, active):
        active_flat = np.asarray(active).reshape(-1).astype(bool)
        self.active_calls.append(active_flat.copy())
        status = active_flat if self.accept_active else np.zeros_like(active_flat)
        return SimpleNamespace(parameters=np.asarray(seeds).reshape(-1, 8)[:, :6], status=status)


def _grid_centres() -> np.ndarray:
    x = np.array([31.5, 95.5, 159.5], dtype=np.float32)
    y = np.array([31.5, 95.5, 159.5], dtype=np.float32)
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)


def test_frontier_prediction_extrapolates_from_neighbor_to_candidate_and_preserves_gradient():
    solver = _FrontierSolver(accept_active=True)
    centres = _grid_centres()
    state = np.zeros((9, 6), dtype=np.float32)
    state[4] = (10.0, 0.1, 0.02, -3.0, 0.03, -0.1)
    status = np.zeros(9, dtype=bool)
    status[4] = True

    result = _frontier_propagation(solver, np.zeros((1, 1), np.uint8), centres, state, status)

    assert len(solver.active_calls) == 1
    assert solver.active_calls[0].sum() == 8
    np.testing.assert_allclose(result["parameters"][5], (16.4, 0.1, 0.02, -1.08, 0.03, -0.1), atol=1e-5)
    assert result["status"].all()
    assert result["rings"] == [{"ring": 1, "frontier_attempts": 8, "newly_accepted": 8}]


def test_frontier_attempts_each_failed_window_at_most_three_times():
    solver = _FrontierSolver(accept_active=False)
    centres = _grid_centres()
    state = np.zeros((9, 6), dtype=np.float32)
    status = np.zeros(9, dtype=bool)
    status[4] = True

    result = _frontier_propagation(solver, np.zeros((1, 1), np.uint8), centres, state, status)

    assert len(solver.active_calls) == 3
    assert all(active.sum() == 8 for active in solver.active_calls)
    assert result["attempts"][4] == 0
    assert np.all(result["attempts"][[0, 1, 2, 3, 5, 6, 7, 8]] == 3)
    assert np.array_equal(result["status"], status)
