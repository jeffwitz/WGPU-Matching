from __future__ import annotations

import numpy as np
import pytest
from qualify_ndf_pair_icgn import rotation_bank_angles, rotation_refinement_candidates


@pytest.mark.parametrize("steps, expected_count", [(6, 6), (9, 10), (18, 18)])
def test_rotation_bank_always_contains_identity(steps: int, expected_count: int):
    angles = rotation_bank_angles(steps)

    assert len(angles) == expected_count
    assert np.any(np.isclose(angles, 0.0))


def test_rotation_refinement_adds_half_step_neighbors_of_best_peak():
    scores = [
        {"angle_rad": -np.pi, "consensus_support": 10, "ranking_score": 0.1},
        {"angle_rad": -np.pi / 2, "consensus_support": 20, "ranking_score": 0.2},
        {"angle_rad": 0.0, "consensus_support": 80, "ranking_score": 0.8},
        {"angle_rad": np.pi / 2, "consensus_support": 30, "ranking_score": 0.3},
    ]

    candidates = rotation_refinement_candidates(scores, coarse_steps=4, top_k=1)

    assert len(candidates) == 2
    assert sorted(candidates) == pytest.approx([-np.pi / 4, np.pi / 4])


def test_rotation_refinement_wraps_at_pi_and_deduplicates():
    scores = [
        {"angle_rad": -np.pi, "consensus_support": 80, "ranking_score": 0.8},
        {"angle_rad": 0.0, "consensus_support": 10, "ranking_score": 0.1},
    ]

    candidates = rotation_refinement_candidates(scores, coarse_steps=2, top_k=2)

    assert len(candidates) == 2
    assert all(-np.pi <= angle < np.pi for angle in candidates)
    assert sorted(candidates) == pytest.approx([-np.pi / 2, np.pi / 2])


def test_rotation_refinement_is_disabled_for_zero_topk():
    assert rotation_refinement_candidates([], coarse_steps=9, top_k=0) == []
