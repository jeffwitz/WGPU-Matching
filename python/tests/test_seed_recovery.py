import numpy as np
from wgpu_matching.seed_recovery import retry_small_subsets


def test_retries_only_unresolved_and_preserves_coarse_sources():
    centres = np.array([(x * 64.0, y * 64.0) for y in range(3) for x in range(3)])
    states = np.zeros((9, 6))
    states[4, 0] = 3.0
    coarse_accepted = np.zeros(9, bool)
    coarse_accepted[4] = True
    calls = []

    def solve(size, indices, initial):
        calls.append((size, indices.copy()))
        status = indices % 2 == (0 if size == 32 else 1)
        return initial, status

    result = retry_small_subsets(centres, (3, 3), np.ones(9, bool), states, coarse_accepted, solve)
    assert result["accepted"].all()
    assert result["stage"][4] == 0
    np.testing.assert_array_equal(result["parameters"][4], states[4])
    assert all(4 not in indices for _, indices in calls)
    assert [size for size, _ in calls] == [32, 16]
    assert np.all(result["source"][result["stage"] > 0] == 4)


def test_inconsistent_solutions_are_rejected_and_suspects_cannot_seed():
    centres = np.array([(x * 64.0, y * 64.0) for y in range(2) for x in range(2)])
    states = np.zeros((4, 6))
    accepted = np.array([True, False, False, True])
    suspect = np.array([True, False, False, False])

    def solve(size, indices, initial):
        del size
        initial[:, 0] += 9
        return initial, np.ones(len(indices), bool)

    result = retry_small_subsets(
        centres, (2, 2), np.ones(4, bool), states, accepted, solve, suspect=suspect, tau_u=8.0
    )
    np.testing.assert_array_equal(result["accepted"], [False, False, False, True])
    assert np.all(result["rejection_reason"][:3] == 2)
    assert np.all(result["attempts"][:3] == 2)
