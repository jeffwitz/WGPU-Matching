"""Tests for image-only hypothesis selection and anchor verification."""

import numpy as np
from wgpu_matching.seed_initialization import initialize, initialize_with_escalation


def test_selects_status_count_and_propagates_confirmed_seeds():
    centres = np.array([(x, y) for y in range(4) for x in range(4)], float)
    wrong = np.zeros((16, 6))
    wrong[:, 0] = 100
    right = np.zeros((16, 6))
    right[:, 0] = 5

    def solve(indices, states):
        return states, np.abs(states[:, 0] - 5) < 1

    result = initialize(centres, (4, 4), np.ones(16, bool),
                        [("wrong", wrong), ("right", right)], solve, sparse_stride=2)
    assert result["selected_hypothesis"] == "right"
    assert result["hypothesis_acceptances"][0]["accepted"] == 0
    assert result["accepted"].all()
    assert result["minimum_anchors_met"]


def test_isolated_seed_that_cannot_grow_is_rejected():
    centres = np.array([(x, y) for y in range(3) for x in range(3)], float)

    def solve(indices, states):
        return states, indices == 0

    result = initialize(centres, (3, 3), np.ones(9, bool),
                        [("single", np.zeros((9, 6)))], solve, sparse_stride=2)
    assert not result["accepted"].any()


def test_invalid_hypothesis_locations_are_never_probed():
    centres = np.array([(x, y) for y in range(3) for x in range(3)], float)
    calls = []

    def solve(indices, states):
        calls.append(indices.copy())
        return states, np.zeros(len(indices), dtype=bool)

    mask = np.zeros(9, bool)
    mask[0] = True
    initialize(centres, (3, 3), np.ones(9, bool), [("local", np.zeros((9, 6)), mask)],
               solve, sparse_stride=1)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0])


def test_escalation_selects_a_separate_motion_for_unresolved_region():
    centres = np.array([(x, y) for y in range(4) for x in range(9)], float)
    left, right = np.zeros((36, 6)), np.zeros((36, 6))
    left[:, 0], right[:, 0] = 1, 2

    def solve(indices, states):
        expected = np.where(centres[indices, 0] < 4, 1, 2)
        return states, np.isclose(states[:, 0], expected)

    valid = centres[:, 0] != 4
    out = initialize_with_escalation(centres, (4, 9), valid,
                                     [("left", left), ("right", right)], solve,
                                     min_region=4, sparse_stride=2)
    np.testing.assert_array_equal(out["accepted"], valid)
    assert (out["source_stage"][:4] == 0).all()
    assert (out["source_stage"][5:9] > 0).all()
    np.testing.assert_array_equal(out["parameters"][valid, 0], np.where(centres[valid, 0] < 4, 1, 2))


def test_resumed_escalation_preserves_accepted_states_and_solves_only_unresolved():
    centres = np.array([(x, y) for y in range(4) for x in range(9)], float)
    valid = centres[:, 0] != 4
    left, right = np.zeros((36, 6)), np.zeros((36, 6))
    left[:, 0], right[:, 0] = 1, 2
    calls = []

    def solve(indices, states):
        calls.extend(indices.tolist())
        expected = np.where(centres[indices, 0] < 4, 1, 2)
        return states, np.isclose(states[:, 0], expected)

    initial = initialize(centres, (4, 9), valid, [("left", left)], solve, sparse_stride=2)
    previously_accepted = initial["accepted"].copy()
    old_parameters = initial["parameters"].copy()
    calls.clear()
    out = initialize_with_escalation(
        centres, (4, 9), valid, [("left", left), ("right", right)], solve,
        min_region=4, sparse_stride=2, initial_result=initial)
    assert out["accepted"][valid].all()
    np.testing.assert_array_equal(out["parameters"][previously_accepted], old_parameters[previously_accepted])
    assert not previously_accepted[np.asarray(calls, dtype=int)].any()
    assert (out["source_stage"][previously_accepted] == 0).all()
    assert (out["source_hypothesis"][out["source_stage"] > 0] == 1).all()
    assert np.array_equal(initial["accepted"], previously_accepted)


def test_optional_unresolved_escalation_continues_within_a_seeded_region():
    centres = np.array([(x, y) for y in range(4) for x in range(8)], float)
    valid = np.ones(32, bool)
    first, second = np.zeros((32, 6)), np.zeros((32, 6))
    first[:, 0], second[:, 0] = 1, 2

    def solve(indices, states):
        expected = np.where(centres[indices, 0] < 4, 1, 2)
        return states, np.isclose(states[:, 0], expected)

    initial = initialize(centres, (4, 8), valid, [("first", first)], solve, sparse_stride=2)
    assert initial["minimum_anchors_met"] and not initial["accepted"].all()
    out = initialize_with_escalation(
        centres, (4, 8), valid, [("first", first), ("second", second)], solve,
        min_region=4, sparse_stride=2, initial_result=initial, fill_unresolved=True)
    assert out["accepted"].all()
    np.testing.assert_array_equal(out["parameters"][:, 0], np.where(centres[:, 0] < 4, 1, 2))


def test_batched_confirmation_replays_serial_decisions_and_unique_indices():
    shape = (9, 11)
    centres = np.array([(x, y) for y in range(shape[0]) for x in range(shape[1])], float)
    valid = np.ones(len(centres), bool)
    valid[[14, 27, 43]] = False
    initial = np.zeros((len(centres), 6))
    initial[:, 0] = .1 * centres[:, 0]
    initial[:, 1] = .1
    calls = []

    def solve(indices, states):
        assert len(indices) == len(np.unique(indices))
        calls.append(indices.copy())
        states[:, 3] = (indices % 7) * .01
        return states, indices % 13 != 0

    serial = initialize(centres, shape, valid, [("candidate", initial)], solve, sparse_stride=2)
    calls.clear()
    batched = initialize(centres, shape, valid, [("candidate", initial)], solve, sparse_stride=2,
                         batch_confirmation=True)
    for key in ("parameters", "accepted", "parent", "origin", "generation", "confirmed_seeds"):
        np.testing.assert_array_equal(serial[key], batched[key])
    assert len(calls) >= 8


def test_batched_confirmation_rejects_isolated_seed():
    centres = np.array([(x, y) for y in range(3) for x in range(3)], float)

    def solve(indices, states):
        return states, indices == 0

    result = initialize(centres, (3, 3), np.ones(9, bool), [("single", np.zeros((9, 6)))], solve,
                        sparse_stride=2, batch_confirmation=True)
    assert not result["accepted"].any()


def test_few_candidate_anchors_do_not_trigger_extra_batches():
    centres = np.array([(x, y) for y in range(4) for x in range(4)], float)
    calls = []

    def solve(indices, states):
        calls.append(indices.copy())
        return states, np.ones(len(indices), bool)

    serial = initialize(centres, (4, 4), np.ones(16, bool), [("single", np.zeros((16, 6)))],
                        solve, sparse_stride=2)
    serial_calls = len(calls)
    calls.clear()
    batched = initialize(centres, (4, 4), np.ones(16, bool), [("single", np.zeros((16, 6)))],
                         solve, sparse_stride=2, batch_confirmation=True)
    assert len(calls) == serial_calls
    np.testing.assert_array_equal(serial["parameters"], batched["parameters"])
