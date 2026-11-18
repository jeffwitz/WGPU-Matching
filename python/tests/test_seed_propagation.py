"""Tests for affine state transport and snapshot propagation."""

import numpy as np
from wgpu_matching.seed_propagation import (
    connected_components,
    propagate,
    transfer_to_fine_grid,
    transport,
    weighted_neighbour_initialization,
)


def test_affine_transport_exact():
    state = np.array([2, .2, -.1, 3, .3, .4])
    np.testing.assert_allclose(transport(state, [10, 5]), [3.5, .2, -.1, 8, .3, .4], atol=1e-6)


def test_frontier_respects_holes_and_is_deterministic():
    centres = np.array([(x, y) for y in range(3) for x in range(5)], dtype=float)
    valid = centres[:, 0] != 2
    seeds = np.zeros(15, dtype=bool)
    seeds[0] = True
    def solve(indices, candidates):
        return candidates, np.ones(len(indices), dtype=bool)
    args = (centres, (3, 5), valid, np.zeros((15, 6)), seeds, solve)
    a, b = propagate(*args), propagate(*args)
    assert np.array_equal(a["accepted"], valid & (centres[:, 0] < 2))
    np.testing.assert_array_equal(a["parent"], b["parent"])


def test_consistency_stops_jump():
    centres = np.array([[0., 0.], [1., 0.], [2., 0.]])
    def solve(indices, candidates):
        candidates[:, 0] += 9
        return candidates, np.ones(len(indices), dtype=bool)
    out = propagate(centres, (1, 3), np.ones(3, bool), np.zeros((3, 6)),
                    [True, False, False], solve)
    assert out["accepted"].tolist() == [True, False, False]
    assert out["rejection_reason"][1] == 2


def test_connected_components_use_diagonals():
    components = connected_components([[True, False, False], [False, True, False],
                                        [False, False, True]])
    assert len(components) == 1
    assert components[0].tolist() == [0, 4, 8]


def test_weighted_fill_reproduces_affine_at_hole_without_using_fill_as_source():
    centres = np.array([(x, y) for y in range(3) for x in range(4)], dtype=float)
    state = np.array([[2 + .2*x - .1*y, .2, -.1, 3 + .3*x + .4*y, .3, .4]
                      for x, y in centres])
    accepted = np.zeros(12, bool)
    accepted[[0, 2]] = True
    fill = weighted_neighbour_initialization(centres, (3, 4), np.ones(12, bool), state, accepted)
    assert fill["available"][1]
    np.testing.assert_allclose(fill["parameters"][1], state[1], atol=1e-12)
    assert not fill["available"][3]
    assert not accepted[1]


def test_weighted_fill_rejects_conflicting_neighbours():
    centres = np.array([[0., 0.], [1., 0.], [2., 0.]])
    state = np.zeros((3, 6))
    state[2, 0] = 20
    fill = weighted_neighbour_initialization(centres, (1, 3), np.ones(3, bool), state,
                                            [True, False, True], max_disagreement=8)
    assert not fill["available"][1]
    assert fill["disagreement_px"][1] == 20


def test_weighted_fill_never_crosses_caller_supplied_regions():
    centres = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    states = np.zeros((4, 6))
    accepted = np.array([True, False, True, False])
    labels = np.array([0, 1, 0, 1])
    result = weighted_neighbour_initialization(centres, (2, 2), np.ones(4, bool), states,
                                               accepted, region_labels=labels)
    assert not result["available"].any()


def test_coarse_to_fine_transports_affine_and_respects_distance_limit():
    coarse = np.array([[0., 0.], [64., 0.], [0., 64.], [64., 64.]])
    states = np.array([[2 + .2*x - .1*y, .2, -.1, 3 + .3*x + .4*y, .3, .4]
                       for x, y in coarse])
    fine = np.array([[16., 16.], [300., 300.]])
    out = transfer_to_fine_grid(coarse, (2, 2), states, [True, False, False, False], fine)
    assert out["available"].tolist() == [True, False]
    np.testing.assert_allclose(out["parameters"][0], [3.6, .2, -.1, 14.2, .3, .4], atol=1e-12)


def test_coarse_to_fine_region_labels_exclude_other_side_and_crossing_support():
    coarse = np.array([[0., 0.], [64., 0.]])
    fine = np.array([[31., 0.], [32., 0.], [33., 0.]])
    out = transfer_to_fine_grid(coarse, (1, 2), np.zeros((2, 6)), [True, True], fine,
                                coarse_region_labels=[0, 1], fine_region_labels=[0, -1, 1])
    assert out["source"].tolist() == [0, -1, 1]
