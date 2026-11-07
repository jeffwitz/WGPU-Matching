from __future__ import annotations

import numpy as np
from wgpu_matching.dic_synthetic import generate_speckle_pair


def test_synthetic_pair_is_deterministic_and_tracks_one_texture() -> None:
    first = generate_speckle_pair(73, 91, width=96, height=96, motion_mode="translation")
    second = generate_speckle_pair(73, 91, width=96, height=96, motion_mode="translation")

    np.testing.assert_array_equal(first.reference, second.reference)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.flow_xy, second.flow_xy)
    assert first.visible.shape == first.reference.shape
    assert first.flow_xy.shape == (*first.reference.shape, 2)
    assert first.metadata["affine_matrix"] == np.eye(2).tolist()
    visible_flow = first.flow_xy[first.visible]
    assert np.all(np.std(visible_flow, axis=0) < 2e-3)
    assert float(np.std(first.target - first.reference)) > 0.05


def test_synthetic_heterogeneous_pair_has_spatially_varying_analytic_flow() -> None:
    pair = generate_speckle_pair(179, 241, width=128, height=96, motion_mode="heterogeneous")

    assert pair.metadata["motion_mode"] == "heterogeneous"
    assert np.any(np.asarray(pair.metadata["heterogeneity_amplitude_px"]) != 0)
    assert np.isfinite(pair.flow_xy).all()
    assert np.count_nonzero(pair.visible) > 0
    assert np.std(pair.flow_xy[..., 0]) > 1.0


def test_synthetic_discontinuous_pair_has_an_unmatched_gap_and_visible_sides() -> None:
    first = generate_speckle_pair(311, 719, width=192, height=128, motion_mode="discontinuous")
    second = generate_speckle_pair(311, 719, width=192, height=128, motion_mode="discontinuous")

    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.target_visible, second.target_visible)
    assert first.target_visible is not None
    assert first.metadata["motion_mode"] == "discontinuous"
    assert first.metadata["discontinuity_gap_px"] > 0
    assert first.target_visible.shape == first.target.shape

    seam = first.metadata["discontinuity_boundary_px"] + first.metadata["translation_px"][0]
    gap = first.metadata["discontinuity_gap_px"]
    gap_left = max(0, int(np.floor(seam)))
    gap_right = min(first.target.shape[1], int(np.ceil(seam + gap)))
    assert gap_right > gap_left
    assert not np.any(first.target_visible[:, gap_left:gap_right])

    boundary = int(first.metadata["discontinuity_boundary_px"])
    margin = max(8, int(np.ceil(2 * first.metadata["characteristic_size_px"])))
    assert not np.any(first.visible[:, boundary - margin : boundary + margin])
    left_flow = first.flow_xy[:, : boundary - margin, 0][first.visible[:, : boundary - margin]]
    right_flow = first.flow_xy[:, boundary + margin :, 0][first.visible[:, boundary + margin :]]
    assert left_flow.size and right_flow.size
    assert np.isclose(np.median(right_flow) - np.median(left_flow), gap, atol=1e-4)
