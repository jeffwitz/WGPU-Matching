from __future__ import annotations

import numpy as np
from qualify_heterogeneous_icgn import analytic_parameters, grid_centres, seed_state
from wgpu_matching.dic_synthetic import generate_speckle_pair


def test_icgn_grid_centres_use_pixel_centre_and_requested_stride() -> None:
    points = grid_centres(192, 128, subset_size=64, stride=64)
    np.testing.assert_array_equal(points[0], [31.5, 31.5])
    np.testing.assert_array_equal(points[1], [95.5, 31.5])
    np.testing.assert_array_equal(points[3], [31.5, 95.5])


def test_analytic_local_affine_parameters_follow_icgn_component_order() -> None:
    pair = generate_speckle_pair(15, 22, 256, 192, motion_mode="affine")
    points = np.array([[63.5, 63.5], [127.5, 127.5]], dtype=np.float32)
    parameters = analytic_parameters(pair, points)
    matrix = np.asarray(pair.metadata["affine_matrix"])
    np.testing.assert_allclose(
        parameters[:, [1, 2]], np.broadcast_to(matrix[0] - [1.0, 0.0], (2, 2)), atol=1e-6
    )
    np.testing.assert_allclose(
        parameters[:, [4, 5]], np.broadcast_to(matrix[1] - [0.0, 1.0], (2, 2)), atol=2e-6
    )
    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    np.testing.assert_allclose(parameters[:, 0], pair.flow_xy[y, x, 0], atol=1e-5)
    np.testing.assert_allclose(parameters[:, 3], pair.flow_xy[y, x, 1], atol=1e-5)


def test_seed_state_packs_six_affine_values_and_active_status() -> None:
    parameters = np.arange(18, dtype=np.float32).reshape(3, 6)
    active = np.array([1, 0, 1], dtype=np.uint32)
    state = seed_state(parameters, active)
    assert state.shape == (3, 8)
    np.testing.assert_array_equal(state[:, :6], parameters)
    np.testing.assert_array_equal(state[:, 6], active)
    np.testing.assert_array_equal(state[:, 7], 0.0)
