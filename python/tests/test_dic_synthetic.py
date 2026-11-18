from __future__ import annotations

import numpy as np
import pytest
from wgpu_matching.dic_synthetic import (
    _forward_map,
    _gaussian_blur,
    _motion_parameters,
    analytic_truth_at_points,
    generate_speckle_pair,
    perturb_target,
)


@pytest.mark.parametrize("mode", ["translation", "rigid", "affine", "heterogeneous", "discontinuous", "shear_band"])
def test_analytic_truth_at_half_pixel_centers_matches_forward_map(mode):
    peak = .67 if mode == "shear_band" else None
    pair = generate_speckle_pair(21, 22, 96, 64, motion_mode=mode, shear_band_peak_gradient=peak)
    motion = _motion_parameters(np.random.default_rng(22), 96, 64, mode, peak, .5)
    points = np.array([[20.5, 17.5], [65.5, 40.5], [5.5, 60.5]])
    truth, visible = analytic_truth_at_points(pair.metadata, points)
    x, y = _forward_map(points[:, 0], points[:, 1], motion, 96, 64)
    np.testing.assert_allclose(truth[:, [0, 3]], np.column_stack((x, y)) - points, atol=1e-10)
    eps = 1e-4
    for axis, columns in ((0, [1, 4]), (1, [2, 5])):
        plus, minus = points.copy(), points.copy()
        plus[:, axis] += eps
        minus[:, axis] -= eps
        xp, yp = _forward_map(plus[:, 0], plus[:, 1], motion, 96, 64)
        xm, ym = _forward_map(minus[:, 0], minus[:, 1], motion, 96, 64)
        expected = np.column_stack(((xp - xm) / (2 * eps), (yp - ym) / (2 * eps)))
        expected[:, axis] -= 1
        np.testing.assert_allclose(truth[:, columns], expected, atol=1e-8)
    expected_visible = (x >= 0) & (x <= 95) & (y >= 0) & (y <= 63)
    if mode == "discontinuous":
        expected_visible &= np.abs(points[:, 0] - 48) > max(8, 2 * pair.metadata["characteristic_size_px"])
    np.testing.assert_array_equal(visible, expected_visible)


def test_discontinuity_boundary_can_be_shifted_without_changing_default():
    default = generate_speckle_pair(10, 11, 96, 64, motion_mode="discontinuous")
    shifted = generate_speckle_pair(10, 11, 96, 64, motion_mode="discontinuous",
                                    discontinuity_boundary_fraction=.55)
    assert default.metadata["discontinuity_boundary_px"] == 48
    np.testing.assert_allclose(shifted.metadata["discontinuity_boundary_px"], 52.8)
    assert shifted.metadata["discontinuity_gap_px"] == default.metadata["discontinuity_gap_px"]


def test_target_camera_perturbation_is_reproducible_and_does_not_mutate_input():
    image = np.full((16, 16), .5, dtype=np.float32)
    first = perturb_target(image, 42, gain=1.2, offset=-.05, blur_sigma=1.2, noise_std=.05)
    second = perturb_target(image, 42, gain=1.2, offset=-.05, blur_sigma=1.2, noise_std=.05)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, .5)
    assert not np.array_equal(first, image)


def test_periodic_texture_is_warped_from_same_reference():
    pair = generate_speckle_pair(12, 13, 128, 96, motion_mode="translation",
                                 texture_options={"periodic_strength": .9, "periodic_period_px": 32.})
    assert pair.metadata["periodic_strength"] == .9
    assert pair.metadata["periodic_period_px"] == 32.
    assert np.any(pair.visible)
    assert np.isfinite(pair.target).all()


def _reference_gaussian_blur(image, sigma):
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.asarray(image, dtype=np.float32)
    for axis in (1, 0):
        padded = np.pad(
            result,
            ((0, 0), (radius, radius)) if axis == 1 else ((radius, radius), (0, 0)),
            mode="reflect",
        )
        if axis == 1:
            result = sum(kernel[i] * padded[:, i : i + result.shape[1]] for i in range(len(kernel)))
        else:
            result = sum(kernel[i] * padded[i : i + result.shape[0], :] for i in range(len(kernel)))
    return np.asarray(result, dtype=np.float32)


def test_large_kernel_fft_blur_matches_direct_separable_blur():
    image = np.random.default_rng(812).random((67, 91), dtype=np.float32)
    expected = _reference_gaussian_blur(image, sigma=11.0)
    actual = _gaussian_blur(image, sigma=11.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


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


def test_synthetic_shear_band_has_requested_peak_displacement_gradient() -> None:
    requested = 0.67
    pair = generate_speckle_pair(
        18051003, 18091003, width=384, height=256,
        motion_mode="shear_band", shear_band_peak_gradient=requested,
    )
    du_dy = np.gradient(pair.flow_xy[..., 0], axis=0)

    assert pair.metadata["motion_mode"] == "shear_band"
    assert pair.metadata["shear_band_peak_gradient"] == requested
    assert np.isclose(np.max(np.abs(du_dy)), requested, rtol=0.03)
    assert pair.visible.any()
