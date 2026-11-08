import numpy as np
from benchmark_affine_candidate_patch_sensitivity import (
    affine_target_samples,
    perturb_jacobian,
    sensitivity_conditions,
)
from benchmark_oracle_affine_descriptor_retrieval import (
    canonical_points,
    ranking_metrics,
    warp_target_to_material_frame,
)


def test_target_point_is_mapped_to_its_source_frame_coordinate():
    jacobian = np.asarray([[1.2, 0.15], [-0.08, 0.9]], dtype=np.float64)
    source_anchor = np.asarray([30.0, 42.0])
    target_anchor = np.asarray([110.0, 76.0])
    source_point = source_anchor + np.asarray([7.0, -5.0])
    target_point = target_anchor + jacobian @ (source_point - source_anchor)

    mapped = canonical_points(
        target_point[None], source_anchor, target_anchor, jacobian
    )

    np.testing.assert_allclose(mapped[0], source_point, atol=1e-6)


def test_affine_pullback_uses_forward_map_and_transforms_validity():
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width]
    target = (xx + 2 * yy).astype(np.float32)
    valid = np.ones((height, width), dtype=bool)
    jacobian = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    anchor = np.asarray([width / 2, height / 2])
    target_anchor = anchor + np.asarray([3.0, -2.0])

    canonical, canonical_valid = warp_target_to_material_frame(
        target, valid, anchor, target_anchor, jacobian
    )

    np.testing.assert_allclose(canonical[40, 50], target[38, 53], atol=1e-5)
    assert canonical_valid[40, 50]
    assert not canonical_valid[1, 1]


def test_retrieval_scores_use_positive_rank_and_visible_query_denominators():
    source = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    target = np.asarray([[50.0, 50.0], [10.5, 0.0], [20.0, 0.0]])
    similarities = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.9, 0.2], [0.1, 0.2, 0.9]])

    result = ranking_metrics(similarities, source, target)

    assert result["queries"] == 3
    assert result["repeatable_le5px"] == 2
    assert result["top1_correct_le2px"] == 2
    assert result["top5_retrieval"] == 2


def test_jacobian_perturbations_are_one_factor_at_a_time():
    base = np.asarray([[1.2, 0.1], [-0.2, 0.9]])
    angle = np.deg2rad(5.0)
    expected_rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    ) @ base
    np.testing.assert_allclose(
        perturb_jacobian(base, "angle_deg", 5.0), expected_rotation
    )
    np.testing.assert_allclose(
        perturb_jacobian(base, "stretch_x_percent", 10.0),
        np.diag([1.1, 1.0]) @ base,
    )
    np.testing.assert_allclose(
        perturb_jacobian(base, "stretch_y_percent", -10.0),
        np.diag([1.0, 0.9]) @ base,
    )
    np.testing.assert_allclose(
        perturb_jacobian(base, "shear", 0.1),
        np.asarray([[1.0, 0.1], [0.0, 1.0]]) @ base,
    )


def test_sensitivity_conditions_include_exact_zero_and_independent_axes():
    conditions = sensitivity_conditions()
    assert conditions[0] == {"parameter": "exact", "delta": 0.0}
    assert {str(row["parameter"]) for row in conditions[1:]} == {
        "angle_deg",
        "stretch_x_percent",
        "stretch_y_percent",
        "shear",
    }
    assert all(float(row["delta"]) != 0.0 for row in conditions[1:])


def test_candidate_centered_affine_patch_sampling_needs_no_translation_input():
    height, width = 96, 128
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    source = (0.002 * xx + 0.003 * yy + 0.00001 * xx * yy).astype(np.float32)
    jacobian = np.asarray([[1.1, 0.1], [-0.05, 0.95]])
    translation = np.asarray([7.0, -4.0])
    center = np.asarray([width / 2, height / 2])
    target_x = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    source_xy = center + (target_x - center - translation) @ np.linalg.inv(jacobian).T
    x0 = np.floor(source_xy[:, 0]).astype(int)
    y0 = np.floor(source_xy[:, 1]).astype(int)
    wx = source_xy[:, 0] - x0
    wy = source_xy[:, 1] - y0
    inside = (
        (x0 >= 0)
        & (y0 >= 0)
        & (x0 + 1 < width)
        & (y0 + 1 < height)
    )
    target = np.full(len(source_xy), 0.5, dtype=np.float32)
    ids = np.flatnonzero(inside)
    target[ids] = (
        source[y0[ids], x0[ids]] * (1 - wx[ids]) * (1 - wy[ids])
        + source[y0[ids], x0[ids] + 1] * wx[ids] * (1 - wy[ids])
        + source[y0[ids] + 1, x0[ids]] * (1 - wx[ids]) * wy[ids]
        + source[y0[ids] + 1, x0[ids] + 1] * wx[ids] * wy[ids]
    )
    target = target.reshape(height, width)
    source_center = np.asarray([[50.0, 45.0]])
    target_center = center + (source_center[0] - center) @ jacobian.T + translation
    target_patch, valid = affine_target_samples(
        target,
        target_center[None],
        jacobian,
        side=5,
        offset_start=-2.0,
        spacing=1.0,
    )
    expected_xy = source_center[0] + np.stack(
        np.meshgrid(np.arange(-2, 3), np.arange(-2, 3)), axis=-1
    ).reshape(-1, 2)
    expected = source[
        np.rint(expected_xy[:, 1]).astype(int),
        np.rint(expected_xy[:, 0]).astype(int),
    ]

    assert valid[0]
    np.testing.assert_allclose(target_patch[0], expected, atol=2e-5)
