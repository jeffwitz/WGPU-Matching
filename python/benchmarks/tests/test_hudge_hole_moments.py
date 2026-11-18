"""Analytical and image-only regression tests for hole-derived hypotheses."""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hudge_hole_moments import (
    Hole,
    affine_parameters,
    holes_from_mask,
    match_holes,
    moment_affine,
    quadratic_mapping,
    rank_sector,
    sample,
    score_affine,
    sector_points,
    spd_power,
)


def test_covariance_family_centroids_and_ellipse_ambiguity():
    a = Hole(1, np.array([20., 30.]), np.array([[8., 2.], [2., 3.]]), 100, 0.)
    b = Hole(2, np.array([78., 91.]), np.array([[13., -4.], [-4., 22.]]), 300, 0.)
    for angle in [-95, 0, 35, 180]:
        f, t = moment_affine(a, b, angle)
        np.testing.assert_allclose(f @ a.covariance @ f.T, b.covariance, atol=1e-12)
        np.testing.assert_allclose(f @ a.centre + t, b.centre, atol=1e-12)
        assert np.linalg.det(f) > 0
        symmetry, shift = moment_affine(a, a, angle)
        np.testing.assert_allclose(symmetry @ a.covariance @ symmetry.T, a.covariance, atol=1e-12)
        np.testing.assert_allclose(symmetry @ a.centre + shift, a.centre, atol=1e-12)


def test_affine_parameters_at_subpixel_centres():
    f = np.array([[1.2, -.7], [.8, 3.2]])
    t = np.array([41., -7.])
    centres = np.array([[5.5, 13.5], [25.25, 11.75]])
    p = affine_parameters(f, t, centres)
    np.testing.assert_allclose(centres + p[:, [0, 3]], centres @ f.T + t)
    np.testing.assert_allclose(p[:, [1, 2, 4, 5]], np.tile((f - np.eye(2)).ravel(), (2, 1)))


def test_holes_exclude_crop_and_assignment_is_not_component_order():
    y, x = np.indices((160, 200))
    mask = (x > 4) & (x < 195) & (y > 4) & (y < 155)
    for cx, cy, radius in [(40, 40, 12), (140, 100, 18)]:
        mask &= (x-cx)**2 + (y-cy)**2 > radius**2
    mask[0:20, 70:85] = False  # Exterior-connected notch is not a hole.
    holes = holes_from_mask(mask)
    assert len(holes) == 2
    targets = list(reversed(holes))
    pairs, _ = match_holes(holes, targets, mask, mask)
    assert [(a, b) for a, b, _, _ in pairs] == [(0, 1), (1, 0)]


def test_ambiguous_holes_are_rejected_and_sectors_keep_material_only():
    y, x = np.indices((100, 100))
    mask = (x-50)**2 + (y-50)**2 > 15**2
    h = Hole(1, np.array([50., 50.]), np.eye(2)*15**2/4, int(np.pi*15**2), 0.)
    pairs, _ = match_holes([h], [h, h], mask, mask)
    assert not pairs
    mask[40:60, 70:80] = False
    points = sector_points(mask, h, (1.02, 2.), 0, 50)
    assert len(points) > 32
    assert mask[points[:, 1].astype(int), points[:, 0].astype(int)].all()


def test_rotation_retrieval_uses_candidate_texture_without_true_angle():
    rng = np.random.default_rng(81232)
    reference = gaussian_filter(rng.normal(size=(256, 256)), 1.2)
    centre = np.array([128., 128.])
    source = Hole(1, centre, np.eye(2)*100, 1257, 0.)
    f = np.array([[np.cos(.61), -np.sin(.61)], [np.sin(.61), np.cos(.61)]])
    target_hole = Hole(2, centre, f @ source.covariance @ f.T, 1257, 0.)
    y, x = np.indices(reference.shape)
    points = np.column_stack((x.ravel(), y.ravel()))
    target = sample(reference, (points-centre) @ f + centre).reshape(reference.shape)
    mask = (x-128)**2 + (y-128)**2 > 400
    sector = sector_points(mask, source, (1.1, 2.), 0, aperture=60)
    ranked = rank_sector(reference, target, np.ones_like(target), sector, source, target_hole)
    best = ranked[np.argmax(ranked[:, 1])]
    assert abs(best[0] - np.rad2deg(.61)) < 2.5
    assert best[1] > .95


def test_invalid_target_cannot_win_by_reducing_sample_count():
    points = np.column_stack((np.arange(10, 50), np.full(40, 20)))
    target = np.indices((64, 64))[1].astype(float)
    mask = np.ones_like(target)
    mask[:, 30:] = 0
    score, fraction = score_affine(points[:, 0], points, target, mask, np.eye(2), np.zeros(2))
    assert score == -1 and fraction == .5


def test_spd_rejects_degenerate_covariance():
    import pytest
    with pytest.raises(ValueError, match='positive definite'):
        spd_power(np.diag([1., 0.]), -.5)


def test_quadratic_spatial_derivatives_and_exact_local_transport():
    points = np.array([[10.25, 31.75], [23., 47.5]])
    origin = np.array([5., 6.])
    f = np.array([[1.2, -.2], [.4, 2.]])
    t = np.array([30., 100.])
    h = np.array([[.02, -.03, .04], [.01, .02, -.01]])
    q, jac = quadratic_mapping(points, origin, f, t, h)
    for dimension in range(2):
        step = np.eye(2)[dimension] * 1e-5
        upper, _ = quadratic_mapping(points+step, origin, f, t, h)
        lower, _ = quadratic_mapping(points-step, origin, f, t, h)
        np.testing.assert_allclose((upper-lower)/2e-5, jac[:, :, dimension], atol=1e-8)
    local_t = q[0] - jac[0] @ points[0]
    transported, transported_jac = quadratic_mapping(points, points[0], jac[0], local_t, h)
    np.testing.assert_allclose(transported, q, atol=1e-12)
    np.testing.assert_allclose(transported_jac, jac, atol=1e-12)
