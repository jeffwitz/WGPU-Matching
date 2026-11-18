import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hudge_affine_restart import affine_states
from hudge_rbf_restart import SeedField, fit_seed_field


def test_derivatives_native_coordinates():
    field = SeedField(np.array([[1.1, .2], [-.1, 1.4]]), np.array([50., -13.]),
                      np.array([[10., 30.], [200., 60.]]), np.array([[5., -7.], [-4., 8.]]), 64.)
    points = np.array([[50.3, 27.7], [150.1, 55.2]])
    values = field.evaluate(points)
    for axis, slots in ((0, [1, 4]), (1, [2, 5])):
        step = np.eye(2)[axis]*1e-3
        numerical = (field.evaluate(points+step)[:, [0, 3]]
                     - field.evaluate(points-step)[:, [0, 3]]) / 2e-3
        np.testing.assert_allclose(values[:, slots], numerical, atol=1e-8)


def test_affine_reproduction_and_unaccepted_exclusion():
    y, x = np.mgrid[0:5, 0:6]*32.
    points = np.column_stack((x.ravel(), y.ravel()))
    states = affine_states(points, np.array([[1.2, .3], [-.2, 1.8]]), np.array([12., -9.]))
    accepted = np.ones(len(points), bool)
    accepted[-1] = False
    corrupted = states.copy()
    corrupted[-1] = 1e8
    for method in ("affine", "rbf_position", "rbf_gradient"):
        field, _ = fit_seed_field(points, corrupted, accepted, method=method)
        np.testing.assert_allclose(field.evaluate(points), states, atol=1e-8)


def test_gradient_constraints_improve_known_gaussian_between_sites():
    y, x = np.mgrid[0:5, 0:6]*32.
    points = np.column_stack((x.ravel(), y.ravel()))
    truth = SeedField(np.eye(2), np.array([50., 30.]), points[[8, 20]],
                      np.array([[10., 20.], [-15., 10.]]), 64.)
    states = truth.evaluate(points)
    accepted = np.arange(len(points)) % 3 != 0
    position, _ = fit_seed_field(points, states, accepted, method="rbf_position")
    hermite, _ = fit_seed_field(points, states, accepted, method="rbf_gradient")
    probes = points[~accepted] + [.3, -.4]
    slots = [1, 2, 4, 5]
    assert np.linalg.norm((hermite.evaluate(probes)-truth.evaluate(probes))[:, slots]) < np.linalg.norm(
        (position.evaluate(probes)-truth.evaluate(probes))[:, slots])
