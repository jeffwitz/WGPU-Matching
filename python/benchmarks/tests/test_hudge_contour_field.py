import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hudge_contour_field import ContourField
from hudge_dense_field import LocalField


def make_field():
    reference = np.zeros((64, 64), bool)
    target = reference.copy()
    reference[10:40, 10:40] = True
    target[14:44, 17:47] = True
    yy, xx = np.mgrid[16:36:8, 16:36:8]
    centres = np.column_stack((xx.ravel(), yy.ravel()))
    p = np.tile([7., 0., 0., 4., 0., 0.], (len(centres), 1))
    return ContourField(LocalField(centres, p), reference, target, centres, max_sites=16)


def test_exact_translation_and_pixel_edge_convention():
    field = make_field()
    np.testing.assert_allclose(field.residual()[0], 0., atol=1e-12)
    np.testing.assert_allclose(field.step(), 0., atol=1e-12)


def test_correction_derivatives_are_derivatives_of_displacement():
    field = make_field()
    field.coefficients = np.random.default_rng(41).normal(0, .2, field.coefficients.shape)
    q = np.array([[20.3, 24.1], [33.1, 39.7]])
    states = field.evaluate(q)
    for axis, slots in ((0, [1, 4]), (1, [2, 5])):
        step = np.eye(2)[axis]*1e-4
        finite = (field.evaluate(q+step)[:, [0, 3]]-field.evaluate(q-step)[:, [0, 3]])/2e-4
        np.testing.assert_allclose(states[:, slots], finite, atol=1e-9)
