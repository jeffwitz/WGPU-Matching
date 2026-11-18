import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hudge_rbf_restart import SeedField
from qualify_hudge_adaptive_support import quadratic_states, sample_patch


def test_quadratic_parameter_order_and_sampling():
    y, x = np.mgrid[:128, :128].astype(float)
    image = 2*x+3*y
    p = np.array([2., .1, .2, -3., -.1, .15, .001, .002, -.001, -.002, .001, .003])
    patch, bounds, determinant = sample_patch(image, [63.5, 63.5], p, 32)
    yy, xx = np.mgrid[:32, :32].astype(float)-15.5
    u = p[0]+p[1]*xx+p[2]*yy+.5*p[6]*xx**2+p[7]*xx*yy+.5*p[8]*yy**2
    v = p[3]+p[4]*xx+p[5]*yy+.5*p[9]*xx**2+p[10]*xx*yy+.5*p[11]*yy**2
    np.testing.assert_allclose(patch, 2*(63.5+xx+u)+3*(63.5+yy+v), atol=1e-10)
    assert bounds and determinant > 0


def test_rbf_quadratic_is_taylor_expansion():
    field = SeedField(np.eye(2), np.zeros(2), np.array([[0., 0.]]), np.array([[10., -5.]]), 64.)
    c = np.array([[33., 27.]])
    p = quadratic_states(field, c)[0]
    h = field.hessian(c)[0]
    np.testing.assert_allclose(p[6:], [h[0, 0, 0], h[0, 0, 1], h[0, 1, 1],
                                     h[1, 0, 0], h[1, 0, 1], h[1, 1, 1]])
    assert np.all(quadratic_states(field, c, False)[0, 6:] == 0)


def test_fine_grid_keeps_exact_coarse_centre_and_support():
    coarse = np.array([143.5, 175.5])
    y, x = np.mgrid[:127, :127]
    fine = np.column_stack((x.ravel()*8+7.5, y.ravel()*8+7.5))
    selected = fine[np.max(np.abs(fine-coarse), axis=1) <= 8]
    assert len(selected) == 9
    assert np.any(np.all(selected == coarse, axis=1))
    assert np.all(np.abs(selected-coarse)+7.5 <= 15.5)
