import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hudge_dense_field import LocalField, ReducedRBF


def test_local_field_reproduces_global_quadratic():
    yy, xx = np.mgrid[:5, :5]*8.
    centres = np.column_stack((xx.ravel(), yy.ravel()))

    def states(q):
        x, y = q.T
        p = np.zeros((len(q), 12))
        p[:, 0] = 7+.2*x+.1*y+.002*x*x+.003*x*y
        p[:, 3] = -4-.1*x+.4*y+.001*y*y
        p[:, 1], p[:, 2] = .2+.004*x+.003*y, .1+.003*x
        p[:, 4], p[:, 5] = -.1, .4+.002*y
        p[:, 6], p[:, 7], p[:, 11] = .004, .003, .002
        return p

    field = LocalField(centres, states(centres))
    q = np.array([[5.1, 7.3], [44.2, 31.7], [-4., 12.]])
    np.testing.assert_allclose(field.evaluate(q), states(q)[:, :6], atol=1e-12)


def test_weight_derivative_and_bounded_chunking():
    rng = np.random.default_rng(18)
    centres = rng.uniform(0, 30, (10, 2))
    parameters = rng.normal(0, .1, (10, 6))
    field = LocalField(centres, parameters)
    q = np.array([[11.3, 15.7], [20.4, 25.2]])
    states = field.evaluate(q)
    for axis, slots in ((0, [1, 4]), (1, [2, 5])):
        step = np.eye(2)[axis]*1e-4
        finite = (field.evaluate(q+step)[:, [0, 3]]-field.evaluate(q-step)[:, [0, 3]])/2e-4
        np.testing.assert_allclose(states[:, slots], finite, atol=1e-9)
    np.testing.assert_allclose(field.evaluate(np.tile(q, (2100, 1))), np.tile(states, (2100, 1)), atol=1e-12)


def test_reduced_rbf_affine_and_derivative():
    rng = np.random.default_rng(143)
    c = rng.uniform(0, 100, (48, 2))
    p = np.tile([7., .2, -.3, -4., .1, .5], (len(c), 1))
    p[:, 0] += .2*c[:, 0]-.3*c[:, 1]
    p[:, 3] += .1*c[:, 0]+.5*c[:, 1]
    field = ReducedRBF(c, p, max_sites=16)
    np.testing.assert_allclose(field.evaluate(c), p, atol=1e-10)
    p[:, 0] += np.sin(c[:, 0]/20)
    field = ReducedRBF(c, p, max_sites=16)
    q = c[:3]
    for axis, slots in ((0, [1, 4]), (1, [2, 5])):
        step = np.eye(2)[axis]*1e-4
        finite = (field.evaluate(q+step)[:, [0, 3]]-field.evaluate(q-step)[:, [0, 3]])/2e-4
        np.testing.assert_allclose(field.evaluate(q)[:, slots], finite, atol=1e-8)
