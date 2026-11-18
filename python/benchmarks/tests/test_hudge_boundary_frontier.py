import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualify_hudge_boundary_frontier import transport


def test_transport_keeps_exact_centres_and_large_affine_gradients():
    matrix = np.array([[.2, -1.7], [2.4, .8]])
    gradient = matrix-np.eye(2)
    translation = np.array([110., -87.])
    a, b = np.array([7.5, 31.5]), np.array([11.5, 27.5])
    u = gradient@a+translation
    seed = np.array([u[0], *gradient[0], u[1], *gradient[1]])
    saved = seed.copy()
    moved = transport(seed, b-a)
    np.testing.assert_allclose(moved[[0, 3]], gradient@b+translation)
    np.testing.assert_array_equal(moved[[1, 2, 4, 5]], gradient.ravel())
    np.testing.assert_array_equal(seed, saved)


def test_fill_only_unobserved_allowed_finite_pixels():
    from complete_hudge_outer_edge import fill_missing
    old = np.ones((3, 4, 2))
    covered = np.zeros((3, 4), bool)
    covered[0, 0] = True
    allowed = np.ones((3, 4), bool)
    allowed[1, 1] = False
    candidate = np.full((3, 4, 2), 5.)
    candidate[2, 2] = np.nan
    result, valid, added = fill_missing(old, covered, candidate, np.ones_like(covered), allowed)
    np.testing.assert_array_equal(result[0, 0], old[0, 0])
    assert not valid[1, 1] and not valid[2, 2]
    assert added.sum() == 9
    np.testing.assert_array_equal(old, np.ones_like(old))
    assert covered.sum() == 1


def test_common_image_padding_preserves_displacement_transport():
    seed = np.array([7., .4, -.2, 8., .3, 1.8])
    source, target = np.array([5.5, 17.5]), np.array([1.5, 13.5])
    for padding in [0, 32, 34]:
        shifted = transport(seed, (target+padding)-(source+padding))
        np.testing.assert_array_equal(shifted, transport(seed,target-source))
