"""Coordinate and missing-truth checks for external image benchmarks."""

import numpy as np
from external_dic_input import affine_truth


def test_external_truth_is_optional_and_uses_centered_stretch():
    points = np.array([[50., 40.], [10., 20.]])
    assert affine_truth(points, (80, 100), None, None) == (None, None)
    field, visible = affine_truth(points, (80, 100), 1.5, .8)
    np.testing.assert_allclose(field[0], [0, .5, 0, 0, 0, -.2], atol=1e-6)
    np.testing.assert_allclose(field[1, [0, 3]], [-20, 4], atol=1e-6)
    np.testing.assert_array_equal(visible, [True, False])
