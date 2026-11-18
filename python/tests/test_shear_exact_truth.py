import numpy as np
from qualify_shear_exact_truth import exact_shear_truth
from wgpu_matching.dic_synthetic import _forward_map, _motion_parameters


def test_exact_shear_truth_matches_generator_at_half_pixel_centers():
    points = np.array([[31.5, 1087.5], [127.5, 1215.5], [500.5, 700.5]])
    gradient, seed = 2.0, 59091005
    motion = _motion_parameters(np.random.default_rng(seed), 3840, 2160, "shear_band", gradient, .5)
    truth, visible = exact_shear_truth(points, seed, gradient)
    x, y = _forward_map(points[:, 0], points[:, 1], motion, 3840, 2160)
    np.testing.assert_allclose(truth[:, 0], x - points[:, 0], atol=1e-10)
    np.testing.assert_allclose(truth[:, 3], y - points[:, 1], atol=1e-10)
    eps = 1e-3
    xp, _ = _forward_map(points[:, 0], points[:, 1] + eps, motion, 3840, 2160)
    xm, _ = _forward_map(points[:, 0], points[:, 1] - eps, motion, 3840, 2160)
    np.testing.assert_allclose(truth[:, 2], (xp - xm) / (2 * eps), atol=1e-8)
    np.testing.assert_array_equal(visible, (x >= 0) & (x <= 3839) & (y >= 0) & (y <= 2159))
