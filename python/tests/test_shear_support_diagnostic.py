import numpy as np
import pytest
from diagnose_shear_support import map_centres_to_grid


def test_64_centres_map_exactly_to_fine_grid():
    points = np.array([[31.5, 31.5], [95.5, 31.5], [31.5, 95.5], [351.5, 223.5]])
    np.testing.assert_array_equal(map_centres_to_grid(points, 64, 64, 6, 4), [0, 1, 6, 23])
    np.testing.assert_array_equal(map_centres_to_grid(points, 32, 16, 24, 16), [25, 29, 121, 333])
    np.testing.assert_array_equal(map_centres_to_grid(points, 16, 8, 48, 32), [147, 155, 531, 1339])


def test_grid_mapping_rejects_shifted_or_out_of_bounds_centers():
    with pytest.raises(ValueError, match="do not coincide"):
        map_centres_to_grid([[32.0, 31.5]], 32, 16, 24, 16)
    with pytest.raises(ValueError, match="do not coincide"):
        map_centres_to_grid([[999.5, 31.5]], 32, 16, 24, 16)
