"""Load experimental or analytically deformed image pairs for DIC qualification."""

import numpy as np
from PIL import Image


def load_gray_pair(reference_path, target_path):
    """Read native-resolution monochrome images in the 0–1 range."""
    with Image.open(reference_path) as image:
        reference = np.asarray(image.convert("L"), dtype=np.float32) / 255.
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("L"), dtype=np.float32) / 255.
    if reference.shape != target.shape:
        raise ValueError("reference and target dimensions differ")
    return reference, target


def affine_truth(centres, image_shape, stretch_x, stretch_y):
    """Return local DIC parameters and centre visibility for a known centered stretch."""
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    height, width = image_shape
    if stretch_x is None and stretch_y is None:
        return None, None
    if stretch_x is None or stretch_y is None or stretch_x <= 0 or stretch_y <= 0:
        raise ValueError("both positive stretches are required for analytic truth")
    mapped_x = width / 2 + stretch_x * (centres[:, 0] - width / 2)
    mapped_y = height / 2 + stretch_y * (centres[:, 1] - height / 2)
    params = np.column_stack((mapped_x - centres[:, 0], np.full(len(centres), stretch_x - 1),
                              np.zeros(len(centres)), mapped_y - centres[:, 1],
                              np.zeros(len(centres)), np.full(len(centres), stretch_y - 1)))
    visible = (mapped_x >= 0) & (mapped_x <= width - 1) & (mapped_y >= 0) & (mapped_y <= height - 1)
    return params.astype(np.float32), visible
