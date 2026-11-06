"""Small image-loading helper; matchers also accept NumPy grayscale arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image as contiguous uint8 Rec. 709 grayscale pixels."""

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb32 = rgb.astype(np.uint32)
    gray = (2126 * rgb32[..., 0] + 7152 * rgb32[..., 1] + 722 * rgb32[..., 2] + 5000) // 10000
    return np.ascontiguousarray(gray.clip(0, 255), dtype=np.uint8)
