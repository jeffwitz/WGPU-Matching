"""Generate deterministic raw grayscale inputs shared by browser/Python parity runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 512, 384
DX, DY = 8, -4
AFFINE = {
    "matrix": [[1.005, 0.001], [-0.001, 0.997]],
    "translation": [4.0, -3.0],
    "center": [(WIDTH - 1) / 2, (HEIGHT - 1) / 2],
}
OUT = Path(__file__).resolve().parent / "assets"


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 0x100000000


def speckle() -> np.ndarray:
    rng = Lcg(42)
    image = np.full((HEIGHT, WIDTH), 128.0, dtype=np.float32)
    for _ in range(WIDTH * HEIGHT // 80):
        x = rng.random() * WIDTH
        y = rng.random() * HEIGHT
        radius = 1.5 + rng.random() * 2.8
        amplitude = (rng.random() - 0.5) * 120
        x0, x1 = max(0, int(x - 3 * radius)), min(WIDTH, int(x + 3 * radius + 1))
        y0, y1 = max(0, int(y - 3 * radius)), min(HEIGHT, int(y + 3 * radius + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance2 = (xx - x) ** 2 + (yy - y) ** 2
        image[y0:y1, x0:x1] += amplitude * np.exp(-distance2 / (2 * radius * radius))
    noise = np.fromiter((rng.random() for _ in range(WIDTH * HEIGHT)), np.float32)
    image += (noise.reshape(HEIGHT, WIDTH) - 0.5) * 8
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def affine_warp(reference: np.ndarray) -> np.ndarray:
    matrix = np.asarray(AFFINE["matrix"], dtype=np.float64)
    translation = np.asarray(AFFINE["translation"], dtype=np.float64)
    center = np.asarray(AFFINE["center"], dtype=np.float64)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    target_coordinates = np.stack((xx - center[0] - translation[0],
                                   yy - center[1] - translation[1]), axis=-1)
    source_coordinates = target_coordinates @ np.linalg.inv(matrix).T + center
    source_x, source_y = source_coordinates[..., 0], source_coordinates[..., 1]
    valid = ((source_x >= 0) & (source_x < WIDTH - 1)
             & (source_y >= 0) & (source_y < HEIGHT - 1))
    x0 = np.clip(np.floor(source_x).astype(np.int32), 0, WIDTH - 2)
    y0 = np.clip(np.floor(source_y).astype(np.int32), 0, HEIGHT - 2)
    fx, fy = source_x - x0, source_y - y0
    top = reference[y0, x0] * (1 - fx) + reference[y0, x0 + 1] * fx
    bottom = reference[y0 + 1, x0] * (1 - fx) + reference[y0 + 1, x0 + 1] * fx
    sampled = top * (1 - fy) + bottom * fy
    return np.where(valid, np.clip(np.rint(sampled), 0, 255), 128).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = speckle()
    target = np.full_like(reference, 128)
    source_x0, source_x1 = max(0, -DX), min(WIDTH, WIDTH - DX)
    source_y0, source_y1 = max(0, -DY), min(HEIGHT, HEIGHT - DY)
    target[source_y0 + DY:source_y1 + DY, source_x0 + DX:source_x1 + DX] = reference[
        source_y0:source_y1, source_x0:source_x1
    ]
    reference.tofile(OUT / "matcher_parity_reference.u8")
    target.tofile(OUT / "matcher_parity_target.u8")
    affine_warp(reference).tofile(OUT / "matcher_parity_affine.u8")
    (OUT / "matcher_parity_affine.json").write_text(
        json.dumps(AFFINE, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {WIDTH}x{HEIGHT} translation and mild affine fixtures")


if __name__ == "__main__":
    main()
