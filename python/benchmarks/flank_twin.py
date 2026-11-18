"""Resolution-limit control for the laterally thinned HudgeStrains plate.

The target is rendered from the reference with an invertible, separable map.
Its lateral stretch approaches 0.05 at both free edges and returns smoothly to
an average stretch of one at mid-width. The vertical map fills the same image
height as frame 191. Exact local affine states are written for solver scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def lateral_profile(length: float, edge_stretch: float, decay: float) -> tuple[float, float]:
    """Return central stretch and its normalizing average for symmetric sides."""
    edge_integral = decay / length * (1.0 - np.exp(-length / decay))
    centre = (1.0 - 2.0 * edge_stretch * edge_integral) / (1.0 - 2.0 * edge_integral)
    return centre, edge_integral


def map_x(
    x: np.ndarray, left: float, right: float, edge_stretch: float, decay: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the edge-thinning map and its exact derivative."""
    length = right - left
    centre, _ = lateral_profile(length, edge_stretch, decay)
    d_left = x - left
    d_right = right - x
    derivative = centre - (centre - edge_stretch) * (
        np.exp(-d_left / decay) + np.exp(-d_right / decay)
    )

    def primitive(distance: np.ndarray) -> np.ndarray:
        return centre * distance - (centre - edge_stretch) * decay * (
            1.0 - np.exp(-distance / decay)
        )

    value = left + primitive(d_left) - primitive(np.zeros_like(d_left))
    return value, derivative


def render(
    reference_path: Path,
    out: Path,
    edge_stretch: float = 0.05,
    decay: float = 38.0,
    mirror: bool = False,
) -> None:
    reference = np.asarray(Image.open(reference_path).convert("L"), dtype=np.float32)
    if mirror:
        reference = reference.copy()
        occupied = np.flatnonzero(np.any(reference != reference[-1, -1], axis=0))
        reference[:, occupied[0] : occupied[-1] + 1] = reference[:, occupied[0] : occupied[-1] + 1][
            :, ::-1
        ]
    h, w = reference.shape
    material = np.any(reference != reference[-1, -1], axis=0)
    columns = np.flatnonzero(material)
    left, right = float(columns[0]) - 0.5, float(columns[-1]) + 0.5
    rows = np.flatnonzero(np.any(reference != reference[-1, -1], axis=1))
    top, bottom = float(rows[0]) - 0.5, float(rows[-1]) + 0.5

    # The profile integral is normalized to preserve the material width.
    # The equivalent source coordinate is found by monotone interpolation.
    x_source = np.arange(left + 0.5, right, dtype=float)
    x_target, stretch = map_x(x_source, left, right, edge_stretch, decay)
    target_x = np.arange(w, dtype=float)
    source_x = np.interp(target_x, x_target, x_source, left=np.nan, right=np.nan)
    vertical_scale = (h - 1.0 - (top + 0.5)) / (bottom - top)
    target_y = np.arange(h, dtype=float)
    source_y = top + 0.5 + (target_y - (top + 0.5)) / vertical_scale
    xx, yy = np.meshgrid(source_x, source_y)
    valid = (
        np.isfinite(xx) & np.isfinite(yy) & (xx >= 0) & (xx <= w - 1) & (yy >= 0) & (yy <= h - 1)
    )
    source_x = np.nan_to_num(xx, nan=0.0)
    source_y = np.nan_to_num(yy, nan=0.0)
    target = map_coordinates(
        reference, [source_y, source_x], order=3, mode="constant", cval=float(reference[-1, -1])
    )
    target[~valid] = reference[-1, -1]

    # Truth is diagnostic-only. Parameters use the solver's ordering
    # (u, ux, uy, v, vx, vy) and pixel-centred window coordinates.
    grid_path = Path("results/hudge_boundary_frontier_20261115/partial12/seeds.npz")
    grid = dict(np.load(grid_path))
    centres = grid["centres"]
    qx, fx = map_x(centres[:, 0], left, right, edge_stretch, decay)
    qy = top + 0.5 + (centres[:, 1] - (top + 0.5)) * vertical_scale
    parameters = np.zeros((len(centres), 6), np.float32)
    parameters[:, 0] = qx - centres[:, 0]
    parameters[:, 1] = fx - 1.0
    parameters[:, 3] = qy - centres[:, 1]
    parameters[:, 5] = vertical_scale - 1.0
    grid["parameters"] = parameters
    grid["accepted"] = np.zeros(len(centres), bool)
    grid["scores"] = np.zeros(len(centres), np.float32)
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(np.uint8(np.rint(reference))).save(out / "reference.png")
    Image.fromarray(np.uint8(np.clip(np.rint(target), 0, 255))).save(out / "target.png")
    np.savez_compressed(
        out / "oracle_seeds.npz",
        centres=centres,
        parameters=parameters,
        selected=grid["selected"],
        accepted=grid["selected"],
    )
    truth = np.stack((qx, qy), axis=-1)
    np.savez_compressed(
        out / "truth_scoring_only.npz",
        centres=centres,
        parameters=parameters,
        target_centres=truth,
        vertical_scale=np.float64(vertical_scale),
    )
    centre, _ = lateral_profile(right - left, edge_stretch, decay)
    (out / "render.json").write_text(
        json.dumps(
            {
                "reference": str(reference_path.resolve()),
                "shape": list(reference.shape),
                "reference_lateral_bounds": [left, right],
                "reference_vertical_bounds": [top, bottom],
                "edge_stretch": edge_stretch,
                "decay_pixels": decay,
                "mirrored_texture": mirror,
                "midpoint_stretch": centre,
                "vertical_stretch": vertical_scale,
                "rendering": "inverse monotone map; cubic interpolation; 8-bit quantization",
                "purpose": "oracle sensitivity control only; not a physical forward model or evidence about the plate's unknown generator",
                "seed_archive": str(grid_path),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_F.tif")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge-stretch", type=float, default=0.05)
    parser.add_argument("--decay", type=float, default=38.0)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="mirror the speckle texture before applying the same map",
    )
    args = parser.parse_args()
    render(args.reference, args.output, args.edge_stretch, args.decay, args.mirror)


if __name__ == "__main__":
    main()
