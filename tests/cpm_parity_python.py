"""Run the Python CPM frontend on the shared deterministic translation fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wgpu_matching import CPMOptions, WGPUMatching

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/assets"
WIDTH, HEIGHT = 512, 384
SEEDS = ((98.25, 90.75), (248.5, 190.25), (402.25, 278.75))


def main() -> None:
    reference = np.fromfile(FIXTURES / "matcher_parity_reference.u8", dtype=np.uint8).reshape(HEIGHT, WIDTH)
    target = np.fromfile(FIXTURES / "matcher_parity_target.u8", dtype=np.uint8).reshape(HEIGHT, WIDTH)
    options = CPMOptions(
        patch_size=64,
        stride=64,
        downscale_levels=1,
        search_radius_px=64,
        matching_support_px=32,
        work_grid_stride=16,
        levels=2,
        propagation_iterations=2,
        random_trials=4,
        initial_hypotheses=8,
        local_search_radius_px=32,
        affine_radius_px=64,
        seed=17,
        reciprocal_threshold_px=8,
        oracle_affine=(1, 0, 8, 0, 1, -4),
        seed_points=SEEDS,
    )
    with WGPUMatching(device_type="gpu") as matcher:
        field = matcher.match_cpm(reference, target, options=options)
        print(json.dumps({
            "adapter": matcher.device_name,
            "width": WIDTH,
            "height": HEIGHT,
            "matches": [
                {
                    "x": match.x,
                    "y": match.y,
                    "dx": match.dx,
                    "dy": match.dy,
                    "usable": match.usable,
                    "score": match.score,
                    "reciprocal": match.reciprocal,
                    "inliers": match.inliers,
                    "candidates": match.candidates,
                    "inlier_fraction": match.inlier_fraction,
                    "residual": match.residual,
                }
                for match in field.matches
            ],
            "metadata": field.metadata,
        }, allow_nan=False))


if __name__ == "__main__":
    main()
