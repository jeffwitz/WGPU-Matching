"""Score the exact-seed IC-GN flank controls after the solver has run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import map_coordinates


def qualify(root: Path) -> dict:
    truth = np.load(root / "truth_scoring_only.npz")
    result = np.load(root / "oracle" / "native_restart_verification.npz")
    reference = np.asarray(Image.open(root / "reference.png"))
    target = np.asarray(Image.open(root / "target.png"))
    reference_mask = material_mask(reference)
    target_mask = material_mask(target).astype(np.float32)
    centres = truth["centres"]
    parameters = truth["parameters"]
    fitted = result["parameters"]
    selected = np.load("results/hudge_boundary_frontier_20261115/partial12/seeds.npz")["selected"]

    half = 5.5
    dy, dx = np.mgrid[:12, :12] - half
    offsets = np.stack((dx, dy), axis=-1)
    support = np.zeros((len(centres), 12, 12), dtype=bool)
    for index in np.flatnonzero(selected):
        x, y = (centres[index] - half).astype(int)
        support[index] = reference_mask[y : y + 12, x : x + 12]

    gradient = np.eye(2) + parameters[:, [1, 2, 4, 5]].reshape(-1, 2, 2)
    cross_pixels = 12.0 * np.linalg.svd(gradient, compute_uv=False)[:, -1]
    target_points = (
        centres[:, None, None]
        + parameters[:, None, None, [0, 3]]
        + np.einsum("nij,hwj->nhwi", gradient, offsets)
    )
    target_support = map_coordinates(
        target_mask,
        [target_points[..., 1].ravel(), target_points[..., 0].ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(len(centres), 12, 12)
    support_fraction = (target_support * support).sum((1, 2)) / np.maximum(1, support.sum((1, 2)))

    # The reference plate's observed support bounds are x=[3.5,426.5], y=[3.5,214.5].
    distances = np.stack(
        (centres[:, 0] - 3.5, 426.5 - centres[:, 0], centres[:, 1] - 3.5, 214.5 - centres[:, 1]),
        axis=1,
    )
    accepted = result["accepted"]
    displacement_error = np.linalg.norm(fitted[:, [0, 3]] - parameters[:, [0, 3]], axis=1)
    rows = []
    bands = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 16)]
    for side, name in enumerate(("left", "right", "top", "bottom")):
        for low, high in bands:
            use = selected & (distances[:, side] >= low) & (distances[:, side] < high)
            if not use.any():
                continue
            rows.append(
                {
                    "side": name,
                    "distance_px": [low, high],
                    "windows": int(use.sum()),
                    "solver_accepted": int((use & accepted).sum()),
                    "target_support_ge_75pct": int((use & (support_fraction >= 0.75)).sum()),
                    "accepted_and_target_support_ge_75pct": int(
                        (use & accepted & (support_fraction >= 0.75)).sum()
                    ),
                    "median_target_material_fraction": float(np.median(support_fraction[use])),
                }
            )
    lateral = selected & (distances[:, :2].min(axis=1) <= 8)
    lateral_accept = lateral & accepted
    report = {
        "eligible_lateral": int(lateral.sum()),
        "accepted_lateral": int(lateral_accept.sum()),
        "lateral_fraction": float(lateral_accept.sum() / max(1, lateral.sum())),
        "accepted_lateral_and_target_support_ge_75pct": int(
            (lateral_accept & (support_fraction >= 0.75)).sum()
        ),
        "accepted_lateral_with_final_displacement_error_le_2px": int(
            (lateral_accept & (displacement_error <= 2.0)).sum()
        ),
        "accepted_lateral_median_final_displacement_error_px": float(
            np.median(displacement_error[lateral_accept])
        )
        if lateral_accept.any()
        else None,
        "lateral_by_target_cross_pixels": [],
        "rows": rows,
        "interpretation": "Exact local affine seeds were used only in this oracle control. The synthetic map is spatially varying, so within-window non-affinity remains. This result does not establish the unknown physical image generator or real-plate accuracy.",
    }
    for low, high in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, np.inf)):
        use = lateral & (cross_pixels >= low) & (cross_pixels < high)
        if use.any():
            report["lateral_by_target_cross_pixels"].append(
                {
                    "target_cross_pixels": [low, high if np.isfinite(high) else None],
                    "windows": int(use.sum()),
                    "solver_accepted": int((use & accepted).sum()),
                    "accepted_error_le_2px": int(
                        (use & accepted & (displacement_error <= 2.0)).sum()
                    ),
                    "accepted_fraction": float((use & accepted).sum() / use.sum()),
                }
            )
    (root / "oracle" / "edge_bands.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.root), indent=2))


if __name__ == "__main__":
    main()
