"""Run Python DeepMatching/HORG on the browser parity fixture and emit JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from wgpu_matching import CPMOptions, DeepMatchingOptions, HORGOptions, WGPUMatching

WIDTH, HEIGHT = 512, 384
PATCH_SIZE, STRIDE = 128, 128
FIXTURES = Path(__file__).resolve().parent / "assets"


def safe(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def main() -> None:
    reference = np.fromfile(FIXTURES / "matcher_parity_reference.u8", dtype=np.uint8)
    targets = {
        "translation": np.fromfile(FIXTURES / "matcher_parity_target.u8", dtype=np.uint8),
        "affine": np.fromfile(FIXTURES / "matcher_parity_affine.u8", dtype=np.uint8),
    }
    if reference.size != WIDTH * HEIGHT or any(
        target.size != WIDTH * HEIGHT for target in targets.values()
    ):
        raise RuntimeError("Parity fixtures must contain exactly 512x384 uint8 pixels")
    reference = reference.reshape(HEIGHT, WIDTH)
    targets = {name: target.reshape(HEIGHT, WIDTH) for name, target in targets.items()}
    affine_truth = json.loads((FIXTURES / "matcher_parity_affine.json").read_text(encoding="utf-8"))
    manual_seeds = ((98.25, 90.75), (248.5, 190.25), (402.25, 278.75))
    cases = {
        "translation": {"target": targets["translation"], "seed_points": None},
        "affine": {"target": targets["affine"], "seed_points": None},
        "manual_translation": {"target": targets["translation"], "seed_points": manual_seeds},
    }
    options = DeepMatchingOptions(
        patch_size=PATCH_SIZE,
        root_patch_size=128,
        stride=STRIDE,
        downscale_levels=1,
        search_radius_px=64,
    )
    horg_options = HORGOptions(
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        reduction=2,
        max_features=1024,
    )
    def normalize(field):
        return [
            {
                "x": safe(match.x), "y": safe(match.y),
                "dx": safe(match.dx), "dy": safe(match.dy),
                "usable": bool(match.usable), "score": safe(match.score),
                "second_score": safe(match.second_score), "margin": safe(match.margin),
                "reciprocal": match.reciprocal,
                "inliers": match.inliers, "candidates": match.candidates,
                "inlier_fraction": safe(match.inlier_fraction),
                "residual": safe(match.residual),
                "local_consistency_px": safe(match.local_consistency_px),
                "local_consistency_neighbors": match.local_consistency_neighbors,
                "seed_x": safe(match.seed_x), "seed_y": safe(match.seed_y),
                "root_x": safe(match.root_x), "root_y": safe(match.root_y),
                "target_x": safe(match.target_x), "target_y": safe(match.target_y),
            }
            for match in field.matches
        ]

    methods = {"deepmatching": {}, "horg": {}}
    with WGPUMatching(device_type="gpu") as matcher:
        for name, scenario in cases.items():
            scenario_options = replace(options, seed_points=scenario["seed_points"])
            scenario_horg_options = replace(horg_options, seed_points=scenario["seed_points"])
            deep = matcher.match_deepmatching(reference, scenario["target"], options=scenario_options)
            horg = matcher.match_horg(reference, scenario["target"], options=scenario_horg_options)
            methods["deepmatching"][name] = normalize(deep)
            methods["horg"][name] = normalize(horg)

        cpm = matcher.match_cpm(reference, targets["translation"], options=CPMOptions(
            patch_size=32,
            stride=32,
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
            seed_points=manual_seeds,
        ))
        methods["cpm"] = {
            "manual_translation": [
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
                    "inlierFraction": match.inlier_fraction,
                    "residual": match.residual,
                    "measuredX": diagnostic["measured_x"],
                    "measuredY": diagnostic["measured_y"],
                    "measuredDx": diagnostic["measured_dx"],
                    "measuredDy": diagnostic["measured_dy"],
                    "gridDistancePx": diagnostic["grid_distance_px"],
                    "reciprocalErrorPx": diagnostic["reciprocal_error_px"],
                }
                for match, diagnostic in zip(cpm.matches, cpm.metadata["match_diagnostics"], strict=True)
            ],
            "metadata": {
                "workGridPointCount": cpm.metadata["work_grid_point_count"],
                "workGridReciprocalCount": cpm.metadata["work_grid_reciprocal_count"],
                "workGridUsableCount": cpm.metadata["work_grid_usable_count"],
                "gridWidth": cpm.grid_width,
                "gridHeight": cpm.grid_height,
            },
            "oracle": {
                "perSeed": [
                    {
                        "requestedX": item["requested_x"],
                        "requestedY": item["requested_y"],
                        "measuredX": item["measured_x"],
                        "measuredY": item["measured_y"],
                        "levels": [
                            {
                                "level": level["level"],
                                "descriptorScalePx": level["descriptor_scale_px"],
                                "foundDxPx": level["found_dx_px"],
                                "foundDyPx": level["found_dy_px"],
                                "foundCost": level["found_cost"],
                                "foundValid": level["found_valid"],
                                "oracleDxPx": level["oracle_dx_px"],
                                "oracleDyPx": level["oracle_dy_px"],
                                "oracleCost": level["oracle_cost"],
                                "oracleValid": level["oracle_valid"],
                                "oracleMinusFoundCost": level["oracle_minus_found_cost"],
                            }
                            for level in item["levels"]
                        ],
                    }
                    for item in cpm.metadata["oracle_diagnostics"]["per_seed"]
                ]
            },
        }

        print(json.dumps({
            "adapter": matcher.device_name,
            "width": WIDTH,
            "height": HEIGHT,
            "seedCounts": {"translation": 12, "affine": 12, "manual_translation": 3},
            "transforms": {
                "translation": [8, -4],
                "affine": affine_truth,
                "manual_translation": [8, -4],
            },
            "apiDefaults": {
                "deepmatching": asdict(DeepMatchingOptions()),
                "horg": asdict(HORGOptions()),
                "cpm": asdict(CPMOptions()),
            },
            "methods": methods,
        }, allow_nan=False))


if __name__ == "__main__":
    main()
