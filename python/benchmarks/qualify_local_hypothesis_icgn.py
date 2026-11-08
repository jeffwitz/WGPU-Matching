"""Qualify image-derived rotation hypotheses followed by local affine IC-GN seeds.

The affine parameters are inferred from detected correspondences and WGSL local
RANSAC. Analytic motion is used only for the final score and the solver oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from benchmark_affine_hypothesis_search import (
    evaluate_rotation_coarse_to_fine,
    prepare_case,
)
from qualify_heterogeneous_icgn import (
    analytic_parameters,
    grid_centres,
    json_safe,
    score_result,
    seed_state,
)
from wgpu_matching import WGPUMatching


def local_affine_seed_parameters(field, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGSL target-map models to Live-DIC's six-parameter ordering."""
    parameters = np.zeros((len(centers), 6), dtype=np.float32)
    valid = np.zeros(len(centers), dtype=bool)
    if field.affine_models is None:
        return parameters, valid
    models = np.asarray(field.affine_models, dtype=np.float32).reshape(-1, 8)
    if len(models) != len(centers):
        raise ValueError("Local affine model count does not match the DIC window grid")
    for index, (match, model, center) in enumerate(zip(field.matches, models, centers, strict=True)):
        if not match.usable or not np.isfinite(model).all():
            continue
        # Model is [a1,a2,a0,unused,b1,b2,b0,unused], mapping source
        # offsets around center to absolute target coordinates.
        a1, a2, a0 = model[:3]
        b1, b2, b0 = model[4:7]
        parameters[index] = (
            a0 - center[0],
            a1 - 1.0,
            a2,
            b0 - center[1],
            b1,
            b2 - 1.0,
        )
        valid[index] = True
    return parameters, valid


def fit_local_and_solve(
    matcher,
    solver,
    pairs: np.ndarray,
    centers: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
    radius_px: float,
    threshold_px: float,
):
    started = time.perf_counter()
    field = matcher.fit_local_affine(
        pairs,
        centers,
        radius_px=radius_px,
        threshold_px=threshold_px,
        min_inliers=6,
        min_candidates=8,
        min_inlier_fraction=0.15,
        max_iterations=256,
        include_affine_models=True,
    )
    fit_s = time.perf_counter() - started
    parameters, valid = local_affine_seed_parameters(field, centers)
    target_centers = centers + parameters[:, [0, 3]]
    valid &= (
        (target_centers[:, 0] >= 0)
        & (target_centers[:, 0] < width)
        & (target_centers[:, 1] >= 0)
        & (target_centers[:, 1] < height)
    )
    # Model activation uses only the predicted target centre and image bounds;
    # analytic visibility is deliberately excluded from hypothesis selection.
    border = 31.5
    valid &= (
        (target_centers[:, 0] >= border)
        & (target_centers[:, 0] < width - border)
        & (target_centers[:, 1] >= border)
        & (target_centers[:, 1] < height - border)
    )
    active = valid.astype(np.uint32)
    started = time.perf_counter()
    result = solver._register_seeded_frame(target, seed_state(parameters, active), active)
    solve_s = time.perf_counter() - started
    return field, parameters, valid, active, result, fit_s, solve_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--live-dic-source",
        type=Path,
        default=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src"),
    )
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--candidate-budget", type=int, default=8192)
    parser.add_argument("--query-budget", type=int, default=2048)
    parser.add_argument("--coarse-queries", type=int, default=128)
    parser.add_argument("--coarse-candidates", type=int, default=2048)
    parser.add_argument("--textures", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--warp-offset", type=int, default=5_000_009)
    parser.add_argument("--radius-px", type=float, default=192.0)
    parser.add_argument("--threshold-px", type=float, default=4.0)
    args = parser.parse_args()
    if not args.cnn_weights.is_file():
        parser.error(f"CNN weights not found: {args.cnn_weights}")
    if not args.live_dic_source.is_dir():
        parser.error(f"IC-GN source not found: {args.live_dic_source}")
    sys.path.insert(0, str(args.live_dic_source))
    from live_dic import ICGNWGPU

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with WGPUMatching(device_type="gpu") as matcher:
        print(f"WebGPU adapter: {matcher.device_name}", flush=True)
        solver = ICGNWGPU(
            device_type="gpu",
            prefer_subgroups=False,
            subset_size=64,
            subset_stride=64,
            interpolation="linear",
            motion_model="affine",
            outlier_policy="disabled",
        )
        for sample in range(args.textures):
            texture_seed = args.seed + sample * 10_007
            warp_seed = args.seed + args.warp_offset + sample * 10_007
            case = prepare_case(
                matcher,
                texture_seed=texture_seed,
                warp_seed=warp_seed,
                family="heterogeneous",
                width=args.width,
                height=args.height,
                query_count=args.query_budget,
                candidate_budget=args.candidate_budget,
                weights_path=args.cnn_weights,
                describe_source=False,
            )
            search = evaluate_rotation_coarse_to_fine(
                matcher,
                case,
                args.cnn_weights,
                coarse_queries=args.coarse_queries,
                coarse_candidates=args.coarse_candidates,
                refine_seeds=2,
                final_hypotheses=2,
            )
            pairs = np.asarray(search["selected_pairs_xyxy"], dtype=np.float32).reshape(-1, 4)
            pair = case["pair"]
            reference, target = case["reference"], case["target"]
            centers = grid_centres(args.width, args.height, 64, 64)
            visible = pair.visible[
                np.rint(centers[:, 1]).astype(int), np.rint(centers[:, 0]).astype(int)
            ].astype(bool)
            reference_start = time.perf_counter()
            solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
            reference_prepare_s = time.perf_counter() - reference_start
            truth = analytic_parameters(pair, centers)

            (
                local_field,
                initial,
                active_bool,
                _active,
                result,
                local_fit_s,
                solve_wall_s,
            ) = fit_local_and_solve(
                matcher,
                solver,
                pairs,
                centers,
                target,
                args.width,
                args.height,
                args.radius_px,
                args.threshold_px,
            )

            # Same detector and candidates, without support compensation. This
            # is the paired ablation for the image-derived hypothesis bank.
            baseline_descriptor_start = time.perf_counter()
            baseline_source = matcher.describe_dic_cnn_keypoints(
                case["reference"], case["source_features"], weights_path=args.cnn_weights
            )
            baseline_target = matcher.describe_dic_cnn_keypoints(
                case["target"], case["target_features"], weights_path=args.cnn_weights
            )
            baseline_descriptions_s = time.perf_counter() - baseline_descriptor_start
            baseline_matches = matcher.match_features(
                baseline_source,
                baseline_target,
                centers,
                min_cosine_similarity=0.0,
                fit_local_models=False,
            ).correspondences.points
            baseline_source_xy = baseline_matches[:, :2]
            baseline_x = np.clip(np.rint(baseline_source_xy[:, 0]).astype(int), 0, args.width - 1)
            baseline_y = np.clip(np.rint(baseline_source_xy[:, 1]).astype(int), 0, args.height - 1)
            baseline_expected = baseline_source_xy + pair.flow_xy[baseline_y, baseline_x]
            baseline_direct_error = np.linalg.norm(
                baseline_matches[:, 2:] - baseline_expected, axis=1
            )
            (
                baseline_field,
                baseline_initial,
                baseline_valid,
                _baseline_active,
                baseline_result,
                baseline_fit_s,
                baseline_solve_s,
            ) = fit_local_and_solve(
                matcher,
                solver,
                baseline_matches,
                centers,
                target,
                args.width,
                args.height,
                args.radius_px,
                args.threshold_px,
            )

            # Controls are separate: zero initialization and analytic oracle are not
            # allowed to influence correspondence or model selection.
            zero_active = visible.astype(np.uint32)
            zero_start = time.perf_counter()
            zero_result = solver._register_seeded_frame(
                target, seed_state(np.zeros_like(truth), zero_active), zero_active
            )
            zero_wall_s = time.perf_counter() - zero_start
            oracle_start = time.perf_counter()
            oracle_result = solver._register_seeded_frame(
                target, seed_state(truth, zero_active), zero_active
            )
            oracle_wall_s = time.perf_counter() - oracle_start

            source = pairs[:, :2]
            sx = np.clip(np.rint(source[:, 0]).astype(int), 0, args.width - 1)
            sy = np.clip(np.rint(source[:, 1]).astype(int), 0, args.height - 1)
            mapped = source + pair.flow_xy[sy, sx]
            direct_error = np.linalg.norm(pairs[:, 2:] - mapped, axis=1)
            end_to_end = (
                case["detection_s"]
                + case["source_descriptor_s"]
                + search["rotation_search_s"]
                + reference_prepare_s
                + local_fit_s
                + solve_wall_s
            )
            row = {
                "family": "heterogeneous",
                "texture_seed": texture_seed,
                "warp_seed": warp_seed,
                "image_size_px": [args.width, args.height],
                "detector_queries": len(case["source_xy"]),
                "target_candidates": len(case["target_xy"]),
                "repeatable_queries_le5px": int(case["repeatable"].sum()),
                "coarse_to_fine_rotation": search,
                "selected_direct_matches": len(pairs),
                "selected_direct_matches_correct_le2px": int(np.count_nonzero(direct_error <= 2)),
                "selected_direct_matches_correct_lt5px": int(np.count_nonzero(direct_error < 5)),
                "selected_direct_match_error_median_px": float(np.median(direct_error))
                if len(direct_error)
                else None,
                "local_affine_models_usable": int(sum(match.usable for match in local_field.matches)),
                "local_affine_models_active_in_frame": int(active_bool.sum()),
                "local_affine_model_inlier_median": float(
                    np.median([match.inliers for match in local_field.matches if match.usable])
                )
                if any(match.usable for match in local_field.matches)
                else None,
                "local_affine_model_rmse_initial_translation_px": float(
                    np.sqrt(np.mean(np.sum((initial[active_bool][:, [0, 3]] - truth[active_bool][:, [0, 3]]) ** 2, axis=1)))
                ) if active_bool.any() else None,
                "local_affine_model_rmse_initial_gradient": float(
                    np.sqrt(np.mean((initial[active_bool][:, [1, 2, 4, 5]] - truth[active_bool][:, [1, 2, 4, 5]]) ** 2))
                ) if active_bool.any() else None,
                "icgn_local_affine_initialization": score_result(result, truth, visible),
                "uncompensated_direct_matches": len(baseline_matches),
                "uncompensated_direct_matches_correct_lt5px": int(
                    np.count_nonzero(baseline_direct_error < 5.0)
                ),
                "uncompensated_local_affine_models_usable": int(
                    sum(match.usable for match in baseline_field.matches)
                ),
                "uncompensated_local_affine_models_active_in_frame": int(baseline_valid.sum()),
                "icgn_uncompensated_local_affine_initialization": score_result(
                    baseline_result, truth, visible
                ),
                "icgn_zero_initialization": score_result(zero_result, truth, visible),
                "icgn_analytic_oracle_control": score_result(oracle_result, truth, visible),
                "timing_s": {
                    "detection": case["detection_s"],
                    "source_descriptor": case["source_descriptor_s"],
                    "rotation_hypothesis_search": search["rotation_search_s"],
                    "reference_preparation": reference_prepare_s,
                    "local_affine_wgpu_ransac": local_fit_s,
                    "icgn_solver": solve_wall_s,
                    "end_to_end": end_to_end,
                    "uncompensated_description_and_matching": baseline_descriptions_s,
                    "uncompensated_local_affine_wgpu_ransac": baseline_fit_s,
                    "uncompensated_icgn_solver": baseline_solve_s,
                    "zero_control": zero_wall_s,
                    "oracle_control": oracle_wall_s,
                },
                "correct_windows_per_second_end_to_end": (
                    score_result(result, truth, visible)["correct_converged_lt2px"]
                    / max(end_to_end, 1e-12)
                ),
                "local_model_fit": {
                    "radius_px": args.radius_px,
                    "threshold_px": args.threshold_px,
                    "minimum_inliers": 6,
                    "minimum_candidates": 8,
                    "minimum_inlier_fraction": 0.15,
                    "coefficients_order": "WGSL target map [a1,a2,a0,unused,b1,b2,b0,unused], converted to IC-GN affine order",
                },
                "weights_sha256": hashlib.sha256(args.cnn_weights.read_bytes()).hexdigest(),
                "webgpu_adapter": matcher.device_name,
            }
            np.savez_compressed(
                args.output / f"heterogeneous_{texture_seed}_{warp_seed}.npz",
                selected_pairs_xyxy=pairs,
                direct_error_px=direct_error,
                local_affine_models=local_field.affine_models,
                uncompensated_pairs_xyxy=baseline_matches,
                uncompensated_direct_error_px=baseline_direct_error,
                uncompensated_local_affine_models=baseline_field.affine_models,
                uncompensated_initial_parameters=baseline_initial,
                uncompensated_active=baseline_valid,
                uncompensated_icgn_parameters=np.asarray(baseline_result.parameters).reshape(-1, 6),
                uncompensated_icgn_status=np.asarray(baseline_result.status).reshape(-1),
                initial_parameters=initial,
                active=active_bool,
                analytic_parameters=truth,
                icgn_parameters=np.asarray(result.parameters).reshape(-1, 6),
                icgn_status=np.asarray(result.status).reshape(-1),
            )
            result_path = args.output / f"heterogeneous_{texture_seed}_{warp_seed}.json"
            result_path.write_text(json.dumps(json_safe(row), indent=2) + "\n")
            results.append(row)
            print(
                f"heterogeneous {sample + 1}/{args.textures}: direct={len(pairs)}, "
                f"local models={active_bool.sum()}, IC-GN correct="
                f"{row['icgn_local_affine_initialization']['correct_converged_lt2px']}/"
                f"{row['icgn_local_affine_initialization']['visible_windows']}; saved {result_path}",
                flush=True,
            )
    summary = {
        "protocol": {
            "method": "image-derived coarse-to-fine rotation support hypotheses, reciprocal WGSL matching, then per-window local affine WGSL RANSAC and seeded IC-GN",
            "analytic_truth_usage": "final scoring and oracle control only; no truth used to select angles, descriptor matches, or local affine models",
            "field_scope": "smooth heterogeneous synthetic fields; discontinuities intentionally excluded from this first local-model test",
            "webgpu_adapter": matcher.device_name,
            "weights_sha256": hashlib.sha256(args.cnn_weights.read_bytes()).hexdigest(),
        },
        "cases": results,
    }
    (args.output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
