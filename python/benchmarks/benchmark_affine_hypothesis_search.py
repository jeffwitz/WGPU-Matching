"""Measure combined affine-Jacobian errors and image-only rotation banks.

The learned compact descriptor samples each affine support directly from the
native target image in WGSL. Synthetic truth anchors the Jacobian sensitivity
grid and scores results; the rotation-bank search never reads the true angle
or translation when building hypotheses.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from benchmark_dic_descriptor_retrieval import spatial_budget, take
from benchmark_oracle_affine_descriptor_retrieval import bilinear
from wgpu_matching import DICDetectorOptions, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair


def combined_error_grid() -> list[dict[str, float]]:
    """Return a 3^4 grid over angle, two stretches, and shear."""
    axes = (
        (-10.0, 0.0, 10.0),
        (-20.0, 0.0, 20.0),
        (-20.0, 0.0, 20.0),
        (-0.2, 0.0, 0.2),
    )
    return [
        {
            "angle_deg": angle,
            "stretch_x_percent": stretch_x,
            "stretch_y_percent": stretch_y,
            "shear": shear,
        }
        for angle, stretch_x, stretch_y, shear in itertools.product(*axes)
    ]


def perturbed_jacobian(true_jacobian: np.ndarray, condition: dict[str, float]) -> np.ndarray:
    """Apply output-frame parameter errors to a known affine Jacobian."""
    angle = np.deg2rad(condition["angle_deg"])
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    local_error = np.asarray(
        [
            [1.0 + condition["stretch_x_percent"] / 100.0, condition["shear"]],
            [0.0, 1.0 + condition["stretch_y_percent"] / 100.0],
        ],
        dtype=np.float64,
    )
    return rotation @ local_error @ np.asarray(true_jacobian, dtype=np.float64)


def rank_scores(
    scores: np.ndarray, distances: np.ndarray, *, shortlist: int = 20
) -> tuple[dict[str, int], np.ndarray]:
    """Return fixed-population retrieval metrics and the directed top-1 ids."""
    repeatable = np.any(distances <= 5.0, axis=1)
    top1 = np.argmax(scores, axis=1)
    count = min(shortlist, scores.shape[1])
    shortlist_ids = np.argpartition(-scores, count - 1, axis=1)[:, :count]
    shortlist_scores = np.take_along_axis(scores, shortlist_ids, axis=1)
    order = np.argsort(-shortlist_scores, axis=1)
    shortlist_ids = np.take_along_axis(shortlist_ids, order, axis=1)
    shortlist_distances = np.take_along_axis(distances, shortlist_ids, axis=1)
    top1_distances = distances[np.arange(len(scores)), top1]
    return (
        {
            "queries": len(scores),
            "repeatable_le5px": int(repeatable.sum()),
            "top1_correct_le2px": int(np.count_nonzero(top1_distances <= 2.0)),
            "top1_correct_le5px": int(np.count_nonzero(top1_distances <= 5.0)),
            "top5_retrieval": int(
                np.count_nonzero(np.any(shortlist_distances[:, :5] <= 5.0, axis=1) & repeatable)
            ),
            "top20_retrieval": int(
                np.count_nonzero(np.any(shortlist_distances <= 5.0, axis=1) & repeatable)
            ),
        },
        top1,
    )


def score_descriptors(
    source_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    target_valid: np.ndarray,
) -> np.ndarray:
    scores = source_descriptors @ target_descriptors.T
    scores[:, ~target_valid] = -1.0
    return scores


def map_pairs_to_truth(
    pairs: np.ndarray, source_queries: np.ndarray, mapped_queries: np.ndarray
) -> np.ndarray:
    if not len(pairs):
        return np.empty((0, 2), dtype=np.float64)
    delta = source_queries[:, None, :] - pairs[None, :, :2]
    nearest = np.argmin(np.sum(delta * delta, axis=2), axis=0)
    return mapped_queries[nearest]


def fit_consensus(
    matcher,
    pairs: np.ndarray,
    source_queries: np.ndarray,
    mapped_queries: np.ndarray,
    *,
    width: int,
    height: int,
    pair_valid: np.ndarray | None = None,
) -> dict:
    if len(pairs) < 3:
        return {"usable": False, "inliers": 0, "fit_s": 0.0, "model": None}
    center = np.asarray([[width / 2.0, height / 2.0]], dtype=np.float32)
    started = time.perf_counter()
    fitted = matcher.fit_local_affine(
        pairs,
        center,
        radius_px=float(np.hypot(width, height)),
        threshold_px=4.0,
        min_inliers=6,
        min_candidates=8,
        min_inlier_fraction=0.05,
        max_iterations=256,
        include_affine_models=True,
    )
    fit_s = time.perf_counter() - started
    match = fitted.matches[0] if fitted.matches else None
    if fitted.affine_models is None or not len(fitted.affine_models):
        return {
            "usable": False,
            "inliers": 0 if match is None else int(match.inliers),
            "candidate_count": 0 if match is None else int(match.candidates),
            "residual_px": None if match is None else float(match.residual),
            "fit_s": fit_s,
            "model": None,
        }
    model = np.asarray(fitted.affine_models[0], dtype=np.float64)
    matrix = np.asarray([[model[0], model[1]], [model[4], model[5]]])
    offset = np.asarray([model[2], model[6]])
    predicted = (source_queries - center) @ matrix.T + offset
    query_error = np.linalg.norm(predicted - mapped_queries, axis=1)
    predicted_pairs = (pairs[:, :2] - center) @ matrix.T + offset
    residual = np.linalg.norm(predicted_pairs - pairs[:, 2:4], axis=1)
    pair_truth = map_pairs_to_truth(pairs, source_queries, mapped_queries)
    truth_error = np.linalg.norm(pairs[:, 2:4] - pair_truth, axis=1)
    accepted = residual <= 4.0 if match.usable else np.zeros(len(residual), dtype=bool)
    if pair_valid is not None:
        accepted &= np.asarray(pair_valid, dtype=bool)
    angle = float(np.arctan2(matrix[1, 0] - matrix[0, 1], matrix[0, 0] + matrix[1, 1]))
    return {
        "usable": bool(match.usable),
        "inliers": int(match.inliers),
        "candidate_count": int(match.candidates),
        "inlier_fraction": float(match.inlier_fraction),
        "residual_px": float(match.residual),
        "fit_s": fit_s,
        "direct_accepted": int(accepted.sum()),
        "direct_correct_le5px": int(np.count_nonzero(accepted & (truth_error <= 5.0))),
        "direct_false_accepted": int(np.count_nonzero(accepted & (truth_error > 5.0))),
        "predicted_queries_correct_le5px": int(np.count_nonzero(query_error <= 5.0)),
        "predicted_query_count": len(query_error),
        "predicted_query_median_error_px": float(np.median(query_error)),
        "estimated_angle_rad": angle,
        "model": model.tolist(),
    }


def prepare_case(
    matcher,
    *,
    texture_seed: int,
    warp_seed: int,
    family: str,
    width: int,
    height: int,
    query_count: int,
    candidate_budget: int,
    weights_path: Path,
) -> dict:
    pair = generate_speckle_pair(texture_seed, warp_seed, width, height, motion_mode=family)
    reference = np.rint(pair.reference * 255).astype(np.uint8)
    target = np.rint(pair.target * 255).astype(np.uint8)
    options = DICDetectorOptions(max_keypoints=candidate_budget * 4, stride_px=4)
    detection_start = time.perf_counter()
    source_all = matcher.detect_dic(reference, options=options).features
    target_all = matcher.detect_dic(target, options=options).features
    detection_s = time.perf_counter() - detection_start
    margin = 32.0

    def in_image_support(features):
        xy = features.xy
        return (
            features.valid
            & (xy[:, 0] >= margin)
            & (xy[:, 1] >= margin)
            & (xy[:, 0] < width - margin)
            & (xy[:, 1] < height - margin)
        )

    source_features = spatial_budget(
        take(source_all, in_image_support(source_all)), query_count, width, height
    )
    target_features = spatial_budget(
        take(target_all, in_image_support(target_all)), candidate_budget, width, height
    )
    source_xy = source_features.xy.astype(np.float64)
    target_xy = target_features.xy.astype(np.float64)
    mapped_queries = source_xy + bilinear(pair.flow_xy, source_xy).astype(np.float64)
    distances = np.linalg.norm(mapped_queries[:, None, :] - target_xy[None, :, :], axis=2)
    repeatable = np.any(distances <= 5.0, axis=1)
    mapped_inside = (
        (mapped_queries[:, 0] >= 0.0)
        & (mapped_queries[:, 0] < width)
        & (mapped_queries[:, 1] >= 0.0)
        & (mapped_queries[:, 1] < height)
    )
    mapped_support_inside = (
        (mapped_queries[:, 0] >= margin)
        & (mapped_queries[:, 0] < width - margin)
        & (mapped_queries[:, 1] >= margin)
        & (mapped_queries[:, 1] < height - margin)
    )
    descriptor_start = time.perf_counter()
    source_described = matcher.describe_dic_cnn_keypoints(
        reference, source_features, weights_path=weights_path
    )
    source_descriptor_s = time.perf_counter() - descriptor_start
    return {
        "pair": pair,
        "reference": reference,
        "target": target,
        "source_features": source_features,
        "target_features": target_features,
        "source_xy": source_xy,
        "target_xy": target_xy,
        "mapped_queries": mapped_queries,
        "distances": distances,
        "repeatable": repeatable,
        "mapped_center_inside_image": mapped_inside,
        "mapped_64px_support_inside_image": mapped_support_inside,
        "source_described": source_described,
        "detection_s": detection_s,
        "source_descriptor_s": source_descriptor_s,
        "family": family,
        "texture_seed": texture_seed,
        "warp_seed": warp_seed,
        "image_size_px": [width, height],
    }


def evaluate_combined_case(matcher, case: dict, weights_path: Path) -> dict:
    pair = case["pair"]
    true_jacobian = np.asarray(pair.metadata["affine_matrix"], dtype=np.float64)
    conditions = []
    baseline_start = time.perf_counter()
    baseline = matcher.describe_dic_cnn_keypoints(
        case["target"],
        case["target_features"],
        weights_path=weights_path,
    )
    baseline_scores = score_descriptors(
        case["source_described"].descriptors,
        baseline.descriptors,
        baseline.features.valid,
    )
    baseline_retrieval, baseline_ids = rank_scores(baseline_scores, case["distances"])
    baseline_pairs = np.column_stack(
        (case["source_xy"], case["target_xy"][baseline_ids])
    ).astype(np.float32)
    baseline_consensus = fit_consensus(
        matcher,
        baseline_pairs,
        case["source_xy"],
        case["mapped_queries"],
        width=case["image_size_px"][0],
        height=case["image_size_px"][1],
        pair_valid=baseline.features.valid[baseline_ids],
    )
    baseline_elapsed = time.perf_counter() - baseline_start
    for parameters in combined_error_grid():
        jacobian = perturbed_jacobian(true_jacobian, parameters)
        descriptor_start = time.perf_counter()
        described = matcher.describe_dic_cnn_keypoints(
            case["target"],
            case["target_features"],
            weights_path=weights_path,
            support_jacobian=jacobian,
        )
        descriptor_s = time.perf_counter() - descriptor_start
        scores = score_descriptors(
            case["source_described"].descriptors,
            described.descriptors,
            described.features.valid,
        )
        retrieval, best_ids = rank_scores(scores, case["distances"])
        pairs = np.column_stack((case["source_xy"], case["target_xy"][best_ids])).astype(np.float32)
        consensus = fit_consensus(
            matcher,
            pairs,
            case["source_xy"],
            case["mapped_queries"],
            width=case["image_size_px"][0],
            height=case["image_size_px"][1],
            pair_valid=described.features.valid[best_ids],
        )
        conditions.append(
            {
                **parameters,
                "jacobian_determinant": float(np.linalg.det(jacobian)),
                "jacobian_condition_number": float(np.linalg.cond(jacobian)),
                "candidate_support_valid_count": int(described.features.valid.sum()),
                "retrieval": retrieval,
                "all_query_denominator": len(case["source_xy"]),
                "repeatable_query_count_fixed": int(case["repeatable"].sum()),
                "consensus": consensus,
                "descriptor_s": descriptor_s,
            }
        )
    return {
        "texture_seed": case["texture_seed"],
        "warp_seed": case["warp_seed"],
        "family": case["family"],
        "image_size_px": case["image_size_px"],
        "source_query_count": len(case["source_xy"]),
        "target_candidate_count": len(case["target_xy"]),
        "repeatable_query_count_fixed": int(case["repeatable"].sum()),
        "mapped_centers_inside_image": int(case["mapped_center_inside_image"].sum()),
        "mapped_64px_supports_inside_image": int(
            case["mapped_64px_support_inside_image"].sum()
        ),
        "visible_but_without_target_candidate_le5px": int(
            np.count_nonzero(case["mapped_center_inside_image"] & ~case["repeatable"])
        ),
        "out_of_image_target_centers": int(np.count_nonzero(~case["mapped_center_inside_image"])),
        "detection_s": case["detection_s"],
        "source_descriptor_s": case["source_descriptor_s"],
        "uncompensated_baseline": {
            "retrieval": baseline_retrieval,
            "consensus": baseline_consensus,
            "target_descriptor_and_evaluation_s": baseline_elapsed,
        },
        "true_jacobian": true_jacobian.tolist(),
        "true_rotation_rad": float(pair.metadata["angle_rad"]),
        "conditions": conditions,
    }


def angular_error(estimate: float, truth: float) -> float:
    return float((estimate - truth + np.pi) % (2.0 * np.pi) - np.pi)


def evaluate_rotation_case(matcher, case: dict, weights_path: Path) -> dict:
    """Test a 72-angle bank; report nested uniform 18- and 36-angle subsets."""
    angles = np.deg2rad(np.arange(-180.0, 180.0, 5.0))
    q_count = len(case["source_xy"])
    c_count = len(case["target_xy"])
    bank_scores = {
        18: np.full((q_count, c_count), -1.0, dtype=np.float32),
        36: np.full((q_count, c_count), -1.0, dtype=np.float32),
        72: np.full((q_count, c_count), -1.0, dtype=np.float32),
    }
    rows = []
    width, height = case["image_size_px"]
    center = np.asarray([[width / 2.0, height / 2.0]], dtype=np.float32)
    true_angle = float(case["pair"].metadata["angle_rad"])
    baseline_target = matcher.describe_dic_cnn_keypoints(
        case["target"], case["target_features"], weights_path=weights_path
    )
    baseline_scores = score_descriptors(
        case["source_described"].descriptors,
        baseline_target.descriptors,
        baseline_target.features.valid,
    )
    baseline_retrieval, _ = rank_scores(baseline_scores, case["distances"])
    for index, angle in enumerate(angles):
        jacobian = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        total_start = time.perf_counter()
        descriptor_start = time.perf_counter()
        target_described = matcher.describe_dic_cnn_keypoints(
            case["target"],
            case["target_features"],
            weights_path=weights_path,
            support_jacobian=jacobian,
        )
        descriptor_s = time.perf_counter() - descriptor_start
        scores = score_descriptors(
            case["source_described"].descriptors,
            target_described.descriptors,
            target_described.features.valid,
        )
        for size, divisor in ((18, 4), (36, 2), (72, 1)):
            if index % divisor == 0:
                np.maximum(bank_scores[size], scores, out=bank_scores[size])
        match_start = time.perf_counter()
        reciprocal = matcher.match_features(
            case["source_described"],
            target_described,
            center,
            radius_px=float(np.hypot(width, height)),
            threshold_px=4.0,
            min_inliers=6,
            min_candidates=8,
            min_inlier_fraction=0.05,
            max_iterations=256,
            fit_local_models=False,
        ).correspondences
        matching_s = time.perf_counter() - match_start
        pairs = reciprocal.points
        fit_start = time.perf_counter()
        consensus = fit_consensus(
            matcher,
            pairs,
            case["source_xy"][reciprocal.reference_indices],
            case["mapped_queries"][reciprocal.reference_indices],
            width=width,
            height=height,
        )
        fit_s = time.perf_counter() - fit_start
        rows.append(
            {
                "angle_hypothesis_deg": float(np.rad2deg(angle)),
                "angle_index": index,
                "descriptor_s": descriptor_s,
                "matching_s": matching_s,
                "consensus_s": fit_s,
                "total_s": time.perf_counter() - total_start,
                "reciprocal_match_count": len(pairs),
                "consensus": consensus,
            }
        )
    banks = {}
    for size, divisor in ((18, 4), (36, 2), (72, 1)):
        retrieval, _ = rank_scores(bank_scores[size], case["distances"])
        eligible = [
            row for row in rows
            if row["angle_index"] % divisor == 0 and row["consensus"].get("usable", False)
        ]
        selected = max(
            eligible,
            key=lambda row: (
                row["consensus"]["inliers"],
                -row["consensus"].get("residual_px", float("inf")),
                -row["angle_index"],
            ),
            default=None,
        )
        selected_model = None if selected is None else selected["consensus"]["model"]
        output = {"usable": selected is not None}
        if selected is not None:
            model = np.asarray(selected_model, dtype=np.float64)
            matrix = np.asarray([[model[0], model[1]], [model[4], model[5]]])
            model_offset = np.asarray([model[2], model[6]])
            predicted = (case["source_xy"] - center) @ matrix.T + model_offset
            error = np.linalg.norm(predicted - case["mapped_queries"], axis=1)
            estimated_angle = selected["consensus"]["estimated_angle_rad"]
            output.update(selected["consensus"])
            output.update(
                {
                    "selected_bank_angle_deg": selected["angle_hypothesis_deg"],
                    "estimated_rotation_error_deg": float(
                        np.rad2deg(angular_error(estimated_angle, true_angle))
                    ),
                    "selected_angle_offset_error_deg": float(
                        np.rad2deg(
                            angular_error(np.deg2rad(selected["angle_hypothesis_deg"]), true_angle)
                        )
                    ),
                    "predicted_all_queries_correct_le5px": int(np.count_nonzero(error <= 5.0)),
                    "predicted_all_queries": len(error),
                    "predicted_all_queries_median_error_px": float(np.median(error)),
                }
            )
        banks[str(size)] = {
            "hypothesis_count": size,
            "angular_spacing_deg": 360.0 / size,
            "retrieval": retrieval,
            "selected_consensus": output,
            "total_bank_s": float(
                sum(row["total_s"] for row in rows if row["angle_index"] % divisor == 0)
            ),
        }
    true_jacobian = np.asarray(case["pair"].metadata["affine_matrix"], dtype=np.float64)
    oracle_target = matcher.describe_dic_cnn_keypoints(
        case["target"],
        case["target_features"],
        weights_path=weights_path,
        support_jacobian=true_jacobian,
    )
    oracle_scores = score_descriptors(
        case["source_described"].descriptors,
        oracle_target.descriptors,
        oracle_target.features.valid,
    )
    oracle_retrieval, _ = rank_scores(oracle_scores, case["distances"])
    return {
        "texture_seed": case["texture_seed"],
        "warp_seed": case["warp_seed"],
        "family": case["family"],
        "image_size_px": case["image_size_px"],
        "source_query_count": len(case["source_xy"]),
        "target_candidate_count": len(case["target_xy"]),
        "repeatable_query_count_fixed": int(case["repeatable"].sum()),
        "true_rotation_deg": float(np.rad2deg(true_angle)),
        "detection_s": case["detection_s"],
        "source_descriptor_s": case["source_descriptor_s"],
        "uncompensated_baseline": baseline_retrieval,
        "exact_jacobian_oracle_ceiling": oracle_retrieval,
        "bank_results": banks,
        "hypotheses": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("combined", "rotation", "both"), default="both")
    parser.add_argument("--cnn-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--candidate-budget", type=int, default=8192)
    parser.add_argument("--query-budget", type=int, default=512)
    parser.add_argument("--textures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    if not args.cnn_weights.is_file():
        parser.error("CNN descriptor weights must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined, rotation = [], []
    with WGPUMatching(device_type="gpu") as matcher:
        print(f"WebGPU adapter: {matcher.device_name}", flush=True)
        jobs = []
        if args.mode in ("combined", "both"):
            jobs.append(("affine", "combined"))
        if args.mode in ("rotation", "both"):
            jobs.append(("rigid", "rotation"))
        for family_index, (family, experiment) in enumerate(jobs):
            for texture_index in range(args.textures):
                texture_seed = args.seed + family_index * 100_003 + texture_index * 10_007
                warp_seed = args.seed + 5_000_009 + family_index * 100_003 + texture_index * 10_007
                case = prepare_case(
                    matcher,
                    texture_seed=texture_seed,
                    warp_seed=warp_seed,
                    family=family,
                    width=args.width,
                    height=args.height,
                    query_count=args.query_budget,
                    candidate_budget=args.candidate_budget,
                    weights_path=args.cnn_weights,
                )
                result = (
                    evaluate_combined_case(matcher, case, args.cnn_weights)
                    if experiment == "combined"
                    else evaluate_rotation_case(matcher, case, args.cnn_weights)
                )
                (combined if experiment == "combined" else rotation).append(result)
                print(
                    f"{experiment} {texture_index + 1}/{args.textures}: "
                    f"queries={result['source_query_count']} candidates={result['target_candidate_count']}",
                    flush=True,
                )
    payload = {
        "protocol": {
            "purpose": "combined Jacobian sensitivity and image-derived rotation-bank diagnostic",
            "image_size_px": [args.width, args.height],
            "candidate_budget": args.candidate_budget,
            "query_budget": args.query_budget,
            "textures_per_family": args.textures,
            "detector": "WGSL structure-tensor score with deterministic spatial quotas; no learned detection weights passed",
            "descriptor": "learned compact 64-value DIC CNN; affine support sampled directly from the native target image in WGSL",
            "affine_error_parameterization": "F_hat = R(delta_theta) @ [[1+delta_x, delta_shear], [0,1+delta_y]] @ F_true",
            "combined_error_levels": {"angle_deg": [-10, 0, 10], "stretch_x_percent": [-20, 0, 20], "stretch_y_percent": [-20, 0, 20], "shear": [-0.2, 0, 0.2]},
            "rotation_bank_sizes": [18, 36, 72],
            "rotation_bank_grid_deg": "-180 to +175 degrees in 5-degree increments; nested uniform subsets provide 18 and 36 hypotheses",
            "rotation_bank_selection": "largest usable WGSL affine-consensus inlier count, then lowest residual; no analytic truth used for selection",
            "candidate_policy": "fixed detector-selected candidate coordinates; no analytic visibility masks; the repeatability denominator remains fixed across support hypotheses",
            "consensus_thresholds": {"residual_px": 4, "correct_error_px": 5, "minimum_inliers": 6, "minimum_candidates": 8, "minimum_inlier_fraction": 0.05},
            "webgpu_adapter": matcher.device_name,
            "cnn_weights_sha256": hashlib.sha256(args.cnn_weights.read_bytes()).hexdigest(),
            "residency_limit": "WGSL descriptor extraction reads native pixels; descriptors are read to NumPy and uploaded for the existing WGSL mutual matcher, and retrieval diagnostic ranking uses NumPy",
        },
        "combined": combined,
        "rotation": rotation,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
