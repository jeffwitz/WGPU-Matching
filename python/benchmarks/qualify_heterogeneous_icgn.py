"""Qualify learned-feature affine/RBF fields by seeded native WebGPU IC-GN."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from geometric_consensus import LocalConsensus
from PIL import Image
from qualify_all_initializers import Extractor
from train_dic_cnn_descriptor import bilinear, support_is_valid, target_support_is_valid
from wgpu_matching import (
    CorrespondenceSet,
    DeDoDeBOnnxWGPUDescriptor,
    DICDetectorOptions,
    FeatureSet,
    ProjectionOptions,
    WGPUMatching,
)
from wgpu_matching.dic_synthetic import generate_speckle_pair


def grid_centres(width: int, height: int, subset_size: int, stride: int) -> np.ndarray:
    x = (subset_size - 1) / 2 + stride * np.arange(1 + (width - subset_size) // stride)
    y = (subset_size - 1) / 2 + stride * np.arange(1 + (height - subset_size) // stride)
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2).astype(np.float32)


def analytic_parameters(pair, points: np.ndarray) -> np.ndarray:
    """Estimate exact local affine parameters from the generator's analytic flow grid."""
    flow = pair.flow_xy.astype(np.float64)
    du_dy, du_dx = np.gradient(flow[..., 0])
    dv_dy, dv_dx = np.gradient(flow[..., 1])
    x = np.clip(np.rint(points[:, 0]).astype(int), 1, flow.shape[1] - 2)
    y = np.clip(np.rint(points[:, 1]).astype(int), 1, flow.shape[0] - 2)
    return np.column_stack(
        (
            flow[y, x, 0],
            du_dx[y, x],
            du_dy[y, x],
            flow[y, x, 1],
            dv_dx[y, x],
            dv_dy[y, x],
        )
    ).astype(np.float32)


def seed_state(parameters: np.ndarray, active: np.ndarray) -> np.ndarray:
    state = np.zeros((len(parameters), 8), dtype=np.float32)
    state[:, :6] = parameters
    state[:, 6] = active  # Valid per-subset warp seeds for the existing IC-GN kernel.
    return state


def json_safe(value):
    """Convert NumPy values and non-finite diagnostics to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def score_result(result, truth: np.ndarray, visible: np.ndarray) -> dict:
    status = np.asarray(result.status).reshape(-1)
    measured = np.asarray(result.parameters).reshape(-1, 6)
    usable = (status > 0.5) & visible
    displacement_error = np.linalg.norm(measured[:, [0, 3]] - truth[:, [0, 3]], axis=1)
    gradient_error = np.linalg.norm(measured[:, [1, 2, 4, 5]] - truth[:, [1, 2, 4, 5]], axis=1)
    correct = usable & (displacement_error < 2.0)
    return {
        "requested_windows": len(status),
        "visible_windows": int(visible.sum()),
        "solver_accepted": int(usable.sum()),
        "correct_converged_lt2px": int(correct.sum()),
        "coverage_correct": float(correct.sum() / max(int(visible.sum()), 1)),
        "accepted_displacement_rmse_px": float(np.sqrt(np.mean(displacement_error[usable] ** 2)))
        if usable.any()
        else None,
        "accepted_displacement_median_px": float(np.median(displacement_error[usable]))
        if usable.any()
        else None,
        "accepted_affine_gradient_rmse": float(np.sqrt(np.mean(gradient_error[usable] ** 2)))
        if usable.any()
        else None,
        "accepted_mean_iterations": float(
            np.mean(np.asarray(result.iterations).reshape(-1)[usable])
        )
        if usable.any()
        else None,
        "solver_time_s": float(result.timing.get("totalWallMs", 0.0) / 1000.0),
        "correct_windows_per_second": float(
            correct.sum() / max(result.timing.get("totalWallMs", 0.0) / 1000.0, 1e-12)
        ),
    }


def diagnostic_correspondence_scores(
    reference, target, true_target_descriptors, source_xy, target_xy, pair, centers
):
    """Report descriptor ranking and observed-point repeatability before filtering."""
    source_xy = np.asarray(source_xy, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    height, width = pair.visible.shape
    ix = np.clip(np.rint(source_xy[:, 0]).astype(int), 0, width - 1)
    iy = np.clip(np.rint(source_xy[:, 1]).astype(int), 0, height - 1)
    expected_xy = source_xy + pair.flow_xy[iy, ix]
    visible = pair.visible[iy, ix].astype(bool)
    ex = np.clip(np.rint(expected_xy[:, 0]).astype(int), 0, width - 1)
    ey = np.clip(np.rint(expected_xy[:, 1]).astype(int), 0, height - 1)
    if pair.target_visible is not None:
        visible &= pair.target_visible[ey, ex]
    if not len(source_xy) or not len(target_xy):
        return {"source_points": len(source_xy), "target_points": len(target_xy)}, np.empty((0,))

    source_descriptor = reference.descriptors.astype(np.float32, copy=False)
    target_descriptor = target.descriptors.astype(np.float32, copy=False)
    source_support_valid = support_is_valid(pair, source_xy)
    target_support_valid = target_support_is_valid(pair, expected_xy)
    fully_valid_support = visible & source_support_valid & target_support_valid
    target_candidate_support_valid = target_support_is_valid(pair, target_xy)
    similarities = source_descriptor @ target_descriptor.T
    top = np.argmax(similarities, axis=1)
    top_score = similarities[np.arange(len(top)), top]
    if similarities.shape[1] > 1:
        second_score = np.partition(similarities, -2, axis=1)[:, -2]
    else:
        second_score = np.full(len(top), -1.0, dtype=np.float32)
    reciprocal = np.argmax(similarities, axis=0)[top] == np.arange(len(top))
    match_error = np.linalg.norm(target_xy[top] - expected_xy, axis=1)
    target_distances = np.linalg.norm(expected_xy[:, None, :] - target_xy[None, :, :], axis=2)
    repeatability_error = target_distances.min(axis=1)
    expected_candidate = target_distances.argmin(axis=1)
    best_is_closest_geometric = top == expected_candidate
    best_cosine = np.clip(top_score, -1.0, 1.0)
    exact_positive = np.sum(source_descriptor * true_target_descriptors, axis=1)
    far_negative = target_distances >= 48.0
    hard_negative = np.max(np.where(far_negative, similarities, -1.0), axis=1)
    exact_rank = 1 + np.sum(far_negative & (similarities >= exact_positive[:, None]), axis=1)
    window_index = np.argmin(
        np.linalg.norm(source_xy[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    correct5 = visible & (match_error < 5.0)
    correct5_full_support = (
        fully_valid_support & target_candidate_support_valid[top] & (match_error < 5.0)
    )
    coverage = np.zeros(len(centers), dtype=bool)
    if correct5.any():
        coverage[np.unique(window_index[correct5])] = True
    second_margin = top_score - second_score
    metrics = {
        "source_points": len(source_xy),
        "target_points": len(target_xy),
        "visible_source_points": int(visible.sum()),
        "fully_valid_source_and_mapped_target_support_points": int(fully_valid_support.sum()),
        "top1_correct_lt5px_with_valid_support": int(correct5_full_support.sum()),
        "target_detection_repeatability_le2px": int(
            np.count_nonzero(visible & (repeatability_error <= 2))
        ),
        "target_detection_repeatability_le5px": int(
            np.count_nonzero(visible & (repeatability_error <= 5))
        ),
        "descriptor_top1_error_le2px": int(np.count_nonzero(visible & (match_error <= 2))),
        "descriptor_top1_error_lt5px": int(np.count_nonzero(correct5)),
        "descriptor_top1_reciprocal": int(np.count_nonzero(visible & reciprocal)),
        "reciprocal_false_positive_ge5px": int(
            np.count_nonzero(visible & reciprocal & (match_error >= 5))
        ),
        "correct_direct_windows_lt5px": int(coverage.sum()),
        "requested_windows": len(centers),
        "best_cosine_correct_lt5_mean": float(best_cosine[correct5].mean())
        if correct5.any()
        else None,
        "best_cosine_incorrect_mean": float(best_cosine[visible & ~correct5].mean())
        if np.any(visible & ~correct5)
        else None,
        "best_second_margin_median": float(np.median(second_margin[visible]))
        if visible.any()
        else None,
        "exact_correspondence_cosine_median": float(np.median(exact_positive[visible]))
        if visible.any()
        else None,
        "same_texture_hard_negative_cosine_median": float(np.median(hard_negative[visible]))
        if visible.any()
        else None,
        "exact_positive_beats_all_far_detected_negatives": int(
            np.count_nonzero(visible & (exact_rank == 1))
        ),
        "geometric_nearest_is_descriptor_top1": int(
            np.count_nonzero(visible & best_is_closest_geometric)
        ),
        "correct_matches_per_window_max": int(
            np.bincount(window_index[correct5], minlength=len(centers)).max()
        )
        if correct5.any()
        else 0,
        "top1_error_px_median": float(np.median(match_error[visible])) if visible.any() else None,
        "top1_error_px_p95": float(np.percentile(match_error[visible], 95))
        if visible.any()
        else None,
    }
    return metrics, match_error


def run_case(
    matcher,
    descriptor,
    consensus,
    solver,
    mode,
    texture_seed,
    warp_seed,
    output,
    size,
    weights,
    reference_preparation_s,
    descriptor_variant="dedode_b",
    compact_descriptor_weights=None,
    cnn_descriptor_weights=None,
):
    width, height = size
    pair = generate_speckle_pair(texture_seed, warp_seed, width, height, motion_mode=mode)
    reference = np.rint(pair.reference * 255).astype(np.uint8)
    target = np.rint(pair.target * 255).astype(np.uint8)
    case_dir = output / f"{mode}_{texture_seed}_{warp_seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reference).save(case_dir / "reference.png")
    Image.fromarray(target).save(case_dir / "target.png")
    (case_dir / "analytic_truth.json").write_text(
        json.dumps(json_safe(pair.metadata), indent=2) + "\n", encoding="utf-8"
    )
    points = grid_centres(width, height, 64, 64)
    visible = pair.visible[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
    mapped_centres = points + bilinear(pair.flow_xy, points).astype(np.float32)
    fully_valid_support_windows = (
        visible
        & support_is_valid(pair, points)
        & target_support_is_valid(pair, mapped_centres)
    )

    detection_start = time.perf_counter()
    detected = [
        matcher.detect_dic(
            image,
            options=DICDetectorOptions(max_keypoints=2048),
            learned_weights_path=weights,
        ).features
        for image in (reference, target)
    ]
    detection_s = time.perf_counter() - detection_start
    description_start = time.perf_counter()
    if descriptor_variant == "dedode_b":
        described = [
            descriptor.describe(image, features)
            for image, features in zip((reference, target), detected, strict=True)
        ]
        descriptor_name = "DeDoDe-B ONNX/WebGPU"
    elif descriptor_variant == "horg64":
        described = [
            matcher.describe_horg_keypoints(image, features, reduction=4)
            for image, features in zip((reference, target), detected, strict=True)
        ]
        descriptor_name = "HORG-64 fixed-scale WGSL"
    elif descriptor_variant == "dic_descriptor":
        if compact_descriptor_weights is None:
            raise ValueError("Trained compact descriptor weights are required")
        described = [
            matcher.describe_dic_keypoints(image, features, weights_path=compact_descriptor_weights)
            for image, features in zip((reference, target), detected, strict=True)
        ]
        descriptor_name = "trained compact DIC-Descriptor WGSL"
    elif descriptor_variant == "dic_cnn_descriptor":
        if cnn_descriptor_weights is None:
            raise ValueError("Trained convolutional descriptor weights are required")
        described = [
            matcher.describe_dic_cnn_keypoints(image, features, weights_path=cnn_descriptor_weights)
            for image, features in zip((reference, target), detected, strict=True)
        ]
        descriptor_name = "trained patch CNN DIC-Descriptor WGSL"
    else:
        raise ValueError(f"Unknown descriptor variant: {descriptor_variant}")
    description_s = time.perf_counter() - description_start
    source_xy = detected[0].xy
    source_x = np.clip(np.rint(source_xy[:, 0]).astype(int), 0, width - 1)
    source_y = np.clip(np.rint(source_xy[:, 1]).astype(int), 0, height - 1)
    exact_target_xy = source_xy + pair.flow_xy[source_y, source_x]
    exact_target_features = FeatureSet(
        exact_target_xy, np.ones(len(exact_target_xy), dtype=np.float32)
    )
    exact_description_start = time.perf_counter()
    if descriptor_variant == "dedode_b":
        exact_target_described = descriptor.describe(target, exact_target_features)
    elif descriptor_variant == "dic_descriptor":
        exact_target_described = matcher.describe_dic_keypoints(
            target, exact_target_features, weights_path=compact_descriptor_weights
        )
    elif descriptor_variant == "dic_cnn_descriptor":
        exact_target_described = matcher.describe_dic_cnn_keypoints(
            target, exact_target_features, weights_path=cnn_descriptor_weights
        )
    else:
        exact_target_described = matcher.describe_horg_keypoints(
            target, exact_target_features, reduction=4
        )
    exact_description_s = time.perf_counter() - exact_description_start
    diagnostic_metrics, diagnostic_error = diagnostic_correspondence_scores(
        described[0],
        described[1],
        exact_target_described.descriptors,
        detected[0].xy,
        detected[1].xy,
        pair,
        points,
    )
    matching_start = time.perf_counter()
    matched = matcher.match_features(
        described[0], described[1], points, min_cosine_similarity=0.0, fit_local_models=False
    )
    feature_pairs = matched.correspondences
    points_xyxy = feature_pairs.points
    pairs = CorrespondenceSet(
        source_xy=points_xyxy[:, :2],
        target_xy=points_xyxy[:, 2:],
        image_size=(width, height),
        matcher_id=f"DIC-Detector + {descriptor_name} + mutual WGSL cosine",
        accepted=np.ones(len(points_xyxy), dtype=bool),
        raw_scores=feature_pairs.cosine_similarity,
        rejection_reasons=(None,) * len(points_xyxy),
    )
    raw_matches = pairs.pairs_xyxy.copy()
    raw_source = raw_matches[:, :2]
    raw_x = np.clip(np.rint(raw_source[:, 0]).astype(int), 0, width - 1)
    raw_y = np.clip(np.rint(raw_source[:, 1]).astype(int), 0, height - 1)
    raw_error = np.linalg.norm(
        raw_matches[:, 2:] - (raw_source + pair.flow_xy[raw_y, raw_x]), axis=1
    )
    pairs, consensus_diagnostics = consensus.filter(pairs, points)
    matching_s = time.perf_counter() - matching_start
    pair_error = np.linalg.norm(
        pairs.target_xy
        - (
            pairs.source_xy
            + pair.flow_xy[
                np.clip(np.rint(pairs.source_xy[:, 1]).astype(int), 0, height - 1),
                np.clip(np.rint(pairs.source_xy[:, 0]).astype(int), 0, width - 1),
            ]
        ),
        axis=1,
    )

    measurements = {
        "case": mode,
        "pipeline": f"dic_detector_{descriptor_variant}",
        "texture_seed": texture_seed,
        "warp_seed": warp_seed,
        "image_size": [width, height],
        "direct_matches": len(raw_matches),
        "direct_matches_accepted": int(pairs.accepted.sum()),
        "direct_matches_correct_le2px": int(np.count_nonzero((pair_error <= 2) & pairs.accepted)),
        "direct_matches_correct_lt5px": int(np.count_nonzero((pair_error < 5) & pairs.accepted)),
        "fully_valid_support_windows": int(fully_valid_support_windows.sum()),
        "raw_direct_match_error_px": {
            "median": float(np.median(raw_error)) if len(raw_error) else None,
            "p95": float(np.percentile(raw_error, 95)) if len(raw_error) else None,
            "correct_le2px": int(np.count_nonzero(raw_error <= 2)),
            "correct_lt5px": int(np.count_nonzero(raw_error < 5)),
            "count": len(raw_error),
        },
        "correspondence_digest": hashlib.sha256(pairs.pairs_xyxy.tobytes()).hexdigest(),
        "detection_pair_s": detection_s,
        "description_pair_s": description_s,
        "exact_coordinate_description_s": exact_description_s,
        "matching_and_consensus_s": matching_s,
        "reference_preparation_s": reference_preparation_s,
        "detected_keypoints": [len(features.xy) for features in detected],
        "consensus": consensus_diagnostics,
        "descriptor_diagnostics": diagnostic_metrics,
    }
    np.savez_compressed(
        case_dir / f"{descriptor_variant}_descriptor_diagnostics.npz",
        source_keypoints_xy=detected[0].xy,
        target_keypoints_xy=detected[1].xy,
        analytically_expected_target_xy=exact_target_xy,
        raw_reciprocal_pairs_xyxy=raw_matches,
        raw_reciprocal_errors_px=raw_error,
        raw_reciprocal_scores=feature_pairs.cosine_similarity,
        consensus_accepted=pairs.accepted,
        analytical_top1_errors_px=diagnostic_error,
    )
    kept = np.asarray(pairs.accepted, dtype=bool)
    accepted_pairs = type(pairs)(
        source_xy=pairs.source_xy[kept],
        target_xy=pairs.target_xy[kept],
        image_size=pairs.image_size,
        matcher_id=pairs.matcher_id,
    )
    truth_parameters = analytic_parameters(pair, points)
    zero_active = visible.astype(np.uint32)
    zero_start = time.perf_counter()
    zero_result = solver._register_seeded_frame(
        target, seed_state(np.zeros_like(truth_parameters), zero_active), zero_active
    )
    zero_wall_s = time.perf_counter() - zero_start
    oracle_start = time.perf_counter()
    oracle_result = solver._register_seeded_frame(
        target, seed_state(truth_parameters, zero_active), zero_active
    )
    oracle_wall_s = time.perf_counter() - oracle_start
    measurements["zero_initialization"] = score_result(zero_result, truth_parameters, visible)
    measurements["analytic_local_affine_oracle"] = score_result(
        oracle_result, truth_parameters, visible
    )
    measurements["control_solver_wall_s"] = {
        "zero": zero_wall_s,
        "analytic_local_affine_oracle": oracle_wall_s,
    }
    for basis in ("affine", "gaussian_rbf"):
        try:
            fit_start = time.perf_counter()
            field = matcher.fit_initialization_field(
                accepted_pairs, (width, height), options=ProjectionOptions(basis=basis)
            )
            field_fit_s = time.perf_counter() - fit_start
            projection_start = time.perf_counter()
            initial = field.initialize_patches(points, patch_size=64, projection="support_fit")
            projection_s = time.perf_counter() - projection_start
            valid_initial = np.asarray([p.parameters is not None for p in initial], dtype=bool)
            projected_parameters = np.asarray(
                [p.parameters if p.parameters is not None else (0.0,) * 6 for p in initial],
                dtype=np.float32,
            )
            # Projection API order is dx,dy,du/dx,dv/dx,du/dy,dv/dy;
            # live-dic's WGSL solver order is dx,du/dx,du/dy,dy,dv/dx,dv/dy.
            initial_parameters = projected_parameters[:, [0, 2, 4, 1, 3, 5]]
            model_truth_error = initial_parameters - truth_parameters
            if valid_initial.any():
                init_translation_rmse = float(
                    np.sqrt(
                        np.mean(np.sum(model_truth_error[valid_initial][:, [0, 3]] ** 2, axis=1))
                    )
                )
                init_gradient_rmse = float(
                    np.sqrt(np.mean(model_truth_error[valid_initial][:, [1, 2, 4, 5]] ** 2))
                )
            else:
                init_translation_rmse = init_gradient_rmse = None
            active = (valid_initial & visible).astype(np.uint32)
            state = seed_state(initial_parameters, active)
            run_start = time.perf_counter()
            result = solver._register_seeded_frame(target, state, active)
            elapsed = time.perf_counter() - run_start
            full_pipeline_s = (
                reference_preparation_s
                + detection_s
                + description_s
                + matching_s
                + field_fit_s
                + projection_s
                + elapsed
            )
            solver_metrics = score_result(result, truth_parameters, visible)
            full_support_solver_metrics = score_result(
                result, truth_parameters, fully_valid_support_windows
            )
            measurements[basis] = {
                "effective_basis": field.effective_basis,
                "accepted_direct_matches": int(pairs.accepted.sum()),
                "valid_patch_initializations": int(valid_initial.sum()),
                "initial_translation_rmse_px": init_translation_rmse,
                "initial_affine_gradient_rmse": init_gradient_rmse,
                "field_fit_s": field_fit_s,
                "patch_projection_s": projection_s,
                "solver": solver_metrics,
                "solver_on_fully_valid_support_windows": full_support_solver_metrics,
                "solver_wall_s_including_python_readback": elapsed,
                "full_pipeline_s": full_pipeline_s,
                "correct_windows_per_second_end_to_end": float(
                    solver_metrics["correct_converged_lt2px"] / max(full_pipeline_s, 1e-12)
                ),
                "parameters_by_window": np.asarray(result.parameters).reshape(-1, 6).tolist(),
                "status_by_window": np.asarray(result.status).reshape(-1).tolist(),
            }
            np.savez_compressed(
                case_dir / f"dic_detector_{descriptor_variant}_{basis}_pairs.npz",
                raw_pairs_xyxy=raw_matches,
                raw_analytic_error_px=raw_error,
                accepted_after_local_consensus=pairs.accepted,
                projected_initial_parameters=initial_parameters,
                analytic_parameters=truth_parameters,
                solver_parameters=np.asarray(result.parameters).reshape(-1, 6),
                solver_status=np.asarray(result.status).reshape(-1),
                solver_iterations=np.asarray(result.iterations).reshape(-1),
                points_xy=points,
                visible=visible,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            measurements[basis] = {"error": str(error)}
    output_path = case_dir / f"dic_detector_{descriptor_variant}.json"
    output_path.write_text(json.dumps(json_safe(measurements), indent=2, allow_nan=False) + "\n")
    print(
        f"{mode} seeds {texture_seed}/{warp_seed}: {len(raw_matches)} raw direct matches, "
        f"{int(pairs.accepted.sum())} retained after local consensus for "
        f"{descriptor_variant}; saved {output_path}",
        flush=True,
    )
    return measurements


def run_baseline_case(
    extractor, solver, method, reference, target, pair, points, output, reference_preparation_s
):
    """Run one established matcher through the same local filter, field fit and IC-GN."""
    height, width = reference.shape
    visible = pair.visible[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
    started = time.perf_counter()
    pairs, extraction = extractor.extract(method, reference, target, points)
    extraction_s = time.perf_counter() - started
    source = pairs.source_xy
    sx = np.clip(np.rint(source[:, 0]).astype(int), 0, width - 1)
    sy = np.clip(np.rint(source[:, 1]).astype(int), 0, height - 1)
    errors = np.linalg.norm(pairs.target_xy - (source + pair.flow_xy[sy, sx]), axis=1)
    selected = np.asarray(pairs.accepted, dtype=bool)
    accepted = CorrespondenceSet(
        source_xy=source[selected],
        target_xy=pairs.target_xy[selected],
        image_size=(width, height),
        matcher_id=pairs.matcher_id,
    )
    truth_parameters = analytic_parameters(pair, points)
    result = {
        "method": method,
        "direct_matches": len(source),
        "direct_matches_correct_lt5px": int(np.count_nonzero(errors < 5)),
        "accepted_after_consensus": int(selected.sum()),
        "accepted_correct_lt5px": int(np.count_nonzero((errors < 5) & selected)),
        "raw_error_median_px": float(np.median(errors)) if len(errors) else None,
        "raw_error_p95_px": float(np.percentile(errors, 95)) if len(errors) else None,
        "extraction_s": extraction_s,
        "reference_preparation_s": reference_preparation_s,
        "extraction_stages": extraction.get("stages_s", {}),
        "basis_results": {},
    }
    for basis in ("affine", "gaussian_rbf"):
        try:
            fit_started = time.perf_counter()
            field = extractor.matcher.fit_initialization_field(
                accepted, (width, height), options=ProjectionOptions(basis=basis)
            )
            field_fit_s = time.perf_counter() - fit_started
            projection_started = time.perf_counter()
            patches = field.initialize_patches(points, patch_size=64, projection="support_fit")
            projection_s = time.perf_counter() - projection_started
            valid = np.asarray([patch.parameters is not None for patch in patches])
            projected = np.asarray(
                [patch.parameters or (0.0,) * 6 for patch in patches], dtype=np.float32
            )[:, [0, 2, 4, 1, 3, 5]]
            active = (valid & visible).astype(np.uint32)
            solver_started = time.perf_counter()
            icgn_result = solver._register_seeded_frame(
                target, seed_state(projected, active), active
            )
            solver_s = time.perf_counter() - solver_started
            metrics = score_result(icgn_result, truth_parameters, visible)
            end_to_end_s = (
                reference_preparation_s + extraction_s + field_fit_s + projection_s + solver_s
            )
            result["basis_results"][basis] = {
                "effective_basis": field.effective_basis,
                "valid_patch_initializations": int(valid.sum()),
                "solver": metrics,
                "field_fit_s": field_fit_s,
                "patch_projection_s": projection_s,
                "solver_wall_s": solver_s,
                "end_to_end_s": end_to_end_s,
                "correct_windows_per_second_end_to_end": metrics["correct_converged_lt2px"]
                / max(end_to_end_s, 1e-12),
            }
            np.savez_compressed(
                output / f"{method}_{basis}_pairs.npz",
                pairs_xyxy=pairs.pairs_xyxy,
                analytic_error_px=errors,
                accepted=selected,
                initial_parameters=projected,
                analytic_parameters=truth_parameters,
                solver_parameters=np.asarray(icgn_result.parameters).reshape(-1, 6),
                solver_status=np.asarray(icgn_result.status).reshape(-1),
                solver_iterations=np.asarray(icgn_result.iterations).reshape(-1),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            result["basis_results"][basis] = {"error": str(error)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor-onnx", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--compact-descriptor", type=Path)
    parser.add_argument("--cnn-descriptor", type=Path)
    parser.add_argument(
        "--independent-eval-per-family",
        type=int,
        default=0,
        help="Append this many new, deterministically seeded fields for each synthetic motion family",
    )
    parser.add_argument(
        "--live-dic-source",
        type=Path,
        default=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src"),
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument(
        "--efficientloftr-onnx",
        type=Path,
        default=Path("/tmp/wgpu-matching-aliked-study/eloftr_border_mask_640x384.onnx"),
    )
    parser.add_argument(
        "--efficientloftr-dynamic-onnx",
        type=Path,
        default=Path("/tmp/wgpu-matching-aliked-study/eloftr_outdoor_opt.onnx"),
    )
    args = parser.parse_args()
    if not args.live_dic_source.is_dir():
        parser.error(f"IC-GN source is missing: {args.live_dic_source}")
    if args.independent_eval_per_family < 0:
        parser.error("--independent-eval-per-family must be non-negative")
    sys.path.insert(0, str(args.live_dic_source))
    from live_dic import ICGNWGPU

    args.output.mkdir(parents=True, exist_ok=True)
    for model_path in (
        args.descriptor_onnx,
        args.efficientloftr_onnx,
        args.efficientloftr_dynamic_onnx,
    ):
        if not model_path.is_file():
            parser.error(f"Required ONNX model is missing: {model_path}")
    with WGPUMatching(device_type="gpu") as matcher:
        descriptor = DeDoDeBOnnxWGPUDescriptor(args.descriptor_onnx, max_side=1024, capacity=8192)
        solver = ICGNWGPU(
            device_type="gpu",
            prefer_subgroups=False,
            subset_size=64,
            subset_stride=64,
            interpolation="linear",
            motion_model="affine",
            outlier_policy="disabled",
        )
        consensus = LocalConsensus(matcher, radius=192.0, threshold=3.0)
        baseline = Extractor(
            matcher,
            ["deepmatching", "horg", "efficientloftr"],
            {
                "efficientloftr": str(args.efficientloftr_onnx),
                "efficientloftr_dynamic": str(args.efficientloftr_dynamic_onnx),
            },
            0,
            1024,
            "local_consensus",
            "wgpu",
            "cpu",
            None,
        )
        scenarios = [
            ("affine", 8101, 9101),
            ("heterogeneous", 8102, 9102),
            ("heterogeneous", 8103, 9103),
            ("discontinuous", 8104, 9104),
            ("discontinuous", 8105, 9105),
            ("rigid", 8106, 9106),
        ]
        if args.independent_eval_per_family:
            independent_modes = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")
            for mode_index, mode in enumerate(independent_modes):
                for sample_index in range(args.independent_eval_per_family):
                    seed = 970_000_000 + mode_index * 1_000_003 + sample_index * 10_007
                    scenarios.append((mode, seed, seed + 4_000_000_001))
        warm_pair = generate_speckle_pair(8911, 9911, args.width, args.height, motion_mode="affine")
        warm_reference = np.rint(warm_pair.reference * 255).astype(np.uint8)
        warm_target = np.rint(warm_pair.target * 255).astype(np.uint8)
        solver.set_reference(warm_reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
        warm_points = grid_centres(args.width, args.height, 64, 64)
        warm_features = [
            matcher.detect_dic(
                image,
                options=DICDetectorOptions(max_keypoints=2048),
                learned_weights_path=args.weights,
            ).features
            for image in (warm_reference, warm_target)
        ]
        warm_described = [
            descriptor.describe(image, features)
            for image, features in zip((warm_reference, warm_target), warm_features, strict=True)
        ]
        matcher.match_features(
            warm_described[0], warm_described[1], warm_points, fit_local_models=False
        )
        warm_horg = [
            matcher.describe_horg_keypoints(image, features, reduction=4)
            for image, features in zip((warm_reference, warm_target), warm_features, strict=True)
        ]
        matcher.match_features(warm_horg[0], warm_horg[1], warm_points, fit_local_models=False)
        if args.compact_descriptor:
            warm_compact = [
                matcher.describe_dic_keypoints(
                    image, features, weights_path=args.compact_descriptor
                )
                for image, features in zip(
                    (warm_reference, warm_target), warm_features, strict=True
                )
            ]
            matcher.match_features(
                warm_compact[0], warm_compact[1], warm_points, fit_local_models=False
            )
        if args.cnn_descriptor:
            warm_cnn = [
                matcher.describe_dic_cnn_keypoints(
                    image, features, weights_path=args.cnn_descriptor
                )
                for image, features in zip(
                    (warm_reference, warm_target), warm_features, strict=True
                )
            ]
            matcher.match_features(warm_cnn[0], warm_cnn[1], warm_points, fit_local_models=False)
        for method in ("deepmatching", "horg", "efficientloftr"):
            baseline.extract(method, warm_reference, warm_target, warm_points)
        warm_active = np.ones(len(warm_points), dtype=np.uint32)
        solver._register_seeded_frame(
            warm_target,
            seed_state(np.zeros((len(warm_points), 6), dtype=np.float32), warm_active),
            warm_active,
        )
        results = []
        for mode, texture_seed, warp_seed in scenarios:
            pair = generate_speckle_pair(
                texture_seed, warp_seed, args.width, args.height, motion_mode=mode
            )
            reference = np.rint(pair.reference * 255).astype(np.uint8)
            reference_start = time.perf_counter()
            solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
            reference_preparation_s = time.perf_counter() - reference_start
            learned = run_case(
                matcher,
                descriptor,
                consensus,
                solver,
                mode,
                texture_seed,
                warp_seed,
                args.output,
                (args.width, args.height),
                args.weights,
                reference_preparation_s,
            )
            dic_horg64 = run_case(
                matcher,
                descriptor,
                consensus,
                solver,
                mode,
                texture_seed,
                warp_seed,
                args.output,
                (args.width, args.height),
                args.weights,
                reference_preparation_s,
                descriptor_variant="horg64",
            )
            if args.compact_descriptor:
                dic_descriptor = run_case(
                    matcher,
                    descriptor,
                    consensus,
                    solver,
                    mode,
                    texture_seed,
                    warp_seed,
                    args.output,
                    (args.width, args.height),
                    args.weights,
                    reference_preparation_s,
                    descriptor_variant="dic_descriptor",
                    compact_descriptor_weights=args.compact_descriptor,
                )
            else:
                dic_descriptor = None
            if args.cnn_descriptor:
                dic_cnn_descriptor = run_case(
                    matcher,
                    descriptor,
                    consensus,
                    solver,
                    mode,
                    texture_seed,
                    warp_seed,
                    args.output,
                    (args.width, args.height),
                    args.weights,
                    reference_preparation_s,
                    descriptor_variant="dic_cnn_descriptor",
                    cnn_descriptor_weights=args.cnn_descriptor,
                )
            else:
                dic_cnn_descriptor = None
            target = np.rint(pair.target * 255).astype(np.uint8)
            points = grid_centres(args.width, args.height, 64, 64)
            case_dir = args.output / f"{mode}_{texture_seed}_{warp_seed}"
            case_pair = pair
            case_results = {
                "case": mode,
                "texture_seed": texture_seed,
                "warp_seed": warp_seed,
                "dic_detector_dedode_b": learned,
                "dic_detector_horg64": dic_horg64,
                "dic_detector_trained_descriptor": dic_descriptor,
                "dic_detector_patch_cnn_descriptor": dic_cnn_descriptor,
                "baselines": {},
            }
            for method in ("deepmatching", "horg", "efficientloftr"):
                baseline_result = run_baseline_case(
                    baseline,
                    solver,
                    method,
                    reference,
                    target,
                    case_pair,
                    points,
                    case_dir,
                    reference_preparation_s,
                )
                case_results["baselines"][method] = baseline_result
                (case_dir / f"{method}.json").write_text(
                    json.dumps(json_safe(baseline_result), indent=2, allow_nan=False) + "\n"
                )
                print(
                    f"{method} / {mode} seeds {texture_seed}/{warp_seed}: "
                    f"{baseline_result['direct_matches']} direct matches",
                    flush=True,
                )
            results.append(case_results)
        safe_results = json_safe(results)
        (args.output / "summary.json").write_text(
            json.dumps(safe_results, indent=2, allow_nan=False) + "\n"
        )
        solver.close()


if __name__ == "__main__":
    main()
