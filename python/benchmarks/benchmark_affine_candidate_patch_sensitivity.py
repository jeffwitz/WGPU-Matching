"""Measure descriptor retrieval with candidate-centred affine patch sampling.

For every target candidate y and source query x, the descriptor compares a
reference support at x+xi with target samples at y+F_hat@xi. The analytic
translation is not used to construct a candidate patch. It is used only to
score correctness. No physical visibility mask is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from benchmark_dic_descriptor_retrieval import spatial_budget, take
from benchmark_oracle_affine_descriptor_retrieval import (
    describe_method,
    ranking_metrics,
    sample_grid_patches,
)
from train_dic_cnn_descriptor import bilinear
from wgpu_matching import DICDetectorOptions, FeatureSet, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair

METHODS = ("zncc", "horg64", "dic_cnn")
ATLAS_TILE = 80
ATLAS_COLUMNS = 96


def perturb_jacobian(jacobian: np.ndarray, parameter: str, delta: float) -> np.ndarray:
    """Apply one output-frame angular, axial-scale, or shear error."""
    if parameter == "angle_deg":
        angle = np.deg2rad(delta)
        perturbation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    elif parameter == "stretch_x_percent":
        perturbation = np.diag([1.0 + delta / 100.0, 1.0])
    elif parameter == "stretch_y_percent":
        perturbation = np.diag([1.0, 1.0 + delta / 100.0])
    elif parameter == "shear":
        perturbation = np.asarray([[1.0, delta], [0.0, 1.0]])
    else:
        raise ValueError(f"Unknown Jacobian perturbation: {parameter}")
    return perturbation @ np.asarray(jacobian, dtype=np.float64)


def sensitivity_conditions() -> list[dict[str, float | str]]:
    """Use exact geometry and symmetric one-axis perturbation sweeps."""
    rows: list[dict[str, float | str]] = [{"parameter": "exact", "delta": 0.0}]
    axes = {
        "angle_deg": (-10, -5, -2, 2, 5, 10),
        "stretch_x_percent": (-20, -10, -5, 5, 10, 20),
        "stretch_y_percent": (-20, -10, -5, 5, 10, 20),
        "shear": (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20),
    }
    rows.extend(
        {"parameter": parameter, "delta": float(delta)}
        for parameter, values in axes.items()
        for delta in values
    )
    return rows


def image_border_support_valid(
    points: np.ndarray, width: int, height: int, *, support: float = 64.0
) -> np.ndarray:
    """Check image bounds only; never read a physical or analytic mask."""
    margin = support / 2.0
    xy = np.asarray(points, dtype=np.float32)
    return (
        (xy[:, 0] >= margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 0] < width - margin)
        & (xy[:, 1] < height - margin)
    )


def affine_target_samples(
    image: np.ndarray,
    centers: np.ndarray,
    jacobian: np.ndarray,
    *,
    side: int,
    offset_start: float,
    spacing: float,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample candidate-centred patches at ``y + F @ xi`` without translation."""
    height, width = image.shape
    coordinates = np.arange(side, dtype=np.float32) * spacing + offset_start
    ox, oy = np.meshgrid(coordinates, coordinates)
    offsets = np.stack((ox.ravel(), oy.ravel()), axis=1).astype(np.float64)
    transformed = offsets @ np.asarray(jacobian, dtype=np.float64).T
    patches = np.empty((len(centers), side * side), dtype=np.float32)
    support_valid = np.ones(len(centers), dtype=bool)
    source = np.asarray(image, dtype=np.float32)
    for start in range(0, len(centers), chunk_size):
        end = min(start + chunk_size, len(centers))
        locations = np.asarray(centers[start:end], dtype=np.float64)[:, None, :] + transformed[None]
        flat = locations.reshape(-1, 2)
        inside = (
            (flat[:, 0] >= 0.0)
            & (flat[:, 0] <= width - 1.0)
            & (flat[:, 1] >= 0.0)
            & (flat[:, 1] <= height - 1.0)
        )
        support_valid[start:end] = inside.reshape(end - start, side * side).all(axis=1)
        values = bilinear(source, flat)
        values[~inside] = 0.5
        patches[start:end] = values.reshape(end - start, side * side)
    return patches, support_valid


def affine_patch_atlas(
    image: np.ndarray,
    centers: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[np.ndarray, FeatureSet, np.ndarray]:
    """Pack 64-pixel affine patches into non-overlapping GPU descriptor tiles."""
    patches, support_valid = affine_target_samples(
        np.asarray(image, dtype=np.float32),
        centers,
        jacobian,
        side=64,
        offset_start=-31.5,
        spacing=1.0,
    )
    rows = int(np.ceil(len(centers) / ATLAS_COLUMNS))
    atlas = np.full(
        (rows * ATLAS_TILE, ATLAS_COLUMNS * ATLAS_TILE), 128, dtype=np.uint8
    )
    positions = np.empty((len(centers), 2), dtype=np.float32)
    for index, patch in enumerate(patches):
        row, column = divmod(index, ATLAS_COLUMNS)
        y0 = row * ATLAS_TILE + (ATLAS_TILE - 64) // 2
        x0 = column * ATLAS_TILE + (ATLAS_TILE - 64) // 2
        atlas[y0 : y0 + 64, x0 : x0 + 64] = np.rint(
            np.clip(patch.reshape(64, 64), 0, 1) * 255
        ).astype(np.uint8)
        positions[index] = [column * ATLAS_TILE + ATLAS_TILE / 2, row * ATLAS_TILE + ATLAS_TILE / 2]
    return atlas, FeatureSet(positions, np.ones(len(positions), dtype=np.float32)), support_valid


def _descriptor_pool(
    matcher,
    image: np.ndarray,
    features: FeatureSet,
    cnn_weights: Path,
) -> dict[str, np.ndarray]:
    return {
        "zncc": sample_grid_patches(image.astype(np.float32) / 255.0, features.xy),
        "horg64": describe_method(matcher, "horg64", image, features, cnn_weights, None),
        "dic_cnn": describe_method(matcher, "dic_cnn", image, features, cnn_weights, None),
    }


def _score_and_rank(
    source_descriptors: dict[str, np.ndarray],
    target_descriptors: dict[str, np.ndarray],
    target_valid: np.ndarray,
    source_xy: np.ndarray,
    mapped_queries: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[dict, dict]:
    results, best_ids = {}, {}
    for method in METHODS:
        scores = source_descriptors[method] @ target_descriptors[method].T
        scores[:, ~target_valid] = -1.0
        order = np.argsort(-scores, axis=1, kind="stable")
        results[method] = ranking_metrics(scores, mapped_queries, target_xy)
        best_ids[method] = order[:, 0]
    return results, best_ids


def gpu_affine_consensus(
    matcher,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    mapped_queries: np.ndarray,
    selected_target_ids: np.ndarray,
    support_valid: np.ndarray,
    *,
    threshold_px: float = 4.0,
) -> dict:
    """Fit a GPU affine consensus and score accepted top-1 pairs against truth."""
    correspondences = np.column_stack((source_xy, target_xy[selected_target_ids])).astype(
        np.float32
    )
    center = np.asarray([[source_xy[:, 0].mean(), source_xy[:, 1].mean()]], dtype=np.float32)
    started = time.perf_counter()
    field = matcher.fit_local_affine(
        correspondences,
        center,
        radius_px=float(np.hypot(*np.ptp(source_xy, axis=0))),
        threshold_px=threshold_px,
        min_inliers=6,
        min_candidates=8,
        min_inlier_fraction=0.05,
        max_iterations=256,
        include_affine_models=True,
    )
    ransac_s = time.perf_counter() - started
    if field.affine_models is None or not len(field.affine_models):
        return {"fit_usable": False, "ransac_s": ransac_s}
    model = np.asarray(field.affine_models[0], dtype=np.float64)
    predicted = np.column_stack(
        (
            model[0] * (source_xy[:, 0] - center[0, 0])
            + model[1] * (source_xy[:, 1] - center[0, 1])
            + model[2],
            model[4] * (source_xy[:, 0] - center[0, 0])
            + model[5] * (source_xy[:, 1] - center[0, 1])
            + model[6],
        )
    )
    observed = target_xy[selected_target_ids]
    residual = np.linalg.norm(predicted - observed, axis=1)
    truth_error = np.linalg.norm(observed - mapped_queries, axis=1)
    accepted = (residual <= threshold_px) & support_valid[selected_target_ids]
    match = field.matches[0]
    return {
        "fit_usable": bool(match.usable),
        "inliers_reported": int(match.inliers),
        "accepted_by_model": int(accepted.sum()),
        "correct_accepted_le5px": int(np.count_nonzero(accepted & (truth_error <= 5.0))),
        "false_accepted": int(np.count_nonzero(accepted & (truth_error > 5.0))),
        "median_residual_px": float(np.median(residual[accepted])) if accepted.any() else None,
        "ransac_s": ransac_s,
        "threshold_px": threshold_px,
    }


def evaluate_texture(
    matcher,
    cnn_weights: Path,
    *,
    texture_seed: int,
    warp_seed: int,
    family: str,
    width: int,
    height: int,
    query_count: int,
    candidate_budget: int,
    run_sensitivity: bool,
) -> dict:
    pair = generate_speckle_pair(texture_seed, warp_seed, width, height, motion_mode=family)
    reference = np.rint(pair.reference * 255).astype(np.uint8)
    target = np.rint(pair.target * 255).astype(np.uint8)
    # Oversample before the image-border support check so the evaluated pool
    # can still contain the full requested budget of 8,192 candidates.
    options = DICDetectorOptions(max_keypoints=candidate_budget * 4, stride_px=4)
    started = time.perf_counter()
    source_all = matcher.detect_dic(reference, options=options).features
    target_all = matcher.detect_dic(target, options=options).features
    detection_s = time.perf_counter() - started

    source_queries = spatial_budget(
        take(
            source_all,
            source_all.valid & image_border_support_valid(source_all.xy, width, height),
        ),
        query_count,
        width,
        height,
    )
    target_candidates = spatial_budget(
        take(
            target_all,
            target_all.valid & image_border_support_valid(target_all.xy, width, height),
        ),
        candidate_budget,
        width,
        height,
    )
    source_xy, target_xy = source_queries.xy, target_candidates.xy
    mapped_queries = source_xy + bilinear(pair.flow_xy, source_xy).astype(np.float64)
    source_descriptors = _descriptor_pool(matcher, reference, source_queries, cnn_weights)
    baseline_target_desc = _descriptor_pool(
        matcher, target, target_candidates, cnn_weights
    )
    baseline_metrics, baseline_best = _score_and_rank(
        source_descriptors,
        baseline_target_desc,
        np.ones(len(target_xy), dtype=bool),
        source_xy,
        mapped_queries,
        target_xy,
    )
    baseline_consensus = {
        method: gpu_affine_consensus(
            matcher,
            source_xy,
            target_xy,
            mapped_queries,
            ids,
            np.ones(len(target_xy), dtype=bool),
        )
        for method, ids in baseline_best.items()
    }

    conditions = sensitivity_conditions() if run_sensitivity else [
        {"parameter": "exact", "delta": 0.0}
    ]
    true_jacobian = np.asarray(pair.metadata["affine_matrix"], dtype=np.float64)
    rows = []
    for condition in conditions:
        parameter, delta = str(condition["parameter"]), float(condition["delta"])
        jacobian = (
            true_jacobian
            if parameter == "exact"
            else perturb_jacobian(true_jacobian, parameter, delta)
        )
        determinant = float(np.linalg.det(jacobian))
        condition_number = float(np.linalg.cond(jacobian))
        if determinant <= 0 or condition_number > 12:
            rows.append(
                {
                    "parameter": parameter,
                    "delta": delta,
                    "skipped": "non-positive determinant or condition number > 12",
                }
            )
            continue
        warp_start = time.perf_counter()
        target_patch_matrix, support_valid = affine_target_samples(
            pair.target,
            target_xy,
            jacobian,
            side=32,
            offset_start=-32.0,
            spacing=2.0,
        )
        sample_s = time.perf_counter() - warp_start
        target_zncc = target_patch_matrix
        target_descriptors = {"zncc": target_zncc}
        descriptor_start = time.perf_counter()
        atlas, atlas_features, atlas_support_valid = affine_patch_atlas(
            pair.target, target_xy, jacobian
        )
        atlas_desc = _descriptor_pool(matcher, atlas, atlas_features, cnn_weights)
        target_descriptors["horg64"] = atlas_desc["horg64"]
        target_descriptors["dic_cnn"] = atlas_desc["dic_cnn"]
        descriptor_s = time.perf_counter() - descriptor_start
        support_valid &= atlas_support_valid
        metrics, best_ids = _score_and_rank(
            source_descriptors,
            target_descriptors,
            support_valid,
            source_xy,
            mapped_queries,
            target_xy,
        )
        consensus = {
            method: gpu_affine_consensus(
                matcher,
                source_xy,
                target_xy,
                mapped_queries,
                ids,
                support_valid,
            )
            for method, ids in best_ids.items()
        }
        rows.append(
            {
                "parameter": parameter,
                "delta": delta,
                "determinant": determinant,
                "condition_number": condition_number,
                "candidate_support_valid_count": int(support_valid.sum()),
                "affine_patch_sampling_s": sample_s,
                "descriptor_atlas_s": descriptor_s,
                "retrieval": metrics,
                "gpu_affine_consensus": consensus,
            }
        )
    return {
        "texture_seed": texture_seed,
        "warp_seed": warp_seed,
        "family": family,
        "image_size_px": [width, height],
        "source_query_count": len(source_xy),
        "source_detector_selected_count": len(source_all.xy),
        "source_valid_in_image_count": int(
            np.count_nonzero(
                source_all.valid
                & image_border_support_valid(source_all.xy, width, height)
            )
        ),
        "target_candidate_count": len(target_xy),
        "target_detector_selected_count": len(target_all.xy),
        "target_valid_in_image_count": int(
            np.count_nonzero(
                target_all.valid
                & image_border_support_valid(target_all.xy, width, height)
            )
        ),
        "candidate_filter": "only detector validity and 64-pixel support inside image; no analytic physical visibility mask",
        "candidate_pool_fixed_across_jacobians": True,
        "detection_s": detection_s,
        "baseline_retrieval": baseline_metrics,
        "baseline_gpu_affine_consensus": baseline_consensus,
        "true_jacobian": true_jacobian.tolist(),
        "conditions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--candidate-budget", type=int, default=8192)
    parser.add_argument("--global-queries", type=int, default=512)
    parser.add_argument("--textures", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--families", nargs="+", default=["rigid", "affine"])
    args = parser.parse_args()
    if not args.cnn_weights.is_file():
        parser.error("CNN weight archive must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with WGPUMatching(device_type="gpu") as matcher:
        print(f"WebGPU adapter: {matcher.device_name}", flush=True)
        for family_index, family in enumerate(args.families):
            for texture_index in range(args.textures):
                texture_seed = args.seed + family_index * 100_003 + texture_index * 10_007
                warp_seed = args.seed + 4_000_001 + family_index * 100_003 + texture_index * 10_007
                row = evaluate_texture(
                    matcher,
                    args.cnn_weights,
                    texture_seed=texture_seed,
                    warp_seed=warp_seed,
                    family=family,
                    width=args.width,
                    height=args.height,
                    query_count=args.global_queries,
                    candidate_budget=args.candidate_budget,
                    run_sensitivity=args.sensitivity and family == "affine",
                )
                results.append(row)
                print(
                    f"{family} texture {texture_index + 1}/{args.textures}: "
                    f"queries={row['source_query_count']} candidates={row['target_candidate_count']}",
                    flush=True,
                )
    payload = {
        "protocol": {
            "purpose": "oracle diagnostic only; analytic truth is used for the Jacobian and scoring, never for candidate construction",
            "image_size_px": [args.width, args.height],
            "candidate_budget": args.candidate_budget,
            "query_budget": args.global_queries,
            "detector": "WGSL structure-tensor score with deterministic spatial quotas; no learned detector weights passed",
            "textures_per_family": args.textures,
            "families": args.families,
            "methods": ["zero-mean normalized cross-correlation", "HORG-64", "DIC-Descriptor CNN"],
            "candidate_patch_equation": "P1(y, xi) = I1(y + F_hat @ xi); unknown translation is absent",
            "candidate_policy": "same detector candidate positions for every Jacobian; no pair.visible or pair.target_visible mask; invalid affine image-border supports remain in the pool with a score floor",
            "support_sizes_px": {"ZNCC": 64, "HORG-64": 64, "DIC-Descriptor CNN": 64},
            "sensitivity_axes": {
                "angle_deg": [-10, -5, -2, 0, 2, 5, 10],
                "stretch_x_percent": [-20, -10, -5, 0, 5, 10, 20],
                "stretch_y_percent": [-20, -10, -5, 0, 5, 10, 20],
                "shear": [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
            },
            "geometry_filter": "top-1 pairs are passed to WebGPU local affine RANSAC; its centred-coordinate model is evaluated against the observed pairs and analytic truth only scores retained pairs",
            "dedode_b_exclusion": "The current fixed-shape DeDoDe-B descriptor samples one dense image feature map and has no candidate-centred affine patch API; warping its whole input would reintroduce unknown-translation cropping, while an atlas would reduce each 64-pixel support to about 8 pixels. It is therefore reported only in the preceding full-image oracle experiment, not this controlled support ablation.",
            "cnn_weights_sha256": hashlib.sha256(args.cnn_weights.read_bytes()).hexdigest(),
            "webgpu_adapter": matcher.device_name,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
