"""Measure the diagnostic ceiling from oracle local-affine compensation.

The analytic motion is used only to rectify the target image and to score
retrieval. Every target candidate receives the same affine hypothesis for a
given query. This is an oracle diagnostic, never an inference method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from benchmark_dic_descriptor_retrieval import spatial_budget, take
from PIL import Image
from train_dic_cnn_descriptor import bilinear, mask_support_is_valid, support_is_valid
from wgpu_matching import (
    DeDoDeBOnnxWGPUDescriptor,
    DICDetectorOptions,
    FeatureSet,
    WGPUMatching,
)
from wgpu_matching.dic_synthetic import generate_speckle_pair


def matrix_for_query(pair, point: np.ndarray, *, radius: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the local forward Jacobian and mapped center from analytic flow."""
    offsets = np.asarray(
        [[-radius, 0], [radius, 0], [0, -radius], [0, radius]], dtype=np.float32
    )
    samples = point[None] + offsets
    flow = bilinear(pair.flow_xy, samples).astype(np.float64)
    du_dx = (flow[1] - flow[0]) / (2 * radius)
    du_dy = (flow[3] - flow[2]) / (2 * radius)
    jacobian = np.eye(2, dtype=np.float64) + np.column_stack((du_dx, du_dy))
    center = point.astype(np.float64) + bilinear(
        pair.flow_xy, point[None]
    )[0].astype(np.float64)
    return jacobian, center


def warp_target_to_material_frame(
    target: np.ndarray,
    target_valid: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull target pixels and validity into one source-centered affine frame."""
    width, height = target.shape[1], target.shape[0]
    linear = jacobian.astype(np.float64)
    offset = target_anchor.astype(np.float64) - linear @ source_anchor.astype(np.float64)
    coefficients = (
        float(linear[0, 0]),
        float(linear[0, 1]),
        float(offset[0]),
        float(linear[1, 0]),
        float(linear[1, 1]),
        float(offset[1]),
    )
    image = Image.fromarray(np.asarray(target, dtype=np.float32), mode="F")
    valid = Image.fromarray(np.asarray(target_valid, dtype=np.uint8) * 255, mode="L")
    canonical = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0.5,
    )
    canonical_valid = valid.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return np.asarray(canonical, dtype=np.float32), np.asarray(canonical_valid) > 0


def canonical_points(
    target_xy: np.ndarray,
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
    jacobian: np.ndarray,
) -> np.ndarray:
    inverse = np.linalg.inv(jacobian)
    return source_anchor + (np.asarray(target_xy) - target_anchor) @ inverse.T


def sample_grid_patches(image: np.ndarray, points: np.ndarray, side: int = 32) -> np.ndarray:
    """Sample a 32x32 lattice spanning 64 pixels around each point."""
    offsets = np.arange(side, dtype=np.float32) * (64.0 / side) - 32.0
    ox, oy = np.meshgrid(offsets, offsets)
    locations = points[:, None, None, :] + np.stack((ox, oy), axis=-1)[None]
    sampled = bilinear(image, locations.reshape(-1, 2))
    return sampled.reshape(len(points), side * side).astype(np.float32)


def zncc_rank(
    source_patch: np.ndarray,
    target_patches: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> dict:
    source = source_patch - source_patch.mean(axis=1, keepdims=True)
    target = target_patches - target_patches.mean(axis=1, keepdims=True)
    source /= np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    similarities = source @ target.T
    return ranking_metrics(similarities, source_xy, target_xy)


def ranking_metrics(
    similarities: np.ndarray, source_xy: np.ndarray, target_xy: np.ndarray
) -> dict:
    errors = np.linalg.norm(source_xy[:, None, :] - target_xy[None, :, :], axis=2)
    order = np.argsort(-similarities, axis=1, kind="stable")
    positive = errors <= 5.0
    has_positive = positive.any(axis=1)
    ranks = np.full(len(source_xy), len(target_xy), dtype=np.int32)
    if np.any(has_positive):
        ranked_positive = np.take_along_axis(positive, order, axis=1)
        rows, cols = np.nonzero(ranked_positive & has_positive[:, None])
        np.minimum.at(ranks, rows, cols)
    top_error = errors[np.arange(len(source_xy)), order[:, 0]]
    evaluated = int(has_positive.sum())
    rank_values = ranks[has_positive]
    return {
        "queries": len(source_xy),
        "repeatable_le5px": evaluated,
        "top1_correct_le2px": int(np.count_nonzero(has_positive & (top_error <= 2.0))),
        "top1_correct_le5px": int(np.count_nonzero(has_positive & (top_error <= 5.0))),
        "top5_retrieval": int(np.count_nonzero((rank_values >= 0) & (rank_values < 5))),
        "top20_retrieval": int(np.count_nonzero((rank_values >= 0) & (rank_values < 20))),
        "median_positive_rank_zero_based": (
            float(np.median(rank_values)) if len(rank_values) else None
        ),
        "top1_correct_le2px_rate_of_repeatable": (
            float(np.count_nonzero(has_positive & (top_error <= 2.0)) / evaluated)
            if evaluated
            else None
        ),
        "top1_correct_le5px_rate_of_repeatable": (
            float(np.count_nonzero(has_positive & (top_error <= 5.0)) / evaluated)
            if evaluated
            else None
        ),
    }


def describe_method(matcher, method, image, features, cnn_weights, dedode_engine):
    if method == "horg64":
        return matcher.describe_horg_keypoints(image, features, reduction=4).descriptors
    if method == "dic_cnn":
        return matcher.describe_dic_cnn_keypoints(
            image, features, weights_path=cnn_weights
        ).descriptors
    if method == "dedode_b":
        return dedode_engine.describe(image, features).descriptors
    raise ValueError(method)


def evaluate_case(
    matcher,
    dedode_engine,
    cnn_weights: Path,
    family: str,
    texture_seed: int,
    warp_seed: int,
    width: int,
    height: int,
    query_count: int,
    candidate_budget: int,
    local: bool,
) -> dict:
    pair = generate_speckle_pair(
        texture_seed, warp_seed, width, height, motion_mode=family
    )
    reference = np.rint(pair.reference * 255).astype(np.uint8)
    target = np.rint(pair.target * 255).astype(np.uint8)
    detector_options = DICDetectorOptions(
        max_keypoints=candidate_budget * 2,
        stride_px=4,
    )
    detect_start = time.perf_counter()
    source_all = matcher.detect_dic(reference, options=detector_options).features
    target_all = matcher.detect_dic(target, options=detector_options).features
    detection_pair_s = time.perf_counter() - detect_start

    source_ok = source_all.valid & support_is_valid(pair, source_all.xy)
    source_all = take(source_all, source_ok)
    source_queries = spatial_budget(source_all, query_count, width, height)
    target_ok = target_all.valid & mask_support_is_valid(
        pair.target_visible, (height, width), target_all.xy
    )
    target_all = take(target_all, target_ok)
    target_candidates = spatial_budget(target_all, candidate_budget, width, height)
    source_xy = source_queries.xy
    target_xy = target_candidates.xy

    methods = ("zncc", "horg64", "dic_cnn", "dedode_b")
    reference_description_s = {}
    source_desc = {}
    for method in methods:
        start = time.perf_counter()
        if method == "zncc":
            source_desc[method] = sample_grid_patches(pair.reference, source_xy)
        else:
            source_desc[method] = describe_method(
                matcher,
                method,
                reference,
                source_queries,
                cnn_weights,
                dedode_engine,
            )
        reference_description_s[method] = time.perf_counter() - start

    start = time.perf_counter()
    baseline_target_desc = {
        "zncc": sample_grid_patches(pair.target, target_xy),
        **{
            method: describe_method(
                matcher,
                method,
                target,
                target_candidates,
                cnn_weights,
                dedode_engine,
            )
            for method in methods
            if method != "zncc"
        },
    }
    baseline_description_s = time.perf_counter() - start
    baseline_rows = {}
    oracle_warp_s = 0.0
    jacobian_diagnostics = []

    if not local:
        jacobian = np.asarray(pair.metadata["affine_matrix"], dtype=np.float64)
        translation = np.asarray(pair.metadata["translation_px"], dtype=np.float64)
        center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
        mapped_queries = (source_xy - center) @ jacobian.T + center + translation
        warp_start = time.perf_counter()
        target_canonical, canonical_mask = warp_target_to_material_frame(
            pair.target, pair.target_visible, center, center + translation, jacobian
        )
        oracle_warp_s = time.perf_counter() - warp_start
        canonical_target_xy = canonical_points(
            target_xy, center, center + translation, jacobian
        )
        canonical_features = FeatureSet(canonical_target_xy, target_candidates.scores)
        oracle_candidate_valid = mask_support_is_valid(
            canonical_mask, (height, width), canonical_target_xy
        )
        canonical_features = take(canonical_features, oracle_candidate_valid)
        canonical_target_xy = canonical_features.xy
        canonical_target = np.rint(np.clip(target_canonical, 0, 1) * 255).astype(np.uint8)
        start = time.perf_counter()
        oracle_target_desc = {
            "zncc": sample_grid_patches(target_canonical, canonical_target_xy),
            **{
                method: describe_method(
                    matcher,
                    method,
                    canonical_target,
                    canonical_features,
                    cnn_weights,
                    dedode_engine,
                )
                for method in methods
                if method != "zncc"
            },
        }
        oracle_description_s = time.perf_counter() - start
        for method in methods:
            base = ranking_metrics(
                np.asarray(source_desc[method], dtype=np.float32)
                @ np.asarray(baseline_target_desc[method], dtype=np.float32).T,
                mapped_queries,
                target_xy,
            )
            oracle = ranking_metrics(
                np.asarray(source_desc[method], dtype=np.float32)
                @ np.asarray(oracle_target_desc[method], dtype=np.float32).T,
                source_xy,
                canonical_target_xy,
            )
            baseline_rows[method] = {"baseline": base, "oracle": oracle}
        oracle_queries = query_count
    else:
        baseline_rows = {method: {"baseline": None, "oracle": None} for method in methods}
        oracle_description_s = 0.0
        oracle_queries = min(query_count, 32)
        mapped_queries = source_xy + bilinear(
            pair.flow_xy, source_xy
        ).astype(np.float32)
        for method in methods:
            base_similarities = (
                np.asarray(source_desc[method], dtype=np.float32)
                @ np.asarray(baseline_target_desc[method], dtype=np.float32).T
            )
            baseline_rows[method]["baseline"] = ranking_metrics(
                base_similarities, mapped_queries, target_xy
            )
        for query_index in range(oracle_queries):
            point = source_xy[query_index].astype(np.float64)
            jacobian, mapped_point = matrix_for_query(pair, point)
            determinant = float(np.linalg.det(jacobian))
            condition = float(np.linalg.cond(jacobian))
            jacobian_diagnostics.append(
                {
                    "query_index": query_index,
                    "determinant": determinant,
                    "condition_number": condition,
                    "matrix": jacobian.tolist(),
                }
            )
            if determinant <= 0.0 or condition > 8.0:
                continue
            warp_start = time.perf_counter()
            canonical_image, canonical_mask = warp_target_to_material_frame(
                pair.target,
                pair.target_visible,
                point,
                mapped_point,
                jacobian,
            )
            oracle_warp_s += time.perf_counter() - warp_start
            candidate_xy = canonical_points(
                target_xy, point, mapped_point, jacobian
            )
            support_valid = mask_support_is_valid(
                canonical_mask, (height, width), candidate_xy
            )
            ids = np.flatnonzero(support_valid)
            if len(ids) == 0:
                continue
            features = FeatureSet(candidate_xy[ids], target_candidates.scores[ids])
            canonical_u8 = np.rint(np.clip(canonical_image, 0, 1) * 255).astype(np.uint8)
            start = time.perf_counter()
            canonical_desc = {"zncc": sample_grid_patches(canonical_image, features.xy)}
            for method in methods:
                if method != "zncc":
                    canonical_desc[method] = describe_method(
                        matcher,
                        method,
                        canonical_u8,
                        features,
                        cnn_weights,
                        dedode_engine,
                    )
                sims = (
                    np.asarray(source_desc[method][query_index : query_index + 1], dtype=np.float32)
                    @ np.asarray(canonical_desc[method], dtype=np.float32).T
                )
                metric = ranking_metrics(sims, point[None], features.xy)
                baseline_rows[method]["oracle"] = baseline_rows[method]["oracle"] or []
                baseline_rows[method]["oracle"].append(metric)
            oracle_description_s += time.perf_counter() - start

    return {
        "family": family,
        "texture_seed": texture_seed,
        "warp_seed": warp_seed,
        "image_size_px": [width, height],
        "oracle_type": "query_local_affine" if local else "global_affine",
        "oracle_queries_requested": oracle_queries,
        "source_detection_count": len(source_xy),
        "target_candidate_count": len(target_xy),
        "detection_pair_s": detection_pair_s,
        "reference_description_s": reference_description_s,
        "baseline_target_description_s": baseline_description_s,
        "oracle_target_description_s": oracle_description_s,
        "oracle_target_warp_s": oracle_warp_s,
        "local_jacobian_diagnostics": jacobian_diagnostics,
        "retrieval": baseline_rows,
        "note": "Analytic motion was used to rectify target image pixels and score retrieval; no analytic candidate was preselected.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-weights", type=Path, required=True)
    parser.add_argument("--dedode-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--candidate-budget", type=int, default=8192)
    parser.add_argument("--global-queries", type=int, default=128)
    parser.add_argument("--local-queries", type=int, default=32)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["rigid", "affine", "heterogeneous"],
    )
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if not args.cnn_weights.is_file() or not args.dedode_onnx.is_file():
        parser.error("CNN weights and the fixed-shape DeDoDe-B ONNX model must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with WGPUMatching(device_type="gpu") as matcher:
        dedode = DeDoDeBOnnxWGPUDescriptor(
            args.dedode_onnx,
            max_side=1024,
            capacity=args.candidate_budget,
        )
        print(f"adapter: {matcher.device_name}", flush=True)
        for family_index, family in enumerate(args.families):
            texture_seed = args.seed + family_index * 10_007
            warp_seed = args.seed + 4_000_001 + family_index * 10_007
            is_local = family == "heterogeneous"
            row = evaluate_case(
                matcher,
                dedode,
                args.cnn_weights,
                family,
                texture_seed,
                warp_seed,
                args.width,
                args.height,
                args.local_queries if is_local else args.global_queries,
                args.candidate_budget,
                is_local,
            )
            rows.append(row)
            print(
                f"{family}: queries={row['source_detection_count']} "
                f"candidates={row['target_candidate_count']} oracle={row['oracle_type']}",
                flush=True,
            )
    result = {
        "protocol": {
            "purpose": "oracle diagnosis only; not an inference pipeline",
            "candidate_budget": args.candidate_budget,
            "support_px": 64,
            "zncc_support_sampling": "32x32 lattice spanning the 64x64 support",
            "methods": ["ZNCC", "HORG-64", "DIC-CNN-64", "DeDoDe-B"],
            "query_policy": "same source detections and target candidate detections for each baseline/oracle pair",
            "oracle_policy": "global exact affine for translation/rigid/affine; per-query finite-difference Jacobian for heterogeneous fields; same Jacobian applied to all candidates for one query",
            "target_resampling": "Pillow bicubic intensity pullback and nearest-neighbour validity pullback",
            "cnn_weights_sha256": hashlib.sha256(args.cnn_weights.read_bytes()).hexdigest(),
            "dedode_graph_sha256": hashlib.sha256(args.dedode_onnx.read_bytes()).hexdigest(),
            "dedode_runtime_metadata": dedode.metadata,
            "webgpu_adapter": matcher.device_name,
        },
        "results": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
