"""Measure DIC-Detector descriptor retrieval against 512/2k/8k target candidates.

This is a descriptor diagnostic, not an IC-GN qualification. A positive is the
nearest detected target keypoint to the analytically mapped source location.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from train_dic_cnn_descriptor import support_is_valid, target_support_is_valid
from wgpu_matching import DICDetectorOptions, FeatureSet, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair


def bilinear(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    x0 = np.floor(points[:, 0]).astype(int)
    y0 = np.floor(points[:, 1]).astype(int)
    wx = points[:, 0] - x0
    wy = points[:, 1] - y0
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    return (
        array[y0, x0] * ((1 - wx) * (1 - wy))[:, None]
        + array[y0, x1] * (wx * (1 - wy))[:, None]
        + array[y1, x0] * ((1 - wx) * wy)[:, None]
        + array[y1, x1] * (wx * wy)[:, None]
    )


def take(features: FeatureSet, valid: np.ndarray) -> FeatureSet:
    ids = np.flatnonzero(valid)
    return FeatureSet(features.xy[ids], features.scores[ids])


def spatial_budget(features: FeatureSet, count: int, width: int, height: int) -> FeatureSet:
    """Keep top-scoring detections round-robin across the detector's image tiles."""
    columns, rows = 16, 9
    tile_x = np.minimum((features.xy[:, 0] * columns / width).astype(int), columns - 1)
    tile_y = np.minimum((features.xy[:, 1] * rows / height).astype(int), rows - 1)
    queues = []
    for tile in range(columns * rows):
        ids = np.flatnonzero(tile_y * columns + tile_x == tile)
        queues.append(ids[np.argsort(-features.scores[ids])])
    selected = []
    depth = 0
    while len(selected) < min(count, len(features.xy)):
        advanced = False
        for queue in queues:
            if depth < len(queue):
                selected.append(int(queue[depth]))
                advanced = True
                if len(selected) == min(count, len(features.xy)):
                    break
        if not advanced:
            break
        depth += 1
    ids = np.asarray(selected, dtype=np.int64)
    return FeatureSet(features.xy[ids], features.scores[ids])


def summarize(source_desc, target_desc, source_xy, target_xy, expected_xy) -> dict:
    similarities = np.asarray(source_desc, dtype=np.float32) @ np.asarray(
        target_desc, dtype=np.float32
    ).T
    order = np.argsort(-similarities, axis=1)
    distances = np.linalg.norm(expected_xy[:, None, :] - target_xy[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    repeat_error = distances[np.arange(len(source_xy)), nearest]
    ranks = np.empty(len(source_xy), dtype=np.int32)
    for row, candidate in enumerate(nearest):
        ranks[row] = np.flatnonzero(order[row] == candidate)[0] + 1
    top = order[:, 0]
    top_error = np.linalg.norm(target_xy[top] - expected_xy, axis=1)
    reverse = np.argmax(similarities, axis=0)
    mutual = reverse[top] == np.arange(len(source_xy))
    visible = repeat_error <= 5.0
    return {
        "source_queries": len(source_xy),
        "target_candidates": len(target_xy),
        "repeatable_le2px": int(np.count_nonzero(repeat_error <= 2.0)),
        "repeatable_le5px": int(np.count_nonzero(visible)),
        "positive_rank_top1_le5px": int(np.count_nonzero(visible & (ranks <= 1))),
        "positive_rank_top5_le5px": int(np.count_nonzero(visible & (ranks <= 5))),
        "positive_rank_top20_le5px": int(np.count_nonzero(visible & (ranks <= 20))),
        "positive_rank_median": float(np.median(ranks[visible])) if visible.any() else None,
        "descriptor_top1_correct_le2px": int(np.count_nonzero(top_error <= 2.0)),
        "descriptor_top1_correct_le5px": int(np.count_nonzero(top_error <= 5.0)),
        "mutual_top1_false_positive_ge5px": int(np.count_nonzero(mutual & (top_error > 5.0))),
        "top1_similarity_margin_median": float(
            np.median(similarities[np.arange(len(top)), top] - np.partition(similarities, -2, axis=1)[:, -2])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--families", nargs="+", default=["rigid", "affine", "heterogeneous", "discontinuous"])
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    budgets = (512, 2048, 8192)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with WGPUMatching(device_type="gpu") as matcher:
        print(f"adapter: {matcher.device_name}", flush=True)
        for family_index, family in enumerate(args.families):
            texture_seed = args.seed + family_index * 10_007
            warp_seed = args.seed + 4_000_001 + family_index * 10_007
            pair = generate_speckle_pair(
                texture_seed, warp_seed, args.width, args.height, motion_mode=family
            )
            reference = np.rint(pair.reference * 255).astype(np.uint8)
            target = np.rint(pair.target * 255).astype(np.uint8)
            # The full pool must exceed the requested final budget because
            # physical-validity filtering removes edge and discontinuity supports.
            pool_budget = max(budgets) * 2
            large = DICDetectorOptions(max_keypoints=pool_budget, stride_px=4)
            source_start = time.perf_counter()
            source_all = matcher.detect_dic(reference, options=large).features
            mapped_all = source_all.xy + bilinear(pair.flow_xy, source_all.xy).astype(np.float32)
            source_valid = (
                source_all.valid
                & support_is_valid(pair, source_all.xy)
                & target_support_is_valid(pair, mapped_all)
            )
            source_all = take(source_all, source_valid)
            if len(source_all.xy) < args.queries:
                raise RuntimeError(f"Only {len(source_all.xy)} valid source keypoints in {family}")
            if len(source_all.xy) < args.queries:
                raise RuntimeError(
                    f"Only {len(source_all.xy)} valid source detections for "
                    f"the requested {args.queries} queries in {family}"
                )
            source_queries = spatial_budget(
                source_all, args.queries, args.width, args.height
            )
            expected = source_queries.xy + bilinear(
                pair.flow_xy, source_queries.xy
            ).astype(np.float32)
            detection_s = time.perf_counter() - source_start
            source_horg = matcher.describe_horg_keypoints(reference, source_queries, reduction=4)
            source_cnn = matcher.describe_dic_cnn_keypoints(
                reference, source_queries, weights_path=args.weights
            )
            start = time.perf_counter()
            target_all = matcher.detect_dic(target, options=large).features
            target_valid = target_all.valid & target_support_is_valid(pair, target_all.xy)
            target_all = take(target_all, target_valid)
            target_detection_s = time.perf_counter() - start
            for budget in budgets:
                target_features = spatial_budget(
                    target_all, budget, args.width, args.height
                )
                target_horg = matcher.describe_horg_keypoints(target, target_features, reduction=4)
                target_cnn = matcher.describe_dic_cnn_keypoints(
                    target, target_features, weights_path=args.weights
                )
                for name, source_desc, target_desc in (
                    ("horg64", source_horg.descriptors, target_horg.descriptors),
                    ("dic_cnn", source_cnn.descriptors, target_cnn.descriptors),
                ):
                    row = {
                        "family": family,
                        "texture_seed": texture_seed,
                        "warp_seed": warp_seed,
                        "descriptor": name,
                        "candidate_budget": budget,
                        "actual_candidate_count": len(target_features.xy),
                        "adapter": matcher.device_name,
                        "source_detection_s": detection_s,
                        "target_detection_s": target_detection_s,
                        "descriptor_and_ranking_s": None,
                    }
                    ranking_start = time.perf_counter()
                    row.update(
                        summarize(
                            source_desc,
                            target_desc,
                            source_queries.xy,
                            target_features.xy,
                            expected,
                        )
                    )
                    row["descriptor_and_ranking_s"] = time.perf_counter() - ranking_start
                    rows.append(row)
                    print(
                        f"{family} {name} candidates={len(target_features.xy)} "
                        f"repeat5={row['repeatable_le5px']}/{args.queries} "
                        f"top1={row['descriptor_top1_correct_le5px']}/{args.queries} "
                        f"top5={row['positive_rank_top5_le5px']}/{args.queries}",
                        flush=True,
                    )
    report = {
        "protocol": {
            "families": args.families,
            "candidate_budgets": budgets,
            "detector_pool_budget": max(budgets) * 2,
            "requested_image_size_px": [args.width, args.height],
            "query_count": args.queries,
            "support_px": 64,
            "validity": "source and target supports must remain within the analytic physical-validity masks",
            "positive_definition": "nearest detected target keypoint to the exact analytic mapped source point; counted repeatable only within 5 px",
            "ranking": "descriptor cosine rank of the nearest geometrically correct detected candidate among all target detections",
            "selection": "DIC-Detector spatially distributed WGSL detections; no geometry thresholds tuned",
        },
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
