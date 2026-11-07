"""Compare DaD, DeDoDe v2, HORG, and DIC-specific detectors on analytic speckles.

This is an optional detector comparison. It deliberately keeps detector
repeatability, common DeDoDe-B descriptor matching, and WGSL affine RANSAC
measurements separate. Optional DaD/DeDoDe ONNX graphs run through ONNX
Runtime's WebGPU provider; descriptor extraction and mutual matching remain
reference stages and are not GPU-resident.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from wgpu_matching import (
    DaDDetector,
    DeDoDeBDescriptor,
    DeDoDeV2Detector,
    DescribedFeatures,
    DICDetectorOptions,
    FeatureSet,
    OnnxWGPUKeypointDetector,
    WGPUMatching,
    mutual_nearest_cosine,
)
from wgpu_matching.images import load_gray

ROOT = Path(__file__).resolve().parents[2]
PROGRESSIVE = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
STRESS = ROOT / "results/uniaxial/tensile_60pct"
DEFAULT_OUTPUT = ROOT / "results/features/learned_detector_speckle_reference.json"
PATCH_SIZE = 128
SEED_STRIDE = 512
DEFAULT_BUDGETS = (512, 1024, 2048, 4096, 8192)


class WGPUTensorDetector:
    """Expose one native-resolution WGSL DIC score variant to the shared protocol."""

    def __init__(self, matcher: WGPUMatching, *, learned_weights_path: Path | None = None) -> None:
        self.matcher = matcher
        self.learned_weights_path = learned_weights_path
        self.name = "DIC-Learned-WGSL" if learned_weights_path else "DIC-Tensor-WGSL"
        self.metadata = {
            "detector": self.name,
            "execution": "native-resolution WGSL; structure tensor, optional learned score model, NMS, and spatial quotas",
            "source": "WGPU-Matching DIC-Detector",
            "weights_path": str(learned_weights_path) if learned_weights_path else None,
        }

    def detect(
        self,
        image: np.ndarray,
        *,
        max_keypoints: int = 4096,
        roi_mask: np.ndarray | None = None,
    ) -> FeatureSet:
        result = self.matcher.detect_dic(
            image,
            options=DICDetectorOptions(max_keypoints=max_keypoints),
            roi_mask=roi_mask,
            learned_weights_path=self.learned_weights_path,
        )
        return result.features


class HORGCommonDescriptorDetector:
    """Expose the existing WebGPU HORG/SIFT detector to the shared descriptor."""

    name = "HORG-SIFT-common-DeDoDe-B"

    def __init__(self, matcher: WGPUMatching) -> None:
        self.matcher = matcher
        self.metadata = {
            "detector": self.name,
            "execution": "HORG WGSL DoG detector at reduction 4; DeDoDe-B is sampled separately",
        }

    def detect(
        self,
        image: np.ndarray,
        *,
        max_keypoints: int = 4096,
        roi_mask: np.ndarray | None = None,
    ) -> FeatureSet:
        if roi_mask is not None:
            raise ValueError(
                "The current HORG common-descriptor adapter does not accept an ROI mask"
            )
        return self.matcher.detect_horg(image, max_keypoints=max_keypoints, reduction=4)


def load_truth(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["frame"]): {key: float(value) for key, value in row.items() if key != "frame"}
            for row in csv.DictReader(stream)
        }


def load_cases(progressive_root: Path, stress_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    progressive_truth_path = progressive_root / "machine_truth.csv"
    if progressive_truth_path.is_file():
        truth = load_truth(progressive_truth_path)
        for frame in (40, 100, 199):
            image_path = progressive_root / f"frame_{frame:03d}.png"
            if frame in truth and image_path.is_file():
                cases.append(
                    {
                        "name": f"progressive_{frame:03d}",
                        "reference": progressive_root / "reference.png",
                        "target": image_path,
                        "truth": truth[frame],
                        "frame": frame,
                        "source": "Live-DIC progressive tensile, read-only external dataset",
                    }
                )
    manifest_path = stress_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        truth_by_frame = load_truth(stress_root / manifest["truth_file"])
        for snapshot in manifest["snapshots"]:
            frame = int(snapshot["frame"])
            cases.append(
                {
                    "name": f"synthetic_{round(snapshot['requested_engineering_strain'] * 100):02d}pct",
                    "reference": stress_root / "reference.png",
                    "target": stress_root / snapshot["file"],
                    "truth": truth_by_frame[frame],
                    "frame": frame,
                    "source": "synthetic tensile extension with analytic affine truth",
                }
            )
    if not cases:
        raise FileNotFoundError("No progressive or synthetic tensile cases were found")
    return cases


def visible_seeds(width: int, height: int, stretch_x: float, stretch_y: float) -> np.ndarray:
    offset = (PATCH_SIZE - 1) / 2
    centers = []
    for row in range(1 + (height - PATCH_SIZE) // SEED_STRIDE):
        for column in range(1 + (width - PATCH_SIZE) // SEED_STRIDE):
            x = column * SEED_STRIDE + offset
            y = row * SEED_STRIDE + offset
            target_x = width / 2 + stretch_x * (x - width / 2)
            target_y = height / 2 + stretch_y * (y - height / 2)
            if 0 <= target_x < width and 0 <= target_y < height:
                centers.append((x, y))
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def nearest_indices(query: np.ndarray, reference: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    if not len(reference):
        return np.full(len(query), -1, dtype=np.int32)
    reference = np.asarray(reference, dtype=np.float32)
    ref_norm = np.einsum("ij,ij->i", reference, reference, dtype=np.float32)
    result = np.empty(len(query), dtype=np.int32)
    for start in range(0, len(query), chunk_size):
        stop = min(len(query), start + chunk_size)
        block = np.asarray(query[start:stop], dtype=np.float32)
        squared = (
            np.einsum("ij,ij->i", block, block, dtype=np.float32)[:, None]
            + ref_norm[None, :]
            - 2.0 * block @ reference.T
        )
        result[start:stop] = squared.argmin(axis=1)
    return result


def repeatability(
    reference_xy: np.ndarray,
    target_xy: np.ndarray,
    width: int,
    height: int,
    stretch_x: float,
    stretch_y: float,
) -> dict[str, Any]:
    cx, cy = width / 2, height / 2
    expected_target = np.column_stack(
        (
            cx + stretch_x * (reference_xy[:, 0] - cx),
            cy + stretch_y * (reference_xy[:, 1] - cy),
        )
    ).astype(np.float32)
    source_visible = (
        (reference_xy[:, 0] >= 0)
        & (reference_xy[:, 0] < width)
        & (reference_xy[:, 1] >= 0)
        & (reference_xy[:, 1] < height)
        & (expected_target[:, 0] >= 0)
        & (expected_target[:, 0] < width)
        & (expected_target[:, 1] >= 0)
        & (expected_target[:, 1] < height)
    )
    inverse_source = np.column_stack(
        (
            cx + (target_xy[:, 0] - cx) / stretch_x,
            cy + (target_xy[:, 1] - cy) / stretch_y,
        )
    ).astype(np.float32)
    target_visible = (
        (target_xy[:, 0] >= 0)
        & (target_xy[:, 0] < width)
        & (target_xy[:, 1] >= 0)
        & (target_xy[:, 1] < height)
        & (inverse_source[:, 0] >= 0)
        & (inverse_source[:, 0] < width)
        & (inverse_source[:, 1] >= 0)
        & (inverse_source[:, 1] < height)
    )
    ref_indices = np.flatnonzero(source_visible)
    tar_indices = np.flatnonzero(target_visible)
    source = reference_xy[ref_indices]
    target = target_xy[tar_indices]
    expected = expected_target[ref_indices]
    target_source = inverse_source[tar_indices]
    denominator = min(len(ref_indices), len(tar_indices))
    if not len(source) or not len(target):
        return {
            "source_visible_points": len(ref_indices),
            "target_visible_points": len(tar_indices),
            "one_to_one_mutual_pairs": 0,
            "denominator_min_visible": denominator,
            "at_1px": 0,
            "at_2px": 0,
            "at_5px": 0,
            "repeatability_1px": 0.0,
            "repeatability_2px": 0.0,
            "repeatability_5px": 0.0,
        }
    nearest_target = nearest_indices(expected, target)
    nearest_source = nearest_indices(target_source, source)
    mutual = nearest_source[nearest_target] == np.arange(len(source))
    errors = np.linalg.norm(expected - target[nearest_target], axis=1)
    errors = errors[mutual]
    return {
        "source_visible_points": len(ref_indices),
        "target_visible_points": len(tar_indices),
        "one_to_one_mutual_pairs": int(mutual.sum()),
        "denominator_min_visible": denominator,
        "at_1px": int(np.count_nonzero(errors <= 1.0)),
        "at_2px": int(np.count_nonzero(errors <= 2.0)),
        "at_5px": int(np.count_nonzero(errors <= 5.0)),
        "repeatability_1px": float(np.count_nonzero(errors <= 1.0) / denominator)
        if denominator
        else 0.0,
        "repeatability_2px": float(np.count_nonzero(errors <= 2.0) / denominator)
        if denominator
        else 0.0,
        "repeatability_5px": float(np.count_nonzero(errors <= 5.0) / denominator)
        if denominator
        else 0.0,
    }


def keypoint_distribution(xy: np.ndarray, width: int, height: int) -> dict[str, Any]:
    cells = np.zeros((9, 16), dtype=np.int32)
    ix = np.clip((xy[:, 0] / width * cells.shape[1]).astype(np.int32), 0, cells.shape[1] - 1)
    iy = np.clip((xy[:, 1] / height * cells.shape[0]).astype(np.int32), 0, cells.shape[0] - 1)
    np.add.at(cells, (iy, ix), 1)
    dic_columns = max(1, 1 + (width - PATCH_SIZE) // SEED_STRIDE)
    dic_rows = max(1, 1 + (height - PATCH_SIZE) // SEED_STRIDE)
    dic_cells = np.zeros((dic_rows, dic_columns), dtype=np.int32)
    dx = np.clip((xy[:, 0] / SEED_STRIDE).astype(np.int32), 0, dic_columns - 1)
    dy = np.clip((xy[:, 1] / SEED_STRIDE).astype(np.int32), 0, dic_rows - 1)
    np.add.at(dic_cells, (dy, dx), 1)
    return {
        "coverage_cells_occupied": int(np.count_nonzero(cells)),
        "coverage_cells_total": int(cells.size),
        "dic_grid_windows_with_points": int(np.count_nonzero(dic_cells)),
        "dic_grid_windows_total": int(dic_cells.size),
        "points_per_dic_window_median": float(np.median(dic_cells)),
        "points_per_dic_window_p10": float(np.percentile(dic_cells, 10)),
        "points_per_dic_window_p90": float(np.percentile(dic_cells, 90)),
    }


def select_indices(
    features: FeatureSet,
    budget: int,
    mode: str,
    width: int,
    height: int,
) -> np.ndarray:
    order = np.argsort(-features.scores, kind="stable")
    if mode == "global_topk":
        return order[:budget]
    columns, rows = 16, 9
    per_cell = max(1, int(np.ceil(budget / (columns * rows))))
    cell_x = np.clip((features.xy[:, 0] / width * columns).astype(np.int32), 0, columns - 1)
    cell_y = np.clip((features.xy[:, 1] / height * rows).astype(np.int32), 0, rows - 1)
    counts = np.zeros((rows, columns), dtype=np.int32)
    selected: list[int] = []
    chosen = np.zeros(len(features.xy), dtype=np.bool_)
    for index in order:
        cell_y_index, cell_x_index = cell_y[index], cell_x[index]
        if counts[cell_y_index, cell_x_index] >= per_cell:
            continue
        selected.append(int(index))
        chosen[index] = True
        counts[cell_y_index, cell_x_index] += 1
        if len(selected) == budget:
            break
    if len(selected) < budget:
        selected.extend(int(index) for index in order[~chosen[order]][: budget - len(selected)])
    return np.asarray(selected, dtype=np.int32)


def subset(described: DescribedFeatures, indices: np.ndarray) -> DescribedFeatures:
    original = described.features
    return DescribedFeatures(
        features=FeatureSet(
            xy=original.xy[indices],
            scores=original.scores[indices],
            valid=original.valid[indices],
            metadata=dict(original.metadata),
        ),
        descriptors=described.descriptors[indices],
        metadata=dict(described.metadata),
    )


def measure_candidate_pool(
    detector: Any,
    descriptor: DeDoDeBDescriptor,
    reference: np.ndarray,
    target: np.ndarray,
    candidate_budget: int,
    repeats: int,
) -> dict[str, Any]:
    detector.detect(reference, max_keypoints=candidate_budget)
    detector.detect(target, max_keypoints=candidate_budget)

    detection_times: dict[str, list[float]] = {"reference": [], "target": []}
    detection_stage_times: dict[str, dict[str, list[float]]] = {
        "reference": {},
        "target": {},
    }
    features: dict[str, FeatureSet | None] = {"reference": None, "target": None}
    for _ in range(repeats):
        for name, image in (("reference", reference), ("target", target)):
            start = time.perf_counter()
            features[name] = detector.detect(image, max_keypoints=candidate_budget)
            detection_times[name].append(time.perf_counter() - start)
            for stage, value in features[name].metadata.items():
                if stage in {
                    "input_preprocessing_s",
                    "onnx_session_run_s",
                    "coordinate_postprocessing_s",
                    "detect_total_s",
                } and isinstance(value, (int, float)):
                    detection_stage_times[name].setdefault(stage, []).append(float(value))

    assert features["reference"] is not None and features["target"] is not None
    described: dict[str, DescribedFeatures | None] = {"reference": None, "target": None}
    descriptor_times: dict[str, list[float]] = {"reference": [], "target": []}
    for name, image in (("reference", reference), ("target", target)):
        descriptor.describe(image, features[name])
    for _ in range(repeats):
        for name, image in (("reference", reference), ("target", target)):
            start = time.perf_counter()
            described[name] = descriptor.describe(image, features[name])
            descriptor_times[name].append(time.perf_counter() - start)

    assert described["reference"] is not None and described["target"] is not None
    return {
        "features_reference": features["reference"],
        "features_target": features["target"],
        "described_reference": described["reference"],
        "described_target": described["target"],
        "detection_times": detection_times,
        "detection_stage_times": detection_stage_times,
        "descriptor_times": descriptor_times,
    }


def run_case(
    case: dict[str, Any],
    detectors: dict[str, Any],
    descriptor: DeDoDeBDescriptor,
    matcher: WGPUMatching,
    budgets: tuple[int, ...],
    max_side: int,
    candidate_multiplier: int,
    repeats: int,
) -> dict[str, Any]:
    reference = load_gray(case["reference"])
    target = load_gray(case["target"])
    if reference.shape != target.shape:
        raise ValueError(f"Mismatched image shapes for {case['name']}")
    height, width = reference.shape
    truth = case["truth"]
    seeds = visible_seeds(width, height, truth["stretch_x"], truth["stretch_y"])
    max_candidates = max(budgets) * candidate_multiplier
    result: dict[str, Any] = {
        "name": case["name"],
        "source": case["source"],
        "frame": case["frame"],
        "reference_path": str(case["reference"]),
        "target_path": str(case["target"]),
        "image_size_px": [width, height],
        "model_max_side_px": max_side,
        "truth": {key: truth[key] for key in ("engineering_axial", "stretch_x", "stretch_y")},
        "visible_dic_seeds": len(seeds),
        "detectors": {},
    }
    for detector_name, detector in detectors.items():
        detector_budget = max_candidates
        selection_pools = {}
        for selection in ("global_topk", "spatial_quotas"):
            if selection == "spatial_quotas":
                selection_pools[selection] = selection_pools["global_topk"]
            else:
                selection_pools[selection] = measure_candidate_pool(
                    detector,
                    descriptor,
                    reference,
                    target,
                    detector_budget,
                    repeats,
                )
        warm_pool = selection_pools["global_topk"]
        warm_indices_a = select_indices(
            warm_pool["features_reference"], max(budgets), "global_topk", width, height
        )
        warm_indices_b = select_indices(
            warm_pool["features_target"], max(budgets), "global_topk", width, height
        )
        warm_pairs = mutual_nearest_cosine(
            subset(warm_pool["described_reference"], warm_indices_a),
            subset(warm_pool["described_target"], warm_indices_b),
            min_cosine_similarity=0.0,
        )
        if len(warm_pairs.points) >= 3 and len(seeds):
            matcher.fit_local_affine(
                warm_pairs.points,
                seeds,
                radius_px=256,
                threshold_px=3,
                min_inliers=6,
                min_candidates=8,
                min_inlier_fraction=0.2,
                max_iterations=256,
            )
        reference_features = warm_pool["features_reference"]
        reference_described = warm_pool["described_reference"]
        method_result = {
            "detector_metadata": reference_features.metadata,
            "detector_metadata_by_selection": {
                selection: pool["features_reference"].metadata
                for selection, pool in selection_pools.items()
            },
            "candidate_count_by_selection": {
                selection: {
                    "reference": len(pool["features_reference"].xy),
                    "target": len(pool["features_target"].xy),
                }
                for selection, pool in selection_pools.items()
            },
            "selection_scope": (
                "Both selectors act on the same returned candidate pool. The WGSL DIC detector "
                "internally creates that pool with its configured spatial tile quotas; the global "
                "top-K metric is not a global top-K over every NMS peak in the full response map."
            ),
            "descriptor_metadata": reference_described.metadata,
            "selection_timing": {
                selection: {
                    "reference_detection_s_median": statistics.median(
                        pool["detection_times"]["reference"]
                    ),
                    "target_detection_s_median": statistics.median(
                        pool["detection_times"]["target"]
                    ),
                    "reference_detection_s_repeats": pool["detection_times"]["reference"],
                    "target_detection_s_repeats": pool["detection_times"]["target"],
                    "reference_detector_stage_s_median": {
                        stage: statistics.median(values)
                        for stage, values in pool["detection_stage_times"]["reference"].items()
                    },
                    "target_detector_stage_s_median": {
                        stage: statistics.median(values)
                        for stage, values in pool["detection_stage_times"]["target"].items()
                    },
                    "reference_descriptor_s_median": statistics.median(
                        pool["descriptor_times"]["reference"]
                    ),
                    "target_descriptor_s_median": statistics.median(
                        pool["descriptor_times"]["target"]
                    ),
                    "reference_descriptor_s_repeats": pool["descriptor_times"]["reference"],
                    "target_descriptor_s_repeats": pool["descriptor_times"]["target"],
                }
                for selection, pool in selection_pools.items()
            },
            "budgets": {},
        }
        for budget in budgets:
            budget_result: dict[str, Any] = {}
            for selection in ("global_topk", "spatial_quotas"):
                pool = selection_pools[selection]
                features_reference = pool["features_reference"]
                features_target = pool["features_target"]
                described_reference = pool["described_reference"]
                described_target = pool["described_target"]
                indices_a = select_indices(features_reference, budget, selection, width, height)
                indices_b = select_indices(features_target, budget, selection, width, height)
                selected_a = subset(described_reference, indices_a)
                selected_b = subset(described_target, indices_b)
                repeat = repeatability(
                    selected_a.features.xy,
                    selected_b.features.xy,
                    width,
                    height,
                    truth["stretch_x"],
                    truth["stretch_y"],
                )
                match_times = []
                pairs = None
                for _ in range(repeats):
                    match_start = time.perf_counter()
                    pairs = mutual_nearest_cosine(selected_a, selected_b, min_cosine_similarity=0.0)
                    match_times.append(time.perf_counter() - match_start)
                assert pairs is not None
                matching_s = statistics.median(match_times)
                if len(pairs.points) >= 3 and len(seeds):
                    fit_times = []
                    compute_times = []
                    affine = None
                    for _ in range(repeats):
                        fit_start = time.perf_counter()
                        affine = matcher.fit_local_affine(
                            pairs.points,
                            seeds,
                            radius_px=256,
                            threshold_px=3,
                            min_inliers=6,
                            min_candidates=8,
                            min_inlier_fraction=0.2,
                            max_iterations=256,
                        )
                        fit_times.append(time.perf_counter() - fit_start)
                        compute_times.append(affine.gpu_compute_ms)
                    ransac_wall_s = statistics.median(fit_times)
                    ransac_compute_ms = statistics.median(compute_times)
                else:
                    affine = None
                    ransac_wall_s = 0.0
                    ransac_compute_ms = 0.0
                accepted_errors: list[float] = []
                fit_rows: list[dict[str, Any]] = []
                if affine is not None:
                    cx, cy = width / 2, height / 2
                    for seed, match in zip(seeds, affine.matches, strict=True):
                        expected_target = np.asarray(
                            (
                                cx + truth["stretch_x"] * (seed[0] - cx),
                                cy + truth["stretch_y"] * (seed[1] - cy),
                            )
                        )
                        estimated_target = np.asarray((match.x + match.dx, match.y + match.dy))
                        error_px = float(np.linalg.norm(estimated_target - expected_target))
                        if match.usable:
                            accepted_errors.append(error_px)
                        fit_rows.append(
                            {
                                "seed_xy_px": seed.tolist(),
                                "estimated_target_xy_px": estimated_target.tolist()
                                if match.usable
                                else None,
                                "analytic_error_px": error_px if match.usable else None,
                                "usable": match.usable,
                                "inliers": match.inliers,
                                "candidates": match.candidates,
                                "residual_px": (
                                    float(match.residual)
                                    if match.residual is not None and np.isfinite(match.residual)
                                    else None
                                ),
                            }
                        )
                raw_errors = []
                if len(pairs.points):
                    raw_source = pairs.points[:, :2]
                    raw_expected = np.column_stack(
                        (
                            width / 2 + truth["stretch_x"] * (raw_source[:, 0] - width / 2),
                            height / 2 + truth["stretch_y"] * (raw_source[:, 1] - height / 2),
                        )
                    )
                    raw_errors = np.linalg.norm(pairs.points[:, 2:] - raw_expected, axis=1)
                raw_denominator = len(pairs.points)
                selection_timing = method_result["selection_timing"][selection]
                detection_pair_s = (
                    selection_timing["reference_detection_s_median"]
                    + selection_timing["target_detection_s_median"]
                )
                descriptor_pair_s = (
                    selection_timing["reference_descriptor_s_median"]
                    + selection_timing["target_descriptor_s_median"]
                )
                total_pair_s = detection_pair_s + descriptor_pair_s + matching_s + ransac_wall_s
                budget_result[selection] = {
                    "selected_keypoints": {
                        "reference": len(selected_a.features.xy),
                        "target": len(selected_b.features.xy),
                    },
                    "reference_distribution": keypoint_distribution(
                        selected_a.features.xy, width, height
                    ),
                    "target_distribution": keypoint_distribution(
                        selected_b.features.xy, width, height
                    ),
                    "detector_repeatability": repeat,
                    "raw_mutual_matches": raw_denominator,
                    "raw_correct_under_1px": int(np.count_nonzero(np.asarray(raw_errors) <= 1.0)),
                    "raw_correct_under_2px": int(np.count_nonzero(np.asarray(raw_errors) <= 2.0)),
                    "raw_correct_under_5px": int(np.count_nonzero(np.asarray(raw_errors) < 5.0)),
                    "raw_precision_under_5px": (
                        float(np.count_nonzero(np.asarray(raw_errors) < 5.0) / raw_denominator)
                        if raw_denominator
                        else 0.0
                    ),
                    "descriptor_matching_s": matching_s,
                    "descriptor_matching_s_repeats": match_times,
                    "detection_pair_s": detection_pair_s,
                    "descriptor_pair_s": descriptor_pair_s,
                    "wgsl_ransac_wall_s": ransac_wall_s,
                    "wgsl_ransac_compute_ms_median": ransac_compute_ms,
                    "accepted_affine_seeds": sum(row["usable"] for row in fit_rows),
                    "affine_seed_correct_under_2px": sum(
                        row["usable"] and row["analytic_error_px"] <= 2.0 for row in fit_rows
                    ),
                    "affine_seed_correct_under_5px": sum(
                        row["usable"] and row["analytic_error_px"] < 5.0 for row in fit_rows
                    ),
                    "accepted_affine_error_rmse_px": (
                        float(np.mean(np.square(accepted_errors)) ** 0.5)
                        if accepted_errors
                        else None
                    ),
                    "accepted_affine_error_median_px": (
                        float(np.median(accepted_errors)) if accepted_errors else None
                    ),
                    "accepted_affine_error_p95_px": (
                        float(np.percentile(accepted_errors, 95)) if accepted_errors else None
                    ),
                    "accepted_affine_error_max_px": (
                        float(np.max(accepted_errors)) if accepted_errors else None
                    ),
                    "total_pair_s_median": total_pair_s,
                    "pair_fps": 1.0 / total_pair_s if total_pair_s else 0.0,
                    "per_seed_affine": fit_rows,
                }
            method_result["budgets"][str(budget)] = budget_result
        result["detectors"][detector_name] = method_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progressive-root", type=Path, default=PROGRESSIVE)
    parser.add_argument("--stress-root", type=Path, default=STRESS)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument(
        "--dad-onnx",
        type=Path,
        help="Optional keypoint-mode DaD ONNX export using ONNX Runtime WebGPU",
    )
    parser.add_argument(
        "--dedode-onnx",
        type=Path,
        help="Optional keypoint-mode DeDoDe-v2 ONNX export using ONNX Runtime WebGPU",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        choices=(
            "DaD",
            "DeDoDe-v2-L",
            "DaD-ONNX-WebGPU",
            "DeDoDe-v2-ONNX-WebGPU",
            "DIC-Tensor-WGSL",
            "HORG-SIFT-common-DeDoDe-B",
            "DIC-Learned-WGSL",
        ),
        help="Optional detector subset; ONNX choices require their corresponding model path",
    )
    parser.add_argument(
        "--dic-weights",
        type=Path,
        default=ROOT / "results/features/dic_detector_training_20261107/dic_detector_weights.npz",
        help="Optional trained DIC-Detector weights; omit to compare only the tensor baseline",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--cases", nargs="*", help="Case names to run; default runs all available cases"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch reference device (default: cpu)")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=8,
        help="Set PyTorch CPU thread count for repeatable timing",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs after one warm-up")
    parser.add_argument(
        "--software-webgpu", action="store_true", help="Allow a software WebGPU adapter for RANSAC"
    )
    args = parser.parse_args()
    if (
        args.max_side < 64
        or args.candidate_multiplier < 1
        or args.torch_threads < 1
        or args.repeats < 1
        or any(value < 1 for value in args.budgets)
    ):
        parser.error(
            "max-side, candidate-multiplier, torch threads, and budgets must be positive "
            "(max-side >=64)"
        )

    cases = load_cases(args.progressive_root, args.stress_root)
    if args.cases:
        cases = [case for case in cases if case["name"] in args.cases]
        missing = sorted(set(args.cases) - {case["name"] for case in cases})
        if missing:
            parser.error(f"Unknown or unavailable cases: {', '.join(missing)}")
    budgets = tuple(sorted(set(args.budgets)))
    import torch

    torch.set_num_threads(args.torch_threads)
    model_load_start = time.perf_counter()
    detector_dad = DaDDetector(max_side=args.max_side, device=args.device)
    dad_load_s = time.perf_counter() - model_load_start
    model_load_start = time.perf_counter()
    detector_dedode = DeDoDeV2Detector(max_side=args.max_side, device=args.device)
    dedode_load_s = time.perf_counter() - model_load_start
    detectors = {
        "DaD": detector_dad,
        "DeDoDe-v2-L": detector_dedode,
    }
    for label, model_path in (
        ("DaD-ONNX-WebGPU", args.dad_onnx),
        ("DeDoDe-v2-ONNX-WebGPU", args.dedode_onnx),
    ):
        if model_path is None:
            continue
        try:
            onnx_detector = OnnxWGPUKeypointDetector(model_path, max_side=args.max_side)
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            parser.error(f"Could not initialize {label}: {error}")
        required_candidates = max(budgets) * args.candidate_multiplier
        if required_candidates > onnx_detector.keypoint_capacity:
            parser.error(
                f"{label} exports K={onnx_detector.keypoint_capacity}, but the selected budgets "
                f"need {required_candidates} candidates; reduce --budgets/--candidate-multiplier "
                "or re-export with a larger --keypoint-capacity"
            )
        detectors[label] = onnx_detector
    if args.dic_weights is not None and not args.dic_weights.is_file():
        parser.error(f"DIC-Detector weights were not found: {args.dic_weights}")
    model_load_start = time.perf_counter()
    descriptor = DeDoDeBDescriptor(max_side=args.max_side, device=args.device)
    descriptor_load_s = time.perf_counter() - model_load_start
    gpu_device_type = "auto" if args.software_webgpu else "gpu"
    rows = []
    checkpoint = args.output.with_suffix(".partial.json")
    with WGPUMatching(device_type=gpu_device_type) as matcher:
        detectors["DIC-Tensor-WGSL"] = WGPUTensorDetector(matcher)
        detectors["HORG-SIFT-common-DeDoDe-B"] = HORGCommonDescriptorDetector(matcher)
        if args.dic_weights is not None:
            detectors["DIC-Learned-WGSL"] = WGPUTensorDetector(
                matcher, learned_weights_path=args.dic_weights
            )
        if args.detectors:
            missing_detectors = sorted(set(args.detectors) - detectors.keys())
            if missing_detectors:
                parser.error(
                    "Requested detectors are not configured; provide their ONNX model paths "
                    f"or DIC weights as needed: {', '.join(missing_detectors)}"
                )
            detectors = {
                name: detector for name, detector in detectors.items() if name in args.detectors
            }
        for case in cases:
            print(f"case={case['name']} on {matcher.device_name}", flush=True)
            rows.append(
                run_case(
                    case,
                    detectors,
                    descriptor,
                    matcher,
                    budgets,
                    args.max_side,
                    args.candidate_multiplier,
                    args.repeats,
                )
            )
            checkpoint_metadata = {
                "protocol": "incomplete per-case checkpoint; do not use as final benchmark",
                "adapter": matcher.device_name,
                "model_max_side_px": args.max_side,
                "candidate_multiplier": args.candidate_multiplier,
                "steady_state_repeats": args.repeats,
                "torch_num_threads": args.torch_threads,
                "budgets": budgets,
                "completed_cases": [row["name"] for row in rows],
                "cases": rows,
            }
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps(checkpoint_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    metadata = {
        "protocol": "official PyTorch detector references; common DeDoDe-B descriptor; mutual cosine matching; existing WGSL local-affine RANSAC",
        "model_max_side_px": args.max_side,
        "candidate_multiplier": args.candidate_multiplier,
        "steady_state_repeats": args.repeats,
        "torch_num_threads": args.torch_threads,
        "budgets": budgets,
        "selection_methods": [
            "global_topk_from_returned_candidate_pool",
            "spatial_quotas_from_returned_candidate_pool",
        ],
        "descriptor_and_matching": "DeDoDe-B with mutual nearest cosine; matching currently runs on CPU",
        "selection_scope": "Global top-K and spatial quotas are applied to each detector's returned candidate pool; DIC-Detector itself uses WGSL per-tile quotas when building that pool. This is not a full-response-map global top-K comparison for DIC-Detector.",
        "ransac": {
            "implementation": "WGPU-Matching WGSL local-affine RANSAC",
            "radius_px": 256,
            "threshold_px": 3,
            "min_inliers": 6,
            "min_candidates": 8,
            "min_inlier_fraction": 0.2,
            "iterations": 256,
        },
        "adapter": matcher.device_name,
        "upstream_revisions": {
            "DaD": {
                key: detector_dad.metadata[key]
                for key in (
                    "source_revision",
                    "checkpoint_revision",
                    "weights_url",
                    "weights_sha256",
                )
            },
            "DeDoDe-v2-L": {
                key: detector_dedode.metadata[key]
                for key in (
                    "source_revision",
                    "checkpoint_revision",
                    "weights_url",
                    "weights_sha256",
                )
            },
            "DeDoDe-B": {
                key: descriptor.metadata[key]
                for key in (
                    "source_revision",
                    "checkpoint_revision",
                    "weights_url",
                    "weights_sha256",
                )
            },
        },
        "detectors": {
            name: {
                "model_load_s": {
                    "DaD": dad_load_s,
                    "DeDoDe-v2-L": dedode_load_s,
                }.get(name, 0.0),
                **detector.metadata,
            }
            for name, detector in detectors.items()
        },
        "descriptor": {"model_load_s": descriptor_load_s, **descriptor.metadata},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    report = args.output.with_suffix(".md")
    lines = [
        "# Shared-descriptor detector comparison",
        "",
        "Detector inference uses the configured reference or ONNX/WebGPU path recorded in each method's metadata. Common DeDoDe-B description and mutual cosine matching run on CPU; local-affine RANSAC runs on the recorded WebGPU adapter. Global top-K and spatial quotas act on each returned candidate pool. The DIC WGSL detector's candidate pool already uses its internal spatial tile quotas, so its global top-K is not over the full NMS response map.",
        "These results separate detector repeatability from descriptor matches and RANSAC-accepted initializations.",
        "",
        f"Model maximum side: {args.max_side}px. Keypoint selection candidates: {args.candidate_multiplier}× each requested budget. Adapter: {metadata['adapter']}. Timings use {args.torch_threads} CPU threads for the reference PyTorch components and are medians over {args.repeats} steady-state repeats after one warm-up.",
        "",
        "| Case | Detector | Budget | Selection | Repeatability ≤2 px | Raw matches | Raw <5 px | Affine accepted | Affine <5 px | Affine RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in rows:
        for detector_name, detector_row in case["detectors"].items():
            for budget, selections in detector_row["budgets"].items():
                for selection, values in selections.items():
                    affine_error = (
                        f"{values['accepted_affine_error_rmse_px']:.2f}/"
                        f"{values['accepted_affine_error_p95_px']:.2f}"
                        if values["accepted_affine_error_rmse_px"] is not None
                        else "—"
                    )
                    lines.append(
                        f"| {case['name']} | {detector_name} | {budget} | {selection} | "
                        f"{values['detector_repeatability']['at_2px']}/"
                        f"{values['detector_repeatability']['denominator_min_visible']} | "
                        f"{values['raw_mutual_matches']} | {values['raw_correct_under_5px']} | "
                        f"{values['accepted_affine_seeds']}/{case['visible_dic_seeds']} | "
                        f"{values['affine_seed_correct_under_5px']} | "
                        f"{affine_error} | "
                        f"{values['detection_pair_s']:.3f} | "
                        f"{values['descriptor_pair_s']:.3f} | "
                        f"{values['descriptor_matching_s']:.3f} | "
                        f"{values['wgsl_ransac_compute_ms_median']:.2f} | "
                        f"{values['total_pair_s_median']:.3f} | {values['pair_fps']:.3f} |"
                    )
    lines.extend(
        (
            "",
            "Raw per-seed data, timing scopes, checkpoint URLs, and SHA-256 digests are in the adjacent JSON file.",
            "",
        )
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {report}", flush=True)


if __name__ == "__main__":
    main()
