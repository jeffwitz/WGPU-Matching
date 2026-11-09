"""Run the learned coarse pair field through the existing WebGPU IC-GN solver."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ndf_field import (
    DenseBilinearDisplacement,
    fit_bspline_field,
    recommended_control_shape,
    uniform_points,
)
from ndf_pair_model import NDFPairMatcher, PairPrediction, refine_local_features, refine_local_ncc
from train_ndf_pair_matcher import local_affine_consensus
from wgpu_matching.api import WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair


def centres(width: int, height: int, subset: int = 64, stride: int = 64):
    x = (subset - 1) / 2 + stride * np.arange(1 + (width - subset) // stride)
    y = (subset - 1) / 2 + stride * np.arange(1 + (height - subset) // stride)
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2).astype(np.float32)


def truth_parameters(pair, points):
    flow = pair.flow_xy.astype(np.float64)
    du_dy, du_dx = np.gradient(flow[..., 0])
    dv_dy, dv_dx = np.gradient(flow[..., 1])
    x = np.clip(np.rint(points[:, 0]).astype(int), 1, flow.shape[1] - 2)
    y = np.clip(np.rint(points[:, 1]).astype(int), 1, flow.shape[0] - 2)
    return np.column_stack(
        (flow[y, x, 0], du_dx[y, x], du_dy[y, x], flow[y, x, 1], dv_dx[y, x], dv_dy[y, x])
    ).astype(np.float32)


def rotate_target_about_image_center(image: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Diagnostic warp that removes only a known global target rotation."""
    _, _, height, width = image.shape
    dtype, device = image.dtype, image.device
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    source_x = cosine * (x - center_x) - sine * (y - center_y) + center_x
    source_y = sine * (x - center_x) + cosine * (y - center_y) + center_y
    grid = torch.stack(
        (2 * (source_x + 0.5) / width - 1, 2 * (source_y + 0.5) / height - 1),
        dim=-1,
    )[None].expand(image.shape[0], -1, -1, -1)
    sampled = torch.nn.functional.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    valid = torch.nn.functional.grid_sample(
        torch.ones_like(image), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled + 0.5 * (1.0 - valid)


def unrotate_target_displacement(
    displacement_xy: torch.Tensor,
    image_size: tuple[int, int],
    angle_rad: float,
) -> torch.Tensor:
    """Map a field estimated in the rotated target raster back to target pixels."""
    _, field_height, field_width, _ = displacement_xy.shape
    image_width, image_height = image_size
    dtype, device = displacement_xy.dtype, displacement_xy.device
    y, x = torch.meshgrid(
        torch.arange(field_height, dtype=dtype, device=device),
        torch.arange(field_width, dtype=dtype, device=device),
        indexing="ij",
    )
    stride_x, stride_y = image_width / field_width, image_height / field_height
    source_x = (x + 0.5) * stride_x - 0.5
    source_y = (y + 0.5) * stride_y - 0.5
    target_x = source_x + displacement_xy[0, ..., 0]
    target_y = source_y + displacement_xy[0, ..., 1]
    center_x, center_y = (image_width - 1) / 2, (image_height - 1) / 2
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    original_x = cosine * (target_x - center_x) - sine * (target_y - center_y) + center_x
    original_y = sine * (target_x - center_x) + cosine * (target_y - center_y) + center_y
    return torch.stack((original_x - source_x, original_y - source_y), dim=-1)[None]


def fit_global_affine_ransac(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    threshold: float = 5.0,
    min_inliers: int = 24,
    iterations: int = 128,
    seed: int = 417,
) -> tuple[np.ndarray | None, np.ndarray, float | None]:
    """Fit a robust source-to-target affine map from image-derived matches.

    Returns coefficients for ``[1, x, y] @ coefficients``, an inlier mask, and
    the median inlier residual. No synthetic truth or visibility mask is used.
    """
    source_xy = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    if source_xy.shape != target_xy.shape:
        raise ValueError("source_xy and target_xy must have matching shapes")
    finite = np.isfinite(source_xy).all(axis=1) & np.isfinite(target_xy).all(axis=1)
    design = np.column_stack((np.ones(len(source_xy)), source_xy))
    best = np.zeros(len(source_xy), dtype=bool)
    best_median = float("inf")
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(finite)
    if len(candidates) < 3:
        return None, best, None
    for _ in range(iterations):
        chosen = rng.choice(candidates, size=3, replace=False)
        sample_design = design[chosen]
        if abs(np.linalg.det(sample_design)) < 1e-7:
            continue
        hypothesis = np.linalg.solve(sample_design, target_xy[chosen])
        residual = np.linalg.norm(design @ hypothesis - target_xy, axis=1)
        inliers = finite & (residual <= threshold)
        median = float(np.median(residual[inliers])) if inliers.any() else float("inf")
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and median < best_median):
            best, best_median = inliers, median
    if int(best.sum()) < min_inliers:
        return None, best, best_median if best.any() else None

    fit_design = design[best]
    fit_target = target_xy[best]
    weights = np.ones(len(fit_design), dtype=np.float64)
    coefficients = np.linalg.lstsq(fit_design, fit_target, rcond=None)[0]
    for _ in range(8):
        residual = np.linalg.norm(fit_design @ coefficients - fit_target, axis=1)
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 0.25)
        cutoff = 1.345 * scale
        weights = np.where(residual <= cutoff, 1.0, cutoff / np.maximum(residual, 1e-12))
        root_weights = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(
            fit_design * root_weights, fit_target * root_weights, rcond=None
        )[0]
    residual = np.linalg.norm(design @ coefficients - target_xy, axis=1)
    best = finite & (residual <= threshold)
    if int(best.sum()) < min_inliers:
        return None, best, None
    return coefficients, best, float(np.median(residual[best]))


def rotation_refinement_candidates(
    candidate_scores: list[dict[str, float | int]],
    *,
    coarse_steps: int,
    top_k: int,
) -> list[float]:
    """Return half-step angles around the best coarse hypotheses.

    Scores must already be computed from image-derived matching. The returned
    angles are wrapped to [-pi, pi) and exclude angles already evaluated.
    """
    if coarse_steps < 1 or top_k < 1:
        return []
    step = 2.0 * np.pi / coarse_steps
    ranked = sorted(
        candidate_scores,
        key=lambda item: (float(item["ranking_score"]), int(item["consensus_support"])),
        reverse=True,
    )[:top_k]
    existing = [float(item["angle_rad"]) for item in candidate_scores]
    refined: list[float] = []
    for item in ranked:
        center = float(item["angle_rad"])
        for offset in (-0.5 * step, 0.5 * step):
            angle = (center + offset + np.pi) % (2.0 * np.pi) - np.pi
            if any(abs((angle - old + np.pi) % (2.0 * np.pi) - np.pi) < 1e-8
                   for old in (*existing, *refined)):
                continue
            refined.append(float(angle))
    return refined


def rotation_bank_angles(steps: int) -> np.ndarray:
    """Build the uniform angular grid and guarantee that identity is tested."""
    if steps < 1:
        raise ValueError("rotation-bank steps must be positive")
    return np.unique(np.r_[np.linspace(-np.pi, np.pi, steps, endpoint=False), 0.0])


def rectify_target_with_affine(
    target: torch.Tensor, coefficients: np.ndarray
) -> torch.Tensor:
    """Sample target at an image-derived affine map of reference coordinates."""
    if target.ndim != 4 or target.shape[:2] != (1, 1):
        raise ValueError("target must have shape [1, 1, H, W]")
    _, _, height, width = target.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=target.device, dtype=target.dtype),
        torch.arange(width, device=target.device, dtype=target.dtype),
        indexing="ij",
    )
    design = torch.stack((torch.ones_like(xx), xx, yy), dim=-1)
    affine = torch.as_tensor(coefficients, dtype=target.dtype, device=target.device)
    mapped = design @ affine
    grid = torch.stack(
        (2 * (mapped[..., 0] + 0.5) / width - 1,
         2 * (mapped[..., 1] + 0.5) / height - 1),
        dim=-1,
    )[None]
    sampled = torch.nn.functional.grid_sample(
        target, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    valid = torch.nn.functional.grid_sample(
        torch.ones_like(target), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled + 0.5 * (1.0 - valid)


def fit_local_bspline_prefit(
    prediction: PairPrediction,
    image_size: tuple[int, int],
    *,
    radius: float,
    threshold: float,
    min_support: int,
    bending: float,
) -> tuple[object | None, dict[str, float | int]]:
    """Build a smooth image-derived prefit from local affine match consensus."""
    _, field_height, field_width, _ = prediction.hard_displacement_xy.shape
    image_width, image_height = image_size
    yy, xx = torch.meshgrid(
        torch.arange(field_height, dtype=torch.float64),
        torch.arange(field_width, dtype=torch.float64),
        indexing="ij",
    )
    source_xy = torch.stack(
        ((xx + 0.5) * image_width / field_width - 0.5,
         (yy + 0.5) * image_height / field_height - 0.5),
        dim=-1,
    ).reshape(-1, 2)
    target_xy = source_xy + prediction.hard_displacement_xy[0].reshape(-1, 2).to(torch.float64)
    parameters, usable = local_affine_consensus(
        source_xy.numpy(), target_xy.numpy(), image_width, image_height,
        radius=radius, threshold=threshold,
    )
    support = int(usable.sum())
    diagnostics: dict[str, float | int] = {
        "support": support,
        "support_fraction": support / max(len(source_xy), 1),
    }
    if support < min_support:
        return None, diagnostics

    affine_target = np.column_stack(
        (
            parameters[:, 0] + (1.0 + parameters[:, 1]) * source_xy[:, 0].numpy()
            + parameters[:, 2] * source_xy[:, 1].numpy(),
            parameters[:, 3] + parameters[:, 4] * source_xy[:, 0].numpy()
            + (1.0 + parameters[:, 5]) * source_xy[:, 1].numpy(),
        )
    )
    local_flow = torch.from_numpy(affine_target[usable] - source_xy.numpy()[usable])
    fit_points = source_xy[torch.from_numpy(usable)]
    shapes = tuple(
        recommended_control_shape(image_size, spacing) for spacing in (32.0, 16.0)
    )
    field = fit_bspline_field(
        fit_points, local_flow, image_size, shapes, bending=bending
    )
    check_xy = torch.tensor(
        [[x, y] for y in np.linspace(0, image_height - 1, 12)
         for x in np.linspace(0, image_width - 1, 16)],
        dtype=torch.float64,
    )
    _, jacobian = field.evaluate(check_xy)
    determinant = torch.linalg.det(jacobian)
    diagnostics.update(
        {
            "jacobian_det_min": float(determinant.min()),
            "jacobian_det_max": float(determinant.max()),
            "jacobian_valid_fraction": float(((determinant > 0.05) & (determinant < 20)).double().mean()),
        }
    )
    if diagnostics["jacobian_valid_fraction"] < 0.98:
        return None, diagnostics
    observed_flow, _ = field.evaluate(fit_points)
    diagnostics["median_fit_residual"] = float(
        torch.linalg.vector_norm(observed_flow - local_flow, dim=1).median()
    )
    return field, diagnostics


def rectify_target_with_field(target: torch.Tensor, field) -> torch.Tensor:
    """Pull the target back through a sampled, image-derived displacement field."""
    _, _, height, width = target.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=target.dtype),
        torch.arange(width, dtype=target.dtype),
        indexing="ij",
    )
    points = torch.stack((xx, yy), dim=-1).reshape(-1, 2).to(torch.float64)
    displacement, _ = field.evaluate(points)
    mapped = points.to(target.dtype) + displacement.to(target.dtype)
    grid = torch.stack(
        (2 * (mapped[:, 0].reshape(height, width) + 0.5) / width - 1,
         2 * (mapped[:, 1].reshape(height, width) + 0.5) / height - 1),
        dim=-1,
    )[None]
    sampled = torch.nn.functional.grid_sample(
        target, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    valid = torch.nn.functional.grid_sample(
        torch.ones_like(target), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled + 0.5 * (1.0 - valid)


def compose_field_residual_flow(
    residual_flow: torch.Tensor,
    prefit_field,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Compose an image-derived spline prefit with the residual pair-model flow."""
    image_width, image_height = image_size
    _, field_height, field_width, _ = residual_flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(field_height, dtype=residual_flow.dtype),
        torch.arange(field_width, dtype=residual_flow.dtype),
        indexing="ij",
    )
    source_xy = torch.stack(
        ((xx + 0.5) * image_width / field_width - 0.5,
         (yy + 0.5) * image_height / field_height - 0.5),
        dim=-1,
    )
    residual_target_xy = source_xy + residual_flow[0]
    prefit_flow, _ = prefit_field.evaluate(residual_target_xy.reshape(-1, 2).to(torch.float64))
    final_target = residual_target_xy.reshape(-1, 2).to(torch.float64) + prefit_flow
    return (final_target.reshape(field_height, field_width, 2).to(residual_flow.dtype)
            - source_xy)[None]


def compose_affine_residual_flow(
    residual_flow: torch.Tensor,
    coefficients: np.ndarray,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Compose a global source-to-target affine map with a dense residual field."""
    if residual_flow.ndim != 4 or residual_flow.shape[0] != 1 or residual_flow.shape[-1] != 2:
        raise ValueError("residual_flow must have shape [1, H, W, 2]")
    image_width, image_height = image_size
    _, field_height, field_width, _ = residual_flow.shape
    dtype, device = residual_flow.dtype, residual_flow.device
    yy, xx = torch.meshgrid(
        torch.arange(field_height, device=device, dtype=dtype),
        torch.arange(field_width, device=device, dtype=dtype),
        indexing="ij",
    )
    source_xy = torch.stack(
        ((xx + 0.5) * image_width / field_width - 0.5,
         (yy + 0.5) * image_height / field_height - 0.5),
        dim=-1,
    )
    residual_target_xy = source_xy + residual_flow[0]
    design = torch.cat((torch.ones_like(residual_target_xy[..., :1]), residual_target_xy), dim=-1)
    affine = torch.as_tensor(coefficients, dtype=dtype, device=device)
    target_xy = design @ affine
    return (target_xy - source_xy)[None]


@torch.no_grad()
def probe_rotation_hypothesis(
    model: NDFPairMatcher,
    reference: torch.Tensor,
    target: torch.Tensor,
    query_budget: int,
    local_radius: float,
    threshold: float,
    consensus_method: str,
    reference_features: torch.Tensor | None = None,
) -> tuple[int, float, int, float, float, float]:
    """Rank one orientation using sparse reciprocal feature matches only."""
    def normalize(image):
        centered = image - image.mean(dim=(-2, -1), keepdim=True)
        scale = centered.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-4)
        return centered / scale

    encoding_start = time.perf_counter()
    if reference_features is None:
        reference_features = model.encoder(normalize(reference))
    target_features = model.encoder(normalize(target))
    encoding_s = time.perf_counter() - encoding_start
    _, _, feature_height, feature_width = reference_features.shape
    query_rows = max(1, round(np.sqrt(query_budget * feature_height / feature_width)))
    query_columns = max(1, query_budget // query_rows)
    rows = torch.linspace(0, feature_height - 1, query_rows).round().long().unique()
    columns = torch.linspace(0, feature_width - 1, query_columns).round().long().unique()
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    query_indices = (yy * feature_width + xx).reshape(-1)
    first = reference_features.flatten(2).transpose(1, 2)[0, query_indices]
    candidates = target_features.flatten(2).transpose(1, 2)[0]
    matching_start = time.perf_counter()
    similarities = first @ candidates.T
    temperature = model.log_temperature.exp().clamp(0.025, 0.5)
    confidence = (similarities / temperature).softmax(dim=-1).amax(dim=-1)
    target_indices = similarities.argmax(dim=-1)
    reverse_indices = similarities.argmax(dim=0)
    reciprocal = reverse_indices[target_indices] == torch.arange(
        len(query_indices), device=target_indices.device
    )
    matching_s = time.perf_counter() - matching_start

    stride = target.new_tensor(
        [target.shape[-1] / feature_width, target.shape[-2] / feature_height]
    )
    grid_y, grid_x = torch.meshgrid(
        rows.to(target.dtype), columns.to(target.dtype), indexing="ij"
    )
    source_xy = (torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2) + 0.5) * stride - 0.5
    candidate_y = torch.div(target_indices, feature_width, rounding_mode="floor")
    candidate_x = target_indices % feature_width
    target_xy = (
        torch.stack((candidate_x, candidate_y), dim=-1).to(target.dtype) + 0.5
    ) * stride - 0.5
    source_matches = source_xy[reciprocal]
    target_matches = target_xy[reciprocal]
    match_confidence = confidence[reciprocal]
    support = 0
    mean_confidence = 0.0
    consensus_s = 0.0
    if len(source_matches) >= 8:
        consensus_start = time.perf_counter()
        source_array = source_matches.cpu().numpy()
        target_array = target_matches.cpu().numpy()
        confidence_array = match_confidence.cpu().numpy()
        if consensus_method == "local":
            _, usable = local_affine_consensus(
                source_array,
                target_array,
                target.shape[-1],
                target.shape[-2],
                radius=local_radius,
                threshold=threshold,
            )
            support = int(usable.sum())
        else:
            in_bounds = (
                (target_array[:, 0] >= 0)
                & (target_array[:, 0] < target.shape[-1])
                & (target_array[:, 1] >= 0)
                & (target_array[:, 1] < target.shape[-2])
            )
            source_array = source_array[in_bounds]
            target_array = target_array[in_bounds]
            confidence_array = confidence_array[in_bounds]
            design = np.column_stack((np.ones(len(source_array)), source_array))
            if len(source_array) >= 8:
                rng = np.random.default_rng(417)
                best = np.zeros(len(source_array), dtype=bool)
                best_median = float("inf")
                for _ in range(32):
                    chosen = rng.choice(len(source_array), size=3, replace=False)
                    if abs(np.linalg.det(design[chosen])) < 1e-7:
                        continue
                    coefficients = np.linalg.solve(design[chosen], target_array[chosen])
                    residual = np.linalg.norm(design @ coefficients - target_array, axis=1)
                    inliers = residual <= threshold
                    median = float(np.median(residual[inliers])) if inliers.any() else float("inf")
                    if inliers.sum() > best.sum() or (
                        inliers.sum() == best.sum() and median < best_median
                    ):
                        best, best_median = inliers, median
                if best.sum() >= 6:
                    coefficients = np.linalg.lstsq(design[best], target_array[best], rcond=None)[0]
                    residual = np.linalg.norm(design @ coefficients - target_array, axis=1)
                    support = int((residual <= threshold).sum())
                    usable = residual <= threshold
                else:
                    usable = np.zeros(len(source_array), dtype=bool)
                    support = 0
            else:
                usable = np.zeros(len(source_array), dtype=bool)
                support = 0
        consensus_s = time.perf_counter() - consensus_start
        if support:
            mean_confidence = float(confidence_array[usable].mean())
    return support, mean_confidence, int(reciprocal.sum()), encoding_s, matching_s, consensus_s


def robust_affine_projection(source_xy, displacement_xy, evaluation_xy):
    """Fit an affine map using deterministic RANSAC followed by robust least squares."""
    source = source_xy.to(dtype=torch.float64)
    target = source + displacement_xy.to(dtype=torch.float64)
    design = torch.cat((torch.ones_like(source[:, :1]), source), dim=1)
    generator = torch.Generator(device=source.device).manual_seed(417)
    best_inliers = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
    best_count = 0
    best_median = float("inf")
    for _ in range(256):
        sample = torch.randperm(source.shape[0], generator=generator, device=source.device)[:3]
        sample_design = design[sample]
        if torch.abs(torch.linalg.det(sample_design)) < 1e-8:
            continue
        hypothesis = torch.linalg.solve(sample_design, target[sample])
        residual = torch.linalg.vector_norm(design @ hypothesis - target, dim=1)
        inliers = residual <= 2.5
        count = int(inliers.sum())
        median = float(torch.median(residual[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and median < best_median):
            best_inliers = inliers
            best_count = count
            best_median = median
    if best_count < 3:
        best_inliers[:] = True
    fit_design = design[best_inliers]
    fit_target = target[best_inliers]
    weights = torch.ones(fit_design.shape[0], dtype=source.dtype, device=source.device)
    for _ in range(32):
        root_weight = weights.sqrt()[:, None]
        coefficients = torch.linalg.lstsq(
            fit_design * root_weight, fit_target * root_weight
        ).solution
        residual = torch.linalg.vector_norm(fit_design @ coefficients - fit_target, dim=1)
        median = torch.median(residual)
        scale = (1.4826 * torch.median(torch.abs(residual - median))).clamp_min(0.25)
        threshold = 1.345 * scale
        weights = torch.where(
            residual <= threshold,
            torch.ones_like(residual),
            threshold / residual.clamp_min(1e-12),
        )
    evaluation = evaluation_xy.to(dtype=torch.float64)
    evaluation_design = torch.cat((torch.ones_like(evaluation[:, :1]), evaluation), dim=1)
    mapped = evaluation_design @ coefficients
    gradient = coefficients[1:, :].T.contiguous()
    return mapped - evaluation, gradient.expand(len(evaluation), -1, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-scale", type=int, default=8)
    parser.add_argument("--native-width", type=int)
    parser.add_argument("--native-height", type=int)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--model-height", type=int, default=64)
    parser.add_argument("--speckle-size", type=float, default=3.5,
                        help="characteristic speckle size at model input resolution")
    parser.add_argument("--local-ncc-radius", type=int, default=0,
                        help="optional local NCC search radius in model-input pixels; zero disables refinement")
    parser.add_argument("--local-ncc-geometry", choices=("translation", "affine", "hybrid"), default="translation",
                        help="compare fixed patches, warp by the predicted Jacobian, or search both")
    parser.add_argument("--local-ncc-jacobian-damping", type=float, default=1.0,
                        help="blend affine patch warps toward identity (0=fixed patches, 1=full predicted Jacobian)")
    parser.add_argument("--local-ncc-backend", choices=("cpu", "wgpu"), default="cpu")
    parser.add_argument("--local-ncc-implementation", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--coarse-estimate", choices=("soft", "topk8", "hard"), default="soft")
    parser.add_argument("--contextual-features", action="store_true",
                        help="load a checkpoint trained with dilated contextual feature convolutions")
    parser.add_argument(
        "--oracle-rotation-rectification",
        action="store_true",
        help="diagnostic only: remove the known synthetic global rotation before matching",
    )
    parser.add_argument(
        "--rotation-bank-steps",
        type=int,
        default=0,
        help=(
            "base grid count for the no-oracle global rotation bank over [-pi, pi); "
            "identity is added when the grid omits zero"
        ),
    )
    parser.add_argument(
        "--rotation-bank-selection",
        choices=("confidence", "support", "confidence_support"),
        default="confidence_support",
        help="rank hypotheses by confidence, support, or their normalized product",
    )
    parser.add_argument(
        "--rotation-bank-batch-size",
        type=int,
        default=1,
        help="number of rotated targets evaluated in one model batch",
    )
    parser.add_argument(
        "--rotation-bank-min-score-gain",
        "--rotation-bank-min-confidence-gain",
        dest="rotation_bank_min_score_gain",
        type=float,
        default=0.1,
        help="fall back to identity unless the selected ranking score improves by this fraction",
    )
    parser.add_argument(
        "--rotation-bank-probe-query-budget",
        type=int,
        default=0,
        help="if positive, rank each angle using sparse reciprocal queries, then run the full model once",
    )
    parser.add_argument(
        "--rotation-bank-refine-topk",
        type=int,
        default=0,
        help=(
            "after sparse coarse angle scoring, evaluate the two half-step angles "
            "around each of the top K hypotheses"
        ),
    )
    parser.add_argument(
        "--rotation-bank-consensus",
        choices=("global", "local"),
        default="global",
        help="use one global affine RANSAC or local affine RANSAC to rank sparse angle probes",
    )
    parser.add_argument(
        "--rotation-bank-min-consensus-support",
        type=int,
        default=24,
        help="reject an angle hypothesis with fewer local-affine consensus nodes",
    )
    parser.add_argument(
        "--rotation-bank-min-angle-deg",
        type=float,
        default=0.0,
        help="ignore nonzero angle hypotheses smaller than this magnitude",
    )
    parser.add_argument(
        "--global-affine-prefit",
        action="store_true",
        help=(
            "fit a robust image-derived global affine map from coarse hard matches, "
            "rectify the target, then predict and compose a nonlinear residual field"
        ),
    )
    parser.add_argument(
        "--local-bspline-prefit",
        action="store_true",
        help=(
            "fit a spatially varying cubic B-spline prefit from local affine consensus, "
            "rectify the target, then predict and compose a nonlinear residual field"
        ),
    )
    parser.add_argument(
        "--local-bspline-prefit-min-support",
        type=int,
        default=96,
        help="minimum number of locally supported grid points before fitting a B-spline prefit",
    )
    parser.add_argument(
        "--local-bspline-prefit-bending",
        type=float,
        default=0.1,
        help="bending penalty used when fitting the local-consensus B-spline prefit",
    )
    parser.add_argument(
        "--affine-prefit-min-inliers",
        type=int,
        default=24,
        help="minimum RANSAC inlier matches required before affine target rectification",
    )
    parser.add_argument(
        "--affine-prefit-min-linear-deviation",
        type=float,
        default=0.03,
        help=(
            "require the fitted 2x2 linear map to differ from identity by this Frobenius "
            "norm before affine rectification; suppresses affine prefit on translations"
        ),
    )
    parser.add_argument("--local-ncc-patch-radius", type=int, default=4,
                        help="half-width of the local NCC patch in model-input pixels")
    parser.add_argument("--local-consensus-radius", type=float, default=20.0,
                        help="source-space radius for local affine consensus, in model-input pixels")
    parser.add_argument("--local-consensus-threshold", type=float, default=5.0,
                        help="RANSAC residual threshold in model-input pixels")
    parser.add_argument("--local-feature-radius-cells", type=int, default=0,
                        help="optional local CNN-feature search radius; zero disables refinement")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument(
        "--projection",
        choices=("multilevel_spline", "single_spline", "raw_bilinear", "local_consensus", "global_affine"),
        default="multilevel_spline",
    )
    args = parser.parse_args()
    if not 0.0 <= args.local_ncc_jacobian_damping <= 1.0:
        parser.error("--local-ncc-jacobian-damping must be between 0 and 1")
    if args.rotation_bank_steps < 0:
        parser.error("--rotation-bank-steps cannot be negative")
    if args.rotation_bank_batch_size < 1:
        parser.error("--rotation-bank-batch-size must be positive")
    if args.rotation_bank_min_score_gain < 0:
        parser.error("--rotation-bank-min-score-gain cannot be negative")
    if args.rotation_bank_probe_query_budget < 0:
        parser.error("--rotation-bank-probe-query-budget cannot be negative")
    if args.rotation_bank_refine_topk < 0:
        parser.error("--rotation-bank-refine-topk cannot be negative")
    if args.rotation_bank_refine_topk and not (
        args.rotation_bank_steps and args.rotation_bank_probe_query_budget
    ):
        parser.error("--rotation-bank-refine-topk requires sparse rotation probes")
    if args.rotation_bank_min_consensus_support < 0:
        parser.error("--rotation-bank-min-consensus-support cannot be negative")
    if not 0.0 <= args.rotation_bank_min_angle_deg <= 180.0:
        parser.error("--rotation-bank-min-angle-deg must be in [0, 180]")
    if args.oracle_rotation_rectification and args.rotation_bank_steps:
        parser.error("oracle rotation and a no-oracle rotation bank are mutually exclusive")
    if args.oracle_rotation_rectification and args.global_affine_prefit:
        parser.error("oracle rotation and image-derived affine prefit cannot be combined")
    if args.global_affine_prefit and args.local_bspline_prefit:
        parser.error("global affine and local B-spline prefit are mutually exclusive")
    if args.oracle_rotation_rectification and args.local_bspline_prefit:
        parser.error("oracle rotation and image-derived B-spline prefit cannot be combined")
    if args.local_bspline_prefit_min_support < 8:
        parser.error("--local-bspline-prefit-min-support must be at least 8")
    if args.local_bspline_prefit_bending < 0:
        parser.error("--local-bspline-prefit-bending cannot be negative")
    if args.affine_prefit_min_inliers < 6:
        parser.error("--affine-prefit-min-inliers must be at least 6")
    if args.affine_prefit_min_linear_deviation < 0:
        parser.error("--affine-prefit-min-linear-deviation cannot be negative")
    sys.path.insert(0, str(args.live_dic_source))
    from live_dic import ICGNWGPU

    small_width, small_height = args.model_width, args.model_height
    width = args.native_width or small_width * args.native_scale
    height = args.native_height or small_height * args.native_scale
    scale_x, scale_y = width / small_width, height / small_height
    texture_scale = 0.5 * (scale_x + scale_y)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = NDFPairMatcher(contextual_features=args.contextual_features)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()
    solver = ICGNWGPU(
        device_type="gpu",
        prefer_subgroups=False,
        subset_size=64,
        subset_stride=64,
        interpolation="linear",
        motion_model="affine",
        outlier_policy="disabled",
    )
    ndf_wgpu = WGPUMatching(device_type="gpu") if args.local_ncc_backend == "wgpu" else None
    report = {
        "protocol": {
            "model_input_px": [small_width, small_height],
            "speckle_size_px_at_model_input": args.speckle_size,
            "generated_native_px": [width, height],
            "scale_xy": [scale_x, scale_y],
            "candidates_per_query": ((small_width + 3) // 4) * ((small_height + 3) // 4),
            "families": ["translation", "heterogeneous"],
            "training_weights": str(args.weights),
            "analytic_truth_used_by_inference": bool(args.oracle_rotation_rectification),
            "oracle_rotation_rectification": bool(args.oracle_rotation_rectification),
            "rotation_bank_steps": args.rotation_bank_steps,
            "rotation_bank_orientation_count": (
                len(rotation_bank_angles(args.rotation_bank_steps))
                if args.rotation_bank_steps else 0
            ),
            "rotation_bank_selection": (
                f"maximum {args.rotation_bank_selection} with "
                f"{args.rotation_bank_consensus} affine consensus on predicted hard matches"
                if args.rotation_bank_steps else None
            ),
            "rotation_bank_batch_size": args.rotation_bank_batch_size,
            "rotation_bank_min_score_gain": args.rotation_bank_min_score_gain,
            "rotation_bank_probe_query_budget": args.rotation_bank_probe_query_budget,
            "rotation_bank_refine_topk": args.rotation_bank_refine_topk,
            "rotation_bank_consensus": args.rotation_bank_consensus,
            "rotation_bank_min_consensus_support": args.rotation_bank_min_consensus_support,
            "rotation_bank_min_angle_deg": args.rotation_bank_min_angle_deg,
            "global_affine_prefit": args.global_affine_prefit,
            "local_bspline_prefit": args.local_bspline_prefit,
            "local_bspline_prefit_min_support": args.local_bspline_prefit_min_support,
            "local_bspline_prefit_bending": args.local_bspline_prefit_bending,
            "affine_prefit_min_inliers": args.affine_prefit_min_inliers,
            "affine_prefit_min_linear_deviation": args.affine_prefit_min_linear_deviation,
            "visibility_mask_used_to_activate_windows": False,
            "solver": "WebGPU IC-GN, affine, 64 px subset and stride",
            "field_projection": args.projection,
            "local_ncc_search_radius_px": args.local_ncc_radius,
            "local_ncc_geometry": args.local_ncc_geometry,
            "local_ncc_jacobian_damping": args.local_ncc_jacobian_damping,
            "local_ncc_backend": args.local_ncc_backend,
            "local_ncc_implementation": args.local_ncc_implementation,
            "coarse_estimate": args.coarse_estimate,
            "contextual_features": args.contextual_features,
            "local_ncc_adapter": ndf_wgpu.device_name if ndf_wgpu is not None else None,
            "local_ncc_patch_radius_px": args.local_ncc_patch_radius,
            "local_consensus_radius_px": args.local_consensus_radius,
            "local_consensus_threshold_px": args.local_consensus_threshold,
            "local_feature_search_radius_cells": args.local_feature_radius_cells,
        },
        "cases": [],
    }
    for mode_index, mode in enumerate(("translation", "heterogeneous")):
        for sample in range(args.cases_per_family):
            texture_seed = 51000 + args.seed_offset + mode_index + 2 * sample
            warp_seed = 91000 + args.seed_offset + mode_index + 2 * sample
            pair = generate_speckle_pair(
                texture_seed,
                warp_seed,
                width,
                height,
                motion_mode=mode,
                texture_options={
                    "characteristic_size_px": args.speckle_size * texture_scale,
                    "coverage": 0.31,
                    "blur_sigma_px": (0.45 * args.speckle_size / 3.5) * texture_scale,
                    # Match the dimensionless speckle edge softness used by
                    # training, independent of native image scale.
                    "softness": 0.612,
                    "noise_std": 0.025,
                },
            )
            reference_tensor = torch.nn.functional.interpolate(
                torch.from_numpy(pair.reference.copy())[None, None],
                size=(small_height, small_width),
                mode="area",
            )
            target_tensor = torch.nn.functional.interpolate(
                torch.from_numpy(pair.target.copy())[None, None],
                size=(small_height, small_width),
                mode="area",
            )
            original_target_tensor = target_tensor
            oracle_angle = float(pair.metadata["angle_rad"]) if args.oracle_rotation_rectification else 0.0
            selected_rotation = oracle_angle
            if args.oracle_rotation_rectification:
                target_tensor = rotate_target_about_image_center(target_tensor, oracle_angle)
            inference_start = time.perf_counter()
            local_ncc = None
            local_feature = None
            consensus_usable_fraction = None
            rotation_bank_candidate_scores = None
            rotation_probe_encoder_s = 0.0
            rotation_probe_matching_s = 0.0
            rotation_probe_consensus_s = 0.0
            affine_prefit_coefficients = None
            affine_prefit_inliers = 0
            affine_prefit_median_residual = None
            affine_prefit_spatial_bins = 0
            affine_prefit_linear_deviation = None
            local_bspline_prefit_field = None
            local_bspline_prefit_diagnostics = None
            with torch.no_grad():
                model_inference_start = time.perf_counter()
                if args.rotation_bank_steps and args.rotation_bank_probe_query_budget:
                    bank_angles = rotation_bank_angles(args.rotation_bank_steps)
                    reference_centered = reference_tensor - reference_tensor.mean(
                        dim=(-2, -1), keepdim=True
                    )
                    reference_scale = reference_centered.square().mean(
                        dim=(-2, -1), keepdim=True
                    ).sqrt().clamp_min(1e-4)
                    cached_reference_features = model.encoder(reference_centered / reference_scale)
                    best_bank_score = (-1, -float("inf"))
                    identity_bank_score = None
                    identity_mean_confidence = 0.0
                    best_mean_confidence = 0.0
                    selected_rotation_target = None
                    rotation_bank_candidate_scores = []
                    for bank_angle in bank_angles:
                        candidate_target = rotate_target_about_image_center(
                            target_tensor, float(bank_angle)
                        )
                        (
                            support,
                            mean_confidence,
                            reciprocal_count,
                            probe_encoder_s,
                            probe_matching_s,
                            probe_consensus_s,
                        ) = probe_rotation_hypothesis(
                            model,
                            reference_tensor,
                            candidate_target,
                            args.rotation_bank_probe_query_budget,
                            args.local_consensus_radius,
                            args.local_consensus_threshold,
                            args.rotation_bank_consensus,
                            reference_features=cached_reference_features,
                        )
                        rotation_probe_encoder_s += probe_encoder_s
                        rotation_probe_matching_s += probe_matching_s
                        rotation_probe_consensus_s += probe_consensus_s
                        support_fraction = support / max(args.rotation_bank_probe_query_budget, 1)
                        primary_score = (
                            mean_confidence
                            if args.rotation_bank_selection == "confidence"
                            else support_fraction
                            if args.rotation_bank_selection == "support"
                            else mean_confidence * support_fraction
                        )
                        bank_score = (primary_score, support)
                        rotation_bank_candidate_scores.append({
                            "angle_rad": float(bank_angle),
                            "consensus_support": support,
                            "mean_confidence_on_support": mean_confidence,
                            "reciprocal_queries": reciprocal_count,
                            "ranking_score": primary_score,
                        })
                        if abs(float(bank_angle)) < 1e-8:
                            identity_bank_score = bank_score
                            identity_mean_confidence = mean_confidence
                        angle_eligible = (
                            abs(float(bank_angle)) < 1e-8
                            or abs(float(bank_angle)) * 180.0 / np.pi >= args.rotation_bank_min_angle_deg
                        )
                        if angle_eligible and bank_score > best_bank_score:
                            best_bank_score = bank_score
                            best_mean_confidence = mean_confidence
                            selected_rotation = float(bank_angle)
                            selected_rotation_target = candidate_target
                    refinement_angles = rotation_refinement_candidates(
                        rotation_bank_candidate_scores,
                        coarse_steps=args.rotation_bank_steps,
                        top_k=args.rotation_bank_refine_topk,
                    )
                    for refine_angle in refinement_angles:
                        candidate_target = rotate_target_about_image_center(
                            target_tensor, refine_angle
                        )
                        (
                            support,
                            mean_confidence,
                            reciprocal_count,
                            probe_encoder_s,
                            probe_matching_s,
                            probe_consensus_s,
                        ) = probe_rotation_hypothesis(
                            model,
                            reference_tensor,
                            candidate_target,
                            args.rotation_bank_probe_query_budget,
                            args.local_consensus_radius,
                            args.local_consensus_threshold,
                            args.rotation_bank_consensus,
                            reference_features=cached_reference_features,
                        )
                        rotation_probe_encoder_s += probe_encoder_s
                        rotation_probe_matching_s += probe_matching_s
                        rotation_probe_consensus_s += probe_consensus_s
                        support_fraction = support / max(
                            args.rotation_bank_probe_query_budget, 1
                        )
                        primary_score = (
                            mean_confidence
                            if args.rotation_bank_selection == "confidence"
                            else support_fraction
                            if args.rotation_bank_selection == "support"
                            else mean_confidence * support_fraction
                        )
                        bank_score = (primary_score, support)
                        rotation_bank_candidate_scores.append({
                            "angle_rad": refine_angle,
                            "consensus_support": support,
                            "mean_confidence_on_support": mean_confidence,
                            "reciprocal_queries": reciprocal_count,
                            "ranking_score": primary_score,
                            "refined_from_coarse_peak": True,
                        })
                        angle_eligible = (
                            abs(refine_angle) < 1e-8
                            or abs(refine_angle) * 180.0 / np.pi
                            >= args.rotation_bank_min_angle_deg
                        )
                        if angle_eligible and bank_score > best_bank_score:
                            best_bank_score = bank_score
                            best_mean_confidence = mean_confidence
                            selected_rotation = refine_angle
                            selected_rotation_target = candidate_target
                    assert selected_rotation_target is not None and identity_bank_score is not None
                    score_gain_over_identity = (
                        best_bank_score[0] - identity_bank_score[0]
                    ) / max(abs(identity_bank_score[0]), 1e-6)
                    best_support = best_bank_score[1]
                    if (
                        best_support < args.rotation_bank_min_consensus_support
                        or score_gain_over_identity < args.rotation_bank_min_score_gain
                    ):
                        selected_rotation = 0.0
                        selected_rotation_target = rotate_target_about_image_center(target_tensor, 0.0)
                        best_bank_score = identity_bank_score
                        best_mean_confidence = identity_mean_confidence
                    target_tensor = selected_rotation_target
                    prediction = model(
                        reference_tensor,
                        target_tensor,
                        include_topk=args.coarse_estimate == "topk8",
                    )
                    rotation_bank_support = best_bank_score[1]
                    rotation_bank_mean_confidence = best_mean_confidence
                elif args.rotation_bank_steps:
                    bank_angles = rotation_bank_angles(args.rotation_bank_steps)
                    feature_height = (small_height + 3) // 4
                    feature_width = (small_width + 3) // 4
                    yy, xx = torch.meshgrid(
                        torch.arange(feature_height, dtype=torch.float64),
                        torch.arange(feature_width, dtype=torch.float64), indexing="ij",
                    )
                    bank_source_xy = torch.stack(
                        ((xx + 0.5) * (small_width / feature_width) - 0.5,
                         (yy + 0.5) * (small_height / feature_height) - 0.5), dim=-1,
                    ).reshape(-1, 2).numpy()
                    best_bank_score = (-1, -float("inf"))
                    prediction = None
                    selected_rotation_target = None
                    rotation_bank_candidate_scores = []
                    identity_prediction = None
                    identity_target = None
                    identity_bank_score = None
                    identity_mean_confidence = 0.0
                    best_mean_confidence = 0.0
                    for batch_start in range(0, len(bank_angles), args.rotation_bank_batch_size):
                        angle_batch = bank_angles[
                            batch_start : batch_start + args.rotation_bank_batch_size
                        ]
                        target_batch = torch.cat(
                            [
                                rotate_target_about_image_center(target_tensor, float(angle))
                                for angle in angle_batch
                            ],
                            dim=0,
                        )
                        prediction_batch = model(
                            reference_tensor.expand(len(angle_batch), -1, -1, -1),
                            target_batch,
                            include_topk=args.coarse_estimate == "topk8",
                        )
                        for batch_index, bank_angle in enumerate(angle_batch):
                            candidate_prediction = PairPrediction(
                                displacement_xy=prediction_batch.displacement_xy[batch_index : batch_index + 1],
                                target_xy=prediction_batch.target_xy[batch_index : batch_index + 1],
                                hard_displacement_xy=prediction_batch.hard_displacement_xy[batch_index : batch_index + 1],
                                hard_target_xy=prediction_batch.hard_target_xy[batch_index : batch_index + 1],
                                topk_displacement_xy=(
                                    prediction_batch.topk_displacement_xy[batch_index : batch_index + 1]
                                    if prediction_batch.topk_displacement_xy is not None else None
                                ),
                                topk_target_xy=(
                                    prediction_batch.topk_target_xy[batch_index : batch_index + 1]
                                    if prediction_batch.topk_target_xy is not None else None
                                ),
                                confidence=prediction_batch.confidence[batch_index : batch_index + 1],
                            )
                            candidate_affine, candidate_usable = local_affine_consensus(
                                bank_source_xy,
                                candidate_prediction.hard_target_xy[0].reshape(-1, 2).numpy(),
                                small_width,
                                small_height,
                                radius=args.local_consensus_radius,
                                threshold=args.local_consensus_threshold,
                            )
                            del candidate_affine
                            support = int(candidate_usable.sum())
                            mean_confidence = (
                                float(candidate_prediction.confidence[0].reshape(-1)[candidate_usable].mean())
                                if support else 0.0
                            )
                            support_fraction = support / max(len(bank_source_xy), 1)
                            primary_score = (
                                mean_confidence
                                if args.rotation_bank_selection == "confidence"
                                else support_fraction
                                if args.rotation_bank_selection == "support"
                                else mean_confidence * support_fraction
                            )
                            bank_score = (primary_score, support)
                            rotation_bank_candidate_scores.append({
                                "angle_rad": float(bank_angle),
                                "consensus_support": support,
                                "mean_confidence_on_support": mean_confidence,
                            })
                            if abs(float(bank_angle)) < 1e-8:
                                identity_prediction = candidate_prediction
                                identity_target = target_batch[batch_index : batch_index + 1]
                                identity_bank_score = bank_score
                                identity_mean_confidence = mean_confidence
                            angle_eligible = (
                                abs(float(bank_angle)) < 1e-8
                                or abs(float(bank_angle)) * 180.0 / np.pi >= args.rotation_bank_min_angle_deg
                            )
                            if angle_eligible and bank_score > best_bank_score:
                                best_bank_score = bank_score
                                best_mean_confidence = mean_confidence
                                prediction = candidate_prediction
                                selected_rotation = float(bank_angle)
                                selected_rotation_target = target_batch[batch_index : batch_index + 1]
                    assert prediction is not None and selected_rotation_target is not None
                    assert identity_prediction is not None and identity_target is not None
                    assert identity_bank_score is not None
                    score_gain_over_identity = (
                        best_bank_score[0] - identity_bank_score[0]
                    ) / max(abs(identity_bank_score[0]), 1e-6)
                    best_support = best_bank_score[1]
                    if (
                        best_support < args.rotation_bank_min_consensus_support
                        or score_gain_over_identity < args.rotation_bank_min_score_gain
                    ):
                        prediction = identity_prediction
                        selected_rotation_target = identity_target
                        selected_rotation = 0.0
                        best_bank_score = identity_bank_score
                        best_mean_confidence = identity_mean_confidence
                    target_tensor = selected_rotation_target
                    rotation_bank_support = best_bank_score[1]
                    rotation_bank_mean_confidence = best_mean_confidence
                else:
                    prediction = model(
                        reference_tensor, target_tensor,
                        include_topk=args.coarse_estimate == "topk8",
                    )
                    rotation_bank_support = None
                    rotation_bank_mean_confidence = None
                    score_gain_over_identity = None
                if args.global_affine_prefit:
                    feature_height, feature_width = prediction.confidence.shape[-2:]
                    yy, xx = torch.meshgrid(
                        torch.arange(feature_height, dtype=torch.float64),
                        torch.arange(feature_width, dtype=torch.float64),
                        indexing="ij",
                    )
                    affine_source_xy = torch.stack(
                        ((xx + 0.5) * (small_width / feature_width) - 0.5,
                         (yy + 0.5) * (small_height / feature_height) - 0.5),
                        dim=-1,
                    ).reshape(-1, 2).numpy()
                    match_flow = prediction.hard_displacement_xy
                    if args.rotation_bank_steps:
                        match_flow = unrotate_target_displacement(
                            match_flow, (small_width, small_height), selected_rotation
                        )
                    affine_target_xy = affine_source_xy + match_flow[0].reshape(-1, 2).numpy()
                    affine_prefit_coefficients, inlier_mask, affine_prefit_median_residual = (
                        fit_global_affine_ransac(
                            affine_source_xy,
                            affine_target_xy,
                            threshold=args.local_consensus_threshold,
                            min_inliers=args.affine_prefit_min_inliers,
                        )
                    )
                    affine_prefit_inliers = int(inlier_mask.sum())
                    if affine_prefit_coefficients is not None:
                        inlier_points = affine_source_xy[inlier_mask]
                        bins_x = np.clip((inlier_points[:, 0] / small_width * 3).astype(int), 0, 2)
                        bins_y = np.clip((inlier_points[:, 1] / small_height * 3).astype(int), 0, 2)
                        affine_prefit_spatial_bins = len(np.unique(bins_y * 3 + bins_x))
                        linear = affine_prefit_coefficients[1:, :].T
                        affine_prefit_linear_deviation = float(
                            np.linalg.norm(linear - np.eye(2), ord="fro")
                        )
                        singular_values = np.linalg.svd(linear, compute_uv=False)
                        determinant = float(np.linalg.det(linear))
                        valid_affine = (
                            affine_prefit_spatial_bins >= 4
                            and affine_prefit_linear_deviation
                            >= args.affine_prefit_min_linear_deviation
                            and 0.05 < determinant < 20.0
                            and singular_values[-1] > 1e-6
                            and singular_values[0] / singular_values[-1] < 20.0
                        )
                        if valid_affine:
                            target_tensor = rectify_target_with_affine(
                                original_target_tensor, affine_prefit_coefficients
                            )
                            prediction = model(
                                reference_tensor,
                                target_tensor,
                                include_topk=args.coarse_estimate == "topk8",
                            )
                        else:
                            affine_prefit_coefficients = None
                elif args.local_bspline_prefit:
                    prefit_hard_flow = prediction.hard_displacement_xy
                    if args.rotation_bank_steps:
                        prefit_hard_flow = unrotate_target_displacement(
                            prefit_hard_flow,
                            (small_width, small_height),
                            selected_rotation,
                        )
                    local_bspline_prefit_field, local_bspline_prefit_diagnostics = (
                        fit_local_bspline_prefit(
                            SimpleNamespace(hard_displacement_xy=prefit_hard_flow),
                            (small_width, small_height),
                            radius=args.local_consensus_radius,
                            threshold=args.local_consensus_threshold,
                            min_support=args.local_bspline_prefit_min_support,
                            bending=args.local_bspline_prefit_bending,
                        )
                    )
                    if local_bspline_prefit_field is not None:
                        target_tensor = rectify_target_with_field(
                            original_target_tensor, local_bspline_prefit_field
                        )
                        prediction = model(
                            reference_tensor,
                            target_tensor,
                            include_topk=args.coarse_estimate == "topk8",
                        )
                model_inference_s = time.perf_counter() - model_inference_start
                if args.coarse_estimate == "hard":
                    prediction.displacement_xy = prediction.hard_displacement_xy
                elif args.coarse_estimate == "topk8":
                    assert prediction.topk_displacement_xy is not None
                    prediction.displacement_xy = prediction.topk_displacement_xy
                refinement_start = time.perf_counter()
                if args.projection == "local_consensus":
                    feature_height, feature_width = prediction.confidence.shape[-2:]
                    yy, xx = torch.meshgrid(
                        torch.arange(feature_height, dtype=torch.float64),
                        torch.arange(feature_width, dtype=torch.float64), indexing="ij",
                    )
                    consensus_source = torch.stack(
                        ((xx + 0.5) * (small_width / feature_width) - 0.5,
                         (yy + 0.5) * (small_height / feature_height) - 0.5), dim=-1,
                    ).reshape(-1, 2)
                    local_parameters, local_usable = local_affine_consensus(
                        consensus_source.numpy(),
                        prediction.hard_target_xy[0].reshape(-1, 2).numpy(),
                        small_width,
                        small_height,
                        radius=args.local_consensus_radius,
                        threshold=args.local_consensus_threshold,
                    )
                    affine_target = np.column_stack(
                        (
                            local_parameters[:, 0]
                            + (1.0 + local_parameters[:, 1]) * consensus_source[:, 0].numpy()
                            + local_parameters[:, 2] * consensus_source[:, 1].numpy(),
                            local_parameters[:, 3]
                            + local_parameters[:, 4] * consensus_source[:, 0].numpy()
                            + (1.0 + local_parameters[:, 5]) * consensus_source[:, 1].numpy(),
                        )
                    )
                    affine_flow = torch.from_numpy(affine_target).to(
                        prediction.displacement_xy.dtype
                    ).reshape_as(prediction.displacement_xy[0]) - consensus_source.to(
                        prediction.displacement_xy.dtype
                    ).reshape_as(prediction.displacement_xy[0])
                    usable_tensor = torch.from_numpy(local_usable).reshape(
                        feature_height, feature_width
                    )
                    prediction.displacement_xy[0] = torch.where(
                        usable_tensor[..., None], affine_flow,
                        prediction.displacement_xy[0],
                    )
                    consensus_usable_fraction = float(local_usable.mean())
                if args.local_feature_radius_cells:
                    local_feature = refine_local_features(
                        model,
                        reference_tensor,
                        target_tensor,
                        prediction.target_xy,
                        search_radius_cells=args.local_feature_radius_cells,
                    )
                    prediction.displacement_xy = local_feature.displacement_xy
                if args.local_ncc_radius:
                    local_jacobian = None
                    if args.local_ncc_geometry in {"affine", "hybrid"}:
                        flow = prediction.displacement_xy[0]
                        stride_x = small_width / flow.shape[1]
                        stride_y = small_height / flow.shape[0]
                        flow_x_y, flow_x_x = torch.gradient(
                            flow[..., 0], spacing=(stride_y, stride_x), dim=(0, 1)
                        )
                        flow_y_y, flow_y_x = torch.gradient(
                            flow[..., 1], spacing=(stride_y, stride_x), dim=(0, 1)
                        )
                        local_jacobian = torch.stack(
                            (
                                torch.stack((1.0 + flow_x_x, flow_x_y), dim=-1),
                                torch.stack((flow_y_x, 1.0 + flow_y_y), dim=-1),
                            ),
                            dim=-2,
                        )
                        identity = torch.eye(2, dtype=local_jacobian.dtype)
                        local_jacobian = identity + args.local_ncc_jacobian_damping * (
                            local_jacobian - identity
                        )
                    if ndf_wgpu is None:
                        local_ncc = refine_local_ncc(
                            reference_tensor,
                            target_tensor,
                            prediction.displacement_xy,
                            search_radius=args.local_ncc_radius,
                            patch_radius=args.local_ncc_patch_radius,
                            initial_jacobian=local_jacobian,
                            geometry=args.local_ncc_geometry,
                        )
                    else:
                        gpu_result = ndf_wgpu.refine_ndf_local_ncc(
                            reference_tensor[0, 0].numpy(),
                            target_tensor[0, 0].numpy(),
                            prediction.displacement_xy[0].numpy(),
                            search_radius=args.local_ncc_radius,
                            patch_radius=args.local_ncc_patch_radius,
                            initial_jacobian=(
                                local_jacobian.numpy() if local_jacobian is not None else None
                            ),
                            geometry=args.local_ncc_geometry,
                            implementation=args.local_ncc_implementation,
                        )
                        local_ncc = SimpleNamespace(
                            displacement_xy=torch.from_numpy(gpu_result.displacement_xy)[None],
                            score=torch.from_numpy(gpu_result.score)[None],
                            margin=torch.from_numpy(gpu_result.margin)[None],
                            valid=torch.from_numpy(gpu_result.valid)[None],
                        )
                    prediction.displacement_xy = local_ncc.displacement_xy
                if affine_prefit_coefficients is not None:
                    if local_ncc is not None:
                        local_ncc.displacement_xy = compose_affine_residual_flow(
                            local_ncc.displacement_xy,
                            affine_prefit_coefficients,
                            (small_width, small_height),
                        )
                    if local_feature is not None:
                        local_feature.displacement_xy = compose_affine_residual_flow(
                            local_feature.displacement_xy,
                            affine_prefit_coefficients,
                            (small_width, small_height),
                        )
                    prediction.displacement_xy = compose_affine_residual_flow(
                        prediction.displacement_xy,
                        affine_prefit_coefficients,
                        (small_width, small_height),
                    )
                elif local_bspline_prefit_field is not None:
                    if local_ncc is not None:
                        local_ncc.displacement_xy = compose_field_residual_flow(
                            local_ncc.displacement_xy,
                            local_bspline_prefit_field,
                            (small_width, small_height),
                        )
                    if local_feature is not None:
                        local_feature.displacement_xy = compose_field_residual_flow(
                            local_feature.displacement_xy,
                            local_bspline_prefit_field,
                            (small_width, small_height),
                        )
                    prediction.displacement_xy = compose_field_residual_flow(
                        prediction.displacement_xy,
                        local_bspline_prefit_field,
                        (small_width, small_height),
                    )
                elif args.oracle_rotation_rectification or args.rotation_bank_steps:
                    prediction.displacement_xy = unrotate_target_displacement(
                        prediction.displacement_xy,
                        (small_width, small_height),
                        selected_rotation,
                    )
                local_refinement_s = time.perf_counter() - refinement_start
            inference_s = time.perf_counter() - inference_start

            points_coarse = uniform_points((small_width, small_height), 4).to(torch.float64)
            flow_coarse = prediction.displacement_xy[0].reshape(-1, 2).to(torch.float64)
            fit_start = time.perf_counter()
            native_centres = centres(width, height)
            coarse_centres = torch.from_numpy(
                native_centres / np.asarray([scale_x, scale_y], dtype=np.float32)
            ).to(torch.float64)
            if args.projection == "raw_bilinear":
                field = DenseBilinearDisplacement(
                    prediction.displacement_xy[0].to(torch.float64),
                    (small_width, small_height),
                )
                initial_u, initial_f = field.evaluate(coarse_centres)
            elif args.projection == "global_affine":
                initial_u, initial_f = robust_affine_projection(
                    points_coarse, flow_coarse, coarse_centres
                )
            else:
                spacings = (32.0, 16.0, 8.0) if args.projection == "multilevel_spline" else (8.0,)
                shapes = tuple(
                    recommended_control_shape((small_width, small_height), spacing)
                    for spacing in spacings
                )
                field = fit_bspline_field(
                    points_coarse, flow_coarse, (small_width, small_height), shapes, bending=1e-3
                )
                initial_u, initial_f = field.evaluate(coarse_centres)
            projection_s = time.perf_counter() - fit_start
            initial_parameters = np.column_stack(
                (
                    (initial_u[:, 0].numpy() * scale_x),
                    (initial_f[:, 0, 0].numpy() - 1),
                    initial_f[:, 0, 1].numpy() * (scale_x / scale_y),
                    (initial_u[:, 1].numpy() * scale_y),
                    initial_f[:, 1, 0].numpy() * (scale_y / scale_x),
                    (initial_f[:, 1, 1].numpy() - 1),
                )
            ).astype(np.float32)
            truth = truth_parameters(pair, native_centres)
            visible = pair.visible[
                np.rint(native_centres[:, 1]).astype(int),
                np.rint(native_centres[:, 0]).astype(int),
            ]
            init_translation = np.linalg.norm(
                initial_parameters[:, [0, 3]] - truth[:, [0, 3]], axis=1
            )
            init_gradient = np.linalg.norm(
                initial_parameters[:, [1, 2, 4, 5]] - truth[:, [1, 2, 4, 5]], axis=1
            )
            local_ncc_metrics = {}
            local_feature_metrics = {}
            local_final_metrics = {}
            measured_refinements = (
                ("local_ncc", local_ncc),
                ("local_feature", local_feature),
            )
            refinement_reference = next(
                (result for _, result in measured_refinements if result is not None), None
            )
            if refinement_reference is not None:
                feature_height, feature_width = refinement_reference.valid.shape[-2:]
                flow_model = torch.nn.functional.interpolate(
                    torch.from_numpy(pair.flow_xy.copy()).permute(2, 0, 1)[None],
                    size=(small_height, small_width), mode="bilinear", align_corners=False,
                )
                flow_model = flow_model / torch.tensor([scale_x, scale_y]).reshape(1, 2, 1, 1)
                flow_feature = torch.nn.functional.interpolate(
                    flow_model, size=(feature_height, feature_width), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                visible_feature = torch.nn.functional.interpolate(
                    torch.from_numpy(pair.visible.copy())[None, None].float(),
                    size=(small_height, small_width), mode="nearest",
                )
                visible_feature = torch.nn.functional.interpolate(
                    visible_feature, size=(feature_height, feature_width), mode="nearest"
                )[0, 0] > 0.5
                for label, refinement in measured_refinements:
                    if refinement is None:
                        continue
                    valid_nodes = refinement.valid[0] & visible_feature
                    if not torch.any(valid_nodes):
                        continue
                    node_error = torch.linalg.vector_norm(
                        refinement.displacement_xy[0] - flow_feature[0], dim=-1
                    ) * texture_scale
                    metrics = {
                        f"{label}_valid_visible_nodes": int(valid_nodes.sum()),
                        f"{label}_node_median_error_native_px": float(
                            torch.median(node_error[valid_nodes])
                        ),
                        f"{label}_node_fraction_within_2_native_px": float(
                            (node_error[valid_nodes] < 2.0).float().mean()
                        ),
                        f"{label}_score_median": float(
                            torch.median(refinement.score[0][valid_nodes])
                        ),
                        f"{label}_margin_median": float(
                            torch.median(refinement.margin[0][valid_nodes])
                        ),
                    }
                    if label == "local_ncc":
                        local_ncc_metrics = metrics
                    else:
                        local_feature_metrics = metrics
                final_valid = refinement_reference.valid[0] & visible_feature
                final_error = torch.linalg.vector_norm(
                    prediction.displacement_xy[0] - flow_feature[0], dim=-1
                ) * texture_scale
                if torch.any(final_valid):
                    local_final_metrics = {
                        "final_local_valid_visible_nodes": int(final_valid.sum()),
                        "final_local_node_median_error_native_px": float(
                            torch.median(final_error[final_valid])
                        ),
                        "final_local_node_fraction_within_2_native_px": float(
                            (final_error[final_valid] < 2.0).float().mean()
                        ),
                    }
            reference = np.rint(pair.reference * 255).astype(np.uint8)
            target = np.rint(pair.target * 255).astype(np.uint8)
            solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
            active = np.ones(len(native_centres), dtype=np.uint32)
            state = np.zeros((len(native_centres), 8), dtype=np.float32)
            state[:, :6] = initial_parameters
            state[:, 6] = 1
            control_start = time.perf_counter()
            zero_state = np.zeros_like(state)
            zero_state[:, 6] = 1
            zero_result = solver._register_seeded_frame(target, zero_state, active)
            zero_s = time.perf_counter() - control_start
            oracle_state = np.zeros_like(state)
            oracle_state[:, :6] = truth
            oracle_state[:, 6] = 1
            oracle_start = time.perf_counter()
            oracle_result = solver._register_seeded_frame(target, oracle_state, active)
            oracle_s = time.perf_counter() - oracle_start
            solve_start = time.perf_counter()
            result = solver._register_seeded_frame(target, state, active)
            solve_s = time.perf_counter() - solve_start
            returned = np.asarray(result.parameters).reshape(-1, 6)
            status = np.asarray(result.status).reshape(-1) > 0.5
            displacement_error = np.linalg.norm(returned[:, [0, 3]] - truth[:, [0, 3]], axis=1)
            correct = visible & status & (displacement_error < 2.0)
            zero_parameters = np.asarray(zero_result.parameters).reshape(-1, 6)
            zero_status = np.asarray(zero_result.status).reshape(-1) > 0.5
            zero_error = np.linalg.norm(zero_parameters[:, [0, 3]] - truth[:, [0, 3]], axis=1)
            oracle_parameters = np.asarray(oracle_result.parameters).reshape(-1, 6)
            oracle_status = np.asarray(oracle_result.status).reshape(-1) > 0.5
            oracle_error = np.linalg.norm(oracle_parameters[:, [0, 3]] - truth[:, [0, 3]], axis=1)
            report["cases"].append(
                {
                    "mode": mode,
                    "projection": args.projection,
                    "texture_seed": texture_seed,
                    "warp_seed": warp_seed,
                    "visible_windows": int(visible.sum()),
                    "initialized_windows": len(native_centres),
                    "initial_translation_rmse_px": float(np.sqrt(np.mean(init_translation**2))),
                    "initial_gradient_rmse": float(np.sqrt(np.mean(init_gradient**2))),
                    "local_consensus_usable_fraction": consensus_usable_fraction,
                    "rotation_bank_selected_angle_rad": (
                        selected_rotation if args.rotation_bank_steps else None
                    ),
                    "rotation_bank_consensus_support": rotation_bank_support,
                    "rotation_bank_mean_confidence_on_support": rotation_bank_mean_confidence,
                    "rotation_bank_candidate_scores": rotation_bank_candidate_scores,
                    "rotation_bank_score_gain_over_identity": score_gain_over_identity,
                    "rotation_bank_probe_encoder_s": rotation_probe_encoder_s,
                    "rotation_bank_probe_matching_s": rotation_probe_matching_s,
                    "rotation_bank_probe_consensus_s": rotation_probe_consensus_s,
                    "rotation_bank_truth_angle_rad_for_diagnostic_only": (
                        float(pair.metadata["angle_rad"]) if args.rotation_bank_steps else None
                    ),
                    "global_affine_prefit_used": affine_prefit_coefficients is not None,
                    "global_affine_prefit_coefficients": (
                        affine_prefit_coefficients.tolist()
                        if affine_prefit_coefficients is not None else None
                    ),
                    "global_affine_prefit_inliers": affine_prefit_inliers,
                    "global_affine_prefit_median_residual_px": affine_prefit_median_residual,
                    "global_affine_prefit_spatial_bins": affine_prefit_spatial_bins,
                    "global_affine_prefit_linear_deviation": affine_prefit_linear_deviation,
                    "local_bspline_prefit_used": local_bspline_prefit_field is not None,
                    "local_bspline_prefit_diagnostics": local_bspline_prefit_diagnostics,
                    **local_ncc_metrics,
                    **local_feature_metrics,
                    **local_final_metrics,
                    "solver_accepted_visible": int((status & visible).sum()),
                    "correct_converged_lt2px": int(correct.sum()),
                    "zero_correct_converged_lt2px": int((zero_status & visible & (zero_error < 2.0)).sum()),
                    "analytic_oracle_correct_converged_lt2px": int((oracle_status & visible & (oracle_error < 2.0)).sum()),
                    "coverage_correct": float(correct.sum() / max(int(visible.sum()), 1)),
                    "correct_windows_per_second": float(correct.sum() / max(inference_s + projection_s + solve_s, 1e-12)),
                    "inference_s_cpu": inference_s,
                    "model_inference_s_cpu": model_inference_s,
                    "local_refinement_s_wall": local_refinement_s,
                    "field_projection_s_cpu": projection_s,
                    "icgn_s": solve_s,
                    "zero_icgn_s": zero_s,
                    "oracle_icgn_s": oracle_s,
                    "mean_iterations_accepted": float(np.mean(np.asarray(result.iterations).reshape(-1)[status])) if status.any() else None,
                }
            )
            print(json.dumps(report["cases"][-1]), flush=True)
    for mode in ("translation", "heterogeneous"):
        cases = [case for case in report["cases"] if case["mode"] == mode]
        report.setdefault("summary", {})[mode] = {
            "cases": len(cases),
            "visible_windows": int(sum(case["visible_windows"] for case in cases)),
            "correct_converged_lt2px": int(sum(case["correct_converged_lt2px"] for case in cases)),
            "coverage_correct": float(
                sum(case["correct_converged_lt2px"] for case in cases)
                / max(sum(case["visible_windows"] for case in cases), 1)
            ),
            "median_correct_windows_per_second": float(
                np.median([case["correct_windows_per_second"] for case in cases])
            ),
        }
    if ndf_wgpu is not None:
        ndf_wgpu.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
