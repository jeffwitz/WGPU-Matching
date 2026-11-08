"""Native wgpu-py orchestration for sparse-grid CPM-HOG PatchMatch."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import wgpu

from ._gpu import (
    ResourceScope,
    create_buffer,
    create_texture,
    dispatch,
    make_bind_group,
    make_params,
    make_pipeline,
    read_array,
    shader_source,
)
from .deepmatching import DeepMatchingBackend


def _centers(width: int, height: int, patch_size: int, stride: int) -> list[tuple[float, float]]:
    half = (patch_size - 1) / 2
    return [
        (half + x_index * stride, half + y_index * stride)
        for y_index in range(max(0, 1 + (height - patch_size) // stride))
        for x_index in range(max(0, 1 + (width - patch_size) // stride))
    ]


def _nearest_index(
    centers: list[tuple[float, float]], x: float, y: float, grid_width: int, stride: int
) -> tuple[int, float]:
    origin_x, origin_y = centers[0]
    gx = min(grid_width - 1, max(0, round((x - origin_x) / stride)))
    grid_height = math.ceil(len(centers) / grid_width)
    gy = min(grid_height - 1, max(0, round((y - origin_y) / stride)))
    estimate = min(len(centers) - 1, gy * grid_width + gx)
    best = estimate
    best_distance_squared = math.inf
    for index in range(max(0, estimate - 2), min(len(centers), estimate + 3)):
        px, py = centers[index]
        distance_squared = (px - x) ** 2 + (py - y) ** 2
        if distance_squared < best_distance_squared:
            best, best_distance_squared = index, distance_squared
    return best, math.sqrt(best_distance_squared)


def _interpolate(
    states: np.ndarray,
    centers: list[tuple[float, float]],
    grid_width: int,
    grid_height: int,
    x: float,
    y: float,
    stride: int,
) -> tuple[float, float] | None:
    origin_x, origin_y = centers[0]
    grid_x, grid_y = (x - origin_x) / stride, (y - origin_y) / stride
    if grid_x < 0 or grid_y < 0 or grid_x > grid_width - 1 or grid_y > grid_height - 1:
        return None
    x0, y0 = math.floor(grid_x), math.floor(grid_y)
    x1, y1 = min(grid_width - 1, x0 + 1), min(grid_height - 1, y0 + 1)
    tx, ty = grid_x - x0, grid_y - y0
    corners = (
        (y0 * grid_width + x0, (1 - tx) * (1 - ty)),
        (y0 * grid_width + x1, tx * (1 - ty)),
        (y1 * grid_width + x0, (1 - tx) * ty),
        (y1 * grid_width + x1, tx * ty),
    )
    dx = dy = 0.0
    for index, weight in corners:
        if weight == 0:
            continue
        if states[index, 3] == 0:
            return None
        dx += int(states[index, 0]) * weight
        dy += int(states[index, 1]) * weight
    return dx, dy


def _fit_local_affine(
    work_matches: list[dict[str, Any]], x: float, y: float, radius_px: float
) -> dict[str, Any] | None:
    candidates = [
        match for match in work_matches
        if match["usable"] and math.hypot(match["x"] - x, match["y"] - y) <= radius_px
    ]
    if len(candidates) < 6:
        return None
    inliers = candidates
    coefficients_x = coefficients_y = None
    for iteration in range(3):
        matrix = np.zeros((3, 3), dtype=np.float64)
        rhs_x = np.zeros(3, dtype=np.float64)
        rhs_y = np.zeros(3, dtype=np.float64)
        for match in inliers:
            local_x, local_y = match["x"] - x, match["y"] - y
            basis = np.asarray((1.0, local_x, local_y))
            ratio = math.hypot(local_x, local_y) / radius_px
            weight = 1 / (1 + ratio * ratio)
            matrix += weight * np.outer(basis, basis)
            rhs_x += weight * basis * match["dx"]
            rhs_y += weight * basis * match["dy"]
        try:
            coefficients_x = np.linalg.solve(matrix, rhs_x)
            coefficients_y = np.linalg.solve(matrix, rhs_y)
        except np.linalg.LinAlgError:
            return None
        residuals = []
        for match in candidates:
            local_x, local_y = match["x"] - x, match["y"] - y
            predicted_x = coefficients_x @ (1.0, local_x, local_y)
            predicted_y = coefficients_y @ (1.0, local_x, local_y)
            residuals.append((match, math.hypot(match["dx"] - predicted_x,
                                                match["dy"] - predicted_y)))
        errors = sorted(error for _, error in residuals)
        cutoff = max(2.0, 2.5 * errors[len(errors) // 2])
        next_inliers = [match for match, error in residuals if error <= cutoff]
        if len(next_inliers) < 6 or len(next_inliers) == len(inliers) or iteration == 2:
            break
        inliers = next_inliers
    if coefficients_x is None or coefficients_y is None or len(inliers) < 6:
        return None
    squared = []
    for match in inliers:
        local_x, local_y = match["x"] - x, match["y"] - y
        residual_x = match["dx"] - coefficients_x @ (1.0, local_x, local_y)
        residual_y = match["dy"] - coefficients_y @ (1.0, local_x, local_y)
        squared.append(residual_x * residual_x + residual_y * residual_y)
    return {
        "coefficients_x": coefficients_x,
        "coefficients_y": coefficients_y,
        "support_count": len(inliers),
        "candidate_count": len(candidates),
        "mean_score": sum(match["score"] for match in inliers) / len(inliers),
        "reciprocal_error_px": max(match["reciprocal_error_px"] for match in inliers),
        "rmse_px": math.sqrt(sum(squared) / len(squared)),
    }


class CPMBackend:
    """Sparse-grid PatchMatch using shared DeepMatching HOG descriptors and WGSL."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self.deepmatching = DeepMatchingBackend(device)

    def _pyramid(self, scope: ResourceScope, descriptors: dict[str, Any], requested: int) -> list[dict[str, Any]]:
        device = self.device
        module = device.create_shader_module(code=shader_source("cpm-pyramid.wgsl"), label="CPM HOG pyramid")
        pipeline = make_pipeline(device, module, "downsample_hog", "CPM HOG pyramid downsample")
        levels = [descriptors]
        texture_usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
        for level_index in range(1, requested):
            source = levels[-1]
            width, height = source["width"] // 2, source["height"] // 2
            if width < 1 or height < 1:
                break
            level: dict[str, Any] = {"width": width, "height": height}
            for image_name in ("reference", "target"):
                level[image_name] = {
                    "first": create_texture(device, scope, width, height, "rgba16float", texture_usage,
                                             f"CPM level {level_index} {image_name} HOG 1-4"),
                    "second": create_texture(device, scope, width, height, "rgba16float", texture_usage,
                                              f"CPM level {level_index} {image_name} HOG 5-8"),
                }
            params = make_params(device, scope, [source["width"], source["height"], width, height],
                                 f"CPM pyramid level {level_index} dimensions", length=4)
            encoder = device.create_command_encoder(label=f"CPM pyramid level {level_index}")
            for image_name in ("reference", "target"):
                input_set, output_set = source[image_name], level[image_name]
                group = make_bind_group(device, pipeline, [
                    {"binding": 0, "resource": input_set["first"].create_view()},
                    {"binding": 1, "resource": input_set["second"].create_view()},
                    {"binding": 2, "resource": output_set["first"].create_view()},
                    {"binding": 3, "resource": output_set["second"].create_view()},
                    {"binding": 4, "resource": {"buffer": params}},
                ])
                dispatch(encoder, pipeline, group, math.ceil(width / 8), math.ceil(height / 8),
                         label=f"CPM pooled HOG {image_name}")
            device.queue.submit([encoder.finish()])
            levels.append(level)
        return levels

    def _run_direction(
        self,
        scope: ResourceScope,
        pipeline: Any,
        levels: list[dict[str, Any]],
        point_buffers: list[Any],
        grid_width: int,
        grid_height: int,
        point_count: int,
        support_px: int,
        base_scale: int,
        radius_px: int,
        seed: int,
        options: Any,
        diagnostics: list[int] | None = None,
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]] | None]:
        device = self.device
        state_bytes = point_count * 16
        previous = create_buffer(device, scope, state_bytes,
                                 wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC, "CPM states A")
        next_state = create_buffer(device, scope, state_bytes,
                                   wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC, "CPM states B")
        readback = create_buffer(device, scope, state_bytes,
                                 wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC, "CPM state readback")
        groups = []
        level_ends = []
        read_state, write_state = previous, next_state
        iteration = 0
        for level_index in reversed(range(len(levels))):
            level = levels[level_index]
            level_scale = base_scale * (2 ** level_index)
            patch_size = max(2, round(support_px / level_scale))
            search_radius = math.floor(radius_px / level_scale) if radius_px > 0 else max(level["width"], level["height"])
            random_radius_at_level = search_radius if level_index == len(levels) - 1 else max(
                1, min(search_radius, math.ceil(options.local_search_radius_px * (2 ** level_index) / level_scale)))

            def append_pass(
                initialize: bool,
                scale_previous: bool,
                random_radius: int,
                *,
                level: dict[str, Any] = level,
                level_index: int = level_index,
                patch_size: int = patch_size,
                search_radius: int = search_radius,
            ) -> None:
                nonlocal read_state, write_state, iteration
                params = make_params(device, scope, [
                    grid_width, grid_height, level["width"], level["height"], patch_size,
                    search_radius, seed, iteration, random_radius, options.random_trials,
                    int(initialize), int(scale_previous), options.initial_hypotheses, 0, 0, 0,
                ], f"CPM direction pass {iteration}")
                source_set, target_set = level["reference"], level["target"]
                group = make_bind_group(device, pipeline, [
                    {"binding": 0, "resource": source_set["first"].create_view()},
                    {"binding": 1, "resource": source_set["second"].create_view()},
                    {"binding": 2, "resource": target_set["first"].create_view()},
                    {"binding": 3, "resource": target_set["second"].create_view()},
                    {"binding": 4, "resource": {"buffer": point_buffers[level_index]}},
                    {"binding": 5, "resource": {"buffer": read_state}},
                    {"binding": 6, "resource": {"buffer": write_state}},
                    {"binding": 7, "resource": {"buffer": params}},
                ])
                groups.append(group)
                read_state, write_state = write_state, read_state
                iteration += 1

            if level_index == len(levels) - 1:
                append_pass(True, False, search_radius)
                for pass_index in range(options.propagation_iterations):
                    append_pass(False, False, max(1, math.ceil(search_radius / (2 ** pass_index))))
            else:
                for pass_index in range(options.propagation_iterations):
                    random_radius = max(1, min(search_radius,
                                               math.ceil(random_radius_at_level / (2 ** pass_index))))
                    append_pass(False, pass_index == 0, random_radius)
            level_ends.append((level_index, len(groups), read_state))

        snapshot_readback = None
        if diagnostics:
            snapshot_readback = create_buffer(
                device, scope, len(levels) * len(diagnostics) * 16,
                wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
                "CPM per-level diagnostic readback",
            )
        encoder = device.create_command_encoder(label="CPM coarse-to-fine PatchMatch")
        for group_index, group in enumerate(groups, 1):
            dispatch(encoder, pipeline, group, math.ceil(point_count / 64), label="CPM PatchMatch pass")
            if snapshot_readback is not None:
                for level_index, after_count, state_buffer in level_ends:
                    if after_count != group_index:
                        continue
                    for sample_index, state_index in enumerate(diagnostics):
                        encoder.copy_buffer_to_buffer(
                            state_buffer, state_index * 16, snapshot_readback,
                            (level_index * len(diagnostics) + sample_index) * 16, 16,
                        )
        encoder.copy_buffer_to_buffer(read_state, 0, readback, 0, state_bytes)
        device.queue.submit([encoder.finish()])
        states = read_array(device, readback, np.dtype(np.int32), point_count * 4).reshape(-1, 4)
        snapshots = None
        if snapshot_readback is not None:
            raw = read_array(device, snapshot_readback, np.dtype(np.int32), len(levels) * len(diagnostics) * 4)
            floats = raw.view(np.float32)
            snapshots = [[] for _ in levels]
            for level_index in range(len(levels)):
                for sample_index in range(len(diagnostics)):
                    offset = (level_index * len(diagnostics) + sample_index) * 4
                    snapshots[level_index].append({
                        "dx": int(raw[offset]), "dy": int(raw[offset + 1]),
                        "cost": float(floats[offset + 2]), "valid": bool(raw[offset + 3]),
                    })
        return states, snapshots

    def compute(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        options: Any,
        *,
        on_progress: Callable[[str], None] | None = None,
        include_correspondences: bool = False,
    ) -> dict[str, Any]:
        progress = on_progress or (lambda _: None)
        reference = np.ascontiguousarray(reference, dtype=np.uint8)
        target = np.ascontiguousarray(target, dtype=np.uint8)
        if reference.ndim != 2 or target.ndim != 2 or reference.shape != target.shape:
            raise ValueError("CPM expects same-sized two-dimensional grayscale uint8 images")
        height, width = reference.shape
        if width > self.device.limits["max-texture-dimension-2d"] or height > self.device.limits["max-texture-dimension-2d"]:
            raise ValueError("Image exceeds the adapter's maximum texture dimension")
        scale = 2 ** options.downscale_levels
        support = options.matching_support_px
        effective_patch = max(2, math.floor(support / scale + 0.5))
        max_levels = max(1, math.floor(math.log2(effective_patch / 4)) + 1)
        requested_levels = options.levels
        effective_levels = min(requested_levels, max_levels)
        work_centers = _centers(width, height, support, options.work_grid_stride)
        if not work_centers:
            raise ValueError("CPM work grid does not fit its matching support in the image")
        if len(work_centers) > 200_000:
            raise ValueError("CPM work grid exceeds the current 200000-point safety limit")
        if options.seed_points:
            seeds = [(float(x), float(y)) for x, y in options.seed_points]
        else:
            seeds = _centers(width, height, options.patch_size, options.stride)
        if not seeds:
            raise ValueError("There are no requested DIC seed centers")
        half = (options.patch_size - 1) / 2
        for index, (x, y) in enumerate(seeds, 1):
            if not (half <= x <= width - 1 - half and half <= y <= height - 1 - half):
                raise ValueError(f"CPM seed {index} does not fit a full DIC patch in the reference image")
        grid_width = 1 + (width - support) // options.work_grid_stride
        grid_height = 1 + (height - support) // options.work_grid_stride
        if grid_width * grid_height != len(work_centers):
            raise RuntimeError("CPM work-grid coordinates do not match its propagation lattice")
        state_bytes = len(work_centers) * 16
        coordinate_bytes = len(work_centers) * 8
        limits = self.device.limits
        if (
            state_bytes > limits["max-buffer-size"]
            or state_bytes > limits["max-storage-buffer-binding-size"]
            or coordinate_bytes > limits["max-storage-buffer-binding-size"]
            or math.ceil(len(work_centers) / 64) > limits["max-compute-workgroups-per-dimension"]
        ):
            raise ValueError("CPM state buffers exceed this adapter's WebGPU buffer limits")
        radius_px = options.search_radius_px
        affine_radius = options.affine_radius_px or (2 * options.work_grid_stride)
        if affine_radius < options.work_grid_stride:
            raise ValueError("CPM affine radius must be at least the work-grid stride")

        scope = ResourceScope()
        start_descriptor = time.perf_counter()
        try:
            progress("CPM-HOG: building shared GPU descriptors")
            descriptors = self.deepmatching.build_hog_descriptors(
                scope, reference, target, options.downscale_levels, progress,
            )
            descriptor_ms = (time.perf_counter() - start_descriptor) * 1000
            if effective_patch > descriptors["width"] or effective_patch > descriptors["height"]:
                raise ValueError("CPM patch is larger than the reduced descriptor image")
            pyramid_start = time.perf_counter()
            levels = self._pyramid(scope, descriptors, effective_levels)
            pyramid_ms = (time.perf_counter() - pyramid_start) * 1000
            if len(levels) < effective_levels:
                progress("CPM pyramid depth reduced because the descriptor image became too small")
            module = self.device.create_shader_module(code=shader_source("cpm-match.wgsl"), label="CPM PatchMatch")
            search_pipeline = make_pipeline(self.device, module, "patchmatch_step", "CPM sparse HOG PatchMatch")
            oracle_pipeline = make_pipeline(self.device, module, "evaluate_oracle", "CPM analytic oracle cost")
            point_buffers = []
            for level_index in range(len(levels)):
                level_scale = scale * (2 ** level_index)
                coordinates = np.asarray([
                    (math.floor(x / level_scale + 0.5), math.floor(y / level_scale + 0.5))
                    for x, y in work_centers
                ], dtype=np.int32)
                point_buffers.append(self.device.create_buffer_with_data(
                    data=coordinates, usage=wgpu.BufferUsage.STORAGE,
                    label=f"CPM source points at level {level_index}",
                ))
                scope.keep(point_buffers[-1])
            diagnostic_indices = [
                _nearest_index(work_centers, x, y, grid_width, options.work_grid_stride)[0]
                for x, y in seeds
            ]
            matching_start = time.perf_counter()
            forward, level_snapshots = self._run_direction(
                scope, search_pipeline, levels, point_buffers, grid_width, grid_height,
                len(work_centers), support, scale, radius_px, options.seed & 0xFFFFFFFF,
                options, diagnostic_indices if options.oracle_affine is not None else None,
            )
            reverse_levels = [{**level, "reference": level["target"], "target": level["reference"]}
                              for level in levels]
            backward, _ = self._run_direction(
                scope, search_pipeline, reverse_levels, point_buffers, grid_width, grid_height,
                len(work_centers), support, scale, radius_px, options.seed & 0xFFFFFFFF, options,
            )
            matching_ms = (time.perf_counter() - matching_start) * 1000

            work_matches = []
            for index, (x, y) in enumerate(work_centers):
                valid = bool(forward[index, 3])
                dx = int(forward[index, 0]) * scale if valid else 0
                dy = int(forward[index, 1]) * scale if valid else 0
                cost = float(forward[index].view(np.float32)[2])
                score = 1 - min(1.0, max(0.0, cost))
                inverse = _interpolate(backward, work_centers, grid_width, grid_height,
                                        x + dx, y + dy, options.work_grid_stride) if valid else None
                reciprocal_error = math.hypot(dx + inverse[0] * scale, dy + inverse[1] * scale) if inverse else math.inf
                reciprocal = inverse is not None and reciprocal_error <= options.reciprocal_threshold_px
                work_matches.append({
                    "x": x, "y": y, "dx": dx, "dy": dy, "score": score,
                    "valid": valid, "reciprocal": reciprocal,
                    "reciprocal_error_px": reciprocal_error,
                    "usable": valid and reciprocal and score >= options.minimum_similarity,
                })
            reciprocal_count = sum(match["reciprocal"] for match in work_matches)
            usable_work_count = sum(match["usable"] for match in work_matches)

            requested_matches = []
            diagnostics: list[dict[str, Any]] = []
            for x, y in seeds:
                nearest, grid_distance = _nearest_index(work_centers, x, y, grid_width, options.work_grid_stride)
                measured = work_matches[nearest]
                fit = _fit_local_affine(work_matches, x, y, affine_radius)
                usable = fit is not None
                dx = float(fit["coefficients_x"][0]) if fit else float(measured["dx"])
                dy = float(fit["coefficients_y"][0]) if fit else float(measured["dy"])
                requested_matches.append({
                    "x": x, "y": y, "dx": dx, "dy": dy, "usable": usable,
                    "seed_x": x, "seed_y": y,
                    "target_x": x + dx, "target_y": y + dy,
                    "score": float(fit["mean_score"]) if fit else measured["score"],
                    "reciprocal": usable,
                    "inliers": fit["support_count"] if fit else 0,
                    "candidates": fit["candidate_count"] if fit else 0,
                    "inlier_fraction": fit["support_count"] / fit["candidate_count"] if fit else 0.0,
                    "residual": fit["rmse_px"] if fit else None,
                })
                affine_parameters = None
                if fit:
                    cx, cy = fit["coefficients_x"], fit["coefficients_y"]
                    affine_parameters = [x + cx[0], 1 + cx[1], cx[2], y + cy[0], cy[1], 1 + cy[2]]
                diagnostics.append({
                    "requested_x": x, "requested_y": y,
                    "measured_x": measured["x"], "measured_y": measured["y"],
                    "measured_dx": measured["dx"], "measured_dy": measured["dy"],
                    "grid_distance_px": grid_distance,
                    "directly_evaluated": False,
                    "projection_method": "local-affine-fit" if fit else "nearest-grid-fallback-rejected",
                    "affine_support_count": fit["support_count"] if fit else 0,
                    "affine_candidate_count": fit["candidate_count"] if fit else 0,
                    "affine_fit_rmse_px": fit["rmse_px"] if fit else None,
                    "affine_parameters": affine_parameters,
                    "reciprocal_error_px": (fit["reciprocal_error_px"] if fit else
                                             measured["reciprocal_error_px"] if math.isfinite(measured["reciprocal_error_px"])
                                             else None),
                })

            oracle_diagnostics = None
            if options.oracle_affine is not None:
                oracle_diagnostics = self._evaluate_oracle(
                    scope, oracle_pipeline, levels, point_buffers, work_centers,
                    grid_width, grid_height, support, scale, options.oracle_affine,
                    seeds, diagnostic_indices, level_snapshots or [],
                )
            raw_export = {}
            if include_correspondences:
                raw_export = {
                    "raw_correspondences_xyxy": [[m["x"], m["y"], m["x"] + m["dx"], m["y"] + m["dy"]] for m in work_matches],
                    "raw_match_scores": [m["score"] for m in work_matches],
                    "raw_correspondence_accepted": [m["usable"] for m in work_matches],
                    "raw_correspondence_rejection_reasons": [None if m["usable"] else "work_grid_rejected" for m in work_matches],
                    "raw_correspondence_kind": "direct_CPM_work_grid_matches",
                }
            return {
                "algorithm": "CPM-WebGPU",
                "width": width, "height": height, "patch_size": options.patch_size,
                "stride": options.stride, "scale": scale,
                "grid_width": grid_width, "grid_height": grid_height,
                "matches": requested_matches,
                **raw_export,
                "matching_support_px": support,
                "affine_radius_px": affine_radius,
                "levels": len(levels), "requested_levels": requested_levels,
                "count": len(requested_matches),
                "seed_mode": "manual" if options.seed_points else "grid",
                "work_grid_stride_px": options.work_grid_stride,
                "work_grid_point_count": len(work_centers),
                "search_radius_px": radius_px,
                "work_grid_reciprocal_count": reciprocal_count,
                "work_grid_usable_count": usable_work_count,
                "random_trials": options.random_trials,
                "initial_hypotheses": options.initial_hypotheses,
                "propagation_iterations": options.propagation_iterations,
                "local_search_radius_px": options.local_search_radius_px,
                "seed": options.seed & 0xFFFFFFFF,
                "oracle_diagnostics": oracle_diagnostics,
                "match_diagnostics": diagnostics,
                "descriptor_ms": descriptor_ms,
                "pyramid_ms": pyramid_ms,
                "matching_ms": matching_ms,
                "method_variant": "CPM-inspired sparse-grid PatchMatch using shared INRIA-style HOG descriptors, separate rigid matching support, mean-pooled HOG pyramid, stratified coarse initialization, and synchronous four-neighbor propagation",
            }
        finally:
            scope.close()

    def _evaluate_oracle(
        self,
        scope: ResourceScope,
        pipeline: Any,
        levels: list[dict[str, Any]],
        point_buffers: list[Any],
        centers: list[tuple[float, float]],
        grid_width: int,
        grid_height: int,
        support: int,
        base_scale: int,
        affine: tuple[float, ...],
        seeds: list[tuple[float, float]],
        sample_indices: list[int],
        found_levels: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if len(affine) != 6 or not all(math.isfinite(value) for value in affine):
            raise ValueError("CPM oracle affine must contain six finite values")
        result_levels = []
        for level_index, level in enumerate(levels):
            level_scale = base_scale * (2 ** level_index)
            candidates = np.zeros((len(centers), 4), dtype=np.int32)
            max_radius = 0
            for index, (x, y) in enumerate(centers):
                sx, sy = math.floor(x / level_scale + 0.5), math.floor(y / level_scale + 0.5)
                tx = math.floor((affine[0] * x + affine[1] * y + affine[2]) / level_scale + 0.5)
                ty = math.floor((affine[3] * x + affine[4] * y + affine[5]) / level_scale + 0.5)
                candidates[index, :2] = (tx - sx, ty - sy)
                max_radius = max(max_radius, abs(tx - sx), abs(ty - sy))
            candidate_buffer = self.device.create_buffer_with_data(
                data=candidates, usage=wgpu.BufferUsage.STORAGE, label="CPM oracle candidate offsets")
            scope.keep(candidate_buffer)
            output = create_buffer(self.device, scope, len(centers) * 16,
                                   wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                                   "CPM oracle cost states")
            readback = create_buffer(self.device, scope, len(sample_indices) * 16,
                                     wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
                                     "CPM oracle result readback")
            params = make_params(self.device, scope, [
                grid_width, grid_height, level["width"], level["height"],
                max(2, round(support / level_scale)), max_radius,
                0, level_index, 0, 0, 0, 0, 0, 0, 0, 0,
            ], "CPM oracle parameters")
            group = make_bind_group(self.device, pipeline, [
                {"binding": 0, "resource": level["reference"]["first"].create_view()},
                {"binding": 1, "resource": level["reference"]["second"].create_view()},
                {"binding": 2, "resource": level["target"]["first"].create_view()},
                {"binding": 3, "resource": level["target"]["second"].create_view()},
                {"binding": 4, "resource": {"buffer": point_buffers[level_index]}},
                {"binding": 5, "resource": {"buffer": candidate_buffer}},
                {"binding": 6, "resource": {"buffer": output}},
                {"binding": 7, "resource": {"buffer": params}},
            ])
            encoder = self.device.create_command_encoder(label="CPM oracle HOG costs")
            dispatch(encoder, pipeline, group, math.ceil(len(centers) / 64), label="CPM oracle costs")
            for output_index, state_index in enumerate(sample_indices):
                encoder.copy_buffer_to_buffer(output, state_index * 16, readback, output_index * 16, 16)
            self.device.queue.submit([encoder.finish()])
            raw = read_array(self.device, readback, np.dtype(np.int32), len(sample_indices) * 4)
            floats = raw.view(np.float32)
            level_result = []
            for sample_index, state_index in enumerate(sample_indices):
                offset = sample_index * 4
                found = found_levels[level_index][sample_index]
                valid = bool(raw[offset + 3])
                level_result.append({
                    "level": level_index,
                    "descriptor_scale_px": level_scale,
                    "found_dx_px": found["dx"] * level_scale,
                    "found_dy_px": found["dy"] * level_scale,
                    "found_cost": found["cost"] if found["valid"] else None,
                    "found_valid": found["valid"],
                    "oracle_dx_px": int(raw[offset]) * level_scale,
                    "oracle_dy_px": int(raw[offset + 1]) * level_scale,
                    "oracle_cost": float(floats[offset + 2]) if valid else None,
                    "oracle_valid": valid,
                    "oracle_minus_found_cost": float(floats[offset + 2]) - found["cost"]
                    if valid and found["valid"] else None,
                })
            result_levels.append(level_result)
        per_seed = []
        for seed_index, state_index in enumerate(sample_indices):
            requested_x, requested_y = seeds[seed_index]
            per_seed.append({
                "requested_x": requested_x,
                "requested_y": requested_y,
                "measured_x": centers[state_index][0],
                "measured_y": centers[state_index][1],
                "levels": [result_levels[level][seed_index] for level in range(len(levels))],
            })
        return {"affine_source_to_target": affine, "per_seed": per_seed}
