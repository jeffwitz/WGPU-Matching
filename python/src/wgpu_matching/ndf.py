"""WebGPU refinements used by the experimental NDF-DIC pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import wgpu

from ._gpu import (
    ResourceScope,
    create_buffer,
    dispatch,
    make_bind_group,
    make_pipeline,
    read_array,
    shader_source,
    upload_buffer,
)


@dataclass(frozen=True, slots=True)
class LocalNCCResult:
    """Locally refined displacement and compact matching diagnostics."""

    displacement_xy: np.ndarray
    score: np.ndarray
    margin: np.ndarray
    valid: np.ndarray


class NDFRefinementBackend:
    """Run coarse-guided local normalized correlation on a WebGPU device."""

    def __init__(self, device: Any) -> None:
        self.device = device
        serial_module = device.create_shader_module(
            label="NDF local normalized cross-correlation",
            code=shader_source("ndf-local-ncc.wgsl"),
        )
        self.serial_pipeline = make_pipeline(
            device, serial_module, "refine", "NDF serial local NCC control"
        )
        parallel_module = device.create_shader_module(
            label="NDF parallel local normalized cross-correlation",
            code=shader_source("ndf-local-ncc-parallel.wgsl"),
        )
        self.prepare_pipeline = make_pipeline(
            device, parallel_module, "prepare_source", "NDF local NCC source preparation"
        )
        self.score_pipeline = make_pipeline(
            device, parallel_module, "score_candidates", "NDF parallel local NCC candidate scores"
        )
        self.reduce_pipeline = make_pipeline(
            device, parallel_module, "reduce_best", "NDF local NCC candidate reduction"
        )

    def refine_local_ncc(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        initial_displacement_xy: np.ndarray,
        *,
        search_radius: int = 8,
        patch_radius: int = 5,
        initial_jacobian: np.ndarray | None = None,
        geometry: str = "translation",
        implementation: str = "parallel",
    ) -> LocalNCCResult:
        """Refine one displacement per feature-grid cell on the GPU.

        Images are floating-point grayscale arrays with the same dimensions.
        The initial field has shape ``(field_height, field_width, 2)`` and is
        expressed in input-image pixels. Only the small output field and
        diagnostics are read back to the CPU.
        """
        image_a = np.asarray(reference, dtype=np.float32)
        image_b = np.asarray(target, dtype=np.float32)
        flow = np.asarray(initial_displacement_xy, dtype=np.float32)
        if image_a.ndim != 2 or image_a.shape != image_b.shape:
            raise ValueError("reference and target must be same-sized 2D grayscale arrays")
        if flow.ndim != 3 or flow.shape[-1] != 2 or not flow.shape[0] or not flow.shape[1]:
            raise ValueError("initial_displacement_xy must have shape (height, width, 2)")
        if initial_jacobian is None:
            jacobian = np.broadcast_to(np.eye(2, dtype=np.float32), (*flow.shape[:2], 2, 2)).copy()
        else:
            jacobian = np.asarray(initial_jacobian, dtype=np.float32)
        if jacobian.shape != (*flow.shape[:2], 2, 2):
            raise ValueError("initial_jacobian must have shape (field_height, field_width, 2, 2)")
        if geometry not in {"translation", "affine", "hybrid"}:
            raise ValueError("geometry must be 'translation', 'affine', or 'hybrid'")
        if geometry == "translation":
            jacobian = np.broadcast_to(
                np.eye(2, dtype=np.float32), (*flow.shape[:2], 2, 2)
            ).copy()
        if (not np.isfinite(image_a).all() or not np.isfinite(image_b).all()
                or not np.isfinite(flow).all() or not np.isfinite(jacobian).all()):
            raise ValueError("images, initial displacement, and Jacobian must contain finite values")
        if not 1 <= int(search_radius) <= 32:
            raise ValueError("search_radius must be between 1 and 32 pixels")
        if not 1 <= int(patch_radius) <= 7:
            raise ValueError("patch_radius must be between 1 and 7 pixels")
        if implementation not in {"parallel", "serial"}:
            raise ValueError("implementation must be 'parallel' or 'serial'")
        if implementation == "serial" and geometry == "hybrid":
            raise ValueError("hybrid geometry is supported by the parallel WebGPU implementation only")
        height, width = image_a.shape
        field_height, field_width = flow.shape[:2]
        if min(width, height) <= 2 * patch_radius + 1:
            raise ValueError("image is too small for the requested correlation patch")
        point_count = field_height * field_width
        offsets_per_geometry = (2 * int(search_radius) + 1) ** 2
        geometry_count = 2 if geometry == "hybrid" else 1
        candidate_count = offsets_per_geometry * geometry_count
        patch_count = (2 * int(patch_radius) + 1) ** 2
        params = np.asarray(
            [
                width, height, field_width, field_height, search_radius, patch_radius,
                candidate_count, patch_count,
            ],
            dtype=np.uint32,
        )
        with ResourceScope() as scope:
            image_a_buffer = upload_buffer(
                self.device, scope, image_a, wgpu.BufferUsage.STORAGE, "NDF reference image"
            )
            image_b_buffer = upload_buffer(
                self.device, scope, image_b, wgpu.BufferUsage.STORAGE, "NDF target image"
            )
            flow_buffer = upload_buffer(
                self.device, scope, flow, wgpu.BufferUsage.STORAGE, "NDF initial displacement"
            )
            jacobian_buffer = upload_buffer(
                self.device, scope, jacobian, wgpu.BufferUsage.STORAGE, "NDF initial Jacobian"
            )
            output_flow = create_buffer(
                self.device,
                scope,
                point_count * 2 * np.dtype(np.float32).itemsize,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "NDF refined displacement",
            )
            output_scores = create_buffer(
                self.device,
                scope,
                point_count * 2 * np.dtype(np.float32).itemsize,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "NDF local NCC score and margin",
            )
            output_valid = create_buffer(
                self.device,
                scope,
                point_count * np.dtype(np.uint32).itemsize,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "NDF local NCC validity",
            )
            params_buffer = upload_buffer(
                self.device, scope, params, wgpu.BufferUsage.UNIFORM, "NDF local NCC parameters"
            )
            encoder = self.device.create_command_encoder(label="NDF local NCC")
            if implementation == "serial":
                group = make_bind_group(
                    self.device,
                    self.serial_pipeline,
                    [
                        {"binding": 0, "resource": {"buffer": image_a_buffer}},
                        {"binding": 1, "resource": {"buffer": image_b_buffer}},
                        {"binding": 2, "resource": {"buffer": flow_buffer}},
                        {"binding": 3, "resource": {"buffer": output_flow}},
                        {"binding": 4, "resource": {"buffer": output_scores}},
                        {"binding": 5, "resource": {"buffer": output_valid}},
                        {"binding": 6, "resource": {"buffer": params_buffer}},
                        {"binding": 7, "resource": {"buffer": jacobian_buffer}},
                    ],
                )
                dispatch(
                    encoder, self.serial_pipeline, group,
                    math.ceil(point_count / 64), label="NDF serial local NCC control",
                )
            else:
                source_values = create_buffer(
                    self.device, scope, point_count * 225 * 4,
                    wgpu.BufferUsage.STORAGE, "NDF prepared reference patches",
                )
                source_stats = create_buffer(
                    self.device, scope, point_count * 2 * 4,
                    wgpu.BufferUsage.STORAGE, "NDF reference patch statistics",
                )
                scores_buffer = create_buffer(
                    self.device, scope, point_count * candidate_count * 4,
                    wgpu.BufferUsage.STORAGE, "NDF candidate NCC scores",
                )
                prepare_group = make_bind_group(
                    self.device,
                    self.prepare_pipeline,
                    [
                        {"binding": 0, "resource": {"buffer": image_a_buffer}},
                        {"binding": 3, "resource": {"buffer": source_values}},
                        {"binding": 4, "resource": {"buffer": source_stats}},
                        {"binding": 9, "resource": {"buffer": params_buffer}},
                    ],
                )
                score_group = make_bind_group(
                    self.device,
                    self.score_pipeline,
                    [
                        {"binding": 1, "resource": {"buffer": image_b_buffer}},
                        {"binding": 2, "resource": {"buffer": flow_buffer}},
                        {"binding": 10, "resource": {"buffer": jacobian_buffer}},
                        {"binding": 3, "resource": {"buffer": source_values}},
                        {"binding": 4, "resource": {"buffer": source_stats}},
                        {"binding": 5, "resource": {"buffer": scores_buffer}},
                        {"binding": 9, "resource": {"buffer": params_buffer}},
                    ],
                )
                reduce_group = make_bind_group(
                    self.device,
                    self.reduce_pipeline,
                    [
                        {"binding": 2, "resource": {"buffer": flow_buffer}},
                        {"binding": 10, "resource": {"buffer": jacobian_buffer}},
                        {"binding": 5, "resource": {"buffer": scores_buffer}},
                        {"binding": 6, "resource": {"buffer": output_flow}},
                        {"binding": 7, "resource": {"buffer": output_scores}},
                        {"binding": 8, "resource": {"buffer": output_valid}},
                        {"binding": 9, "resource": {"buffer": params_buffer}},
                    ],
                )
                dispatch(
                    encoder, self.prepare_pipeline, prepare_group,
                    math.ceil(point_count / 64), label="NDF prepare reference patches",
                )
                dispatch(
                    encoder, self.score_pipeline, score_group,
                    math.ceil(point_count * candidate_count / 64),
                    label="NDF score local candidate matches",
                )
                dispatch(
                    encoder, self.reduce_pipeline, reduce_group,
                    math.ceil(point_count / 64), label="NDF reduce local match scores",
                )
            self.device.queue.submit([encoder.finish()])
            displacement = read_array(
                self.device, output_flow, np.dtype(np.float32), point_count * 2
            ).reshape(field_height, field_width, 2)
            scores = read_array(
                self.device, output_scores, np.dtype(np.float32), point_count * 2
            ).reshape(field_height, field_width, 2)
            valid = read_array(
                self.device, output_valid, np.dtype(np.uint32), point_count
            ).reshape(field_height, field_width).astype(bool)
        return LocalNCCResult(displacement, scores[..., 0], scores[..., 1], valid)
