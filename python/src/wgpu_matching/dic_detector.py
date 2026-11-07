"""WGSL structure-tensor point selector for speckle images.

This is a non-learned baseline. Its structure-tensor score measures local
photometric conditioning; it does not measure whether a point is unique across
images or whether a descriptor can match it after a large displacement.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import wgpu

from ._gpu import (
    ResourceScope,
    create_buffer,
    create_texture,
    dispatch,
    make_bind_group,
    make_pipeline,
    read_array,
    shader_source,
    uniform_buffer,
    upload_buffer,
)
from .features import FeatureSet


@dataclass(frozen=True, slots=True)
class DICDetectorOptions:
    """Parameters for the WGSL structure-tensor detector."""

    stride_px: int = 4
    tensor_radius_px: int = 6
    max_keypoints: int = 4096
    tile_columns: int = 16
    tile_rows: int = 9
    minimum_eigenvalue: float = 1e-5
    minimum_score: float = 0.05


@dataclass(slots=True)
class DICDetectorResult:
    """Selected DIC texture points and local information diagnostics."""

    features: FeatureSet
    minimum_eigenvalues: np.ndarray
    condition_numbers: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.minimum_eigenvalues = np.ascontiguousarray(
            self.minimum_eigenvalues, dtype=np.float32
        ).reshape(-1)
        self.condition_numbers = np.ascontiguousarray(
            self.condition_numbers, dtype=np.float32
        ).reshape(-1)
        count = len(self.features.xy)
        if self.minimum_eigenvalues.shape != (count,) or self.condition_numbers.shape != (count,):
            raise ValueError("DIC tensor diagnostics must align with the selected points")


class DICDetectorBackend:
    """Run tensor scoring, NMS, and spatial quotas using only WGSL kernels."""

    PARAM_BYTES = 48

    def __init__(self, device: Any) -> None:
        self.device = device
        self._pipelines: tuple[Any, Any, Any] | None = None
        self._learned_pipeline: Any | None = None

    def _get_pipelines(self) -> tuple[Any, Any, Any]:
        if self._pipelines is None:
            module = self.device.create_shader_module(
                label="DIC structure-tensor detector", code=shader_source("dic-detector.wgsl")
            )
            self._pipelines = (
                make_pipeline(
                    self.device, module, "compute_tensor_response", "DIC tensor response"
                ),
                make_pipeline(self.device, module, "suppress_nonmaxima", "DIC tensor NMS"),
                make_pipeline(self.device, module, "select_spatial_quotas", "DIC spatial quotas"),
            )
        return self._pipelines

    def _get_learned_pipeline(self) -> Any:
        if self._learned_pipeline is None:
            module = self.device.create_shader_module(
                label="DIC learned score network", code=shader_source("dic-learned.wgsl")
            )
            self._learned_pipeline = make_pipeline(
                self.device, module, "score_keypoints", "DIC learned point scores"
            )
        return self._learned_pipeline

    @staticmethod
    def _as_gray(image: np.ndarray) -> np.ndarray:
        pixels = np.asarray(image)
        if pixels.ndim != 2:
            raise ValueError("DIC detector expects a two-dimensional grayscale image")
        if not np.issubdtype(pixels.dtype, np.number) or not np.isfinite(pixels).all():
            raise ValueError("Image pixels must be finite numeric values")
        return np.ascontiguousarray(np.clip(pixels, 0, 255).astype(np.uint8))

    @staticmethod
    def _pack_mask(mask: np.ndarray) -> np.ndarray:
        packed = np.packbits(np.asarray(mask, dtype=np.bool_).reshape(-1), bitorder="little")
        padded_size = (packed.size + 3) & ~3
        if padded_size != packed.size:
            packed = np.pad(packed, (0, padded_size - packed.size))
        return np.ascontiguousarray(packed).view(np.uint32)

    def _params(
        self,
        width: int,
        height: int,
        grid_width: int,
        grid_height: int,
        options: DICDetectorOptions,
        tile_columns: int,
        tile_rows: int,
        max_keypoints: int,
        roi_enabled: bool,
        learned_enabled: bool,
        scope: ResourceScope,
    ) -> Any:
        raw = bytearray(self.PARAM_BYTES)
        np.frombuffer(raw, dtype=np.uint32, count=10)[:] = (
            width,
            height,
            grid_width,
            grid_height,
            options.stride_px,
            options.tensor_radius_px,
            tile_columns,
            tile_rows,
            max_keypoints,
            int(roi_enabled),
        )
        threshold = options.minimum_score if learned_enabled else options.minimum_eigenvalue
        np.frombuffer(raw, dtype=np.float32, count=1, offset=40)[0] = threshold
        np.frombuffer(raw, dtype=np.uint32, count=1, offset=44)[0] = int(learned_enabled)
        return uniform_buffer(self.device, scope, bytes(raw), "DIC detector parameters")

    def compute(
        self,
        image: np.ndarray,
        *,
        options: DICDetectorOptions | None = None,
        roi_mask: np.ndarray | None = None,
        learned_weights_path: str | Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> DICDetectorResult:
        settings = options or DICDetectorOptions()
        if settings.stride_px < 2:
            raise ValueError("stride_px must be at least 2")
        if settings.tensor_radius_px < 2 or settings.tensor_radius_px > 16:
            raise ValueError("tensor_radius_px must be between 2 and 16")
        if settings.max_keypoints < 1:
            raise ValueError("max_keypoints must be positive")
        if settings.tile_columns < 1 or settings.tile_rows < 1:
            raise ValueError("spatial quota tile counts must be positive")
        if not math.isfinite(settings.minimum_eigenvalue) or settings.minimum_eigenvalue < 0:
            raise ValueError("minimum_eigenvalue must be finite and non-negative")
        if not math.isfinite(settings.minimum_score) or not 0.0 <= settings.minimum_score <= 1.0:
            raise ValueError("minimum_score must be finite and between zero and one")

        learned_weights = None
        learned_weights_sha256 = None
        if learned_weights_path is not None:
            weights_path = Path(learned_weights_path).expanduser().resolve()
            if not weights_path.is_file():
                raise FileNotFoundError(f"DIC detector weights do not exist: {weights_path}")
            with np.load(weights_path, allow_pickle=False) as archive:
                names = (
                    "conv1.weight",
                    "conv1.bias",
                    "depthwise.weight",
                    "depthwise.bias",
                    "score.weight",
                    "score.bias",
                )
                missing = [name for name in names if name not in archive]
                if missing:
                    raise ValueError(f"DIC learned weights are missing arrays: {missing}")
                arrays = [np.asarray(archive[name], dtype=np.float32) for name in names]
            learned_weights = np.concatenate([array.reshape(-1) for array in arrays])
            if learned_weights.shape != (169,) or not np.isfinite(learned_weights).all():
                raise ValueError(
                    "DIC learned weight bundle has an invalid layout or non-finite values"
                )
            learned_weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()

        pixels = self._as_gray(image)
        height, width = pixels.shape
        mask = None if roi_mask is None else np.asarray(roi_mask) != 0
        if mask is not None and mask.shape != pixels.shape:
            raise ValueError("roi_mask must have the same shape as the grayscale image")
        if (
            width > self.device.limits["max-texture-dimension-2d"]
            or height > self.device.limits["max-texture-dimension-2d"]
        ):
            raise ValueError("Image exceeds the adapter's maximum 2D texture size")

        grid_width = math.ceil(width / settings.stride_px)
        grid_height = math.ceil(height / settings.stride_px)
        candidate_count = grid_width * grid_height
        tile_columns = min(settings.tile_columns, grid_width)
        tile_rows = min(settings.tile_rows, grid_height)
        tile_count = tile_columns * tile_rows
        selected_count = min(settings.max_keypoints, candidate_count)
        estimated_selection_work = candidate_count * math.ceil(selected_count / tile_count)
        if estimated_selection_work > 180_000_000:
            raise ValueError(
                "DIC detector top-score selection would require an impractically large serial scan; "
                "increase the spatial tile count or reduce max_keypoints"
            )
        response_bytes = candidate_count * 16
        flag_bytes = candidate_count * 4
        point_bytes = max(32, selected_count * 32)
        max_binding = self.device.limits["max-storage-buffer-binding-size"]
        if max(response_bytes, flag_bytes, point_bytes) > max_binding:
            raise ValueError("DIC detector buffers exceed the adapter storage-buffer limit")

        callback = on_progress or (lambda _: None)
        callback("Preparing DIC structure-tensor detection")
        response_pipeline, nms_pipeline, quota_pipeline = self._get_pipelines()
        resources = ResourceScope()
        started = time.perf_counter()
        try:
            texture = create_texture(
                self.device,
                resources,
                width,
                height,
                "r8unorm",
                wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                "DIC detector grayscale image",
            )
            self.device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                pixels,
                {"offset": 0, "bytes_per_row": width, "rows_per_image": height},
                (width, height, 1),
            )
            if mask is None:
                packed_mask = np.zeros(1, dtype=np.uint32)
            else:
                packed_mask = self._pack_mask(mask)
            mask_buffer = upload_buffer(
                self.device,
                resources,
                packed_mask,
                wgpu.BufferUsage.STORAGE,
                "Packed DIC ROI mask",
            )
            response_buffer = create_buffer(
                self.device,
                resources,
                response_bytes,
                wgpu.BufferUsage.STORAGE,
                "DIC tensor response",
            )
            peaks_buffer = create_buffer(
                self.device,
                resources,
                response_bytes,
                wgpu.BufferUsage.STORAGE,
                "DIC NMS tensor peaks",
            )
            flags_buffer = create_buffer(
                self.device,
                resources,
                flag_bytes,
                wgpu.BufferUsage.STORAGE,
                "DIC selected-point flags",
            )
            points_buffer = create_buffer(
                self.device,
                resources,
                point_bytes,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "DIC spatial-quota points",
            )
            params_buffer = self._params(
                width,
                height,
                grid_width,
                grid_height,
                settings,
                tile_columns,
                tile_rows,
                selected_count,
                mask is not None,
                learned_weights is not None,
                resources,
            )
            response_group = make_bind_group(
                self.device,
                response_pipeline,
                [
                    {"binding": 0, "resource": texture.create_view()},
                    {"binding": 1, "resource": {"buffer": mask_buffer}},
                    {"binding": 2, "resource": {"buffer": response_buffer}},
                    {"binding": 4, "resource": {"buffer": flags_buffer}},
                    {"binding": 6, "resource": {"buffer": params_buffer}},
                ],
            )
            nms_group = make_bind_group(
                self.device,
                nms_pipeline,
                [
                    {"binding": 2, "resource": {"buffer": response_buffer}},
                    {"binding": 3, "resource": {"buffer": peaks_buffer}},
                    {"binding": 6, "resource": {"buffer": params_buffer}},
                ],
            )
            quota_group = make_bind_group(
                self.device,
                quota_pipeline,
                [
                    {"binding": 2, "resource": {"buffer": response_buffer}},
                    {"binding": 3, "resource": {"buffer": peaks_buffer}},
                    {"binding": 4, "resource": {"buffer": flags_buffer}},
                    {"binding": 5, "resource": {"buffer": points_buffer}},
                    {"binding": 6, "resource": {"buffer": params_buffer}},
                ],
            )

            learned_group = None
            if learned_weights is not None:
                learned_pipeline = self._get_learned_pipeline()
                weight_buffer = upload_buffer(
                    self.device,
                    resources,
                    learned_weights,
                    wgpu.BufferUsage.STORAGE,
                    "DIC learned score network weights",
                )
                learned_group = make_bind_group(
                    self.device,
                    learned_pipeline,
                    [
                        {"binding": 0, "resource": texture.create_view()},
                        {"binding": 1, "resource": {"buffer": mask_buffer}},
                        {"binding": 2, "resource": {"buffer": response_buffer}},
                        {"binding": 6, "resource": {"buffer": params_buffer}},
                        {"binding": 7, "resource": {"buffer": weight_buffer}},
                    ],
                )

            callback("Scoring local gradient information on WebGPU")
            dispatch_started = time.perf_counter()
            encoder = self.device.create_command_encoder(label="DIC texture detector")
            dispatch(
                encoder,
                response_pipeline,
                response_group,
                math.ceil(grid_width / 8),
                math.ceil(grid_height / 8),
                label="DIC structure tensor response",
            )
            if learned_group is not None:
                dispatch(
                    encoder,
                    learned_pipeline,
                    learned_group,
                    math.ceil(grid_width / 8),
                    math.ceil(grid_height / 8),
                    label="DIC learned point scores",
                )
            dispatch(
                encoder,
                nms_pipeline,
                nms_group,
                math.ceil(candidate_count / 64),
                label="DIC tensor non-maximum suppression",
            )
            dispatch(
                encoder,
                quota_pipeline,
                quota_group,
                tile_count,
                label="DIC spatial keypoint quotas",
            )
            self.device.queue.submit([encoder.finish()])
            values = read_array(
                self.device,
                points_buffer,
                np.dtype(np.float32),
                selected_count * 8,
            ).reshape(selected_count, 2, 4)
            dispatch_readback_ms = (time.perf_counter() - dispatch_started) * 1000.0
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        finally:
            resources.close()

        valid = values[:, 0, 2] >= 0.0
        selected = values[valid]
        detector_name = (
            "DIC-Detector-learned-v0"
            if learned_weights is not None
            else "DIC-Detector-tensor-baseline"
        )
        features = FeatureSet(
            xy=selected[:, 0, :2],
            scores=selected[:, 0, 2],
            metadata={
                "detector": detector_name,
                "coordinate_convention": "source-image pixel centres; integer values identify pixel centres",
                "weights_sha256": learned_weights_sha256,
                "quota_tiles": [tile_columns, tile_rows],
                "candidate_count": candidate_count,
                "selected_keypoints": len(selected),
                "selection_scope": "WGSL per-tile top-score candidate pool",
            },
        )
        metadata = {
            "detector": detector_name,
            "algorithm": (
                "small learned convolutional score network; 3x3 grid NMS; deterministic per-tile quotas"
                if learned_weights is not None
                else "local structure-tensor minimum eigenvalue; 3x3 grid NMS; deterministic per-tile top-score quotas"
            ),
            "source_size_px": [width, height],
            "stride_px": settings.stride_px,
            "tensor_radius_px": settings.tensor_radius_px,
            "minimum_eigenvalue": settings.minimum_eigenvalue,
            "requested_keypoints": settings.max_keypoints,
            "candidate_count": candidate_count,
            "selected_keypoints": len(selected),
            "quota_tiles": [tile_columns, tile_rows],
            "roi_mask_applied": mask is not None,
            "readback_bytes": int(selected_count * 32),
            "full_detection_wall_ms": elapsed_ms,
            "dispatch_and_readback_ms": dispatch_readback_ms,
            "score_interpretation": (
                "sigmoid learned score; trained with a structure-tensor teacher and analytic-flow equivariance, not IC-GN convergence"
                if learned_weights is not None
                else "minimum eigenvalue of the mean local gradient structure tensor; larger values indicate better-conditioned photometric gradients, not unique cross-image correspondences"
            ),
            "weights_sha256": learned_weights_sha256,
        }
        return DICDetectorResult(
            features=features,
            minimum_eigenvalues=selected[:, 1, 0],
            condition_numbers=selected[:, 1, 2],
            metadata=metadata,
        )
