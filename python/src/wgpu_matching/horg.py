"""Native wgpu-py host orchestration for the HORG WGSL kernels."""

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
    uniform_buffer,
    upload_buffer,
)
from .features import FeatureSet

FEATURE_STRIDE = 68
DESCRIPTOR_SIZE = 64
MAX_DETECTIONS_PER_OCTAVE = 60_000
MAX_RANSAC_CANDIDATES = 96


class HORGBackend:
    def __init__(self, device: Any) -> None:
        self.device = device

    def _shader(self, name: str) -> Any:
        return self.device.create_shader_module(label=name, code=shader_source(name))

    def _pyramid_params(
        self,
        scope: ResourceScope,
        dims: tuple[int, int, int, int],
        config: tuple[int, int, int, int],
        weights: np.ndarray | None = None,
        label: str = "HORG pyramid parameters",
    ) -> Any:
        raw = bytearray(96)
        np.frombuffer(raw, dtype=np.uint32, count=4, offset=0)[:] = dims
        np.frombuffer(raw, dtype=np.uint32, count=4, offset=16)[:] = config
        if weights is not None:
            np.frombuffer(raw, dtype=np.float32, count=16, offset=32)[:] = weights
        return uniform_buffer(self.device, scope, bytes(raw), label)

    @staticmethod
    def _gaussian_weights(sigma: float) -> tuple[int, np.ndarray]:
        radius = min(7, max(1, math.ceil(3.0 * sigma)))
        values = np.zeros(16, dtype=np.float32)
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
        weights /= weights.sum()
        values[: len(weights)] = weights.astype(np.float32)
        return radius, values

    def _new_texture(
        self,
        scope: ResourceScope,
        width: int,
        height: int,
        label: str,
        *,
        format: str = "r32float",
        usage: int | None = None,
    ) -> Any:
        if width < 1 or height < 1:
            raise ValueError("HORG generated an empty pyramid texture")
        if usage is None:
            usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
        return create_texture(self.device, scope, width, height, format, usage, label)

    @staticmethod
    def _view_entry(binding: int, texture: Any) -> dict[str, Any]:
        return {"binding": binding, "resource": texture.create_view()}

    @staticmethod
    def _buffer_entry(binding: int, buffer: Any) -> dict[str, Any]:
        return {"binding": binding, "resource": {"buffer": buffer}}

    def _blur(
        self,
        scope: ResourceScope,
        encoder: Any,
        pipeline: Any,
        source: Any,
        destination: Any,
        width: int,
        height: int,
        sigma: float,
        axis: int,
        uniforms: list[Any],
    ) -> None:
        radius, weights = self._gaussian_weights(sigma)
        params = self._pyramid_params(scope, (width, height, 0, 0), (radius, axis, 0, 0), weights)
        uniforms.append(params)
        group = make_bind_group(
            self.device,
            pipeline,
            [
                self._view_entry(0, source),
                self._view_entry(1, destination),
                self._buffer_entry(2, params),
            ],
        )
        dispatch(
            encoder,
            pipeline,
            group,
            math.ceil(width / 8),
            math.ceil(height / 8),
            label="HORG Gaussian blur",
        )

    def _subtract(
        self,
        scope: ResourceScope,
        encoder: Any,
        pipeline: Any,
        lower: Any,
        upper: Any,
        output: Any,
        width: int,
        height: int,
        uniforms: list[Any],
    ) -> None:
        params = self._pyramid_params(scope, (width, height, 0, 0), (0, 0, 0, 0))
        uniforms.append(params)
        group = make_bind_group(
            self.device,
            pipeline,
            [
                self._view_entry(0, lower),
                self._buffer_entry(2, params),
                self._view_entry(3, upper),
                self._view_entry(4, output),
            ],
        )
        dispatch(
            encoder, pipeline, group, math.ceil(width / 8), math.ceil(height / 8), label="HORG DoG"
        )

    def _image_features(
        self,
        pixels: np.ndarray,
        width: int,
        height: int,
        reduction: int,
        image_name: str,
        on_progress: Callable[[str], None],
    ) -> dict[str, Any]:
        limits = self.device.limits
        if (
            width > limits["max-texture-dimension-2d"]
            or height > limits["max-texture-dimension-2d"]
        ):
            raise ValueError("Image exceeds the adapter's maximum 2D texture size")

        resources = ResourceScope()
        segments: list[np.ndarray] = []
        total_detected = 0
        overflow = 0
        octave_base = None
        base_width = max(1, width // reduction)
        base_height = max(1, height // reduction)
        octave_count = min(6, max(1, math.floor(math.log2(min(base_width, base_height))) - 2))
        gaussian_sigma = 1.6
        contrast_threshold = 0.04 / 3.0
        edge_limit = (10 + 1) * (10 + 1) / 10

        input_texture = self._new_texture(
            resources,
            width,
            height,
            f"{image_name} source",
            format="r8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": input_texture, "mip_level": 0, "origin": (0, 0, 0)},
            pixels,
            {"offset": 0, "bytes_per_row": width, "rows_per_image": height},
            (width, height, 1),
        )

        pyramid_module = self._shader("horg-pyramid.wgsl")
        detect_module = self._shader("horg-detect.wgsl")
        descriptor_module = self._shader("horg-descriptor.wgsl")
        downsample_base = make_pipeline(
            self.device, pyramid_module, "downsample_base", "HORG input reduction"
        )
        downsample_half = make_pipeline(
            self.device, pyramid_module, "downsample_half", "HORG octave reduction"
        )
        blur_pipeline = make_pipeline(
            self.device, pyramid_module, "gaussian_blur", "HORG Gaussian blur"
        )
        subtract_pipeline = make_pipeline(
            self.device, pyramid_module, "subtract_gaussian", "HORG DoG"
        )
        detect_pipeline = make_pipeline(
            self.device, detect_module, "detect_extrema", "HORG SIFT extrema"
        )
        descriptor_pipeline = make_pipeline(
            self.device, descriptor_module, "build_horg", "HORG descriptor"
        )

        try:
            for octave in range(octave_count):
                if min(base_width, base_height) < 28:
                    break
                gaussian: list[Any] = []
                dogs: list[Any] = []
                scratch_textures: list[Any] = []
                temporary_uniforms: list[Any] = []
                raw_base = None
                blur_scratch = None
                encoder = self.device.create_command_encoder(
                    label=f"{image_name} HORG octave {octave}"
                )

                if octave == 0:
                    raw_base = self._new_texture(
                        resources, base_width, base_height, "HORG initial downsample"
                    )
                    blur_scratch = self._new_texture(
                        resources, base_width, base_height, "HORG initial blur scratch"
                    )
                    down_params = self._pyramid_params(
                        resources, (width, height, base_width, base_height), (reduction, 0, 0, 0)
                    )
                    temporary_uniforms.append(down_params)
                    down_group = make_bind_group(
                        self.device,
                        downsample_base,
                        [
                            self._view_entry(0, input_texture),
                            self._view_entry(1, raw_base),
                            self._buffer_entry(2, down_params),
                        ],
                    )
                    dispatch(
                        encoder,
                        downsample_base,
                        down_group,
                        math.ceil(base_width / 8),
                        math.ceil(base_height / 8),
                        label="HORG initial reduction",
                    )
                    first = self._new_texture(
                        resources, base_width, base_height, "HORG Gaussian level 0"
                    )
                    gaussian.append(first)
                    self._blur(
                        resources,
                        encoder,
                        blur_pipeline,
                        raw_base,
                        blur_scratch,
                        base_width,
                        base_height,
                        gaussian_sigma,
                        0,
                        temporary_uniforms,
                    )
                    self._blur(
                        resources,
                        encoder,
                        blur_pipeline,
                        blur_scratch,
                        first,
                        base_width,
                        base_height,
                        gaussian_sigma,
                        1,
                        temporary_uniforms,
                    )
                else:
                    gaussian.append(octave_base)
                    octave_base = None

                for level in range(1, 6):
                    sigma_before = gaussian_sigma * 2 ** ((level - 1) / 3)
                    sigma_after = gaussian_sigma * 2 ** (level / 3)
                    sigma = math.sqrt(sigma_after * sigma_after - sigma_before * sigma_before)
                    scratch = self._new_texture(
                        resources, base_width, base_height, f"HORG blur scratch {level}"
                    )
                    scratch_textures.append(scratch)
                    next_level = self._new_texture(
                        resources, base_width, base_height, f"HORG Gaussian {level}"
                    )
                    gaussian.append(next_level)
                    self._blur(
                        resources,
                        encoder,
                        blur_pipeline,
                        gaussian[level - 1],
                        scratch,
                        base_width,
                        base_height,
                        sigma,
                        0,
                        temporary_uniforms,
                    )
                    self._blur(
                        resources,
                        encoder,
                        blur_pipeline,
                        scratch,
                        next_level,
                        base_width,
                        base_height,
                        sigma,
                        1,
                        temporary_uniforms,
                    )

                for level in range(5):
                    dog = self._new_texture(resources, base_width, base_height, f"HORG DoG {level}")
                    dogs.append(dog)
                    self._subtract(
                        resources,
                        encoder,
                        subtract_pipeline,
                        gaussian[level],
                        gaussian[level + 1],
                        dog,
                        base_width,
                        base_height,
                        temporary_uniforms,
                    )

                counter = create_buffer(
                    self.device,
                    resources,
                    4,
                    wgpu.BufferUsage.STORAGE
                    | wgpu.BufferUsage.COPY_SRC
                    | wgpu.BufferUsage.COPY_DST,
                    "HORG keypoint count",
                )
                self.device.queue.write_buffer(counter, 0, np.zeros(1, dtype=np.uint32))
                keypoints = create_buffer(
                    self.device,
                    resources,
                    MAX_DETECTIONS_PER_OCTAVE * 16,
                    wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                    "HORG keypoints",
                )
                detect_params_bytes = bytearray(48)
                np.frombuffer(detect_params_bytes, dtype=np.uint32, count=4, offset=0)[:] = [
                    base_width,
                    base_height,
                    0,
                    0,
                ]
                np.frombuffer(detect_params_bytes, dtype=np.float32, count=4, offset=16)[:] = [
                    contrast_threshold,
                    edge_limit,
                    0,
                    0,
                ]
                np.frombuffer(detect_params_bytes, dtype=np.uint32, count=4, offset=32)[:] = [
                    MAX_DETECTIONS_PER_OCTAVE,
                    0,
                    0,
                    0,
                ]
                detect_params = uniform_buffer(
                    self.device, resources, bytes(detect_params_bytes), "HORG detect params"
                )
                temporary_uniforms.append(detect_params)
                detect_group = make_bind_group(
                    self.device,
                    detect_pipeline,
                    [
                        *[self._view_entry(index, dog) for index, dog in enumerate(dogs)],
                        self._buffer_entry(5, counter),
                        self._buffer_entry(6, keypoints),
                        self._buffer_entry(7, detect_params),
                    ],
                )
                dispatch(
                    encoder,
                    detect_pipeline,
                    detect_group,
                    math.ceil(base_width / 8),
                    math.ceil(base_height / 8),
                    3,
                    "HORG SIFT scale space",
                )
                self.device.queue.submit([encoder.finish()])
                count = int(
                    np.frombuffer(self.device.queue.read_buffer(counter, 0, 4), dtype=np.uint32)[0]
                )
                feature_count = min(count, MAX_DETECTIONS_PER_OCTAVE)
                total_detected += count
                overflow += max(0, count - MAX_DETECTIONS_PER_OCTAVE)
                on_progress(
                    f"HORG {image_name}: octave {octave + 1}/{octave_count}, {feature_count} keypoints"
                )

                for uniform in temporary_uniforms:
                    resources.release(uniform)
                if raw_base is not None:
                    resources.release(raw_base)
                if blur_scratch is not None:
                    resources.release(blur_scratch)
                for texture in scratch_textures:
                    resources.release(texture)

                if feature_count:
                    feature_bytes = feature_count * FEATURE_STRIDE * 4
                    if feature_bytes > limits["max-storage-buffer-binding-size"]:
                        raise ValueError(
                            "HORG descriptors exceed the adapter storage buffer binding size"
                        )
                    output = create_buffer(
                        self.device,
                        resources,
                        feature_bytes,
                        wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                        "HORG features",
                    )
                    descriptor_params = make_params(
                        self.device,
                        resources,
                        [
                            base_width,
                            base_height,
                            feature_count,
                            0,
                            reduction * (1 << octave),
                            0,
                            0,
                            base_width,
                        ],
                        "HORG descriptor params",
                    )
                    descriptor_group = make_bind_group(
                        self.device,
                        descriptor_pipeline,
                        [
                            self._view_entry(0, gaussian[2]),
                            self._view_entry(1, gaussian[3]),
                            self._view_entry(2, gaussian[4]),
                            self._buffer_entry(3, keypoints),
                            self._buffer_entry(4, output),
                            self._buffer_entry(5, descriptor_params),
                        ],
                    )
                    descriptor_encoder = self.device.create_command_encoder(
                        label="HORG descriptors"
                    )
                    dispatch(
                        descriptor_encoder,
                        descriptor_pipeline,
                        descriptor_group,
                        feature_count,
                        label="HORG orientation and descriptors",
                    )
                    self.device.queue.submit([descriptor_encoder.finish()])
                    raw = self.device.queue.read_buffer(output, 0, feature_bytes)
                    segments.append(np.frombuffer(raw, dtype=np.float32).copy())
                    resources.release(output)
                    resources.release(descriptor_params)

                has_next = octave < octave_count - 1 and min(base_width, base_height) >= 56
                if has_next:
                    next_width, next_height = base_width // 2, base_height // 2
                    next_base = self._new_texture(
                        resources, next_width, next_height, "HORG next octave base"
                    )
                    down_params = self._pyramid_params(
                        resources, (base_width, base_height, next_width, next_height), (2, 0, 0, 0)
                    )
                    down_group = make_bind_group(
                        self.device,
                        downsample_half,
                        [
                            self._view_entry(0, gaussian[3]),
                            self._view_entry(1, next_base),
                            self._buffer_entry(2, down_params),
                        ],
                    )
                    down_encoder = self.device.create_command_encoder(label="HORG octave reduction")
                    dispatch(
                        down_encoder,
                        downsample_half,
                        down_group,
                        math.ceil(next_width / 8),
                        math.ceil(next_height / 8),
                        label="HORG octave reduction",
                    )
                    self.device.queue.submit([down_encoder.finish()])
                    # A synchronous queue read is the portable wgpu-py fence;
                    # some wgpu-native builds have a broken callback ABI for
                    # on_submitted_work_done_sync().
                    self.device.queue.read_buffer(counter, 0, 4)
                    resources.release(down_params)
                    octave_base = next_base
                else:
                    octave_base = None

                resources.release(counter)
                resources.release(keypoints)

                for texture in dogs:
                    resources.release(texture)
                for texture in gaussian:
                    if texture is not octave_base:
                        resources.release(texture)
                if not has_next:
                    break
                base_width //= 2
                base_height //= 2

            if octave_base is not None:
                resources.release(octave_base)
            resources.release(input_texture)
        finally:
            resources.close()

        total = sum(part.size // FEATURE_STRIDE for part in segments)
        data = (
            np.concatenate(segments).astype(np.float32, copy=False)
            if segments
            else np.empty(0, dtype=np.float32)
        )
        return {
            "data": data.reshape(total, FEATURE_STRIDE),
            "count": total,
            "detected": total_detected,
            "overflow": overflow,
        }

    @staticmethod
    def _select_features(
        features: np.ndarray, max_count: int, width: int, height: int
    ) -> np.ndarray:
        if not features.size:
            return np.empty((0, FEATURE_STRIDE), dtype=np.float32)
        valid = np.isfinite(features[:, :2]).all(axis=1)
        valid &= (features[:, 0] >= 0) & (features[:, 1] >= 0)
        valid &= (features[:, 0] < width) & (features[:, 1] < height)
        order = np.flatnonzero(valid)
        order = order[np.argsort(-features[order, 3], kind="stable")]
        limit = min(max_count, order.size)
        selected: list[int] = []
        used = np.zeros(features.shape[0], dtype=np.bool_)
        per_cell: dict[int, int] = {}
        cells_per_row = math.ceil(width / 128)
        for index in order:
            if len(selected) >= limit:
                break
            cell = int(features[index, 1] // 128) * cells_per_row + int(features[index, 0] // 128)
            hits = per_cell.get(cell, 0)
            if hits >= 4:
                continue
            selected.append(int(index))
            used[index] = True
            per_cell[cell] = hits + 1
        if len(selected) < limit:
            for index in order:
                if len(selected) >= limit:
                    break
                if not used[index]:
                    selected.append(int(index))
        return np.ascontiguousarray(features[selected], dtype=np.float32)

    def detect_keypoints(
        self,
        pixels: np.ndarray,
        *,
        max_keypoints: int = 4096,
        reduction: int = 4,
        on_progress: Callable[[str], None] | None = None,
    ) -> FeatureSet:
        """Expose HORG/SIFT locations for common-descriptor experiments.

        The existing HORG DoG detector and descriptor extraction stay on WGSL.
        This adapter returns floating-point locations and discards HORG's own
        descriptor only after its normal extraction stage has completed.
        """
        image = np.ascontiguousarray(pixels, dtype=np.uint8)
        if image.ndim != 2:
            raise ValueError("HORG detection expects a grayscale image")
        if max_keypoints < 1 or reduction < 1:
            raise ValueError("max_keypoints and reduction must be positive")
        height, width = image.shape
        callback = on_progress or (lambda _: None)
        callback("Extracting HORG/SIFT keypoints for common-descriptor evaluation")
        extracted = self._image_features(image, width, height, reduction, "HORG detector", callback)
        selected = self._select_features(extracted["data"], max_keypoints, width, height)
        return FeatureSet(
            xy=selected[:, :2],
            scores=selected[:, 3],
            scales=selected[:, 2],
            metadata={
                "detector": "HORG-SIFT-DoG",
                "source_size_px": [width, height],
                "reduction": reduction,
                "max_keypoints_requested": max_keypoints,
                "keypoints_returned": len(selected),
                "upstream_detected": extracted["detected"],
                "overflow": extracted["overflow"],
                "coordinate_convention": "source-image pixel centres; integer values identify pixel centres",
            },
        )

    def fit_local_correspondences(
        self,
        correspondences: np.ndarray,
        seed_points: np.ndarray,
        *,
        radius_px: float = 256.0,
        threshold_px: float = 3.0,
        max_iterations: int = 256,
        candidate_capacity: int = MAX_RANSAC_CANDIDATES,
        include_affine_models: bool = False,
    ) -> dict[str, Any]:
        """Fit local affine models to supplied ``(x0, y0, x1, y1)`` pairs.

        This reuses HORG's WGSL local-candidate and RANSAC kernels without
        running HORG detection or descriptor matching. It is intended for
        alternate correspondence sources such as XFeat.
        """
        pairs = np.ascontiguousarray(correspondences, dtype=np.float32)
        centers = np.ascontiguousarray(seed_points, dtype=np.float32)
        if pairs.ndim != 2 or pairs.shape[1] != 4 or pairs.shape[0] < 3:
            raise ValueError("correspondences must be an Nx4 array of (x0, y0, x1, y1) pairs")
        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] < 1:
            raise ValueError("seed_points must be a non-empty Nx2 array")
        if not np.isfinite(pairs).all() or not np.isfinite(centers).all():
            raise ValueError("correspondences and seed_points must contain only finite coordinates")
        if not math.isfinite(radius_px) or radius_px <= 0:
            raise ValueError("radius_px must be finite and positive")
        if not math.isfinite(threshold_px) or threshold_px <= 0:
            raise ValueError("threshold_px must be finite and positive")
        if max_iterations < 1 or not 1 <= candidate_capacity <= MAX_RANSAC_CANDIDATES:
            raise ValueError("RANSAC iterations and candidate_capacity are out of range")

        pair_count, seed_count = len(pairs), len(centers)
        result_stride = 16 if include_affine_models else 8
        max_binding_size = self.device.limits["max-storage-buffer-binding-size"]
        candidate_bytes = seed_count * candidate_capacity * np.dtype(np.uint32).itemsize
        if (
            max(pair_count * 16, seed_count * 16, candidate_bytes, seed_count * result_stride * 4)
            > max_binding_size
        ):
            raise ValueError("Local affine buffers exceed the adapter storage-buffer limit")
        seeds = np.zeros((seed_count, 4), dtype=np.float32)
        seeds[:, :2] = centers
        seeds[:, 3] = np.arange(seed_count, dtype=np.float32)

        resources = ResourceScope()
        try:
            prepare_module = self._shader("horg-prepare.wgsl")
            ransac_module = self._shader("horg-ransac.wgsl")
            build_candidates = make_pipeline(
                self.device,
                prepare_module,
                "build_local_candidates",
                "Local affine candidate lists",
            )
            ransac = make_pipeline(
                self.device, ransac_module, "ransac_local", "Local affine RANSAC"
            )
            pair_buffer = upload_buffer(
                self.device,
                resources,
                pairs,
                wgpu.BufferUsage.STORAGE,
                "Input correspondence pairs",
            )
            seed_buffer = upload_buffer(
                self.device, resources, seeds, wgpu.BufferUsage.STORAGE, "Local affine POI centers"
            )
            candidate_buffer = create_buffer(
                self.device,
                resources,
                candidate_bytes,
                wgpu.BufferUsage.STORAGE,
                "Local affine candidates",
            )
            result_buffer = create_buffer(
                self.device,
                resources,
                seed_count * result_stride * 4,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "Local affine RANSAC results",
            )
            candidate_params = self._ransac_uniform(
                resources, [seed_count, pair_count, candidate_capacity, 0], [radius_px, 0, 0, 0]
            )
            candidate_group = make_bind_group(
                self.device,
                build_candidates,
                [
                    self._buffer_entry(3, pair_buffer),
                    self._buffer_entry(5, seed_buffer),
                    self._buffer_entry(6, candidate_buffer),
                    self._buffer_entry(7, candidate_params),
                ],
            )
            ransac_params = self._ransac_uniform(
                resources,
                [seed_count, pair_count, candidate_capacity, max_iterations],
                [threshold_px, float(include_affine_models), 0, 0],
            )
            ransac_group = make_bind_group(
                self.device,
                ransac,
                [
                    self._buffer_entry(0, pair_buffer),
                    self._buffer_entry(1, seed_buffer),
                    self._buffer_entry(2, candidate_buffer),
                    self._buffer_entry(3, result_buffer),
                    self._buffer_entry(4, ransac_params),
                ],
            )

            start = time.perf_counter()
            encoder = self.device.create_command_encoder(label="Local affine correspondence RANSAC")
            dispatch(
                encoder,
                build_candidates,
                candidate_group,
                seed_count,
                label="Local affine GPU candidates",
            )
            dispatch(encoder, ransac, ransac_group, seed_count, label="Local affine GPU RANSAC")
            self.device.queue.submit([encoder.finish()])
            values = read_array(
                self.device, result_buffer, np.dtype(np.float32), seed_count * result_stride
            )
            compute_ms = (time.perf_counter() - start) * 1000.0

            matches = []
            for index, (x, y) in enumerate(centers):
                raw = values[index * result_stride : (index + 1) * result_stride]
                inliers = round(float(raw[2]))
                candidates = round(float(raw[3]))
                fraction = float(raw[4])
                residual = float(raw[5])
                fitted = bool(raw[6] > 0.5)
                matches.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "dx": float(raw[0]) if fitted else 0.0,
                        "dy": float(raw[1]) if fitted else 0.0,
                        "inliers": inliers if fitted else 0,
                        "candidates": candidates if fitted else 0,
                        "inlier_fraction": fraction if fitted else 0.0,
                        "residual": residual if fitted else math.nan,
                        "fitted": fitted,
                    }
                )
            return {
                "matches": matches,
                "gpu_compute_ms": compute_ms,
                "candidate_capacity": candidate_capacity,
                "affine_models": (
                    values.reshape(seed_count, result_stride)[:, 8:16].copy()
                    if include_affine_models
                    else None
                ),
            }
        finally:
            resources.close()

    def compute(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        *,
        patch_size: int = 128,
        stride: int = 512,
        reduction: int = 4,
        max_features: int = 1024,
        seed_points: tuple[tuple[float, float], ...] | None = None,
        include_correspondences: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        progress = on_progress or (lambda _: None)
        reference = np.ascontiguousarray(reference, dtype=np.uint8)
        target = np.ascontiguousarray(target, dtype=np.uint8)
        if reference.ndim != 2 or target.ndim != 2 or reference.shape != target.shape:
            raise ValueError("HORG expects same-sized two-dimensional grayscale uint8 images")
        height, width = reference.shape
        if min(patch_size, stride, reduction, max_features) < 1:
            raise ValueError("patch_size, stride, reduction and max_features must be positive")
        if not isinstance(include_correspondences, bool):
            raise TypeError("include_correspondences must be a boolean")
        extraction_start = time.perf_counter()
        reference_all = self._image_features(
            reference, width, height, reduction, "reference", progress
        )
        target_all = self._image_features(target, width, height, reduction, "target", progress)
        ref_features = self._select_features(reference_all["data"], max_features, width, height)
        tar_features = self._select_features(target_all["data"], max_features, width, height)
        extraction_ms = (time.perf_counter() - extraction_start) * 1000
        if ref_features.shape[0] < 1 or tar_features.shape[0] < 2:
            raise ValueError("HORG needs at least one reference and two target keypoints")
        return self._match_features(
            ref_features,
            tar_features,
            width,
            height,
            patch_size,
            stride,
            reduction,
            max_features,
            extraction_ms,
            reference_all["detected"],
            target_all["detected"],
            reference_all["overflow"] + target_all["overflow"],
            seed_points,
            include_correspondences,
        )

    def _match_features(
        self,
        ref_features: np.ndarray,
        tar_features: np.ndarray,
        width: int,
        height: int,
        patch_size: int,
        stride: int,
        reduction: int,
        max_features: int,
        extraction_ms: float,
        keypoints_reference: int,
        keypoints_target: int,
        keypoint_overflow: int,
        seed_points: tuple[tuple[float, float], ...] | None,
        include_correspondences: bool = False,
    ) -> dict[str, Any]:
        ratio_limit = 0.75
        ransac_threshold = 1.5
        min_inliers, min_candidates, min_fraction = 4, 8, 0.2
        max_iterations = 256
        resources = ResourceScope()
        try:
            matcher_module = self._shader("horg-match.wgsl")
            prepare_module = self._shader("horg-prepare.wgsl")
            ransac_module = self._shader("horg-ransac.wgsl")
            confidence_module = self._shader("horg-confidence.wgsl")
            nearest = make_pipeline(
                self.device, matcher_module, "nearest_two", "HORG nearest descriptors"
            )
            filter_ratio = make_pipeline(
                self.device, prepare_module, "filter_lowe_ratio", "HORG Lowe ratio"
            )
            build_candidates = make_pipeline(
                self.device, prepare_module, "build_local_candidates", "HORG candidate lists"
            )
            ransac = make_pipeline(self.device, ransac_module, "ransac_local", "HORG local RANSAC")
            confidence = make_pipeline(
                self.device, confidence_module, "local_affine_leave_one_out", "HORG local LOO"
            )

            ref_desc = np.ascontiguousarray(ref_features[:, 4:68], dtype=np.float32)
            tar_desc = np.ascontiguousarray(tar_features[:, 4:68], dtype=np.float32)
            ref_xy = np.ascontiguousarray(ref_features[:, :2], dtype=np.float32)
            tar_xy = np.ascontiguousarray(tar_features[:, :2], dtype=np.float32)
            ref_count, tar_count = ref_desc.shape[0], tar_desc.shape[0]
            ref_buffer = upload_buffer(
                self.device,
                resources,
                ref_desc,
                wgpu.BufferUsage.STORAGE,
                "HORG reference descriptors",
            )
            tar_buffer = upload_buffer(
                self.device,
                resources,
                tar_desc,
                wgpu.BufferUsage.STORAGE,
                "HORG target descriptors",
            )
            ref_xy_buffer = upload_buffer(
                self.device,
                resources,
                ref_xy,
                wgpu.BufferUsage.STORAGE,
                "HORG reference coordinates",
            )
            tar_xy_buffer = upload_buffer(
                self.device, resources, tar_xy, wgpu.BufferUsage.STORAGE, "HORG target coordinates"
            )
            knn_buffer = create_buffer(
                self.device,
                resources,
                ref_count * 16,
                wgpu.BufferUsage.STORAGE,
                "HORG top two neighbors",
            )
            ratio_matches_buffer = create_buffer(
                self.device,
                resources,
                ref_count * 16,
                wgpu.BufferUsage.STORAGE,
                "HORG filtered matches",
            )
            raw_match_buffer = create_buffer(
                self.device,
                resources,
                ref_count * 16,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "HORG direct descriptor proposals",
            )
            ratio_buffer = create_buffer(
                self.device,
                resources,
                ref_count * 4,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "HORG ratio diagnostics",
            )
            knn_params = make_params(
                self.device, resources, [ref_count, tar_count, 0, 0], "HORG kNN params", 4
            )
            knn_group = make_bind_group(
                self.device,
                nearest,
                [
                    self._buffer_entry(0, ref_buffer),
                    self._buffer_entry(1, tar_buffer),
                    self._buffer_entry(2, knn_buffer),
                    self._buffer_entry(3, knn_params),
                ],
            )
            ratio_params = self._ransac_uniform(
                resources, [ref_count, tar_count, 0, 0], [ratio_limit, 0, 0, 0]
            )
            ratio_group = make_bind_group(
                self.device,
                filter_ratio,
                [
                    self._buffer_entry(0, knn_buffer),
                    self._buffer_entry(1, ref_xy_buffer),
                    self._buffer_entry(2, tar_xy_buffer),
                    self._buffer_entry(3, ratio_matches_buffer),
                    self._buffer_entry(4, ratio_buffer),
                    self._buffer_entry(7, ratio_params),
                    self._buffer_entry(8, raw_match_buffer),
                ],
            )

            if seed_points is None:
                grid_width = 1 + (width - patch_size) // stride
                grid_height = 1 + (height - patch_size) // stride
                if grid_width < 1 or grid_height < 1:
                    raise ValueError("DIC window is larger than the image")
                seed_count = grid_width * grid_height
                seeds = np.zeros((seed_count, 4), dtype=np.float32)
                for gy in range(grid_height):
                    for gx in range(grid_width):
                        seed = gy * grid_width + gx
                        seeds[seed] = [
                            gx * stride + (patch_size - 1) / 2,
                            gy * stride + (patch_size - 1) / 2,
                            0,
                            seed,
                        ]
            else:
                points = np.asarray(seed_points, dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
                    raise ValueError(
                        "HORG seed_points must be a non-empty sequence of (x, y) pairs"
                    )
                if (
                    not np.isfinite(points).all()
                    or np.any(points < 0)
                    or np.any(points[:, 0] >= width)
                    or np.any(points[:, 1] >= height)
                ):
                    raise ValueError(
                        "HORG seed_points must contain finite coordinates inside the image"
                    )
                seed_count = points.shape[0]
                grid_width, grid_height = seed_count, 1
                seeds = np.zeros((seed_count, 4), dtype=np.float32)
                seeds[:, :2] = points
                default_grid_width = max(1, 1 + (width - patch_size) // stride)
                half = (patch_size - 1) / 2
                for index, (x, y) in enumerate(points):
                    gx, gy = round((float(x) - half) / stride), round((float(y) - half) / stride)
                    on_grid = (
                        0 <= gx < default_grid_width
                        and abs(float(x) - (gx * stride + half)) < 1e-3
                        and abs(float(y) - (gy * stride + half)) < 1e-3
                    )
                    # Keep the original per-POI RANSAC stream when a visible subset
                    # is selected from the default grid; arbitrary points use order.
                    seeds[index, 3] = gy * default_grid_width + gx if on_grid else index
            seed_buffer = upload_buffer(
                self.device, resources, seeds, wgpu.BufferUsage.STORAGE, "HORG POI centers"
            )
            candidate_buffer = create_buffer(
                self.device,
                resources,
                seed_count * MAX_RANSAC_CANDIDATES * 4,
                wgpu.BufferUsage.STORAGE,
                "HORG candidate ids",
            )
            candidate_params = self._ransac_uniform(
                resources, [seed_count, ref_count, MAX_RANSAC_CANDIDATES, 0], [2 * stride, 0, 0, 0]
            )
            candidate_group = make_bind_group(
                self.device,
                build_candidates,
                [
                    self._buffer_entry(3, ratio_matches_buffer),
                    self._buffer_entry(5, seed_buffer),
                    self._buffer_entry(6, candidate_buffer),
                    self._buffer_entry(7, candidate_params),
                ],
            )
            result_bytes = seed_count * 32
            result_buffer = create_buffer(
                self.device,
                resources,
                result_bytes,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "HORG affine results",
            )
            loo_buffer = create_buffer(
                self.device,
                resources,
                seed_count * 16,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "HORG local consistency",
            )
            ransac_params = self._ransac_uniform(
                resources,
                [seed_count, ref_count, MAX_RANSAC_CANDIDATES, max_iterations],
                [ransac_threshold * reduction, 0, 0, 0],
            )
            ransac_group = make_bind_group(
                self.device,
                ransac,
                [
                    self._buffer_entry(0, ratio_matches_buffer),
                    self._buffer_entry(1, seed_buffer),
                    self._buffer_entry(2, candidate_buffer),
                    self._buffer_entry(3, result_buffer),
                    self._buffer_entry(4, ransac_params),
                ],
            )
            confidence_params = self._ransac_uniform(
                resources,
                [seed_count, 0, 0, 0],
                [ransac_threshold, min_inliers, min_candidates, min_fraction],
            )
            confidence_group = make_bind_group(
                self.device,
                confidence,
                [
                    self._buffer_entry(0, seed_buffer),
                    self._buffer_entry(1, result_buffer),
                    self._buffer_entry(2, loo_buffer),
                    self._buffer_entry(3, confidence_params),
                ],
            )

            match_start = time.perf_counter()
            encoder = self.device.create_command_encoder(
                label="HORG GPU feature match and candidate setup"
            )
            dispatch(
                encoder, nearest, knn_group, ref_count, label="HORG exact k=2 descriptor search"
            )
            dispatch(
                encoder,
                filter_ratio,
                ratio_group,
                math.ceil(ref_count / 64),
                label="HORG Lowe ratio GPU",
            )
            dispatch(
                encoder, build_candidates, candidate_group, seed_count, label="HORG candidates GPU"
            )
            self.device.queue.submit([encoder.finish()])
            ratio_values = read_array(self.device, ratio_buffer, np.dtype(np.float32), ref_count)
            correspondence_readback_ms = 0.0
            raw_correspondences = None
            raw_correspondence_ids = None
            raw_match_scores = None
            raw_correspondence_accepted = None
            if include_correspondences:
                correspondence_start = time.perf_counter()
                raw_matches = read_array(
                    self.device, raw_match_buffer, np.dtype(np.float32), ref_count * 4
                ).reshape(ref_count, 4)
                measured = raw_matches[:, 0] >= 0
                raw_correspondences = raw_matches[measured].tolist()
                raw_correspondence_ids = np.flatnonzero(measured).tolist()
                raw_match_scores = ratio_values[measured].tolist()
                raw_correspondence_accepted = (ratio_values[measured] < ratio_limit).tolist()
                correspondence_readback_ms = (time.perf_counter() - correspondence_start) * 1000
            match_ms = (time.perf_counter() - match_start) * 1000
            valid_ratios = ratio_values[(ratio_values >= 0) & (ratio_values <= 1)]
            ratio_matches_count = int(np.count_nonzero(valid_ratios < ratio_limit))
            ratio_stats = {
                "min": float(np.min(valid_ratios)) if valid_ratios.size else math.nan,
                "p25": float(np.quantile(valid_ratios, 0.25, method="lower"))
                if valid_ratios.size
                else math.nan,
                "median": float(np.quantile(valid_ratios, 0.5, method="lower"))
                if valid_ratios.size
                else math.nan,
                "p90": float(np.quantile(valid_ratios, 0.9, method="lower"))
                if valid_ratios.size
                else math.nan,
            }

            ransac_start = time.perf_counter()
            encoder = self.device.create_command_encoder(label="HORG local RANSAC and LOO")
            dispatch(encoder, ransac, ransac_group, seed_count, label="HORG deterministic RANSAC")
            dispatch(
                encoder, confidence, confidence_group, seed_count, label="HORG local LOO diagnostic"
            )
            self.device.queue.submit([encoder.finish()])
            ransac_values = read_array(
                self.device, result_buffer, np.dtype(np.float32), seed_count * 8
            )
            loo_values = read_array(
                self.device, loo_buffer, np.dtype(np.float32), seed_count * 4
            ).reshape(seed_count, 4)
            ransac_ms = (time.perf_counter() - ransac_start) * 1000

            matches = []
            for seed in range(seed_count):
                raw = ransac_values[seed * 8 : seed * 8 + 8]
                inliers = round(float(raw[2]))
                candidates = round(float(raw[3]))
                fraction = float(raw[4])
                residual = float(raw[5])
                fit = raw[6] > 0.5
                usable = bool(
                    fit
                    and inliers >= min_inliers
                    and candidates >= min_candidates
                    and fraction >= min_fraction
                    and residual <= ransac_threshold
                )
                consistency = float(loo_values[seed, 0])
                matches.append(
                    {
                        "x": float(seeds[seed, 0]),
                        "y": float(seeds[seed, 1]),
                        "dx": float(raw[0]) if fit else 0.0,
                        "dy": float(raw[1]) if fit else 0.0,
                        "score": fraction if fit else 0.0,
                        "second_score": residual if fit else math.nan,
                        "margin": fraction if fit else 0.0,
                        "usable": usable,
                        "inliers": inliers if fit else 0,
                        "candidates": candidates if fit else 0,
                        "inlier_fraction": fraction if fit else 0.0,
                        "residual": residual if fit else math.nan,
                        "local_consistency_px": consistency if consistency >= 0 else math.nan,
                        "local_consistency_neighbors": round(float(loo_values[seed, 1])),
                    }
                )
            accepted = sum(match["usable"] for match in matches)
            result = {
                "algorithm": "HORG-wgpu-py",
                "width": width,
                "height": height,
                "patch_size": patch_size,
                "stride": stride,
                "scale": reduction,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "count": len(matches),
                "matches": matches,
                "keypoints_reference": int(keypoints_reference),
                "keypoints_target": int(keypoints_target),
                "keypoint_overflow": int(keypoint_overflow),
                "descriptors_reference": int(ref_count),
                "descriptors_target": int(tar_count),
                "ratio_matches": ratio_matches_count,
                "ratio_stats": ratio_stats,
                "accepted_initializations": accepted,
                "extraction_ms": extraction_ms,
                "match_compute_ms": match_ms,
                "ransac_gpu_ms": ransac_ms,
                "ransac_threshold": ransac_threshold * reduction,
                "ratio_limit": ratio_limit,
                "max_features": max_features,
                "manual_seeds": seed_points is not None,
            }
            if include_correspondences:
                result.update(
                    {
                        "raw_correspondences_xyxy": raw_correspondences,
                        "raw_correspondence_ids": raw_correspondence_ids,
                        "raw_match_scores": raw_match_scores,
                        "raw_correspondence_accepted": raw_correspondence_accepted,
                        "raw_correspondence_rejection_reasons": [
                            None if accepted else "lowe_ratio_rejected"
                            for accepted in raw_correspondence_accepted
                        ],
                        "raw_correspondence_readback_ms": correspondence_readback_ms,
                        "raw_correspondence_kind": "direct_HORG_ratio_matches",
                    }
                )
            return result
        finally:
            resources.close()

    def _ransac_uniform(
        self, scope: ResourceScope, counts: list[int], settings: list[float]
    ) -> Any:
        raw = bytearray(32)
        np.frombuffer(raw, dtype=np.uint32, count=4, offset=0)[:] = counts
        np.frombuffer(raw, dtype=np.float32, count=4, offset=16)[:] = settings
        return uniform_buffer(self.device, scope, bytes(raw), "HORG compute parameters")
