"""Small patch-convolution descriptor executed with native WGSL kernels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import wgpu

from ._gpu import (
    ResourceScope,
    create_buffer,
    make_bind_group,
    make_pipeline,
    read_array,
    shader_source,
    upload_buffer,
)
from .features import DescribedFeatures, FeatureSet


@dataclass(slots=True)
class DICConvDescriptorWeights:
    """Validated convolution and projection weights exported by the trainer."""

    metadata: dict[str, Any]
    conv1_weight: np.ndarray
    conv1_bias: np.ndarray
    depthwise_weight: np.ndarray
    depthwise_bias: np.ndarray
    projection_weight: np.ndarray
    projection_bias: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> DICConvDescriptorWeights:
        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(bytes(archive["metadata_json"].tolist()).decode("utf-8"))
            arrays = [
                np.asarray(archive[name], dtype=np.float32)
                for name in (
                    "conv1_weight",
                    "conv1_bias",
                    "depthwise_weight",
                    "depthwise_bias",
                    "projection_weight",
                    "projection_bias",
                )
            ]
        if metadata.get("architecture") != "dic_patch_cnn_v1" or metadata.get("support_px") != 64:
            raise ValueError("Unsupported DIC convolutional descriptor architecture")
        expected_shapes = ((8, 1, 3, 3), (8,), (8, 1, 3, 3), (8,), (64, 128), (64,))
        if tuple(array.shape for array in arrays) != expected_shapes:
            raise ValueError("Convolutional descriptor weights have invalid shapes")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Convolutional descriptor weights must be finite")
        metadata["weights_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        return cls(metadata, *arrays)

    def packed_weights(self) -> np.ndarray:
        return np.concatenate(
            (
                self.conv1_weight.ravel(),
                self.conv1_bias,
                self.depthwise_weight.ravel(),
                self.depthwise_bias,
                self.projection_weight.ravel(),
                self.projection_bias,
            )
        ).astype(np.float32, copy=False)


class DICConvDescriptorImage:
    """Resident image and model buffers shared by multiple keypoint sets."""

    def __init__(
        self,
        device: Any,
        weights: DICConvDescriptorWeights,
        width: int,
        height: int,
        scope: ResourceScope,
        image_buffer: Any,
        weight_buffer: Any,
        first_pipeline: Any,
        pool_pipeline: Any,
        weights_path: str | None,
    ) -> None:
        self.device = device
        self.weights = weights
        self.width = width
        self.height = height
        self.scope = scope
        self.first_pipeline = first_pipeline
        self.pool_pipeline = pool_pipeline
        self.weights_path = weights_path
        self.closed = False
        self.active_sessions = 0
        self.image_buffer = image_buffer
        self.weight_buffer = weight_buffer

    def prepare(self, features: FeatureSet, *, owns_image: bool = False) -> DICConvDescriptorSession:
        """Create per-point work buffers while sharing this image and model."""
        if self.closed:
            raise RuntimeError("Prepared DIC descriptor image is closed")
        count = len(features.xy)
        if count == 0:
            raise ValueError("A reusable DIC descriptor session requires at least one keypoint")
        max_workgroups = int(
            self.device.limits.get("max-compute-workgroups-per-dimension", 65535)
        )
        if count > max_workgroups:
            raise ValueError("Keypoint count exceeds this adapter's two-dimensional dispatch limit")
        feature_buffer_size = count * 2048 * np.dtype(np.float32).itemsize
        max_binding_size = int(
            self.device.limits.get(
                "max-storage-buffer-binding-size", self.device.limits.get("max-buffer-size", 0)
            )
        )
        if feature_buffer_size > max_binding_size:
            raise ValueError("Point-feature buffer exceeds this adapter's storage-buffer limit")
        scope = ResourceScope()
        try:
            point_buffer = upload_buffer(
                self.device,
                scope,
                features.xy.ravel(),
                wgpu.BufferUsage.STORAGE,
                "DIC descriptor keypoints",
            )
            convolution = create_buffer(
                self.device,
                scope,
                feature_buffer_size,
                wgpu.BufferUsage.STORAGE,
                "DIC descriptor convolution features",
            )
            output = create_buffer(
                self.device,
                scope,
                count * 64 * 4,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "DIC convolutional descriptors",
            )
            params = create_buffer(
                self.device,
                scope,
                48,
                wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                "DIC convolutional descriptor parameters",
            )
            validity = create_buffer(
                self.device,
                scope,
                count * 4,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                "DIC descriptor support validity",
            )
            packed_params = np.zeros(12, dtype=np.uint32)
            packed_params[:4] = [self.width, self.height, count, 64]
            packed_params[4:8].view(np.float32)[:] = np.eye(2, dtype=np.float32).ravel()
            self.device.queue.write_buffer(params, 0, packed_params)
            session = DICConvDescriptorSession(
                self,
                features,
                scope,
                (point_buffer, convolution, output, params, validity),
                owns_image=owns_image,
            )
            self.active_sessions += 1
            return session
        except Exception:
            scope.close()
            raise

    def close(self) -> None:
        """Release image/model buffers after all dependent sessions are closed."""
        if self.closed:
            return
        if self.active_sessions:
            raise RuntimeError("Close all DIC descriptor sessions before their prepared image")
        self.scope.close()
        self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Prepared DIC descriptor image is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DICConvDescriptorSession:
    """Reusable GPU keypoint and workspace buffers for many support matrices."""

    def __init__(
        self,
        image: DICConvDescriptorImage,
        features: FeatureSet,
        scope: ResourceScope,
        buffers: tuple[Any, Any, Any, Any, Any],
        *,
        owns_image: bool = False,
    ) -> None:
        self.image = image
        self.device = image.device
        self.features = features
        self.weights = image.weights
        self.width = image.width
        self.height = image.height
        self.scope = scope
        self.point_buffer, self.convolution, self.output, self.params, self.validity = buffers
        self.first_pipeline = image.first_pipeline
        self.pool_pipeline = image.pool_pipeline
        self.image_buffer = image.image_buffer
        self.weight_buffer = image.weight_buffer
        self.weights_path = image.weights_path
        self.owns_image = owns_image
        self.closed = False
        self.described_features: FeatureSet | None = None
        self.first_group = make_bind_group(
            self.device,
            self.first_pipeline,
            [
                {"binding": 0, "resource": {"buffer": self.image_buffer}},
                {"binding": 1, "resource": {"buffer": self.point_buffer}},
                {"binding": 2, "resource": {"buffer": self.weight_buffer}},
                {"binding": 3, "resource": {"buffer": self.convolution}},
                {"binding": 5, "resource": {"buffer": self.params}},
            ],
        )
        self.pool_group = make_bind_group(
            self.device,
            self.pool_pipeline,
            [
                {"binding": 1, "resource": {"buffer": self.point_buffer}},
                {"binding": 2, "resource": {"buffer": self.weight_buffer}},
                {"binding": 3, "resource": {"buffer": self.convolution}},
                {"binding": 4, "resource": {"buffer": self.output}},
                {"binding": 5, "resource": {"buffer": self.params}},
            ],
        )

    def describe(self, support_jacobian: np.ndarray | None = None) -> DescribedFeatures:
        """Evaluate one support transform while reusing all resident buffers."""
        described_features = self.compute_resident(support_jacobian)
        descriptors = read_array(
            self.device, self.output, np.dtype(np.float32), len(self.features.xy) * 64
        ).reshape(len(self.features.xy), 64)
        return DescribedFeatures(
            described_features,
            descriptors,
            {
                "descriptor": "DIC-Descriptor patch CNN WGSL",
                "dimension": 64,
                "support_px": int(self.weights.metadata["support_px"]),
                "support_jacobian": self.support_jacobian.tolist(),
                "weights_sha256": self.weights.metadata["weights_sha256"],
                "execution": "resident image, points, weights, and workspace on WebGPU; output copied to host",
                "weights_path": self.weights_path,
            },
        )

    def compute_resident(self, support_jacobian: np.ndarray | None = None) -> FeatureSet:
        """Run inference without reading descriptor vectors back to the host."""
        if self.closed:
            raise RuntimeError("DIC descriptor session is closed")
        jacobian = _validated_jacobian(support_jacobian)
        self.device.queue.write_buffer(self.params, 16, jacobian.ravel())
        encoder = self.device.create_command_encoder(label="DIC convolutional descriptor")
        first_pass = encoder.begin_compute_pass(label="Patch convolution feature extraction")
        first_pass.set_pipeline(self.first_pipeline)
        first_pass.set_bind_group(0, self.first_group)
        first_pass.dispatch_workgroups(32, len(self.features.xy), 1)
        first_pass.end()
        second_pass = encoder.begin_compute_pass(label="Patch pooling and descriptor projection")
        second_pass.set_pipeline(self.pool_pipeline)
        second_pass.set_bind_group(0, self.pool_group)
        second_pass.dispatch_workgroups(len(self.features.xy), 1, 1)
        second_pass.end()
        self.device.queue.submit([encoder.finish()])

        support = int(self.weights.metadata["support_px"])
        half_support = support / 2.0
        extent_x = half_support * (abs(float(jacobian[0, 0])) + abs(float(jacobian[0, 1])))
        extent_y = half_support * (abs(float(jacobian[1, 0])) + abs(float(jacobian[1, 1])))
        xy = self.features.xy
        valid = (
            self.features.valid
            & (xy[:, 0] >= extent_x)
            & (xy[:, 1] >= extent_y)
            & (xy[:, 0] + extent_x < self.width)
            & (xy[:, 1] + extent_y < self.height)
        )
        described_features = FeatureSet(
            xy,
            self.features.scores,
            valid=valid,
            scales=self.features.scales,
            orientations=self.features.orientations,
            metadata=self.features.metadata,
        )
        self.device.queue.write_buffer(
            self.validity, 0, valid.astype(np.uint32, copy=False)
        )
        self.described_features = described_features
        self.support_jacobian = jacobian
        return described_features

    def close(self) -> None:
        """Release resident buffers; safe to call repeatedly."""
        if not self.closed:
            self.scope.close()
            self.closed = True
            self.image.active_sessions -= 1
            if self.owns_image:
                self.image.close()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("DIC descriptor session is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validated_jacobian(support_jacobian: np.ndarray | None) -> np.ndarray:
    jacobian = (
        np.eye(2, dtype=np.float32)
        if support_jacobian is None
        else np.asarray(support_jacobian, dtype=np.float32)
    )
    if jacobian.shape != (2, 2) or not np.isfinite(jacobian).all():
        raise ValueError("support_jacobian must be a finite 2x2 matrix")
    if float(np.linalg.det(jacobian)) <= 1e-6:
        raise ValueError("support_jacobian must preserve orientation and be invertible")
    return np.ascontiguousarray(jacobian)


class DICConvDescriptorBackend:
    """Run shared patch convolutions and descriptor projection on WebGPU."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self.first_pipeline = None
        self.pool_pipeline = None

    def describe(
        self,
        image: np.ndarray,
        features: FeatureSet,
        weights: DICConvDescriptorWeights,
        support_jacobian: np.ndarray | None = None,
    ) -> DescribedFeatures:
        jacobian = _validated_jacobian(support_jacobian)
        if len(features.xy) == 0:
            return DescribedFeatures(
                features,
                np.empty((0, 64), dtype=np.float32),
                {
                    "descriptor": "DIC-Descriptor patch CNN WGSL",
                    "dimension": 64,
                    "support_px": int(weights.metadata["support_px"]),
                    "support_jacobian": jacobian.tolist(),
                    "weights_sha256": weights.metadata["weights_sha256"],
                },
            )
        with self.prepare(image, features, weights) as session:
            return session.describe(jacobian)

    def prepare(
        self,
        image: np.ndarray,
        features: FeatureSet,
        weights: DICConvDescriptorWeights,
        *,
        weights_path: str | None = None,
    ) -> DICConvDescriptorSession:
        """Upload invariant data once and prepare a reusable support session."""
        prepared = self.prepare_image(image, weights, weights_path=weights_path)
        try:
            return prepared.prepare(features, owns_image=True)
        except Exception:
            prepared.close()
            raise

    def prepare_image(
        self,
        image: np.ndarray,
        weights: DICConvDescriptorWeights,
        *,
        weights_path: str | None = None,
    ) -> DICConvDescriptorImage:
        """Upload image and weights once for sequential coarse and full point sets."""
        pixels = np.ascontiguousarray(image, dtype=np.float32)
        if pixels.ndim != 2 or not np.isfinite(pixels).all():
            raise ValueError("DIC convolutional descriptor expects a finite grayscale image")
        height, width = pixels.shape
        if max(width, height) > 8192:
            raise ValueError("Current convolutional descriptor supports images up to 8192 pixels")
        limits = self.device.limits
        max_binding_size = int(
            limits.get("max-storage-buffer-binding-size", limits.get("max-buffer-size", 0))
        )
        if pixels.nbytes > max_binding_size:
            raise ValueError(
                "Image buffer exceeds this adapter's storage-buffer limit"
            )
        if self.first_pipeline is None:
            module = self.device.create_shader_module(code=shader_source("dic-cnn-descriptor.wgsl"))
            self.first_pipeline = make_pipeline(
                self.device, module, "first_convolution", "DIC descriptor first convolution"
            )
            self.pool_pipeline = make_pipeline(
                self.device, module, "pool_project", "DIC descriptor pooling and projection"
            )
        scope = ResourceScope()
        try:
            image_buffer = upload_buffer(
                self.device,
                scope,
                pixels.ravel(),
                wgpu.BufferUsage.STORAGE,
                "DIC descriptor grayscale image",
            )
            weight_buffer = upload_buffer(
                self.device,
                scope,
                weights.packed_weights(),
                wgpu.BufferUsage.STORAGE,
                "DIC convolutional descriptor weights",
            )
            return DICConvDescriptorImage(
                self.device,
                weights,
                width,
                height,
                scope,
                image_buffer,
                weight_buffer,
                self.first_pipeline,
                self.pool_pipeline,
                weights_path,
            )
        except Exception:
            scope.close()
            raise
