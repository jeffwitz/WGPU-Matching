"""Small patch-convolution descriptor executed with native WGSL kernels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    ) -> DescribedFeatures:
        pixels = np.ascontiguousarray(image, dtype=np.float32)
        if pixels.ndim != 2 or not np.isfinite(pixels).all():
            raise ValueError("DIC convolutional descriptor expects a finite grayscale image")
        height, width = pixels.shape
        count = len(features.xy)
        if count == 0:
            return DescribedFeatures(features, np.empty((0, 64), dtype=np.float32))
        if max(width, height) > 8192:
            raise ValueError("Current convolutional descriptor supports images up to 8192 pixels")
        limits = self.device.limits
        max_binding_size = int(
            limits.get("max-storage-buffer-binding-size", limits.get("max-buffer-size", 0))
        )
        feature_buffer_size = count * 2048 * np.dtype(np.float32).itemsize
        if pixels.nbytes > max_binding_size or feature_buffer_size > max_binding_size:
            raise ValueError(
                "Image or point-feature buffer exceeds this adapter's storage-buffer limit"
            )
        max_workgroups = int(limits.get("max-compute-workgroups-per-dimension", 65535))
        if count > max_workgroups:
            raise ValueError("Keypoint count exceeds this adapter's two-dimensional dispatch limit")
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
            point_buffer = upload_buffer(
                self.device,
                scope,
                features.xy.ravel(),
                wgpu.BufferUsage.STORAGE,
                "DIC descriptor keypoints",
            )
            weight_buffer = upload_buffer(
                self.device,
                scope,
                weights.packed_weights(),
                wgpu.BufferUsage.STORAGE,
                "DIC convolutional descriptor weights",
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
                32,
                wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                "DIC convolutional descriptor parameters",
            )
            self.device.queue.write_buffer(
                params,
                0,
                np.asarray([width, height, count, 64, 0, 0, 0, 0], dtype=np.uint32),
            )
            all_entries = [
                {"binding": index, "resource": {"buffer": buffer}}
                for index, buffer in enumerate(
                    (image_buffer, point_buffer, weight_buffer, convolution, output, params)
                )
            ]
            first_group = make_bind_group(
                self.device, self.first_pipeline, [all_entries[index] for index in (0, 1, 2, 3, 5)]
            )
            pool_group = make_bind_group(
                self.device, self.pool_pipeline, [all_entries[index] for index in (1, 2, 3, 4, 5)]
            )
            encoder = self.device.create_command_encoder(label="DIC convolutional descriptor")
            first_pass = encoder.begin_compute_pass(label="Patch convolution feature extraction")
            first_pass.set_pipeline(self.first_pipeline)
            first_pass.set_bind_group(0, first_group)
            first_pass.dispatch_workgroups(32, count, 1)
            first_pass.end()
            second_pass = encoder.begin_compute_pass(
                label="Patch pooling and descriptor projection"
            )
            second_pass.set_pipeline(self.pool_pipeline)
            second_pass.set_bind_group(0, pool_group)
            second_pass.dispatch_workgroups(count, 1, 1)
            second_pass.end()
            self.device.queue.submit([encoder.finish()])
            descriptors = read_array(self.device, output, np.dtype(np.float32), count * 64).reshape(
                count, 64
            )
        finally:
            scope.close()

        support = int(weights.metadata["support_px"])
        margin = support // 2
        valid = (
            features.valid
            & (features.xy[:, 0] >= margin - 1)
            & (features.xy[:, 1] >= margin - 1)
            & (features.xy[:, 0] + margin < width)
            & (features.xy[:, 1] + margin < height)
        )
        result_features = FeatureSet(
            features.xy,
            features.scores,
            valid=valid,
            scales=features.scales,
            orientations=features.orientations,
            metadata=features.metadata,
        )
        return DescribedFeatures(
            result_features,
            descriptors,
            {
                "descriptor": "DIC-Descriptor patch CNN WGSL",
                "dimension": 64,
                "support_px": support,
                "weights_sha256": weights.metadata["weights_sha256"],
                "execution": "native grayscale image and descriptor kernels on WebGPU; output copied to host",
            },
        )
