"""Compact descriptor weight loading and WGSL projection support."""

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
    dispatch,
    make_bind_group,
    make_pipeline,
    read_array,
    shader_source,
    upload_buffer,
)


@dataclass(slots=True)
class CompactDescriptorWeights:
    """Validated NumPy weights exported by the optional training workflow."""

    metadata: dict[str, Any]
    first_weight: np.ndarray
    first_bias: np.ndarray
    second_weight: np.ndarray
    second_bias: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> CompactDescriptorWeights:
        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(bytes(archive["metadata_json"].tolist()).decode("utf-8"))
            first_weight = np.asarray(archive["first_weight"], dtype=np.float32)
            first_bias = np.asarray(archive["first_bias"], dtype=np.float32)
            second_weight = np.asarray(archive["second_weight"], dtype=np.float32)
            second_bias = np.asarray(archive["second_bias"], dtype=np.float32)
        input_dim = int(metadata["input_dimension"])
        output_dim = int(metadata["output_dimension"])
        hidden_dim = int(metadata["hidden_dimension"])
        mode = str(metadata["architecture"])
        if output_dim != 64 or mode not in {"linear", "residual_mlp", "multiscale_mlp"}:
            raise ValueError("Unsupported compact descriptor architecture or output dimension")
        if first_weight.shape != (hidden_dim or output_dim, input_dim):
            raise ValueError("First projection weight shape does not match metadata")
        if first_bias.shape != (hidden_dim or output_dim,):
            raise ValueError("First projection bias shape does not match metadata")
        if hidden_dim:
            if second_weight.shape != (output_dim, hidden_dim) or second_bias.shape != (output_dim,):
                raise ValueError("Second projection weight shape does not match metadata")
        elif second_weight.size or second_bias.size:
            raise ValueError("Linear descriptor must not contain a second projection")
        arrays = (first_weight, first_bias, second_weight, second_bias)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Descriptor weights must be finite")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata["weights_sha256"] = digest
        return cls(metadata, *arrays)

    def packed_weights(self) -> np.ndarray:
        return np.concatenate(
            (self.first_weight.ravel(), self.first_bias, self.second_weight.ravel(), self.second_bias)
        ).astype(np.float32, copy=False)


class CompactDescriptorBackend:
    """Project HORG or multiscale HORG features with a compact WGSL model."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self.pipeline = None

    def project(self, descriptors: np.ndarray, weights: CompactDescriptorWeights) -> np.ndarray:
        values = np.ascontiguousarray(descriptors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != weights.metadata["input_dimension"]:
            raise ValueError("Input descriptors do not match the trained projection dimension")
        if not np.isfinite(values).all():
            raise ValueError("Input descriptors must be finite")
        if len(values) == 0:
            return np.empty((0, 64), dtype=np.float32)
        hidden_dim = int(weights.metadata["hidden_dimension"])
        architecture = str(weights.metadata["architecture"])
        mode = {"linear": 0, "residual_mlp": 1, "multiscale_mlp": 2}[architecture]
        if self.pipeline is None:
            module = self.device.create_shader_module(code=shader_source("dic-descriptor-project.wgsl"))
            self.pipeline = make_pipeline(
                self.device, module, "project", "Compact DIC descriptor projection"
            )
        parameter_values = np.asarray(
            [len(values), values.shape[1], hidden_dim, mode, 64, 0, 0, 0], dtype=np.uint32
        )
        scope = ResourceScope()
        try:
            inputs = upload_buffer(
                self.device, scope, values, wgpu.BufferUsage.STORAGE, "Descriptor input features"
            )
            packed = upload_buffer(
                self.device,
                scope,
                weights.packed_weights(),
                wgpu.BufferUsage.STORAGE,
                "Compact descriptor weights",
            )
            output = create_buffer(
                self.device,
                scope,
                len(values) * 64 * 4,
                wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                "Compact DIC descriptors",
            )
            params = create_buffer(
                self.device,
                scope,
                32,
                wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                "Compact descriptor parameters",
            )
            self.device.queue.write_buffer(params, 0, parameter_values)
            group = make_bind_group(
                self.device,
                self.pipeline,
                [
                    {"binding": 0, "resource": {"buffer": inputs}},
                    {"binding": 1, "resource": {"buffer": packed}},
                    {"binding": 2, "resource": {"buffer": output}},
                    {"binding": 3, "resource": {"buffer": params}},
                ],
            )
            encoder = self.device.create_command_encoder(label="Compact DIC descriptor projection")
            dispatch(encoder, self.pipeline, group, len(values), label="Compact DIC descriptor projection")
            self.device.queue.submit([encoder.finish()])
            return read_array(self.device, output, np.dtype(np.float32), len(values) * 64).reshape(
                len(values), 64
            )
        finally:
            scope.close()
