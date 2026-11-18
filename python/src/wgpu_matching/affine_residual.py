"""WebGPU diagnostic photometric scores for final affine DIC states."""

import struct

import numpy as np
import wgpu

from ._gpu import ResourceScope, shader_source, upload_buffer


class AffineResidualWGPU:
    """Evaluate global and quadrant normalized correlation, without classifying windows.

    Input images are uploaded from the host and the compact scores are read
    back. This diagnostic is not a zero-copy solver integration.
    """

    def __init__(self, device):
        self.device = device
        module = device.create_shader_module(code=shader_source("dic-affine-residual.wgsl"))
        kinds = (
            "read-only-storage",
            "read-only-storage",
            "read-only-storage",
            "storage",
            "uniform",
        )
        entries = [
            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
            for i, kind in enumerate(kinds)
        ]
        self.layout = device.create_bind_group_layout(entries=entries)
        layout = device.create_pipeline_layout(bind_group_layouts=[self.layout])
        self.pipeline = device.create_compute_pipeline(
            layout=layout, compute={"module": module, "entry_point": "main"}
        )

    def evaluate(self, reference, target, parameters, accepted, *, subset=64, stride=64):
        reference = np.asarray(reference)
        target = np.asarray(target)
        if (
            reference.ndim != 2
            or reference.shape != target.shape
            or reference.dtype != np.uint8
            or target.dtype != np.uint8
        ):
            raise ValueError("expected matching two-dimensional uint8 images")
        height, width = reference.shape
        if subset < 2 or subset % 2 or stride < 1 or min(width, height) < subset:
            raise ValueError("subset must be even and fit the images; stride must be positive")
        gw = 1 + (width - subset) // stride
        gh = 1 + (height - subset) // stride
        count = gw * gh
        parameters = np.asarray(parameters, dtype=np.float32).reshape(-1, 6)
        accepted = np.asarray(accepted, dtype=bool).reshape(-1)
        if (
            len(parameters) != count
            or len(accepted) != count
            or not np.isfinite(parameters[accepted]).all()
        ):
            raise ValueError("parameters and acceptance mask must match the regular grid")
        states = np.zeros((count, 8), np.float32)
        states[accepted, :6] = parameters[accepted]
        states[:, 6] = accepted
        device = self.device
        with ResourceScope() as scope:
            refs = upload_buffer(
                device,
                scope,
                reference.astype(np.float32) / 255,
                wgpu.BufferUsage.STORAGE,
                "residual reference",
            )
            tgts = upload_buffer(
                device,
                scope,
                target.astype(np.float32) / 255,
                wgpu.BufferUsage.STORAGE,
                "residual target",
            )
            state = upload_buffer(
                device, scope, states, wgpu.BufferUsage.STORAGE, "residual states"
            )
            output = scope.keep(
                device.create_buffer(
                    size=count * 32, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                )
            )
            config = upload_buffer(
                device,
                scope,
                np.frombuffer(
                    struct.pack("<8I", width, height, gw, count, subset, stride, 0, 0), np.uint8
                ),
                wgpu.BufferUsage.UNIFORM,
                "residual config",
            )
            buffers = (refs, tgts, state, output, config)
            group = device.create_bind_group(
                layout=self.layout,
                entries=[
                    {"binding": i, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                    for i, buf in enumerate(buffers)
                ],
            )
            encoder = device.create_command_encoder(label="DIC affine residual")
            phase = encoder.begin_compute_pass()
            phase.set_pipeline(self.pipeline)
            phase.set_bind_group(0, group)
            phase.dispatch_workgroups(count)
            phase.end()
            device.queue.submit([encoder.finish()])
            raw = (
                np.frombuffer(device.queue.read_buffer(output), np.float32).reshape(count, 8).copy()
            )
        valid = accepted & (raw[:, 5] > 0.5)
        correlations = np.column_stack((raw[:, 0], raw[:, 1:4], raw[:, 4]))
        correlations[~valid] = np.nan
        return {
            "correlations": correlations,
            "valid": valid,
            "readback_bytes": count * 32,
            "uploaded_image_bytes": reference.size * 8,
        }
