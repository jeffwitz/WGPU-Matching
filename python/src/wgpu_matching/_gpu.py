"""wgpu-py resource helpers shared by the native matcher implementations."""

from __future__ import annotations

from contextlib import AbstractContextManager
from importlib.resources import files
from typing import Any

import numpy as np
import wgpu


class ResourceScope(AbstractContextManager["ResourceScope"]):
    def __init__(self) -> None:
        self.items: list[Any] = []

    def keep(self, resource: Any) -> Any:
        self.items.append(resource)
        return resource

    def release(self, resource: Any) -> None:
        try:
            self.items.remove(resource)
        except ValueError:
            return
        resource.destroy()

    def close(self) -> None:
        while self.items:
            resource = self.items.pop()
            try:
                resource.destroy()
            except Exception:  # noqa: BLE001, S110 - continue releasing remaining GPU resources
                pass

    def __exit__(self, *_: object) -> None:
        self.close()


def shader_source(name: str) -> str:
    return files("wgpu_matching").joinpath("shaders", name).read_text(encoding="utf-8")


def create_buffer(device: Any, scope: ResourceScope, size: int, usage: int, label: str) -> Any:
    if size <= 0:
        size = 4
    return scope.keep(device.create_buffer(size=size, usage=usage, label=label))


def upload_buffer(device: Any, scope: ResourceScope, data: np.ndarray, usage: int, label: str) -> Any:
    array = np.ascontiguousarray(data)
    return scope.keep(device.create_buffer_with_data(data=array, usage=usage, label=label))


def uniform_buffer(device: Any, scope: ResourceScope, data: bytes | np.ndarray, label: str) -> Any:
    array = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else np.ascontiguousarray(data)
    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    buffer = create_buffer(device, scope, max(4, array.nbytes), usage, label)
    device.queue.write_buffer(buffer, 0, array)
    return buffer


def create_texture(
    device: Any,
    scope: ResourceScope,
    width: int,
    height: int,
    format: str,
    usage: int,
    label: str,
) -> Any:
    return scope.keep(
        device.create_texture(
            label=label,
            size=(width, height, 1),
            mip_level_count=1,
            sample_count=1,
            dimension="2d",
            format=format,
            usage=usage,
        )
    )


def make_pipeline(device: Any, module: Any, entry_point: str, label: str) -> Any:
    return device.create_compute_pipeline(
        label=label,
        layout="auto",
        compute={"module": module, "entry_point": entry_point},
    )


def make_bind_group(device: Any, pipeline: Any, entries: list[dict[str, Any]]) -> Any:
    return device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)


def dispatch(encoder: Any, pipeline: Any, group: Any, x: int, y: int = 1, z: int = 1, label: str = "") -> None:
    compute = encoder.begin_compute_pass(label=label or None)
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, group)
    compute.dispatch_workgroups(x, y, z)
    compute.end()


def read_array(device: Any, buffer: Any, dtype: np.dtype[Any], count: int) -> np.ndarray:
    return np.frombuffer(device.queue.read_buffer(buffer, 0, count * np.dtype(dtype).itemsize), dtype=dtype).copy()


def make_params(device: Any, scope: ResourceScope, words: list[int], label: str, length: int = 16) -> Any:
    values = np.zeros(length, dtype=np.uint32)
    count = min(length, len(words))
    if count:
        values[:count] = np.asarray([int(value) & 0xFFFFFFFF for value in words[:count]], dtype=np.uint32)
    return uniform_buffer(device, scope, values, label)
