"""Public synchronous API for native WebGPU matchers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import wgpu

from .horg import HORGBackend
from .images import load_gray


@dataclass(frozen=True, slots=True)
class HORGOptions:
    """Parameters for HORG keypoint matching and local affine fitting.

    ``patch_size`` and ``stride`` define the DIC point-of-interest (POI) window
    and grid in source pixels. ``reduction`` is the image pyramid reduction
    factor. ``max_features`` caps retained DoG keypoints per image. Set
    ``seed_points`` to a tuple of explicit ``(x, y)`` source coordinates to
    replace the regular grid.
    """

    patch_size: int = 128
    stride: int = 512
    reduction: int = 4
    max_features: int = 1024
    seed_points: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class DeepMatchingOptions:
    """Parameters for hierarchical, sparse hierarchical, or direct matching.

    ``mode="inria"`` selects the hierarchical DeepMatching path;
    ``mode="inria_sparse"`` evaluates only the hierarchy roots covering the
    explicit ``seed_points``; its reciprocal competition is restricted to
    those roots. ``mode="direct"`` selects patch correlation without the
    hierarchy.
    Sizes, stride, and ``search_radius_px`` are expressed in source-image
    pixels. ``seed_points`` optionally replaces the regular source grid.
    ``minimum_score`` and ``minimum_margin`` control the direct-mode
    acceptance test; they do not alter the INRIA reciprocal-match test.
    """

    patch_size: int = 128
    root_patch_size: int = 512
    stride: int = 512
    downscale_levels: int = 2
    search_radius_px: int = 256
    minimum_score: float = 0.0
    minimum_margin: float = 0.0
    seed_points: tuple[tuple[float, float], ...] | None = None
    mode: Literal["inria", "inria_sparse", "direct"] = "inria"


@dataclass(slots=True)
class Match:
    """One source-to-target displacement estimate.

    ``x, y`` identify the reported source coordinate; ``dx, dy`` are the
    displacement from that coordinate. ``usable`` records the matcher's own
    acceptance decision. Scores and algorithm-specific diagnostics are not
    calibrated across HORG and DeepMatching.
    """

    x: float
    y: float
    dx: float
    dy: float
    usable: bool
    score: float = 0.0
    second_score: float | None = None
    margin: float | None = None
    reciprocal: bool | None = None
    inliers: int | None = None
    candidates: int | None = None
    inlier_fraction: float | None = None
    residual: float | None = None
    local_consistency_px: float | None = None
    local_consistency_neighbors: int | None = None
    seed_x: float | None = None
    seed_y: float | None = None
    root_x: float | None = None
    root_y: float | None = None
    target_x: float | None = None
    target_y: float | None = None


@dataclass(slots=True)
class MatchField:
    """Result grid and metadata returned by one matcher call."""

    algorithm: str
    width: int
    height: int
    patch_size: int
    stride: int
    scale: int
    grid_width: int
    grid_height: int
    matches: list[Match]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> list[Match]:
        """Return the matches marked usable by their matcher."""
        return [match for match in self.matches if match.usable]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the field, matches, and algorithm metadata as a dictionary."""
        return {
            "algorithm": self.algorithm,
            "width": self.width,
            "height": self.height,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "scale": self.scale,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "count": len(self.matches),
            "matches": [asdict(match) for match in self.matches],
            **self.metadata,
        }


def _select_adapter(adapters: list[Any], device_type: str, power_preference: str) -> Any:
    if device_type not in {"auto", "gpu", "cpu"}:
        raise ValueError("device_type must be 'auto', 'gpu' or 'cpu'")
    if power_preference not in {"high-performance", "low-power"}:
        raise ValueError("power_preference must be 'high-performance' or 'low-power'")
    hardware = [adapter for adapter in adapters if adapter.info.get("adapter_type") != "CPU"]
    software = [adapter for adapter in adapters if adapter.info.get("adapter_type") == "CPU"]
    candidates = software if device_type == "cpu" else hardware
    if device_type == "auto" and not candidates:
        candidates = software
    if not candidates:
        kind = "CPU software" if device_type == "cpu" else "hardware GPU"
        raise RuntimeError(f"No compatible {kind} WebGPU adapter was found")
    preferred = "IntegratedGPU" if power_preference == "low-power" else "DiscreteGPU"
    return min(
        candidates,
        key=lambda adapter: (
            adapter.info.get("backend_type") != "Vulkan",
            adapter.info.get("adapter_type") != preferred,
        ),
    )


def _as_gray(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        array = load_gray(image)
    else:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image or a path to an image")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Image array must contain finite uint8-compatible values")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array, dtype=np.uint8)


class WGPUMatching:
    """Own one native WebGPU device and run the HORG and DeepMatching pipelines.

    The class is a context manager so the device can be released explicitly.
    All matching kernels are WGSL; installation uses the prebuilt ``wgpu`` wheel
    and does not build a project-specific native extension.
    """

    def __init__(
        self,
        *,
        device_type: Literal["auto", "gpu", "cpu"] = "auto",
        power_preference: Literal["high-performance", "low-power"] = "high-performance",
    ) -> None:
        """Create a native WebGPU device.

        Args:
            device_type: ``"auto"`` prefers a hardware adapter and falls back
                to a software CPU adapter; ``"gpu"`` requires hardware and
                ``"cpu"`` explicitly selects a software adapter.
            power_preference: Prefer an integrated low-power or discrete
                high-performance adapter when the backend exposes that detail.

        Raises:
            RuntimeError: If no compatible adapter of the requested type is
                available.
            ValueError: If an option is not one of the supported values.
        """
        adapters = wgpu.gpu.enumerate_adapters_sync()
        self.adapter = _select_adapter(adapters, device_type, power_preference)
        self.device = self.adapter.request_device_sync(
            label="WGPU-Matching native device",
            required_limits={"max-storage-buffers-per-shader-stage": 8},
        )
        self.adapter_info = dict(self.adapter.info)
        self.closed = False
        self._horg = HORGBackend(self.device)

    @property
    def device_name(self) -> str:
        return f"{self.adapter_info.get('backend_type', 'WebGPU')} ({self.adapter_info.get('device', 'unknown')})"

    def match_horg(
        self,
        reference: np.ndarray | str | Path,
        target: np.ndarray | str | Path,
        *,
        options: HORGOptions | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> MatchField:
        """Compute HORG initial displacements for a pair of grayscale images.

        Images may be NumPy arrays or paths to files readable by Pillow. Both
        images must have the same dimensions. The returned field contains all
        requested POIs, including rejected fits; inspect ``usable`` or
        ``field.accepted`` before using displacements downstream.
        """
        self._ensure_open()
        settings = options or HORGOptions()
        ref = _as_gray(reference)
        tar = _as_gray(target)
        result = self._horg.compute(
            ref,
            tar,
            patch_size=settings.patch_size,
            stride=settings.stride,
            reduction=settings.reduction,
            max_features=settings.max_features,
            seed_points=settings.seed_points,
            on_progress=on_progress,
        )
        return _field_from_dict(result)

    def match_deepmatching(
        self,
        reference: np.ndarray | str | Path,
        target: np.ndarray | str | Path,
        *,
        options: DeepMatchingOptions | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> MatchField:
        """Compute hierarchical, sparse hierarchical, or direct correspondences.

        ``DeepMatchingOptions.mode`` chooses the algorithm variant. Images may
        be NumPy arrays or Pillow-readable paths and must have equal dimensions.
        The result keeps rejected grid entries so callers can distinguish a
        rejected seed from a seed that was never requested. Sparse INRIA mode
        requires explicit seeds and limits reciprocal competition to the roots
        selected for those seeds.
        """
        self._ensure_open()
        from .deepmatching import DeepMatchingBackend

        settings = options or DeepMatchingOptions()
        ref, tar = _as_gray(reference), _as_gray(target)
        backend = DeepMatchingBackend(self.device)
        result = backend.compute(ref, tar, settings, on_progress=on_progress)
        return _field_from_dict(result)

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("WGPUMatching device is closed")

    def close(self) -> None:
        """Release the owned WebGPU device; repeated calls are safe."""
        if not self.closed:
            self.device.destroy()
            self.closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _field_from_dict(result: dict[str, Any]) -> MatchField:
    matches = [Match(**item) for item in result.pop("matches")]
    base = {name: result.pop(name) for name in (
        "algorithm", "width", "height", "patch_size", "stride", "scale", "grid_width", "grid_height"
    )}
    return MatchField(**base, matches=matches, metadata=result)
