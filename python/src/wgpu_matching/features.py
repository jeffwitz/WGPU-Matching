"""Optional reference implementations for learned keypoint detectors.

The published PyTorch models are kept outside the core WebGPU package. This
module imports PyTorch and the upstream model packages only when a learned
detector or descriptor is instantiated.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from .images import load_gray

_REGISTERED_WEBGPU_LIBRARIES: set[str] = set()


@dataclass(slots=True)
class FeatureSet:
    """Detected keypoints in source-image pixel-centre coordinates."""

    xy: np.ndarray
    scores: np.ndarray
    valid: np.ndarray | None = None
    scales: np.ndarray | None = None
    orientations: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xy = np.ascontiguousarray(self.xy, dtype=np.float32)
        self.scores = np.ascontiguousarray(self.scores, dtype=np.float32).reshape(-1)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("Feature coordinates must have shape (N, 2)")
        if self.scores.shape != (len(self.xy),):
            raise ValueError("Feature scores must have shape (N,)")
        if not np.isfinite(self.xy).all() or not np.isfinite(self.scores).all():
            raise ValueError("Feature coordinates and scores must be finite")
        if self.valid is None:
            self.valid = np.ones(len(self.xy), dtype=np.bool_)
        else:
            self.valid = np.ascontiguousarray(self.valid, dtype=np.bool_).reshape(-1)
            if self.valid.shape != (len(self.xy),):
                raise ValueError("Feature validity must have shape (N,)")
        if self.scales is not None:
            self.scales = np.ascontiguousarray(self.scales, dtype=np.float32).reshape(-1)
            if self.scales.shape != (len(self.xy),):
                raise ValueError("Feature scales must have shape (N,)")
        if self.orientations is not None:
            self.orientations = np.ascontiguousarray(self.orientations, dtype=np.float32).reshape(
                -1
            )
            if self.orientations.shape != (len(self.xy),):
                raise ValueError("Feature orientations must have shape (N,)")


@dataclass(slots=True)
class DescribedFeatures:
    """A feature set with one descriptor row per detected keypoint."""

    features: FeatureSet
    descriptors: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.descriptors = np.ascontiguousarray(self.descriptors, dtype=np.float32)
        if self.descriptors.ndim != 2 or self.descriptors.shape[0] != len(self.features.xy):
            raise ValueError("Descriptors must have shape (N, D), aligned with keypoints")
        if not np.isfinite(self.descriptors).all():
            raise ValueError("Descriptors must be finite")


@runtime_checkable
class KeypointDetector(Protocol):
    """Common detector interface used by reference experiments."""

    name: str

    def detect(
        self,
        image: np.ndarray | str | Path,
        *,
        max_keypoints: int = 4096,
        roi_mask: np.ndarray | None = None,
    ) -> FeatureSet: ...


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torchvision import transforms
    except ImportError as error:
        raise ImportError(
            "Learned detector references require the optional research dependencies; "
            "install the 'research' extra and the official DeDoDe source package."
        ) from error
    return torch, transforms


def _gray_image(image: np.ndarray | str | Path) -> np.ndarray:
    gray = load_gray(image) if isinstance(image, (str, Path)) else np.asarray(image)
    if gray.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image")
    if not np.issubdtype(gray.dtype, np.number) or not np.isfinite(gray).all():
        raise ValueError("Image pixels must be finite numeric values")
    return np.ascontiguousarray(np.clip(gray, 0, 255).astype(np.uint8))


def _device(torch: Any, name: str) -> Any:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The requested MPS device is unavailable")
    return device


def _resize_shape(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = max_side / max(width, height)
    resized_width = max(8, int((scale * width) // 8 * 8))
    resized_height = max(8, int((scale * height) // 8 * 8))
    return resized_width, resized_height


def _prepare_image(
    gray: np.ndarray,
    max_side: int,
    device: Any,
    transforms: Any,
) -> tuple[Any, tuple[int, int]]:
    """Apply the official ImageNet normalization and aspect-preserving resize."""
    import torch

    height, width = gray.shape
    resized_width, resized_height = _resize_shape(width, height, max_side)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    pil_image = Image.fromarray(rgb, mode="RGB").resize(
        (resized_width, resized_height), resample=Image.Resampling.BICUBIC
    )
    values = np.asarray(pil_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return normalizer(tensor)[None].to(device), (resized_width, resized_height)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(package: str, checkpoint: Path | None, url: str) -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "reference_package": package,
        "reference_package_version": package_version,
        "weights_url": url,
        "weights_path": str(checkpoint) if checkpoint else None,
        "weights_sha256": _sha256(checkpoint),
        "coordinates": "source-image pixel centres; integer values identify pixel centres",
    }


class _TorchKeypointDetector:
    name = "abstract-torch-detector"

    def __init__(self, *, max_side: int = 1024, device: str = "auto") -> None:
        if max_side < 64:
            raise ValueError("max_side must be at least 64 pixels")
        self.torch, self.transforms = _torch_modules()
        self.device = _device(self.torch, device)
        self.max_side = int(max_side)

    def _infer(self, batch: dict[str, Any], max_keypoints: int) -> tuple[Any, Any]:
        raise NotImplementedError

    def detect(
        self,
        image: np.ndarray | str | Path,
        *,
        max_keypoints: int = 4096,
        roi_mask: np.ndarray | None = None,
    ) -> FeatureSet:
        if max_keypoints < 1:
            raise ValueError("max_keypoints must be positive")
        gray = _gray_image(image)
        height, width = gray.shape
        batch, (resized_width, resized_height) = _prepare_image(
            gray, self.max_side, self.device, self.transforms
        )
        count = min(int(max_keypoints), resized_width * resized_height)
        with self.torch.inference_mode():
            normalized_xy, raw_scores = self._infer({"image": batch}, count)
        normalized_xy = normalized_xy.reshape(-1, 2).detach().float().cpu().numpy()
        raw_scores = raw_scores.reshape(-1).detach().float().cpu().numpy()
        # Upstream normalized coordinates map grid_sample locations to the
        # [0, W] boundary convention. Convert that half-pixel representation
        # to this repository's integer-pixel-centre convention, then undo resize.
        model_half_centres = (
            (normalized_xy + 1.0)
            * np.array([resized_width, resized_height], dtype=np.float32)
            * 0.5
        )
        source_xy = (
            model_half_centres
            * np.array([width / resized_width, height / resized_height], dtype=np.float32)
            - 0.5
        )
        keep = np.isfinite(source_xy).all(axis=1) & (source_xy[:, 0] >= -0.5)
        keep &= source_xy[:, 0] < width - 0.5
        keep &= source_xy[:, 1] >= -0.5
        keep &= source_xy[:, 1] < height - 0.5
        if roi_mask is not None:
            mask = np.asarray(roi_mask)
            if mask.shape != gray.shape:
                raise ValueError("roi_mask must have the same shape as the source image")
            px = np.clip(np.rint(source_xy[:, 0]).astype(np.int64), 0, width - 1)
            py = np.clip(np.rint(source_xy[:, 1]).astype(np.int64), 0, height - 1)
            keep &= mask[py, px] != 0
        source_xy = source_xy[keep]
        scores = raw_scores[keep]
        order = np.argsort(-scores, kind="stable")[:max_keypoints]
        source_xy, scores = source_xy[order], scores[order]
        return FeatureSet(
            xy=source_xy,
            scores=scores,
            metadata={
                **self.metadata,
                "source_size_px": [width, height],
                "model_size_px": [resized_width, resized_height],
                "max_keypoints_requested": int(max_keypoints),
                "keypoints_returned": len(source_xy),
                "preprocessing": "grayscale replicated to RGB; bicubic aspect-preserving resize; ImageNet normalization",
                "roi_mask_applied_after_detection": roi_mask is not None,
            },
        )


class DaDDetector(_TorchKeypointDetector):
    """Official DaD detector wrapped as an optional PyTorch reference."""

    name = "DaD"
    weights_url = "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth"

    def __init__(
        self,
        *,
        max_side: int = 1024,
        device: str = "auto",
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__(max_side=max_side, device=device)
        try:
            import dad
        except ImportError as error:
            raise ImportError("Install the optional official 'dad-detector' package") from error
        path = Path(weights_path).expanduser() if weights_path else None
        self.model = (
            dad.load_DaD(
                resize=self.max_side,
                keep_aspect_ratio=True,
                weights_path=str(path) if path else None,
            )
            .to(self.device)
            .eval()
        )
        cache_path = path or Path(self.torch.hub.get_dir()) / "checkpoints" / "dad.pth"
        self.metadata = _source_metadata("dad-detector", cache_path, self.weights_url)
        self.metadata.update(
            {
                "detector": self.name,
                "device": str(self.device),
                "source_revision": "41b9e86a1a3ae52128485cc1094aa4952ef489ba",
                "checkpoint_revision": "v0.1.0 release asset dad.pth",
            }
        )

    def _infer(self, batch: dict[str, Any], max_keypoints: int) -> tuple[Any, Any]:
        detected = self.model.detect(batch, num_keypoints=max_keypoints, return_dense_probs=False)
        return detected["keypoints"], detected["keypoint_probs"]


class DeDoDeV2Detector(_TorchKeypointDetector):
    """Official DeDoDe-L detector with the released v2 checkpoint."""

    name = "DeDoDe-v2-L"
    weights_url = "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth"

    def __init__(
        self,
        *,
        max_side: int = 1024,
        device: str = "auto",
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__(max_side=max_side, device=device)
        try:
            from DeDoDe.model_zoo import dedode_detector_L
        except ImportError as error:
            raise ImportError(
                "Install the official DeDoDe source package from Parskatt/DeDoDe"
            ) from error
        path = Path(weights_path).expanduser() if weights_path else None
        if path is None:
            state = self.torch.hub.load_state_dict_from_url(
                self.weights_url,
                map_location=self.device,
                file_name="dedode_detector_L_v2.pth",
                weights_only=False,
            )
            path = Path(self.torch.hub.get_dir()) / "checkpoints" / "dedode_detector_L_v2.pth"
        else:
            state = self.torch.load(path, map_location=self.device, weights_only=False)
        self.model = dedode_detector_L(
            device=self.device, weights=state, remove_borders=True
        ).eval()
        self.metadata = _source_metadata("DeDoDe", path, self.weights_url)
        self.metadata.update(
            {
                "detector": self.name,
                "source_revision": "6d156183f4dc84cd704ae779eebc8350995c5b06",
                "checkpoint_revision": "v2 release asset dedode_detector_L_v2.pth",
                "device": str(self.device),
            }
        )

    def _infer(self, batch: dict[str, Any], max_keypoints: int) -> tuple[Any, Any]:
        detected = self.model.detect(batch, num_keypoints=max_keypoints)
        return detected["keypoints"], detected["confidence"]


class OnnxWGPUKeypointDetector:
    """Run a DaD or DeDoDe-v2 keypoint ONNX graph with ONNX Runtime WebGPU.

    The model must have been exported in ``keypoints`` mode by
    ``python/benchmarks/export_learned_detector_onnx.py``. The graph returns a
    fixed top-K point budget, while this adapter handles image preparation,
    coordinate conversion, and optional ROI filtering without importing
    PyTorch.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_side: int = 640,
        keypoint_capacity: int | None = None,
    ) -> None:
        if max_side < 64:
            raise ValueError("max_side must be at least 64 pixels")
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX detector model not found: {self.model_path}")
        try:
            import onnxruntime as ort
            import onnxruntime_ep_webgpu as webgpu_ep
        except ImportError as error:
            raise ImportError(
                "Install the optional 'onnx-webgpu' dependencies to use this detector"
            ) from error

        library_name = "wgpu_matching_keypoint_detector"
        if library_name not in _REGISTERED_WEBGPU_LIBRARIES:
            ort.register_execution_provider_library(library_name, webgpu_ep.get_library_path())
            _REGISTERED_WEBGPU_LIBRARIES.add(library_name)
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()
        ]
        if not devices:
            raise RuntimeError("ONNX Runtime found no compatible WebGPU adapter")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
        session_start = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=None
        )
        self.session_initialization_s = time.perf_counter() - session_start
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if [item.name for item in inputs] != ["image"]:
            raise ValueError(f"Expected an ONNX detector input named 'image'; got {inputs}")
        if [item.name for item in outputs] != ["keypoints", "scores"]:
            raise ValueError(
                "Expected a keypoint-mode ONNX detector with outputs 'keypoints' and 'scores'"
            )
        output_shape = outputs[0].shape
        score_shape = outputs[1].shape
        if (
            len(output_shape) != 3
            or output_shape[-1] != 2
            or len(score_shape) != 2
            or (
                isinstance(output_shape[1], int)
                and isinstance(score_shape[1], int)
                and output_shape[1] != score_shape[1]
            )
        ):
            raise ValueError(
                "The ONNX detector must return fixed-size (1, K, 2) keypoints and (1, K) scores"
            )

        model_metadata = self.session.get_modelmeta().custom_metadata_map
        output_mode = model_metadata.get("wgpu_matching.output_mode")
        if output_mode is not None and output_mode != "keypoints":
            raise ValueError("The ONNX model must be exported with --output-mode keypoints")
        metadata_capacity = model_metadata.get("wgpu_matching.keypoint_capacity")
        if keypoint_capacity is not None:
            self.keypoint_capacity = int(keypoint_capacity)
        elif isinstance(output_shape[1], int):
            self.keypoint_capacity = int(output_shape[1])
        elif metadata_capacity:
            self.keypoint_capacity = int(metadata_capacity)
        else:
            raise ValueError(
                "The ONNX graph has a dynamic keypoint dimension; pass keypoint_capacity or "
                "re-export it with python/benchmarks/export_learned_detector_onnx.py"
            )
        if self.keypoint_capacity < 1:
            raise ValueError("keypoint_capacity must be positive")
        self.max_side = int(max_side)
        self.adapter = {
            "vendor_id": getattr(devices[0].device, "vendor_id", None),
            "device_id": getattr(devices[0].device, "device_id", None),
        }
        self.name = "ONNX-WebGPU-keypoint-detector"
        self.metadata = {
            "detector": self.name,
            "model_path": str(self.model_path),
            "model_sha256": _sha256(self.model_path),
            "onnxruntime_version": ort.__version__,
            "providers": self.session.get_providers(),
            "adapter": self.adapter,
            "max_side_px": self.max_side,
            "compiled_keypoint_capacity": self.keypoint_capacity,
            "model_metadata": model_metadata,
            "session_initialization_s": self.session_initialization_s,
            "coordinate_convention": "source-image integer pixel centres; output converted from align_corners=False normalized coordinates",
            "output_transfer": "keypoints and scores are returned as CPU NumPy arrays; dense maps are not exported",
        }

    def detect(
        self,
        image: np.ndarray | str | Path,
        *,
        max_keypoints: int | None = None,
        roi_mask: np.ndarray | None = None,
    ) -> FeatureSet:
        """Return source-image keypoints and confidence scores from the ONNX graph."""
        requested = self.keypoint_capacity if max_keypoints is None else int(max_keypoints)
        if requested < 1:
            raise ValueError("max_keypoints must be positive")
        if requested > self.keypoint_capacity:
            raise ValueError(
                f"The ONNX graph contains {self.keypoint_capacity} points; re-export it for a larger budget"
            )

        call_start = time.perf_counter()
        preprocessing_start = call_start
        gray = _gray_image(image)
        height, width = gray.shape
        resized_width, resized_height = _resize_shape(width, height, self.max_side)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        resized = (
            np.asarray(
                Image.fromarray(rgb, mode="RGB").resize(
                    (resized_width, resized_height), resample=Image.Resampling.BICUBIC
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        batch = np.ascontiguousarray(((resized - mean) / std).transpose(2, 0, 1)[None])
        preprocessing_s = time.perf_counter() - preprocessing_start

        inference_start = time.perf_counter()
        raw_normalized_xy, raw_scores = self.session.run(["keypoints", "scores"], {"image": batch})
        inference_s = time.perf_counter() - inference_start
        raw_output_bytes = int(raw_normalized_xy.nbytes + raw_scores.nbytes)
        postprocessing_start = time.perf_counter()
        normalized_xy = np.asarray(raw_normalized_xy[0, :requested], dtype=np.float32)
        scores = np.asarray(raw_scores[0, :requested], dtype=np.float32)
        model_half_centres = (
            (normalized_xy + 1.0)
            * np.asarray([resized_width, resized_height], dtype=np.float32)
            * 0.5
        )
        source_xy = (
            model_half_centres
            * np.asarray([width / resized_width, height / resized_height], dtype=np.float32)
            - 0.5
        )
        keep = np.isfinite(source_xy).all(axis=1) & (source_xy[:, 0] >= -0.5)
        keep &= source_xy[:, 0] < width - 0.5
        keep &= source_xy[:, 1] >= -0.5
        keep &= source_xy[:, 1] < height - 0.5
        if roi_mask is not None:
            mask = np.asarray(roi_mask)
            if mask.shape != gray.shape:
                raise ValueError("roi_mask must have the same shape as the source image")
            px = np.clip(np.rint(source_xy[:, 0]).astype(np.int64), 0, width - 1)
            py = np.clip(np.rint(source_xy[:, 1]).astype(np.int64), 0, height - 1)
            keep &= mask[py, px] != 0

        source_xy = source_xy[keep]
        scores = scores[keep]
        postprocessing_s = time.perf_counter() - postprocessing_start
        total_s = time.perf_counter() - call_start
        return FeatureSet(
            xy=source_xy,
            scores=scores,
            metadata={
                **self.metadata,
                "source_size_px": [width, height],
                "model_size_px": [resized_width, resized_height],
                "max_keypoints_requested": requested,
                "keypoints_returned": len(source_xy),
                "preprocessing": "grayscale replicated to RGB; bicubic resize to multiples of 8; ImageNet normalization",
                "roi_mask_applied_after_detection": roi_mask is not None,
                "onnx_session_run_s": inference_s,
                "input_preprocessing_s": preprocessing_s,
                "coordinate_postprocessing_s": postprocessing_s,
                "detect_total_s": total_s,
                "output_bytes_returned_to_cpu": raw_output_bytes,
                "feature_bytes_returned": int(source_xy.nbytes + scores.nbytes),
            },
        )


class DeDoDeBDescriptor:
    """Official DeDoDe-B descriptor sampled at arbitrary source-image points."""

    name = "DeDoDe-B"
    weights_url = (
        "https://github.com/Parskatt/DeDoDe/releases/download/"
        "dedode_pretrained_models/dedode_descriptor_B.pth"
    )

    def __init__(
        self,
        *,
        max_side: int = 1024,
        device: str = "auto",
        weights_path: str | Path | None = None,
    ) -> None:
        if max_side < 64:
            raise ValueError("max_side must be at least 64 pixels")
        self.torch, self.transforms = _torch_modules()
        self.device = _device(self.torch, device)
        self.max_side = int(max_side)
        try:
            from DeDoDe.model_zoo import dedode_descriptor_B
        except ImportError as error:
            raise ImportError(
                "Install the official DeDoDe source package from Parskatt/DeDoDe"
            ) from error
        path = Path(weights_path).expanduser() if weights_path else None
        if path is None:
            state = self.torch.hub.load_state_dict_from_url(
                self.weights_url,
                map_location=self.device,
                file_name="dedode_descriptor_B.pth",
                weights_only=False,
            )
            path = Path(self.torch.hub.get_dir()) / "checkpoints" / "dedode_descriptor_B.pth"
        else:
            state = self.torch.load(path, map_location=self.device, weights_only=False)
        self.model = dedode_descriptor_B(device=self.device, weights=state).eval()
        self.metadata = _source_metadata("DeDoDe", path, self.weights_url)
        self.metadata.update(
            {
                "descriptor": self.name,
                "source_revision": "6d156183f4dc84cd704ae779eebc8350995c5b06",
                "checkpoint_revision": "dedode_pretrained_models release asset dedode_descriptor_B.pth",
                "device": str(self.device),
            }
        )

    def describe(
        self,
        image: np.ndarray | str | Path,
        features: FeatureSet,
    ) -> DescribedFeatures:
        gray = _gray_image(image)
        height, width = gray.shape
        batch, (resized_width, resized_height) = _prepare_image(
            gray, self.max_side, self.device, self.transforms
        )
        # grid_sample(align_corners=False) locates the pixel centre at
        # 2 * (index + 0.5) / extent - 1. Convert source-space centre
        # coordinates back into the resized model image before normalization.
        model_x = (features.xy[:, 0] + 0.5) * (resized_width / width)
        model_y = (features.xy[:, 1] + 0.5) * (resized_height / height)
        normalized = np.column_stack(
            (2.0 * model_x / resized_width - 1.0, 2.0 * model_y / resized_height - 1.0)
        ).astype(np.float32)
        points = self.torch.from_numpy(normalized)[None].to(self.device)
        with self.torch.inference_mode():
            result = self.model.describe_keypoints({"image": batch}, points)["descriptions"]
        descriptors = result[0].detach().float().cpu().numpy()
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / np.maximum(norms, 1e-12)
        return DescribedFeatures(
            features=features,
            descriptors=descriptors,
            metadata={
                **self.metadata,
                "source_size_px": [width, height],
                "model_size_px": [resized_width, resized_height],
                "normalization": "L2",
            },
        )


@dataclass(slots=True)
class FeatureCorrespondences:
    """Mutual nearest-neighbour pairs produced by a shared cosine matcher."""

    points: np.ndarray
    reference_indices: np.ndarray
    target_indices: np.ndarray
    cosine_similarity: np.ndarray

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        self.reference_indices = np.ascontiguousarray(self.reference_indices, dtype=np.int32)
        self.target_indices = np.ascontiguousarray(self.target_indices, dtype=np.int32)
        self.cosine_similarity = np.ascontiguousarray(self.cosine_similarity, dtype=np.float32)
        count = len(self.reference_indices)
        if self.points.shape != (count, 4):
            raise ValueError("Correspondence points must have shape (N, 4)")
        if self.target_indices.shape != (count,) or self.cosine_similarity.shape != (count,):
            raise ValueError("Correspondence arrays must have equal lengths")


def mutual_nearest_cosine(
    reference: DescribedFeatures,
    target: DescribedFeatures,
    *,
    min_cosine_similarity: float = 0.0,
    source_chunk_size: int = 512,
) -> FeatureCorrespondences:
    """Return reciprocal cosine-nearest descriptor pairs using bounded memory.

    Only the source-by-target similarity block for one source chunk is held at
    once. Ties are resolved deterministically by NumPy's first-maximum rule.
    This CPU reference matcher is shared across all detector experiments;
    geometric verification is performed separately by the repository's WGSL
    local-affine RANSAC.
    """
    if not np.isfinite(min_cosine_similarity) or not -1.0 <= min_cosine_similarity <= 1.0:
        raise ValueError("min_cosine_similarity must be in [-1, 1]")
    if source_chunk_size < 1:
        raise ValueError("source_chunk_size must be positive")
    desc_a = np.asarray(reference.descriptors, dtype=np.float32)
    desc_b = np.asarray(target.descriptors, dtype=np.float32)
    if desc_a.ndim != 2 or desc_b.ndim != 2 or desc_a.shape[1] != desc_b.shape[1]:
        raise ValueError("Descriptor matrices must have the same feature dimension")
    original_a = np.flatnonzero(reference.features.valid)
    original_b = np.flatnonzero(target.features.valid)
    desc_a = desc_a[original_a]
    desc_b = desc_b[original_b]
    count_a, count_b = len(desc_a), len(desc_b)
    if not count_a or not count_b:
        return FeatureCorrespondences(
            np.empty((0, 4), np.float32),
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty(0, np.float32),
        )
    desc_a = desc_a / np.maximum(np.linalg.norm(desc_a, axis=1, keepdims=True), 1e-12)
    desc_b = desc_b / np.maximum(np.linalg.norm(desc_b, axis=1, keepdims=True), 1e-12)
    best_target = np.empty(count_a, dtype=np.int32)
    best_score = np.empty(count_a, dtype=np.float32)
    best_reference_for_target = np.full(count_b, -1, dtype=np.int32)
    best_score_for_target = np.full(count_b, -np.inf, dtype=np.float32)
    for start in range(0, count_a, source_chunk_size):
        stop = min(count_a, start + source_chunk_size)
        similarities = desc_a[start:stop] @ desc_b.T
        row_best = similarities.argmax(axis=1)
        row_scores = similarities[np.arange(stop - start), row_best]
        best_target[start:stop] = row_best
        best_score[start:stop] = row_scores
        chunk_col_best = similarities.argmax(axis=0)
        chunk_col_scores = similarities[chunk_col_best, np.arange(count_b)]
        improve = chunk_col_scores > best_score_for_target
        best_score_for_target[improve] = chunk_col_scores[improve]
        best_reference_for_target[improve] = start + chunk_col_best[improve]
    source_indices = np.arange(count_a, dtype=np.int32)
    reciprocal = best_reference_for_target[best_target] == source_indices
    keep = reciprocal & (best_score >= min_cosine_similarity)
    source_indices = original_a[source_indices[keep]]
    target_indices = original_b[best_target[keep]]
    points = np.column_stack(
        (reference.features.xy[source_indices], target.features.xy[target_indices])
    ).astype(np.float32, copy=False)
    return FeatureCorrespondences(
        points=points,
        reference_indices=source_indices,
        target_indices=target_indices,
        cosine_similarity=best_score[keep],
    )
