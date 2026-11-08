"""Public synchronous API for native WebGPU matchers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import wgpu

from ._gpu import make_pipeline, shader_source
from .contracts import validate_matcher_options
from .dic_detector import DICDetectorBackend, DICDetectorOptions, DICDetectorResult
from .features import (
    DescribedFeatures,
    FeatureCorrespondences,
    FeatureSet,
    mutual_nearest_cosine_wgpu,
)
from .horg import HORGBackend
from .images import load_gray
from .projection import (
    CorrespondenceSet,
    ProjectionModel,
    ProjectionOptions,
    fit_initialization_field,
    release_projection_pipelines,
)

_MATCHER_CONTRACT = json.loads(
    files("wgpu_matching").joinpath("matcher-contracts.json").read_text(encoding="utf-8")
)
_DEEP_DEFAULTS = _MATCHER_CONTRACT["algorithms"]["deepmatching"]["defaults"]
_HORG_DEFAULTS = _MATCHER_CONTRACT["algorithms"]["horg"]["defaults"]
_CPM_DEFAULTS = _MATCHER_CONTRACT["algorithms"]["cpm"]["defaults"]


@dataclass(frozen=True, slots=True)
class HORGOptions:
    """Parameters for HORG keypoint matching and local affine fitting.

    ``patch_size`` and ``stride`` define the DIC point-of-interest (POI) window
    and grid in source pixels. ``reduction`` is the image pyramid reduction
    factor. ``max_features`` caps retained DoG keypoints per image. Set
    ``seed_points`` to a tuple of explicit ``(x, y)`` source coordinates to
    replace the regular grid.
    """

    patch_size: int = _HORG_DEFAULTS["patch_size_px"]
    stride: int = _HORG_DEFAULTS["stride_px"]
    reduction: int = _HORG_DEFAULTS["reduction"]
    max_features: int = _HORG_DEFAULTS["max_features"]
    seed_points: tuple[tuple[float, float], ...] | None = _HORG_DEFAULTS["seed_points"]


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

    patch_size: int = _DEEP_DEFAULTS["patch_size_px"]
    root_patch_size: int = _DEEP_DEFAULTS["root_patch_size_px"]
    stride: int = _DEEP_DEFAULTS["stride_px"]
    downscale_levels: int = _DEEP_DEFAULTS["downscale_levels"]
    search_radius_px: int = _DEEP_DEFAULTS["search_radius_px"]
    minimum_score: float = _DEEP_DEFAULTS["minimum_score"]
    minimum_margin: float = _DEEP_DEFAULTS["minimum_margin"]
    seed_points: tuple[tuple[float, float], ...] | None = _DEEP_DEFAULTS["seed_points"]
    mode: Literal["inria", "inria_sparse", "direct"] = _DEEP_DEFAULTS["mode"]


@dataclass(frozen=True, slots=True)
class CPMOptions:
    """Parameters for sparse-grid coarse-to-fine CPM-HOG initialization."""

    patch_size: int = _CPM_DEFAULTS["patch_size_px"]
    stride: int = _CPM_DEFAULTS["stride_px"]
    downscale_levels: int = _CPM_DEFAULTS["downscale_levels"]
    search_radius_px: int = _CPM_DEFAULTS["search_radius_px"]
    matching_support_px: int = _CPM_DEFAULTS["matching_support_px"]
    work_grid_stride: int = _CPM_DEFAULTS["work_grid_stride_px"]
    levels: int = _CPM_DEFAULTS["levels"]
    propagation_iterations: int = _CPM_DEFAULTS["propagation_iterations"]
    random_trials: int = _CPM_DEFAULTS["random_trials"]
    initial_hypotheses: int = _CPM_DEFAULTS["initial_hypotheses"]
    local_search_radius_px: float = _CPM_DEFAULTS["local_search_radius_px"]
    affine_radius_px: float | None = _CPM_DEFAULTS["affine_radius_px"]
    seed: int = _CPM_DEFAULTS["seed"]
    reciprocal_threshold_px: float = _CPM_DEFAULTS["reciprocal_threshold_px"]
    minimum_similarity: float = _CPM_DEFAULTS["minimum_similarity"]
    seed_points: tuple[tuple[float, float], ...] | None = _CPM_DEFAULTS["seed_points"]
    oracle_affine: tuple[float, float, float, float, float, float] | None = _CPM_DEFAULTS[
        "oracle_affine"
    ]


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


@dataclass(slots=True)
class LocalAffineField:
    """GPU affine fits built from an external set of point correspondences.

    Optional ``affine_models`` has shape (seed_count, 8). Each row stores
    ``a1,a2,a0,1,b1,b2,b0,1`` for the target coordinates
    ``x'=a1*(x-xc)+a2*(y-yc)+a0`` and its y counterpart. Only models whose
    corresponding match is usable should be consumed.
    """

    matches: list[Match]
    correspondence_count: int
    radius_px: float
    threshold_px: float
    gpu_compute_ms: float
    affine_models: np.ndarray | None = None

    @property
    def accepted(self) -> list[Match]:
        """Return local affine estimates that meet the requested confidence criteria."""
        return [match for match in self.matches if match.usable]


@dataclass(slots=True)
class FeatureMatchField:
    """Common descriptor matches and their WGSL local-affine initializations."""

    correspondences: FeatureCorrespondences
    initializations: LocalAffineField
    minimum_cosine_similarity: float

    @property
    def matches(self) -> list[Match]:
        """Return local affine estimates, including rejected seeds."""
        return self.initializations.matches

    @property
    def accepted(self) -> list[Match]:
        """Return local affine estimates accepted by the configured RANSAC checks."""
        return self.initializations.accepted


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
    """Own one native WebGPU device and run the HORG, DeepMatching, and CPM pipelines.

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
        self._dic_detector = DICDetectorBackend(self.device)
        self._feature_match_pipeline = None

    @property
    def device_name(self) -> str:
        return f"{self.adapter_info.get('backend_type', 'WebGPU')} ({self.adapter_info.get('device', 'unknown')})"

    def match_horg(
        self,
        reference: np.ndarray | str | Path,
        target: np.ndarray | str | Path,
        *,
        options: HORGOptions | None = None,
        include_correspondences: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> MatchField:
        """Compute HORG initial displacements for a pair of grayscale images.

        Images may be NumPy arrays or paths to files readable by Pillow. Both
        images must have the same dimensions. The returned field contains all
        requested POIs, including rejected fits; inspect ``usable`` or
        ``field.accepted`` before using displacements downstream.
        """
        self._ensure_open()
        if not isinstance(include_correspondences, bool):
            raise TypeError("include_correspondences must be a boolean")
        settings = options or HORGOptions()
        ref = _as_gray(reference)
        tar = _as_gray(target)
        if ref.shape != tar.shape:
            raise ValueError("HORG expects same-sized grayscale images")
        if settings.seed_points is not None and len(settings.seed_points) == 0:
            settings = replace(settings, seed_points=None)
        validate_matcher_options("horg", asdict(settings), ref.shape[1], ref.shape[0])
        result = self._horg.compute(
            ref,
            tar,
            patch_size=settings.patch_size,
            stride=settings.stride,
            reduction=settings.reduction,
            max_features=settings.max_features,
            seed_points=settings.seed_points,
            include_correspondences=include_correspondences,
            on_progress=on_progress,
        )
        return _field_from_dict(result)

    def detect_horg(
        self,
        image: np.ndarray | str | Path,
        *,
        max_keypoints: int = 4096,
        reduction: int = 4,
        on_progress: Callable[[str], None] | None = None,
    ) -> FeatureSet:
        """Return HORG/SIFT points for matching with a common descriptor.

        The point extractor and the original HORG descriptor stage execute on
        WebGPU. This adapter exposes only the floating-point locations, scores,
        and scales so an experiment can describe them with DeDoDe-B.
        """
        self._ensure_open()
        reference = _as_gray(image)
        settings = HORGOptions(reduction=reduction, max_features=max_keypoints)
        validate_matcher_options("horg", asdict(settings), reference.shape[1], reference.shape[0])
        return self._horg.detect_keypoints(
            reference,
            max_keypoints=max_keypoints,
            reduction=reduction,
            on_progress=on_progress,
        )

    def match_deepmatching(
        self,
        reference: np.ndarray | str | Path,
        target: np.ndarray | str | Path,
        *,
        options: DeepMatchingOptions | None = None,
        include_correspondences: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> MatchField:
        """Compute hierarchical, sparse hierarchical, or direct correspondences.

        ``DeepMatchingOptions.mode`` chooses the algorithm variant. Images may
        be NumPy arrays or Pillow-readable paths and must have equal dimensions.
        The result keeps rejected grid entries so callers can distinguish a
        rejected seed from a seed that was never requested. Sparse INRIA mode
        requires explicit seeds and limits reciprocal competition to the roots
        selected for those seeds.

        In hierarchical mode, ``include_correspondences=True`` exports the
        full observed atomic candidate set and reciprocal acceptance mask in
        metadata. These are actual atomic source coordinates, not fitted
        displacements at requested window centres.
        """
        self._ensure_open()
        if not isinstance(include_correspondences, bool):
            raise TypeError("include_correspondences must be a boolean")
        from .deepmatching import DeepMatchingBackend

        settings = options or DeepMatchingOptions()
        ref, tar = _as_gray(reference), _as_gray(target)
        if ref.shape != tar.shape:
            raise ValueError("DeepMatching expects same-sized grayscale images")
        validate_matcher_options("deepmatching", asdict(settings), ref.shape[1], ref.shape[0])
        backend = DeepMatchingBackend(self.device)
        result = backend.compute(
            ref,
            tar,
            settings,
            on_progress=on_progress,
            include_correspondences=include_correspondences,
        )
        return _field_from_dict(result)

    def match_cpm(
        self,
        reference: np.ndarray | str | Path,
        target: np.ndarray | str | Path,
        *,
        options: CPMOptions | None = None,
        include_correspondences: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> MatchField:
        """Compute sparse-grid CPM-HOG correspondences and local affine seeds.

        The matcher runs HOG extraction, HOG pyramids, PatchMatch propagation,
        random search, and reciprocal checking on WebGPU. Requested DIC seeds
        are fitted from nearby accepted work-grid matches when the local affine
        support is adequate; rejected fits remain in the returned field.

        ``include_correspondences=True`` additionally exports the observed
        work-grid pairs and rejection mask. The normal seed outputs are local
        affine estimates and must not be reused as independent measurements.
        """
        self._ensure_open()
        if not isinstance(include_correspondences, bool):
            raise TypeError("include_correspondences must be a boolean")
        settings = options or CPMOptions()
        ref, tar = _as_gray(reference), _as_gray(target)
        if ref.shape != tar.shape:
            raise ValueError("CPM expects same-sized grayscale images")
        if settings.seed_points is not None and len(settings.seed_points) == 0:
            settings = replace(settings, seed_points=None)
        validate_matcher_options("cpm", asdict(settings), ref.shape[1], ref.shape[0])
        from .cpm import CPMBackend

        result = CPMBackend(self.device).compute(
            ref,
            tar,
            settings,
            on_progress=on_progress,
            include_correspondences=include_correspondences,
        )
        return _field_from_dict(result)

    def detect_dic(
        self,
        image: np.ndarray | str | Path,
        *,
        options: DICDetectorOptions | None = None,
        roi_mask: np.ndarray | None = None,
        learned_weights_path: str | Path | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> DICDetectorResult:
        """Detect DIC texture points using the tensor or learned WGSL score model.

        Without a weight path, local structure-tensor eigenvalues report
        photometric conditioning, not cross-image uniqueness. Providing the
        small NumPy weight bundle produced by ``train_dic_detector.py`` runs
        its convolutional score network on WebGPU after the same tensor pass.
        The returned feature set can be described separately and matched through
        ``match_features``.
        """
        self._ensure_open()
        return self._dic_detector.compute(
            _as_gray(image),
            options=options,
            roi_mask=roi_mask,
            learned_weights_path=learned_weights_path,
            on_progress=on_progress,
        )

    def fit_local_affine(
        self,
        correspondences: np.ndarray,
        seed_points: np.ndarray,
        *,
        radius_px: float = 256.0,
        threshold_px: float = 3.0,
        min_inliers: int = 6,
        min_candidates: int = 8,
        min_inlier_fraction: float = 0.2,
        max_iterations: int = 256,
        include_affine_models: bool = False,
    ) -> LocalAffineField:
        """Fit one robust affine model per requested center on WebGPU.

        ``correspondences`` is an Nx4 float array ordered as
        ``(x_reference, y_reference, x_target, y_target)``. This method reuses
        HORG's WGSL candidate-selection and RANSAC kernels, allowing a different
        detector/descriptor or external learned matcher to provide the pairs.

        ``include_affine_models=True`` retains the six target-map coefficients
        of each fitted model, in addition to its centre translation. This
        permits testing original measured pairs against the local model.
        """
        self._ensure_open()
        result = self._horg.fit_local_correspondences(
            correspondences,
            seed_points,
            radius_px=radius_px,
            threshold_px=threshold_px,
            max_iterations=max_iterations,
            include_affine_models=include_affine_models,
        )
        matches = []
        for item in result["matches"]:
            usable = bool(
                item["fitted"]
                and item["inliers"] >= min_inliers
                and item["candidates"] >= min_candidates
                and item["inlier_fraction"] >= min_inlier_fraction
                and item["residual"] <= threshold_px
            )
            matches.append(
                Match(
                    x=item["x"],
                    y=item["y"],
                    dx=item["dx"],
                    dy=item["dy"],
                    usable=usable,
                    score=item["inlier_fraction"],
                    second_score=item["residual"],
                    margin=item["inlier_fraction"],
                    inliers=item["inliers"],
                    candidates=item["candidates"],
                    inlier_fraction=item["inlier_fraction"],
                    residual=item["residual"],
                )
            )
        return LocalAffineField(
            matches=matches,
            correspondence_count=len(correspondences),
            radius_px=radius_px,
            threshold_px=threshold_px,
            gpu_compute_ms=result["gpu_compute_ms"],
            affine_models=result["affine_models"],
        )

    def match_features(
        self,
        reference_features: DescribedFeatures,
        target_features: DescribedFeatures,
        seed_points: np.ndarray,
        *,
        min_cosine_similarity: float = 0.0,
        radius_px: float = 256.0,
        threshold_px: float = 3.0,
        min_inliers: int = 6,
        min_candidates: int = 8,
        min_inlier_fraction: float = 0.2,
        max_iterations: int = 256,
        fit_local_models: bool = True,
    ) -> FeatureMatchField:
        """Match a common descriptor and fit local affine seeds on WebGPU.

        Mutual cosine nearest neighbours execute on WebGPU. Optional local-affine
        RANSAC also runs there; disable it when a later geometric stage owns
        acceptance. Descriptors are uploaded from host memory and only nearest-index
        and score arrays are read back; detector and descriptor inference are
        independent stages and may still execute on the CPU.
        """
        self._ensure_open()
        if self._feature_match_pipeline is None:
            module = self.device.create_shader_module(
                code=shader_source("feature-mutual-nearest.wgsl")
            )
            self._feature_match_pipeline = make_pipeline(
                self.device, module, "nearest", "Mutual cosine nearest descriptor"
            )
        pairs = mutual_nearest_cosine_wgpu(
            self.device,
            reference_features,
            target_features,
            min_cosine_similarity=min_cosine_similarity,
            pipeline=self._feature_match_pipeline,
        )
        if not isinstance(fit_local_models, bool):
            raise TypeError("fit_local_models must be a boolean")
        if not fit_local_models:
            initialization = LocalAffineField(
                matches=[],
                correspondence_count=len(pairs.points),
                radius_px=radius_px,
                threshold_px=threshold_px,
                gpu_compute_ms=0.0,
            )
        elif len(pairs.points) >= 3:
            initialization = self.fit_local_affine(
                pairs.points,
                seed_points,
                radius_px=radius_px,
                threshold_px=threshold_px,
                min_inliers=min_inliers,
                min_candidates=min_candidates,
                min_inlier_fraction=min_inlier_fraction,
                max_iterations=max_iterations,
            )
        else:
            centers = np.ascontiguousarray(seed_points, dtype=np.float32)
            if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers):
                raise ValueError("seed_points must be a non-empty Nx2 array")
            if not np.isfinite(centers).all():
                raise ValueError("seed_points must contain finite coordinates")
            initialization = LocalAffineField(
                matches=[
                    Match(x=float(x), y=float(y), dx=0.0, dy=0.0, usable=False) for x, y in centers
                ],
                correspondence_count=len(pairs.points),
                radius_px=radius_px,
                threshold_px=threshold_px,
                gpu_compute_ms=0.0,
            )
        return FeatureMatchField(
            correspondences=pairs,
            initializations=initialization,
            minimum_cosine_similarity=min_cosine_similarity,
        )

    def fit_initialization_field(
        self,
        correspondences: np.ndarray | CorrespondenceSet,
        image_size: tuple[int, int],
        *,
        options: ProjectionOptions | None = None,
    ) -> ProjectionModel:
        """Project direct feature correspondences onto an affine/RBF field.

        Accept a provenance-aware :class:`CorrespondenceSet`, or an ``Nx4``
        array ordered ``(x_reference, y_reference, x_target, y_target)``.
        Coordinates are native image pixels. With this matcher,
        robust fitting and patch projection run on its WebGPU device; bounded
        Gaussian-RBF candidate selection currently runs on the float64 host.
        """
        self._ensure_open()
        settings = options or ProjectionOptions()
        validate_matcher_options("projection", asdict(settings), image_size[0], image_size[1])
        return fit_initialization_field(
            correspondences, image_size, options=settings, device=self.device
        )

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("WGPUMatching device is closed")

    def close(self) -> None:
        """Release the owned WebGPU device; repeated calls are safe."""
        if not self.closed:
            release_projection_pipelines(self.device)
            self.device.destroy()
            self.closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _field_from_dict(result: dict[str, Any]) -> MatchField:
    matches = [Match(**item) for item in result.pop("matches")]
    base = {
        name: result.pop(name)
        for name in (
            "algorithm",
            "width",
            "height",
            "patch_size",
            "stride",
            "scale",
            "grid_width",
            "grid_height",
        )
    }
    return MatchField(**base, matches=matches, metadata=result)
