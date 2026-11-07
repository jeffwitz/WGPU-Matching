"""Shared robust affine/Gaussian-RBF projection for DIC initialization."""

from __future__ import annotations

import json
import struct
import threading
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, Literal

import numpy as np
import wgpu

from ._gpu import (
    ResourceScope,
    create_buffer,
    make_bind_group,
    read_array,
    shader_source,
    upload_buffer,
)

_PIPELINE_CACHE: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
_PIPELINE_CACHE_LOCK = threading.Lock()
_PROJECTION_DEFAULTS = json.loads(
    files("wgpu_matching").joinpath("matcher-contracts.json").read_text(encoding="utf-8")
)["algorithms"]["projection"]["defaults"]


def _cached_pipelines(
    device: Any,
    cache_key: str,
    shader_name: str,
    entries: tuple[str, ...],
) -> dict[str, Any]:
    """Compile each projection shader pipeline once per live WebGPU device."""
    with _PIPELINE_CACHE_LOCK:
        device_cache = _PIPELINE_CACHE.setdefault(device, {})
        if cache_key not in device_cache:
            module = device.create_shader_module(
                label=shader_name,
                code=shader_source(shader_name),
            )
            device_cache[cache_key] = {
                entry: device.create_compute_pipeline(
                    label=f"projection {entry}",
                    layout="auto",
                    compute={"module": module, "entry_point": entry},
                )
                for entry in entries
            }
        return device_cache[cache_key]


def release_projection_pipelines(device: Any) -> None:
    """Drop cached pipeline handles before the owning device is destroyed."""
    with _PIPELINE_CACHE_LOCK:
        _PIPELINE_CACHE.pop(device, None)


def _fit_model_webgpu(
    device: Any,
    x: np.ndarray,
    d: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    ell: float,
    ridge: float,
    huber: float,
    max_iterations: int,
    tolerance: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool, float]:
    """Run the bounded IRLS solve on WebGPU and return final weights and timing."""
    nparam = 3 + len(centers)
    if nparam > 67 or len(x) > 8192:
        raise ValueError("WebGPU projection solver supports at most 64 RBF centers and 8192 pairs")
    obs = np.column_stack((x, d)).astype(np.float32)
    center_data = np.zeros((max(1, len(centers)), 2), dtype=np.float32)
    if len(centers):
        center_data[: len(centers)] = centers.astype(np.float32)
    density = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(obs).all() or not np.isfinite(density).all():
        raise ValueError("WebGPU projection inputs must be finite")
    if len(x) != len(density):
        raise ValueError("each WebGPU projection pair needs one density weight")
    params = struct.pack(
        "<II6f", len(x), len(centers), ell, ridge, huber, tolerance, damping, 0.0
    )
    max_params = 67
    matrix_size = max_params * (max_params + 2) * np.dtype(np.float32).itemsize
    design_size = len(x) * nparam * np.dtype(np.float32).itemsize
    started = time.perf_counter()
    with ResourceScope() as scope:
        obs_b = upload_buffer(
            device, scope, obs, wgpu.BufferUsage.STORAGE, "projection observations"
        )
        ctr_b = upload_buffer(
            device, scope, center_data, wgpu.BufferUsage.STORAGE, "projection RBF centers"
        )
        wei_b = upload_buffer(
            device, scope, density, wgpu.BufferUsage.STORAGE, "projection density weights"
        )
        robust_b = upload_buffer(
            device,
            scope,
            np.ones(len(x), dtype=np.float32),
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            "projection robust weights",
        )
        normal_b = create_buffer(
            device,
            scope,
            matrix_size,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            "projection augmented normal system",
        )
        design_b = create_buffer(
            device,
            scope,
            design_size,
            wgpu.BufferUsage.STORAGE,
            "projection GPU basis matrix",
        )
        status_b = create_buffer(
            device, scope, 12, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            "projection solver status",
        )
        change_b = create_buffer(
            device, scope, 4, wgpu.BufferUsage.STORAGE, "projection IRLS convergence reduction"
        )
        residual_b = create_buffer(
            device,
            scope,
            len(x) * np.dtype(np.float32).itemsize,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            "projection GPU residuals",
        )
        prm_b = upload_buffer(
            device, scope, np.frombuffer(params, dtype=np.uint8),
            wgpu.BufferUsage.UNIFORM, "projection solver parameters",
        )
        pipelines = _cached_pipelines(
            device,
            "projection_fit",
            "projection-fit.wgsl",
            (
                "build_design", "clear_status", "clear_change", "accumulate",
                "solve", "update_weights", "check_convergence",
            ),
        )
        shared = {
            0: obs_b,
            1: ctr_b,
            2: wei_b,
            3: robust_b,
            4: normal_b,
            5: prm_b,
            6: design_b,
            7: status_b,
            8: change_b,
            9: residual_b,
        }
        bind_entries = {
            "build_design": (0, 1, 5, 6),
            "clear_status": (7,),
            "clear_change": (8,),
            "accumulate": (0, 2, 3, 4, 5, 6, 7),
            "solve": (4, 5, 7),
            "update_weights": (0, 3, 4, 5, 6, 7, 8, 9),
            "check_convergence": (5, 7, 8),
        }
        groups = {
            entry: make_bind_group(
                device,
                pipeline,
                [
                    {"binding": binding, "resource": {"buffer": shared[binding]}}
                    for binding in bind_entries[entry]
                ],
            )
            for entry, pipeline in pipelines.items()
        }
        encoder = device.create_command_encoder(label="projection fit")

        def run(entry: str, x_groups: int, y_groups: int = 1) -> None:
            compute = encoder.begin_compute_pass(label=f"projection {entry}")
            compute.set_pipeline(pipelines[entry])
            compute.set_bind_group(0, groups[entry])
            compute.dispatch_workgroups(x_groups, y_groups, 1)
            compute.end()

        run("build_design", (len(x) * nparam + 63) // 64)
        run("clear_status", 1)
        matrix_stride = nparam + 2
        matrix_x = (nparam + 7) // 8
        matrix_y = (matrix_stride + 7) // 8
        for _ in range(max_iterations):
            run("accumulate", matrix_x, matrix_y)
            run("solve", 1)
            run("clear_change", 1)
            run("update_weights", (len(x) + 63) // 64)
            run("check_convergence", 1)
        # Refit once using the last robust weights, then make the returned weights
        # consistent with those final coefficients. All iterations stay in one submit.
        run("accumulate", matrix_x, matrix_y)
        run("solve", 1)
        run("update_weights", (len(x) + 63) // 64)
        device.queue.submit([encoder.finish()])
        statuses = read_array(device, status_b, np.dtype(np.uint32), 3)
        if statuses[0]:
            raise RuntimeError(f"WebGPU projection solver failed (status {statuses[0]})")
        normal = read_array(
            device, normal_b, np.dtype(np.float32), nparam * matrix_stride
        ).reshape(nparam, matrix_stride)
        result = normal[:, nparam : nparam + 2].astype(np.float64)
        robust = read_array(device, robust_b, np.dtype(np.float32), len(x)).astype(np.float64)
        residuals = read_array(device, residual_b, np.dtype(np.float32), len(x)).astype(np.float64)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not np.isfinite(result).all() or not np.isfinite(robust).all() or not np.isfinite(residuals).all():
        raise RuntimeError("WebGPU projection solver returned non-finite values")
    final_delta = float(np.asarray(statuses[2], dtype=np.uint32).view(np.float32))
    return result[:3], result[3:], robust, residuals, elapsed_ms, 4, bool(statuses[1]), final_delta


@dataclass(frozen=True, slots=True)
class ProjectionOptions:
    """Bounded affine/Gaussian-RBF model selection settings."""

    basis: Literal["affine", "gaussian_rbf"] = _PROJECTION_DEFAULTS["basis"]
    center_counts: tuple[int, ...] = tuple(_PROJECTION_DEFAULTS["center_counts"])
    width_factors: tuple[float, ...] = tuple(_PROJECTION_DEFAULTS["width_factors"])
    ridge_values: tuple[float, ...] = tuple(_PROJECTION_DEFAULTS["ridge_values"])
    folds: int = _PROJECTION_DEFAULTS["folds"]
    max_irls_iterations: int = _PROJECTION_DEFAULTS["max_irls_iterations"]
    irls_tolerance: float = _PROJECTION_DEFAULTS["irls_tolerance"]
    irls_damping: float = _PROJECTION_DEFAULTS["irls_damping"]
    huber_delta: float = _PROJECTION_DEFAULTS["huber_delta"]
    minimum_scale_px: float = _PROJECTION_DEFAULTS["minimum_scale_px"]
    density_cells: tuple[int, int] = tuple(_PROJECTION_DEFAULTS["density_cells"])
    random_seed: int = _PROJECTION_DEFAULTS["random_seed"]


@dataclass(frozen=True, slots=True)
class CorrespondenceSet:
    """Provenance-aware direct point pairs in native reference/target pixels.

    ``source_xy`` and ``target_xy`` must already use the original image pixel
    frame. ``coordinate_mapping`` records any preprocessing transforms applied
    by an upstream matcher; this class never guesses or silently rescales them.
    Arbitrary matcher scores are retained as metadata and are not treated as
    calibrated probabilities or used as fit weights.
    """

    source_xy: np.ndarray
    target_xy: np.ndarray
    image_size: tuple[int, int]
    matcher_id: str
    match_ids: tuple[str | int, ...] | None = None
    accepted: np.ndarray | None = None
    measurement_kind: tuple[str, ...] | str = "direct_measured"
    raw_scores: np.ndarray | None = None
    reciprocal_error_px: np.ndarray | None = None
    rejection_reasons: tuple[str | None, ...] | None = None
    confidence: np.ndarray | None = None
    covariance: np.ndarray | None = None
    coordinate_frame: str = "native_pixels"
    coordinate_mapping: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = np.asarray(self.source_xy, dtype=np.float64)
        target = np.asarray(self.target_xy, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
            raise ValueError("source_xy and target_xy must both have shape (N, 2)")
        object.__setattr__(self, "source_xy", np.ascontiguousarray(source))
        object.__setattr__(self, "target_xy", np.ascontiguousarray(target))
        count = len(source)
        if self.match_ids is None:
            object.__setattr__(self, "match_ids", tuple(range(count)))
        elif len(self.match_ids) != count:
            raise ValueError("match_ids must have one stable identifier per correspondence")
        if len(set(self.match_ids or ())) != count:
            raise ValueError("match_ids must be unique within one correspondence set")
        if isinstance(self.measurement_kind, str):
            object.__setattr__(self, "measurement_kind", (self.measurement_kind,) * count)
        elif len(self.measurement_kind) != count:
            raise ValueError("measurement_kind must have one value per correspondence")
        if self.accepted is not None and np.asarray(self.accepted).shape != (count,):
            raise ValueError("accepted must have one boolean per correspondence")
        if self.rejection_reasons is not None and len(self.rejection_reasons) != count:
            raise ValueError("rejection_reasons must have one entry per correspondence")
        for name in ("raw_scores", "reciprocal_error_px", "confidence"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != (count,):
                raise ValueError(f"{name} must have one value per correspondence")
        if self.covariance is not None and np.asarray(self.covariance).shape not in {
            (count, 2, 2),
            (count, 4, 4),
        }:
            raise ValueError("covariance must have shape (N,2,2) or (N,4,4)")
        if min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive native pixel dimensions")

    @property
    def pairs_xyxy(self) -> np.ndarray:
        """Return columns ``x_ref,y_ref,x_target,y_target`` without filtering rows."""
        return np.column_stack((self.source_xy, self.target_xy)).astype(np.float64, copy=False)

    @classmethod
    def from_match_field(cls, field: Any) -> CorrespondenceSet:
        """Adapt directly measured pairs from a native ``MatchField``.

        DeepMatching and CPM fields may carry an actual target coordinate for
        each accepted work-grid match. HORG's usual output is already a local
        affine estimate; use ``match_horg(..., include_correspondences=True)``
        to request its underlying descriptor pairs instead of treating those
        affine estimates as independent measurements.
        """
        metadata = getattr(field, "metadata", {})
        image_size = (int(field.width), int(field.height))
        matcher_id = str(field.algorithm)
        raw_pairs = metadata.get("raw_correspondences_xyxy")
        if raw_pairs is not None:
            pairs = np.asarray(raw_pairs, dtype=np.float64).reshape(-1, 4)
            ids = tuple(metadata.get("raw_correspondence_ids", range(len(pairs))))
            scores = metadata.get("raw_match_scores")
            accepted = metadata.get("raw_correspondence_accepted")
            reasons = metadata.get("raw_correspondence_rejection_reasons")
            return cls(
                source_xy=pairs[:, :2],
                target_xy=pairs[:, 2:],
                image_size=image_size,
                matcher_id=matcher_id,
                match_ids=ids,
                accepted=None if accepted is None else np.asarray(accepted, dtype=bool),
                raw_scores=None if scores is None else np.asarray(scores, dtype=np.float64),
                rejection_reasons=None if reasons is None else tuple(reasons),
                coordinate_mapping={"frame": "native image pixels", "source": "HORG detector", "target": "HORG detector"},
            )

        matches = list(field.matches)
        source = np.asarray([[match.x, match.y] for match in matches], dtype=np.float64).reshape(-1, 2)
        target = np.full_like(source, np.nan)
        accepted = np.zeros(len(matches), dtype=bool)
        kinds = []
        reasons = []
        scores = []
        for index, match in enumerate(matches):
            tx, ty = getattr(match, "target_x", None), getattr(match, "target_y", None)
            has_direct_target = tx is not None and ty is not None
            if has_direct_target:
                target[index] = [tx, ty]
                kinds.append("direct_measured")
                accepted[index] = bool(match.usable)
                reasons.append(None if accepted[index] else "matcher_rejected")
            else:
                kinds.append("unavailable")
                reasons.append("no_direct_target_coordinate")
            scores.append(float(getattr(match, "score", np.nan)))
        return cls(
            source_xy=source,
            target_xy=target,
            image_size=image_size,
            matcher_id=matcher_id,
            match_ids=tuple(range(len(matches))),
            accepted=accepted,
            measurement_kind=tuple(kinds),
            raw_scores=np.asarray(scores, dtype=np.float64),
            rejection_reasons=tuple(reasons),
            coordinate_mapping={"frame": "native image pixels", "method_scale": getattr(field, "scale", 1)},
        )

    @classmethod
    def from_preprocessed_pairs(
        cls,
        source_xy: np.ndarray,
        target_xy: np.ndarray,
        image_size: tuple[int, int],
        matcher_id: str,
        *,
        source_scale_xy: tuple[float, float],
        target_scale_xy: tuple[float, float],
        source_padding_xy: tuple[float, float] = (0.0, 0.0),
        target_padding_xy: tuple[float, float] = (0.0, 0.0),
        source_crop_origin_xy: tuple[float, float] = (0.0, 0.0),
        target_crop_origin_xy: tuple[float, float] = (0.0, 0.0),
        **metadata: Any,
    ) -> CorrespondenceSet:
        """Map anisotropically resized/padded/cropped pairs to native pixels.

        Preprocessed coordinates obey ``p = scale * (native - crop) + padding``
        independently for source and target images. All mapping vectors use
        ``(x, y)`` order.
        """
        pairs = (source_xy, target_xy)
        scales = (source_scale_xy, target_scale_xy)
        paddings = (source_padding_xy, target_padding_xy)
        crops = (source_crop_origin_xy, target_crop_origin_xy)
        mapped = []
        for points, scale_xy, padding_xy, crop_xy in zip(pairs, scales, paddings, crops, strict=True):
            scale_array = np.asarray(scale_xy, dtype=np.float64)
            padding_array = np.asarray(padding_xy, dtype=np.float64)
            crop_array = np.asarray(crop_xy, dtype=np.float64)
            if scale_array.shape != (2,) or np.any(scale_array <= 0) or not np.isfinite(scale_array).all():
                raise ValueError("each preprocessing scale must contain two finite positive factors")
            if (
                padding_array.shape != (2,)
                or crop_array.shape != (2,)
                or not np.isfinite(padding_array).all()
                or not np.isfinite(crop_array).all()
            ):
                raise ValueError("padding and crop origin must contain two finite (x, y) values")
            points_array = np.asarray(points, dtype=np.float64)
            mapped.append((points_array - padding_array) / scale_array + crop_array)
        coordinate_mapping = {
            "input_frame": "preprocessed_pixels",
            "source_scale_xy": list(source_scale_xy),
            "target_scale_xy": list(target_scale_xy),
            "source_padding_xy": list(source_padding_xy),
            "target_padding_xy": list(target_padding_xy),
            "source_crop_origin_xy": list(source_crop_origin_xy),
            "target_crop_origin_xy": list(target_crop_origin_xy),
            **metadata.pop("coordinate_mapping", {}),
        }
        return cls(
            source_xy=mapped[0],
            target_xy=mapped[1],
            image_size=image_size,
            matcher_id=matcher_id,
            coordinate_frame="native_pixels",
            coordinate_mapping=coordinate_mapping,
            **metadata,
        )


@dataclass(slots=True)
class PatchInitialization:
    """Six affine IC-GN parameters for one requested patch."""

    seed_id: int
    x: float
    y: float
    parameters: tuple[float, float, float, float, float, float] | None
    status: str
    support_rms_px: float | None
    support_max_px: float | None


@dataclass(slots=True)
class ProjectionModel:
    """Fitted displacement model with provenance and patch projection."""

    basis: str
    effective_basis: str
    origin: np.ndarray
    scale: float
    affine: np.ndarray
    centers: np.ndarray
    coefficients: np.ndarray
    ell: float | None
    ridge: float
    robust_weights: np.ndarray
    residuals_px: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    _device: Any | None = field(default=None, repr=False, compare=False)
    correspondence_ids: tuple[str | int, ...] = ()
    source_matcher: str | None = None
    input_row_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def evaluate(self, points_xy: np.ndarray, *, device: Any | None = None) -> np.ndarray:
        """Evaluate displacement at native-pixel coordinates.

        When this model was fitted through :class:`WGPUMatching`, evaluation
        uses that same WebGPU device unless another device is supplied.
        """
        points = np.asarray(points_xy, dtype=np.float64)
        target_device = device or self._device
        if target_device is not None:
            return _evaluate_field_webgpu(self, points, target_device)
        z = (points - self.origin) / self.scale
        design = np.column_stack((np.ones(len(z)), z))
        result = design @ self.affine
        if self.effective_basis == "gaussian_rbf" and len(self.centers):
            delta = z[:, None, :] - self.centers[None, :, :]
            kernel = np.exp(-np.sum(delta * delta, axis=2) / (2.0 * float(self.ell) ** 2))
            result += kernel @ self.coefficients
        return result * self.scale

    def to_dict(self) -> dict[str, Any]:
        """Serialize coefficients, input provenance, and per-pair diagnostics."""
        return {
            "basis": self.basis,
            "effective_basis": self.effective_basis,
            "origin_xy": self.origin.tolist(),
            "coordinate_scale_px": self.scale,
            "affine_coefficients_normalized": self.affine.tolist(),
            "rbf_centers_normalized": self.centers.tolist(),
            "rbf_coefficients_normalized": self.coefficients.tolist(),
            "ell_normalized": self.ell,
            "ridge": self.ridge,
            "source_matcher": self.source_matcher,
            "correspondence_ids": list(self.correspondence_ids),
            "input_row_indices": self.input_row_indices.tolist(),
            "robust_weights": self.robust_weights.tolist(),
            "residuals_px": self.residuals_px.tolist(),
            "diagnostics": self.diagnostics,
        }

    def initialize_patches(
        self,
        seed_points: np.ndarray,
        patch_size: int | tuple[int, int] = 128,
        *,
        projection: Literal["support_fit", "taylor"] = "support_fit",
        samples_per_axis: int = 5,
        seed_ids: Sequence[int] | None = None,
    ) -> list[PatchInitialization]:
        seeds = np.asarray(seed_points, dtype=np.float64).reshape(-1, 2)
        if isinstance(patch_size, int):
            width = height = patch_size
        else:
            width, height = patch_size
        if width < 2 or height < 2:
            raise ValueError("patch dimensions must be at least two pixels")
        if projection not in {"support_fit", "taylor"}:
            raise ValueError("projection must be 'support_fit' or 'taylor'")
        if samples_per_axis < 2:
            raise ValueError("samples_per_axis must be at least two")
        if not np.isfinite(seeds).all():
            raise ValueError("seed points must be finite")
        identifiers = tuple(range(len(seeds))) if seed_ids is None else tuple(seed_ids)
        if len(identifiers) != len(seeds):
            raise ValueError("seed_ids must have one id per requested center")
        if self._device is not None:
            return _project_patches_webgpu(
                self,
                seeds,
                width,
                height,
                projection=projection,
                samples_per_axis=samples_per_axis,
                seed_ids=identifiers,
            )
        offset_x = np.linspace(-(width - 1) / 2, (width - 1) / 2, samples_per_axis)
        offset_y = np.linspace(-(height - 1) / 2, (height - 1) / 2, samples_per_axis)
        offsets = np.asarray([(x, y) for y in offset_y for x in offset_x])
        output = []
        for seed_id, center in zip(identifiers, seeds, strict=True):
            center_displacement = self.evaluate(center[None, :])[0]
            if projection == "taylor":
                eps = max(1e-3, self.scale * 1e-4)
                dx = (
                    self.evaluate((center + (eps, 0))[None, :])[0]
                    - self.evaluate((center - (eps, 0))[None, :])[0]
                ) / (2 * eps)
                dy = (
                    self.evaluate((center + (0, eps))[None, :])[0]
                    - self.evaluate((center - (0, eps))[None, :])[0]
                ) / (2 * eps)
                gradient = np.column_stack((dx, dy))
            else:
                predicted = self.evaluate(center[None, :] + offsets) - center_displacement
                gradient = np.linalg.lstsq(offsets, predicted, rcond=None)[0]
            fitted = offsets @ gradient
            actual = self.evaluate(center[None, :] + offsets) - center_displacement
            errors = np.linalg.norm(actual - fitted, axis=1)
            params = (
                float(center_displacement[0]),
                float(center_displacement[1]),
                float(gradient[0, 0]),
                float(gradient[1, 0]),
                float(gradient[0, 1]),
                float(gradient[1, 1]),
            )
            output.append(
                PatchInitialization(
                    seed_id,
                    float(center[0]),
                    float(center[1]),
                    params,
                    "estimated_from_field",
                    float(np.sqrt(np.mean(errors**2))),
                    float(np.max(errors)),
                )
            )
        return output


def _run_projection_evaluation(
    model: ProjectionModel,
    points: np.ndarray,
    device: Any,
    *,
    patch_size: tuple[int, int] | None = None,
    projection: Literal["support_fit", "taylor"] = "support_fit",
    samples_per_axis: int = 5,
) -> np.ndarray:
    """Evaluate points or fit patchwise affine approximations in one GPU dispatch."""
    if len(points) == 0:
        return np.empty((0, 8), dtype=np.float64)
    width, height = patch_size or (1, 1)
    mode = int(projection == "taylor")
    center_data = np.zeros((max(1, len(model.centers)), 2), dtype=np.float32)
    coeff_data = np.zeros_like(center_data)
    if len(model.centers):
        center_data[: len(model.centers)] = model.centers.astype(np.float32)
        coeff_data[: len(model.centers)] = model.coefficients.astype(np.float32)
    params = struct.pack(
        "<4I8f",
        len(points),
        len(model.centers),
        samples_per_axis,
        mode,
        float(model.origin[0]),
        float(model.origin[1]),
        float(model.scale),
        float(model.ell or 1.0),
        (width - 1) / 2,
        (height - 1) / 2,
        0.0,
        0.0,
    )
    started = time.perf_counter()
    with ResourceScope() as scope:
        point_b = upload_buffer(
            device, scope, np.asarray(points, dtype=np.float32),
            wgpu.BufferUsage.STORAGE, "projection evaluation points",
        )
        affine_b = upload_buffer(
            device, scope, model.affine.astype(np.float32),
            wgpu.BufferUsage.STORAGE, "projection affine coefficients",
        )
        center_b = upload_buffer(
            device, scope, center_data, wgpu.BufferUsage.STORAGE, "projection RBF centers"
        )
        coefficient_b = upload_buffer(
            device, scope, coeff_data,
            wgpu.BufferUsage.STORAGE, "projection RBF coefficients",
        )
        output_b = create_buffer(
            device,
            scope,
            len(points) * 8 * np.dtype(np.float32).itemsize,
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            "projection patch results",
        )
        params_b = upload_buffer(
            device,
            scope,
            np.frombuffer(params, dtype=np.uint8),
            wgpu.BufferUsage.UNIFORM,
            "projection evaluation parameters",
        )
        entry = "evaluate_points" if patch_size is None else "project_patches"
        pipeline = _cached_pipelines(
            device,
            "projection_evaluate",
            "projection-evaluate.wgsl",
            ("evaluate_points", "project_patches"),
        )
        pipeline = pipeline[entry]
        group = make_bind_group(
            device,
            pipeline,
            [
                {"binding": 0, "resource": {"buffer": point_b}},
                {"binding": 1, "resource": {"buffer": affine_b}},
                {"binding": 2, "resource": {"buffer": center_b}},
                {"binding": 3, "resource": {"buffer": coefficient_b}},
                {"binding": 4, "resource": {"buffer": output_b}},
                {"binding": 5, "resource": {"buffer": params_b}},
            ],
        )
        encoder = device.create_command_encoder(label="projection evaluation")
        compute = encoder.begin_compute_pass(label=f"projection {entry}")
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups((len(points) + 63) // 64)
        compute.end()
        device.queue.submit([encoder.finish()])
        output = read_array(device, output_b, np.dtype(np.float32), len(points) * 8)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    model.diagnostics["last_field_evaluation_wall_ms"] = elapsed_ms
    model.diagnostics["last_field_evaluation_host_sync_count"] = 1
    return output.reshape(len(points), 8).astype(np.float64)


def _evaluate_field_webgpu(
    model: ProjectionModel, points: np.ndarray, device: Any
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points_xy must be a finite Nx2 array")
    return _run_projection_evaluation(model, points, device)[:, :2]


def _project_patches_webgpu(
    model: ProjectionModel,
    seeds: np.ndarray,
    width: int,
    height: int,
    *,
    projection: Literal["support_fit", "taylor"],
    samples_per_axis: int,
    seed_ids: Sequence[int],
) -> list[PatchInitialization]:
    rows = _run_projection_evaluation(
        model,
        seeds,
        model._device,
        patch_size=(width, height),
        projection=projection,
        samples_per_axis=samples_per_axis,
    )
    output = []
    for seed_id, center, row in zip(seed_ids, seeds, rows, strict=True):
        displacement = row[:2]
        gradient = row[4:8]
        params = (
            float(displacement[0]),
            float(displacement[1]),
            float(gradient[0]),
            float(gradient[1]),
            float(gradient[2]),
            float(gradient[3]),
        )
        valid = bool(np.isfinite(row).all())
        output.append(
            PatchInitialization(
                seed_id,
                float(center[0]),
                float(center[1]),
                params if valid else None,
                "estimated_from_field" if valid else "non_finite_gpu_result",
                float(row[2]) if valid else None,
                float(row[3]) if valid else None,
            )
        )
    return output


def _density_weights(
    x: np.ndarray, extent: tuple[float, float], cells: tuple[int, int]
) -> np.ndarray:
    ij = np.floor(x / np.asarray(extent) * np.asarray(cells)).astype(int)
    ij = np.clip(ij, 0, np.asarray(cells) - 1)
    keys, inverse, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    del keys
    weights = 1.0 / counts[inverse]
    return weights / np.mean(weights)


def _farthest_centers(x: np.ndarray, count: int) -> np.ndarray:
    if len(x) <= count:
        return x.copy()
    chosen = [int(np.argmin(np.sum((x - np.mean(x, axis=0)) ** 2, axis=1)))]
    distance = np.sum((x - x[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        idx = int(np.argmax(distance))
        chosen.append(idx)
        distance = np.minimum(distance, np.sum((x - x[idx]) ** 2, axis=1))
    return x[np.asarray(chosen)]


def _fit_model(
    x: np.ndarray,
    d: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    ell: float | None,
    ridge: float,
    options: ProjectionOptions,
    minimum_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    affine_design = np.column_stack((np.ones(len(x)), x))
    kernel = (
        np.exp(-np.sum((x[:, None] - centers[None]) ** 2, axis=2) / (2 * float(ell) ** 2))
        if len(centers)
        else np.empty((len(x), 0))
    )
    design = np.column_stack((affine_design, kernel))
    robust = np.ones(len(x))
    affine = np.zeros((3, 2))
    coeff = np.zeros((len(centers), 2))
    for _ in range(options.max_irls_iterations):
        w = np.maximum(weights * robust, 1e-12)
        lhs = design.T @ (w[:, None] * design)
        penalty = np.zeros(lhs.shape[0])
        penalty[3:] = ridge * np.sum(w)
        lhs.flat[:: lhs.shape[0] + 1] += penalty
        rhs = design.T @ (w[:, None] * d)
        beta = np.linalg.lstsq(lhs, rhs, rcond=1e-12)[0]
        residual = np.linalg.norm(d - design @ beta, axis=1)
        scale_floor = options.minimum_scale_px if minimum_scale is None else minimum_scale
        scale = max(scale_floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        cutoff = options.huber_delta * scale
        target = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
        updated = (1.0 - options.irls_damping) * target + options.irls_damping * robust
        if np.max(np.abs(updated - robust)) < options.irls_tolerance:
            robust = updated
            break
        robust = updated
    affine = beta[:3]
    coeff = beta[3:]
    return affine, coeff, robust, scale


def fit_initialization_field(
    correspondences: np.ndarray | CorrespondenceSet,
    image_size: tuple[int, int],
    *,
    options: ProjectionOptions | None = None,
    device: Any | None = None,
) -> ProjectionModel:
    """Fit a shared affine or automatically selected Gaussian-RBF field.

    Input columns are ``x_ref, y_ref, x_target, y_target`` in native pixels,
    or a :class:`CorrespondenceSet` carrying the same coordinates and provenance.
    """
    cfg = options or ProjectionOptions()
    if cfg.basis not in {"affine", "gaussian_rbf"}:
        raise ValueError("basis must be 'affine' or 'gaussian_rbf'")
    if (
        cfg.max_irls_iterations < 1
        or cfg.irls_tolerance <= 0
        or cfg.huber_delta <= 0
        or cfg.minimum_scale_px <= 0
    ):
        raise ValueError("IRLS iterations, tolerance, Huber delta, and minimum scale must be positive")
    if not 0.0 <= cfg.irls_damping < 1.0:
        raise ValueError("irls_damping must be in [0, 1)")
    if len(cfg.density_cells) != 2 or min(cfg.density_cells) < 1:
        raise ValueError("density_cells must contain two positive integers")
    if any(count < 1 or count > 64 for count in cfg.center_counts):
        raise ValueError("RBF center_counts must be between 1 and 64")
    if any(width_factor <= 0 for width_factor in cfg.width_factors):
        raise ValueError("RBF width_factors must be positive")
    if any(ridge < 0 for ridge in cfg.ridge_values):
        raise ValueError("RBF ridge values must be non-negative")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain positive (width, height) dimensions")
    width, height = image_size
    if isinstance(correspondences, CorrespondenceSet):
        if correspondences.image_size != image_size:
            raise ValueError("CorrespondenceSet dimensions must match image_size in native pixels")
        if correspondences.coordinate_frame != "native_pixels":
            raise ValueError("correspondences must be mapped to native_pixels before fitting")
        raw_pairs = correspondences.pairs_xyxy
        input_ids = tuple(correspondences.match_ids or ())
        matcher_id = correspondences.matcher_id
        accepted_rows = (
            np.ones(len(raw_pairs), dtype=bool)
            if correspondences.accepted is None
            else np.asarray(correspondences.accepted, dtype=bool)
        )
        kinds = np.asarray(correspondences.measurement_kind, dtype=object)
        score_values = correspondences.raw_scores
        reciprocal_values = correspondences.reciprocal_error_px
        rejection_reasons = correspondences.rejection_reasons
        coordinate_mapping = correspondences.coordinate_mapping
    else:
        raw_pairs = np.asarray(correspondences, dtype=np.float64)
        if raw_pairs.ndim != 2 or raw_pairs.shape[1] != 4:
            raise ValueError("correspondences must be an Nx4 array: x_ref,y_ref,x_target,y_target")
        input_ids = tuple(range(len(raw_pairs)))
        matcher_id = None
        accepted_rows = np.ones(len(raw_pairs), dtype=bool)
        kinds = np.full(len(raw_pairs), "direct_measured", dtype=object)
        score_values = reciprocal_values = None
        rejection_reasons = None
        coordinate_mapping = {}
    finite = np.isfinite(raw_pairs).all(axis=1)
    direct = kinds == "direct_measured"
    in_bounds = (
        (raw_pairs[:, 0] >= 0)
        & (raw_pairs[:, 0] < width)
        & (raw_pairs[:, 1] >= 0)
        & (raw_pairs[:, 1] < height)
        & (raw_pairs[:, 2] >= 0)
        & (raw_pairs[:, 2] < width)
        & (raw_pairs[:, 3] >= 0)
        & (raw_pairs[:, 3] < height)
    )
    eligible = finite & accepted_rows & direct & in_bounds
    eligible_rows = np.flatnonzero(eligible)
    unique_pairs, first_occurrences = np.unique(
        raw_pairs[eligible], axis=0, return_index=True
    )
    selected_rows = eligible_rows[first_occurrences]
    pairs = unique_pairs
    if len(pairs) < 3:
        raise ValueError("at least three finite in-image correspondences are required")
    x = pairs[:, :2]
    disp = pairs[:, 2:] - x
    origin = np.mean(x, axis=0)
    scale = float(max(width, height))
    z = (x - origin) / scale
    dn = disp / scale
    density = _density_weights(x, (width, height), cfg.density_cells)
    host_selection_started = time.perf_counter()
    best = None
    candidate_log = []
    selection_reason = "affine_requested"
    affine, coeff, robust, residual_scale = _fit_model(
        z, dn, density, np.empty((0, 2)), None, 0.0, cfg,
        minimum_scale=cfg.minimum_scale_px / scale,
    )
    best = (affine, coeff, np.empty((0, 2)), None, 0.0, robust, "affine")
    affine_fallback = (affine.copy(), coeff.copy(), np.empty((0, 2)), None, 0.0, robust.copy(), "affine")
    affine_residual_scale = residual_scale
    scores: list[tuple[float, int, float, float, np.ndarray, list[float], np.ndarray]] = []
    if cfg.basis == "gaussian_rbf" and len(z) >= 12:
        # Spatial block CV; candidate choices are bounded and deterministic.
        if cfg.folds < 2:
            raise ValueError("Gaussian-RBF model selection requires at least two folds")
        gx = np.clip((x[:, 0] / width * cfg.folds).astype(int), 0, cfg.folds - 1)
        gy = np.clip((x[:, 1] / height * cfg.folds).astype(int), 0, cfg.folds - 1)
        fold = (gx + 2 * gy) % cfg.folds
        fold_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for f in range(cfg.folds):
            train = fold != f
            test = ~train
            if train.sum() < 6 or not test.any():
                continue
            af, cf, _, _ = _fit_model(
                z[train], dn[train], density[train], np.empty((0, 2)), None, 0.0, cfg,
                minimum_scale=cfg.minimum_scale_px / scale,
            )
            prediction = np.column_stack((np.ones(test.sum()), z[test])) @ af
            fold_errors = np.linalg.norm(prediction - dn[test], axis=1) * scale
            fold_rows.append((train, test, fold_errors))
        affine_errors = np.concatenate([row[2] for row in fold_rows]) if fold_rows else np.empty(0)
        affine_fold_scores = [float(np.median(row[2])) for row in fold_rows]
        affine_score = float(np.median(affine_errors)) if len(affine_errors) else float("inf")
        candidate_log.append(
            {
                "basis": "affine",
                "centers": 0,
                "cv_median_px": affine_score,
                "cv_p95_px": float(np.percentile(affine_errors, 95)) if len(affine_errors) else None,
                "fold_medians_px": affine_fold_scores,
            }
        )
        for count in cfg.center_counts:
            ncenter = min(int(count), max(1, len(z) // 3))
            if ncenter < 1:
                continue
            center_ids = _farthest_centers(z, ncenter)
            spacing = (
                float(
                    np.median(
                        np.linalg.norm(center_ids[:, None] - center_ids[None], axis=2)[
                            np.triu_indices(ncenter, 1)
                        ]
                    )
                )
                if ncenter > 1
                else 0.25
            )
            for wf in cfg.width_factors:
                ell = max(1e-3, spacing * wf)
                for ridge in cfg.ridge_values:
                    fold_errors: list[float] = []
                    fold_medians: list[float] = []
                    for train, test, _ in fold_rows:
                        if train.sum() < max(6, ncenter // 2) or not test.any():
                            continue
                        centers = _farthest_centers(
                            z[train],
                            min(ncenter, max(1, int(train.sum() // 3))),
                        )
                        af, cf, _, _ = _fit_model(
                            z[train], dn[train], density[train], centers, ell, ridge, cfg,
                            minimum_scale=cfg.minimum_scale_px / scale,
                        )
                        kd = np.exp(
                            -np.sum((z[test, None] - centers[None]) ** 2, axis=2) / (2 * ell * ell)
                        )
                        pred = np.column_stack((np.ones(test.sum()), z[test])) @ af + kd @ cf
                        errors = np.linalg.norm(pred - dn[test], axis=1) * scale
                        fold_errors.extend(errors.tolist())
                        fold_medians.append(float(np.median(errors)))
                    error_array = np.asarray(fold_errors, dtype=np.float64)
                    score = float(np.median(error_array)) if len(error_array) else float("inf")
                    candidate_log.append(
                        {
                            "basis": "gaussian_rbf",
                            "centers": ncenter,
                            "ell_normalized": ell,
                            "ridge": ridge,
                            "cv_median_px": score,
                            "cv_p95_px": float(np.percentile(error_array, 95)) if len(error_array) else None,
                            "fold_medians_px": fold_medians,
                        }
                    )
                    scores.append((score, ncenter, ell, float(ridge), center_ids, fold_medians, error_array))
        if scores:
            best_rbf = min(scores, key=lambda row: (row[0], row[1], row[2], row[3]))
            best_folds = best_rbf[5]
            standard_error = (
                float(np.std(best_folds, ddof=1) / np.sqrt(len(best_folds)))
                if len(best_folds) > 1
                else 0.0
            )
            tolerance = max(standard_error, 1e-6)
            eligible_affine = affine_score <= best_rbf[0] + tolerance
            if eligible_affine:
                selection_reason = "affine_within_one_standard_error_of_best_rbf"
            else:
                score, ncenter, ell, ridge, selected, _, _ = best_rbf
                affine, coeff, robust, residual_scale = _fit_model(
                    z, dn, density, selected, ell, ridge, cfg,
                    minimum_scale=cfg.minimum_scale_px / scale,
                )
                best = (affine, coeff, selected, ell, ridge, robust, "gaussian_rbf")
                selection_reason = "gaussian_rbf_lowest_spatial_cv_error"
        else:
            selection_reason = "selection_unavailable_affine_fallback"
    elif cfg.basis == "gaussian_rbf":
        selection_reason = "insufficient_correspondences_affine_fallback"
    affine, coeff, centers, ell, ridge, robust, effective = best
    host_selection_ms = (time.perf_counter() - host_selection_started) * 1000.0
    backend = "numpy_reference"
    gpu_fit_ms = None
    gpu_syncs = 0
    irls_converged = None
    irls_final_delta = None
    gpu_residuals = None
    gpu_solver_failures = []
    if device is not None:
        # Candidate ordering is CPU float64 CV; solve failures trigger explicit
        # attempts of the next scored RBF candidate, then an affine fallback.
        if effective == "gaussian_rbf":
            candidate_rows = sorted(scores, key=lambda row: (row[0], row[1], row[2], row[3]))
        else:
            candidate_rows = [(0.0, 0, 1.0, 0.0, np.empty((0, 2)), [], np.empty(0))]
        gpu_attempt_ms = 0.0
        selected_successfully = False
        for candidate_index, candidate in enumerate(candidate_rows):
            _, ncenter, candidate_ell, candidate_ridge, candidate_centers, _, _ = candidate
            if effective == "gaussian_rbf":
                if candidate_index == 0:
                    candidate_scale = residual_scale
                else:
                    pilot_started = time.perf_counter()
                    _, _, _, candidate_scale = _fit_model(
                        z,
                        dn,
                        density,
                        candidate_centers,
                        candidate_ell,
                        candidate_ridge,
                        cfg,
                        minimum_scale=cfg.minimum_scale_px / scale,
                    )
                    host_selection_ms += (time.perf_counter() - pilot_started) * 1000.0
                candidate_basis = "gaussian_rbf"
            else:
                candidate_scale = affine_residual_scale
                candidate_centers = np.empty((0, 2))
                candidate_ell = 1.0
                candidate_ridge = 0.0
                candidate_basis = "affine"
            if candidate_index > 0 and candidate_basis == "gaussian_rbf":
                residual_scale = candidate_scale
            attempt_started = time.perf_counter()
            try:
                fit_result = _fit_model_webgpu(
                    device,
                    z,
                    dn,
                    density,
                    candidate_centers,
                    float(candidate_ell),
                    float(candidate_ridge),
                    cfg.huber_delta * max(cfg.minimum_scale_px / scale, candidate_scale),
                    cfg.max_irls_iterations,
                    cfg.irls_tolerance,
                    cfg.irls_damping,
                )
            except RuntimeError as error:
                gpu_attempt_ms += (time.perf_counter() - attempt_started) * 1000.0
                gpu_syncs += 1
                gpu_solver_failures.append(
                    {
                        "basis": candidate_basis,
                        "centers": int(ncenter),
                        "ell_normalized": float(candidate_ell),
                        "ridge": float(candidate_ridge),
                        "reason": str(error),
                    }
                )
                continue
            (
                affine,
                coeff,
                robust,
                gpu_residuals,
                successful_fit_ms,
                successful_syncs,
                irls_converged,
                irls_final_delta,
            ) = fit_result
            gpu_attempt_ms += successful_fit_ms
            gpu_syncs += successful_syncs
            centers = candidate_centers
            ell = float(candidate_ell) if candidate_basis == "gaussian_rbf" else None
            ridge = float(candidate_ridge)
            effective = candidate_basis
            selected_successfully = True
            if candidate_index > 0:
                selection_reason = "gpu_solver_fallback_to_next_spatial_cv_candidate"
            break
        if not selected_successfully and effective == "gaussian_rbf":
            affine, coeff, centers, ell, ridge, robust, effective = affine_fallback
            candidate_scale = affine_residual_scale
            residual_scale = affine_residual_scale
            attempt_started = time.perf_counter()
            try:
                (
                    affine,
                    coeff,
                    robust,
                    gpu_residuals,
                    successful_fit_ms,
                    successful_syncs,
                    irls_converged,
                    irls_final_delta,
                ) = _fit_model_webgpu(
                    device,
                    z,
                    dn,
                    density,
                    centers,
                    1.0,
                    0.0,
                    cfg.huber_delta * max(cfg.minimum_scale_px / scale, candidate_scale),
                    cfg.max_irls_iterations,
                    cfg.irls_tolerance,
                    cfg.irls_damping,
                )
                gpu_attempt_ms += successful_fit_ms
                gpu_syncs += successful_syncs
                selected_successfully = True
                selection_reason = "gpu_rbf_solve_failed_affine_fallback"
            except RuntimeError as error:
                gpu_attempt_ms += (time.perf_counter() - attempt_started) * 1000.0
                gpu_syncs += 1
                gpu_solver_failures.append(
                    {"basis": "affine", "centers": 0, "reason": str(error)}
                )
                raise RuntimeError("WebGPU affine and Gaussian-RBF projection fallbacks failed") from error
        gpu_fit_ms = gpu_attempt_ms
        backend = (
            "webgpu_irls_and_patch_projection_cpu_candidate_selection"
            if cfg.basis == "gaussian_rbf"
            else "webgpu_irls_and_patch_projection_cpu_scale_pilot"
        )
    design = np.column_stack((np.ones(len(z)), z))
    prediction = design @ affine
    if effective == "gaussian_rbf":
        prediction += (
            np.exp(-np.sum((z[:, None] - centers[None]) ** 2, axis=2) / (2 * float(ell) ** 2))
            @ coeff
        )
    residual = (
        gpu_residuals * scale
        if gpu_residuals is not None
        else np.linalg.norm((prediction - dn) * scale, axis=1)
    )
    excluded_rows = []
    selected_row_set = set(selected_rows.tolist())
    source_groups: dict[tuple[float, float], list[int]] = {}
    for index in eligible_rows:
        source_key = tuple(raw_pairs[index, :2].tolist())
        source_groups.setdefault(source_key, []).append(int(index))
    source_conflicts = []
    for source_key, row_ids in source_groups.items():
        targets = {tuple(raw_pairs[index, 2:].tolist()) for index in row_ids}
        if len(targets) > 1:
            source_conflicts.append(
                {
                    "source_xy": list(source_key),
                    "rows": row_ids,
                    "match_ids": [input_ids[index] for index in row_ids],
                    "distinct_target_count": len(targets),
                }
            )
    for index in range(len(raw_pairs)):
        reason = None
        if not accepted_rows[index]:
            reason = (
                rejection_reasons[index]
                if rejection_reasons and rejection_reasons[index]
                else "matcher_rejected"
            )
        elif not direct[index]:
            reason = f"measurement_kind:{kinds[index]}"
        elif not finite[index]:
            reason = "non_finite_coordinate"
        elif not in_bounds[index]:
            reason = "outside_native_image"
        elif index not in selected_row_set:
            reason = "duplicate_pair"
        if reason is not None:
            excluded_rows.append({"row": index, "match_id": input_ids[index], "reason": reason})
    retained_ids = tuple(input_ids[index] for index in selected_rows)
    retained_scores = (
        np.asarray(score_values)[selected_rows].tolist() if score_values is not None else None
    )
    retained_reciprocal = (
        np.asarray(reciprocal_values)[selected_rows].tolist()
        if reciprocal_values is not None
        else None
    )
    diagnostics = {
        "input_count": len(raw_pairs),
        "eligible_direct_count": int(eligible.sum()),
        "retained_count": len(pairs),
        "duplicate_count": int(eligible.sum() - len(pairs)),
        "incompatible_duplicate_sources": source_conflicts,
        "excluded_rows": excluded_rows,
        "retained_input_rows": selected_rows.tolist(),
        "retained_match_ids": list(retained_ids),
        "retained_raw_scores": retained_scores,
        "retained_reciprocal_error_px": retained_reciprocal,
        "source_matcher": matcher_id,
        "coordinate_frame": "native_pixels",
        "coordinate_mapping": coordinate_mapping,
        "candidate_scores": candidate_log,
        "residual_scale_px": residual_scale * scale,
        "backend": backend,
        "candidate_selection_backend": "numpy_float64" if cfg.basis == "gaussian_rbf" else "not_applicable",
        "cpu_selection_and_scale_pilot_ms": host_selection_ms,
        "selection_reason": selection_reason,
        "gpu_final_fit_wall_ms": gpu_fit_ms,
        "gpu_fit_host_sync_count": gpu_syncs,
        "irls_converged": irls_converged,
        "irls_final_max_weight_delta": irls_final_delta,
        "irls_max_iterations": cfg.max_irls_iterations if device is not None else None,
        "irls_damping": cfg.irls_damping,
        "irls_scale_policy": (
            "fixed_scale_from_float64_cpu_pilot" if device is not None else "reestimated_each_cpu_iteration"
        ),
        "solver": (
            "WGSL Cholesky solve on FP32 normal equations"
            if device is not None
            else "NumPy float64 reference solve"
        ),
        "gpu_input_staging": (
            "host arrays uploaded to the matcher WebGPU device" if device is not None else "not_applicable"
        ),
        "gpu_design_buffer_bytes": len(z) * (3 + len(centers)) * 4 if device is not None else 0,
        "selected_center_count": len(centers),
        "selected_ell_normalized": ell,
        "selected_ridge": ridge,
        "gpu_solver_failures": gpu_solver_failures,
    }
    return ProjectionModel(
        basis=cfg.basis,
        effective_basis=effective,
        origin=origin,
        scale=scale,
        affine=affine,
        centers=centers,
        coefficients=coeff,
        ell=ell,
        ridge=ridge,
        robust_weights=robust,
        residuals_px=residual,
        diagnostics=diagnostics,
        _device=device,
        correspondence_ids=retained_ids,
        source_matcher=matcher_id,
        input_row_indices=selected_rows,
    )
