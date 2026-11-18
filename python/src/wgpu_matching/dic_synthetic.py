"""Deterministic speckle image pairs and analytic motion fields for detector work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SyntheticSpecklePair:
    """A speckle pair, forward displacement, visibility, and reproducible metadata."""

    reference: np.ndarray
    target: np.ndarray
    flow_xy: np.ndarray
    visible: np.ndarray
    metadata: dict[str, Any]
    target_visible: np.ndarray | None = None


def analytic_truth_at_points(metadata: dict[str, Any], points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the generator's forward displacement and gradient at float centers.

    Validity follows the generator's center-visibility convention, including
    the discontinuity exclusion margin. It does not certify a full subset.
    """
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    width, height = int(metadata["width"]), int(metadata["height"])
    matrix = np.asarray(metadata["affine_matrix"], dtype=np.float64).reshape(2, 2)
    translation = np.asarray(metadata["translation_px"], dtype=np.float64)
    amplitude = np.asarray(metadata["heterogeneity_amplitude_px"], dtype=np.float64)
    center = np.array([width / 2, height / 2], dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    px, py = 2 * np.pi * x / width, 2 * np.pi * y / height
    hx = amplitude[0] * np.sin(py) * np.cos(px)
    hy = amplitude[1] * np.sin(px) * np.cos(py)
    displacement = (xy - center) @ (matrix - np.eye(2)).T + translation
    displacement += np.column_stack((hx, hy))
    gradient = np.broadcast_to(matrix - np.eye(2), (len(xy), 2, 2)).copy()
    gradient[:, 0, 0] -= amplitude[0] * (2 * np.pi / width) * np.sin(py) * np.sin(px)
    gradient[:, 0, 1] += amplitude[0] * (2 * np.pi / height) * np.cos(py) * np.cos(px)
    gradient[:, 1, 0] += amplitude[1] * (2 * np.pi / width) * np.cos(px) * np.cos(py)
    gradient[:, 1, 1] -= amplitude[1] * (2 * np.pi / height) * np.sin(px) * np.sin(py)
    if metadata["motion_mode"] == "shear_band":
        band_width = float(metadata["shear_band_width_px"])
        z = (y - float(metadata["shear_band_center_y_px"])) / band_width
        peak = float(metadata["shear_band_peak_gradient"])
        displacement[:, 0] += peak * band_width * np.tanh(z)
        gradient[:, 0, 1] += peak * (1 - np.tanh(z) ** 2)
    if metadata["motion_mode"] == "discontinuous":
        boundary = float(metadata["discontinuity_boundary_px"])
        displacement[:, 0] += float(metadata["discontinuity_gap_px"]) * (x >= boundary)
    target = xy + displacement
    visible = ((target[:, 0] >= 0) & (target[:, 0] <= width - 1)
               & (target[:, 1] >= 0) & (target[:, 1] <= height - 1))
    if metadata["motion_mode"] == "discontinuous":
        margin = max(8., 2 * float(metadata["characteristic_size_px"]))
        visible &= np.abs(x - float(metadata["discontinuity_boundary_px"])) > margin
    states = np.column_stack((displacement[:, 0], gradient[:, 0, 0], gradient[:, 0, 1],
                              displacement[:, 1], gradient[:, 1, 0], gradient[:, 1, 1]))
    return states, visible


def perturb_target(image, warp_seed, *, gain=1., offset=0., blur_sigma=0., noise_std=0.):
    """Apply deterministic target-only camera perturbations in a fixed order."""
    if gain <= 0 or blur_sigma < 0 or noise_std < 0:
        raise ValueError("invalid target perturbation")
    result = np.asarray(image, dtype=np.float32)
    if blur_sigma:
        result = _gaussian_blur(result, blur_sigma)
    if gain != 1.0 or offset != 0.0:
        result = np.clip(result * gain + offset, 0, 1).astype(np.float32)
    if noise_std:
        noise = np.random.default_rng(warp_seed + 700000001).normal(
            0, noise_std, result.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 1).astype(np.float32)
    return result


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.asarray(image, dtype=np.float32)
    if radius >= 32:
        # For large, scale-consistent synthetic textures, the direct separable
        # loop repeatedly streams the full UHD image once per kernel tap. Use
        # blocked FFT convolution instead. Reflection padding and the same
        # truncated, normalized kernel preserve the direct path's boundary
        # convention and mathematical result while bounding temporary memory.
        for axis in (1, 0):
            lines = np.moveaxis(result, axis, -1)
            filtered = np.empty(lines.shape, dtype=np.float32)
            line_length = lines.shape[-1]
            padded_length = line_length + 2 * radius
            fft_length = 1 << (padded_length + len(kernel) - 2).bit_length()
            kernel_spectrum = np.fft.rfft(kernel, n=fft_length)
            for start in range(0, lines.shape[0], 128):
                block = np.ascontiguousarray(lines[start : start + 128])
                padded = np.pad(block, ((0, 0), (radius, radius)), mode="reflect")
                spectrum = np.fft.rfft(padded, n=fft_length, axis=-1)
                convolved = np.fft.irfft(spectrum * kernel_spectrum, n=fft_length, axis=-1)
                filtered[start : start + len(block)] = convolved[
                    :, 2 * radius : 2 * radius + line_length
                ]
            result = np.moveaxis(filtered, -1, axis)
        return np.asarray(result, dtype=np.float32)
    for axis in (1, 0):
        padded = np.pad(
            result,
            ((0, 0), (radius, radius)) if axis == 1 else ((radius, radius), (0, 0)),
            mode="reflect",
        )
        if axis == 1:
            result = sum(kernel[i] * padded[:, i : i + result.shape[1]] for i in range(len(kernel)))
        else:
            result = sum(kernel[i] * padded[i : i + result.shape[0], :] for i in range(len(kernel)))
    return np.asarray(result, dtype=np.float32)


def generate_speckle(
    seed: int,
    width: int = 256,
    height: int = 256,
    *,
    characteristic_size_px: float | None = None,
    coverage: float | None = None,
    blur_sigma_px: float | None = None,
    softness: float | None = None,
    noise_std: float = 0.025,
    illumination_variation: float = 0.12,
    periodic_strength: float = 0.0,
    periodic_period_px: float = 128.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Create a repeatable multi-scale random-dot texture in float32 [0, 1]."""
    if min(width, height) < 32:
        raise ValueError("Speckle images must be at least 32 pixels wide and tall")
    rng = np.random.default_rng(seed)
    characteristic_size_px = characteristic_size_px or float(rng.uniform(2.0, 7.0))
    coverage = coverage if coverage is not None else float(rng.uniform(0.18, 0.48))
    blur_sigma_px = blur_sigma_px if blur_sigma_px is not None else float(rng.uniform(0.0, 1.2))
    if (characteristic_size_px <= 0 or not 0 < coverage < 1 or blur_sigma_px < 0
            or not 0 <= periodic_strength <= 1 or periodic_period_px <= 0):
        raise ValueError("Invalid speckle size, coverage, or blur")

    impulse = rng.standard_normal((height, width), dtype=np.float32)
    texture_field = _gaussian_blur(impulse, max(0.6, characteristic_size_px / 2.5))
    texture_field -= np.mean(texture_field)
    texture_field /= max(float(np.std(texture_field)), 1e-6)
    threshold = float(np.quantile(texture_field, coverage))
    # ``softness`` controls the dimensionless logistic edge of the dot mask.
    # Keep the historical resolution-dependent default for compatibility, but
    # allow scale-consistent datasets to set it explicitly.
    softness = softness if softness is not None else (0.12 if blur_sigma_px == 0 else 0.18 + 0.12 * blur_sigma_px)
    if softness <= 0:
        raise ValueError("softness must be positive")
    dots = 1.0 / (1.0 + np.exp(-np.clip((texture_field - threshold) / softness, -30, 30)))
    if blur_sigma_px > 0:
        dots = _gaussian_blur(dots, blur_sigma_px)
    illumination_x = rng.uniform(-illumination_variation, illumination_variation)
    illumination_y = rng.uniform(-illumination_variation, illumination_variation)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    illumination = (
        0.5
        + illumination_x * (xx / max(width - 1, 1) - 0.5)
        + illumination_y * (yy / max(height - 1, 1) - 0.5)
    )
    gain = rng.uniform(0.72, 1.28)
    image = np.clip(illumination + gain * (dots - 0.5) * 0.78, 0.0, 1.0)
    image += rng.normal(0.0, noise_std, size=(height, width)).astype(np.float32)
    if periodic_strength:
        pattern = .5 + .25 * (np.cos(2 * np.pi * xx / periodic_period_px)
                              + np.cos(2 * np.pi * yy / periodic_period_px))
        image = (1 - periodic_strength) * image + periodic_strength * pattern
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, {
        "characteristic_size_px": float(characteristic_size_px),
        "coverage": float(coverage),
        "blur_sigma_px": float(blur_sigma_px),
        "softness": float(softness),
        "noise_std": float(noise_std),
        "illumination_variation": float(illumination_variation),
        "gain": float(gain),
        "periodic_strength": float(periodic_strength),
        "periodic_period_px": float(periodic_period_px),
    }


def _motion_parameters(
    rng: np.random.Generator,
    width: int,
    height: int,
    mode: str | None = None,
    shear_band_peak_gradient: float | None = None,
    discontinuity_boundary_fraction: float = 0.5,
) -> dict[str, Any]:
    mode = mode or str(
        rng.choice(("translation", "rigid", "affine", "heterogeneous", "discontinuous"))
    )
    angle = float(rng.uniform(-np.pi, np.pi) if mode == "rigid" else rng.uniform(-1.1, 1.1))
    stretch_x = float(rng.uniform(0.72, 1.65))
    stretch_y = float(rng.uniform(0.72, 1.35))
    shear = float(rng.uniform(-0.35, 0.35))
    if mode in {"translation", "shear_band"}:
        stretch_x = stretch_y = 1.0
        shear = angle = 0.0
    if mode == "rigid":
        stretch_x = stretch_y = 1.0
        shear = 0.0
    if mode == "discontinuous":
        stretch_x = stretch_y = 1.0
        shear = angle = 0.0
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    strain = np.array([[stretch_x, shear], [0.0, stretch_y]], dtype=np.float64)
    matrix = rotation @ strain
    translation = rng.uniform(-0.30, 0.30, size=2) * np.array([width, height])
    heterogeneity = (
        rng.uniform(-0.035, 0.035, size=2) * np.array([width, height])
        if mode == "heterogeneous"
        else np.zeros(2)
    )
    if mode == "shear_band":
        heterogeneity = np.zeros(2, dtype=np.float64)
    discontinuity_gap = float(rng.uniform(0.02, 0.06) * width) if mode == "discontinuous" else 0.0
    return {
        "mode": mode,
        "matrix": matrix,
        "translation": translation,
        "heterogeneity": heterogeneity,
        "angle_rad": angle,
        "stretch_x": stretch_x,
        "stretch_y": stretch_y,
        "shear": shear,
        "discontinuity_boundary_px": width * discontinuity_boundary_fraction,
        "discontinuity_gap_px": discontinuity_gap,
        "shear_band_peak_gradient": (
            float(shear_band_peak_gradient) if shear_band_peak_gradient is not None else 0.0
        ),
        "shear_band_center_y_px": 0.5 * height,
        "shear_band_width_px": max(8.0, 0.04 * height),
    }


def _heterogeneous_displacement(
    x: np.ndarray, y: np.ndarray, motion: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = motion["heterogeneity"]
    phase_x = 2.0 * np.pi * x / max(width, 1)
    phase_y = 2.0 * np.pi * y / max(height, 1)
    dx = amplitude[0] * np.sin(phase_y) * np.cos(phase_x)
    dy = amplitude[1] * np.sin(phase_x) * np.cos(phase_y)
    if motion["mode"] == "shear_band":
        band_width = motion["shear_band_width_px"]
        normalized = (y - motion["shear_band_center_y_px"]) / band_width
        dx = dx + (
            motion["shear_band_peak_gradient"] * band_width * np.tanh(normalized)
        )
    return dx, dy


def _forward_map(
    x: np.ndarray, y: np.ndarray, motion: dict[str, Any], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    coords = np.stack((x, y), axis=0).reshape(2, -1)
    mapped = center[:, None] + motion["matrix"] @ (coords - center[:, None])
    mapped += motion["translation"][:, None]
    mx, my = mapped[0].reshape(x.shape), mapped[1].reshape(y.shape)
    hx, hy = _heterogeneous_displacement(x, y, motion, width, height)
    if motion["mode"] == "discontinuous":
        mx = mx + motion["discontinuity_gap_px"] * (x >= motion["discontinuity_boundary_px"])
    return mx + hx, my + hy


def _bilinear_sample(
    image: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    visible = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int32)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = np.clip(x - x0, 0.0, 1.0)
    wy = np.clip(y - y0, 0.0, 1.0)
    value = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    return np.where(visible, value, 0.5).astype(np.float32), visible


def generate_speckle_pair(
    texture_seed: int,
    warp_seed: int,
    width: int = 256,
    height: int = 256,
    *,
    motion_mode: str | None = None,
    shear_band_peak_gradient: float | None = None,
    discontinuity_boundary_fraction: float = 0.5,
    texture_options: dict[str, float] | None = None,
) -> SyntheticSpecklePair:
    """Warp one speckle texture through analytic affine, smooth, or discontinuous motion."""
    reference, texture_metadata = generate_speckle(
        texture_seed, width, height, **(texture_options or {})
    )
    rng = np.random.default_rng(warp_seed)
    if motion_mode is not None and motion_mode not in {
        "translation",
        "rigid",
        "affine",
        "heterogeneous",
        "discontinuous",
        "shear_band",
    }:
        raise ValueError("Unknown synthetic motion mode")
    if shear_band_peak_gradient is not None and motion_mode != "shear_band":
        raise ValueError("shear_band_peak_gradient requires motion_mode='shear_band'")
    if not 0.0 < discontinuity_boundary_fraction < 1.0:
        raise ValueError("discontinuity boundary fraction must be between zero and one")
    motion = _motion_parameters(
        rng, width, height, motion_mode, shear_band_peak_gradient, discontinuity_boundary_fraction
    )

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    target_valid = None
    if motion["mode"] == "discontinuous":
        translation_x, translation_y = motion["translation"]
        seam = motion["discontinuity_boundary_px"] + translation_x
        gap = motion["discontinuity_gap_px"]
        left = xx < seam
        right = xx >= seam + gap
        target_x = np.where(left, xx - translation_x, xx - translation_x - gap)
        target_y = yy - translation_y
        target, sampled = _bilinear_sample(reference, target_x, target_y)
        target_valid = sampled & (left | right)
        target = np.where(target_valid, target, 0.5).astype(np.float32)
    else:
        target_x, target_y = xx.copy(), yy.copy()
        inverse_matrix = np.linalg.inv(motion["matrix"])
        center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
        # Solve p = c + M(q-c) + t + h(q) by fixed-point iteration.
        for _ in range(12):
            hx, hy = _heterogeneous_displacement(target_x, target_y, motion, width, height)
            rhs = np.stack((xx - hx, yy - hy), axis=0).reshape(2, -1)
            source = center[:, None] + inverse_matrix @ (
                rhs - center[:, None] - motion["translation"][:, None]
            )
            target_x = source[0].reshape(height, width)
            target_y = source[1].reshape(height, width)
        target, visible_target = _bilinear_sample(reference, target_x, target_y)
        target_valid = visible_target
    target = np.clip(
        target * float(rng.uniform(0.86, 1.14)) + float(rng.uniform(-0.06, 0.06)), 0, 1
    )
    target = np.where(target_valid, target, 0.5).astype(np.float32)

    source_x, source_y = xx.astype(np.float32), yy.astype(np.float32)
    mapped_x, mapped_y = _forward_map(source_x, source_y, motion, width, height)
    flow = np.stack((mapped_x - source_x, mapped_y - source_y), axis=-1).astype(np.float32)
    visible = (
        (mapped_x >= 0.0)
        & (mapped_x <= width - 1.0)
        & (mapped_y >= 0.0)
        & (mapped_y <= height - 1.0)
    )
    target_visible = target_valid.copy()
    if motion["mode"] == "discontinuous":
        boundary = motion["discontinuity_boundary_px"]
        margin = max(8.0, 2.0 * texture_metadata["characteristic_size_px"])
        visible &= np.abs(source_x - boundary) > margin
        seam = boundary + motion["translation"][0]
        gap = motion["discontinuity_gap_px"]
        left_support = xx < seam - margin
        right_support = xx >= seam + gap + margin
        target_visible &= left_support | right_support
    metadata = {
        "texture_seed": int(texture_seed),
        "warp_seed": int(warp_seed),
        "width": int(width),
        "height": int(height),
        "motion_mode": motion["mode"],
        "affine_matrix": motion["matrix"].tolist(),
        "translation_px": motion["translation"].tolist(),
        "angle_rad": float(motion["angle_rad"]),
        "stretch_x": float(motion["stretch_x"]),
        "stretch_y": float(motion["stretch_y"]),
        "shear": float(motion["shear"]),
        "discontinuity_boundary_px": float(motion["discontinuity_boundary_px"]),
        "discontinuity_gap_px": float(motion["discontinuity_gap_px"]),
        "shear_band_peak_gradient": float(motion["shear_band_peak_gradient"]),
        "shear_band_center_y_px": float(motion["shear_band_center_y_px"]),
        "shear_band_width_px": float(motion["shear_band_width_px"]),
        "heterogeneity_amplitude_px": motion["heterogeneity"].tolist(),
        "target_visible_fraction": float(np.mean(target_visible)),
        **texture_metadata,
    }
    return SyntheticSpecklePair(
        reference,
        target,
        flow,
        visible.astype(np.bool_),
        metadata,
        target_visible.astype(np.bool_),
    )
