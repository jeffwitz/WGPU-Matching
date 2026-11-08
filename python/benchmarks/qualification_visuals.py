"""Render auditable image pairs for the common initializer qualification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def statistics(values: np.ndarray) -> dict[str, float | None]:
    """Summarize finite observations without silently replacing empty sets by zero."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "rmse": None, "p95": None, "maximum": None}
    return {
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def restore_centres(
    points: np.ndarray, native: tuple[int, int], content: tuple[int, int]
) -> np.ndarray:
    """Invert a half-pixel-centred raster resize; bottom/right padding adds no offset."""
    return (np.asarray(points, dtype=np.float64) + 0.5) * (
        np.asarray(native, dtype=np.float64) / np.asarray(content, dtype=np.float64)
    ) - 0.5


def assess_parameters(
    parameters: list[Any], seeds: np.ndarray, truth: dict | None, image_size: tuple[int, int]
) -> dict:
    """Evaluate results only; analytic truth is never supplied to field fitting."""
    available = np.asarray(
        [p is not None and np.isfinite(np.asarray(p)).all() for p in parameters], dtype=bool
    )
    result = {
        "requested": len(seeds),
        "estimated": int(available.sum()),
        "pass_below_5px": None,
        "correct_le2": None,
        "correct_lt5": None,
        "false_estimates_ge5": None,
        "per_seed": [],
    }
    errors = np.full(len(seeds), np.nan)
    if truth is not None:
        expected = (seeds - np.asarray(image_size) / 2) * (
            np.asarray([truth["stretch_x"], truth["stretch_y"]]) - 1
        )
        for i, p in enumerate(parameters):
            if available[i]:
                errors[i] = np.linalg.norm(np.asarray(p[:2]) - expected[i])
        good = available & (errors < 5)
        result.update(
            pass_below_5px=bool(len(seeds) and good.all()),
            correct_le2=int(np.count_nonzero(available & (errors <= 2))),
            correct_lt5=int(good.sum()),
            false_estimates_ge5=int(np.count_nonzero(available & (errors >= 5))),
            correct_coverage=float(good.sum() / len(seeds)) if len(seeds) else None,
            conditional_precision=float(good.sum() / available.sum()) if available.any() else None,
            errors_px=statistics(errors),
        )
    for i, (point, p) in enumerate(zip(seeds, parameters, strict=True)):
        result["per_seed"].append(
            {
                "seed_id": i,
                "x": float(point[0]),
                "y": float(point[1]),
                "parameters": list(p) if available[i] else None,
                "error_px": float(errors[i]) if np.isfinite(errors[i]) else None,
                "estimated": bool(available[i]),
            }
        )
    return result


def warp_preview(
    reference: np.ndarray,
    target: np.ndarray,
    model: Any | None,
    width: int = 1280,
    mask: np.ndarray | None = None,
):
    """Sample native target pixels at the estimated forward warp, in bounded chunks."""
    height_native, width_native = reference.shape
    width = min(width, width_native)
    height = round(height_native * width / width_native)
    preview = np.asarray(
        Image.fromarray(reference).resize((width, height), Image.Resampling.BILINEAR)
    )
    corrected = np.zeros((height, width), dtype=np.uint8)
    valid_map = np.zeros((height, width), dtype=bool)
    relevant_map = np.ones((height, width), dtype=bool)
    errors = []
    for start in range(0, height, 64):
        end = min(start + 64, height)
        yy, xx = np.meshgrid(np.arange(start, end), np.arange(width), indexing="ij")
        x = (xx + 0.5) * width_native / width - 0.5
        y = (yy + 0.5) * height_native / height - 0.5
        points = np.column_stack((x.ravel(), y.ravel()))
        displacement = model.evaluate(points) if model is not None else np.zeros_like(points)
        warped = points + displacement
        sx, sy = warped[:, 0].reshape(x.shape), warped[:, 1].reshape(y.shape)
        finite = np.isfinite(sx) & np.isfinite(sy)
        valid = finite & (sx >= 0) & (sx <= width_native - 1)
        valid &= (sy >= 0) & (sy <= height_native - 1)
        bx = np.floor(np.clip(np.nan_to_num(sx), 0, width_native - 1)).astype(np.int64)
        by = np.floor(np.clip(np.nan_to_num(sy), 0, height_native - 1)).astype(np.int64)
        ex, ey = np.minimum(bx + 1, width_native - 1), np.minimum(by + 1, height_native - 1)
        fx, fy = np.where(valid, sx - bx, 0), np.where(valid, sy - by, 0)
        # Out-of-bounds samples stay black; no identity fill can hide divergence.
        values = (
            target[by, bx] * (1 - fx) * (1 - fy)
            + target[by, ex] * fx * (1 - fy)
            + target[ey, bx] * (1 - fx) * fy
            + target[ey, ex] * fx * fy
        )
        values[~valid] = 0
        corrected[start:end] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        relevant = np.ones(x.shape, dtype=bool)
        if mask is not None:
            rx = np.clip(np.rint(x).astype(int), 0, width_native - 1)
            ry = np.clip(np.rint(y).astype(int), 0, height_native - 1)
            relevant = mask[ry, rx]
            valid &= mask[by, bx]
        relevant_map[start:end] = relevant
        valid_map[start:end] = valid & relevant
        errors.extend(
            np.abs(corrected[start:end].astype(float) - preview[start:end].astype(float))[
                valid & relevant
            ].tolist()
        )
    denominator = int(relevant_map.sum())
    return (
        preview,
        corrected,
        {
            "preview_width": width,
            "preview_height": height,
            "valid_warp_fraction_of_reference_roi": (
                float(valid_map.sum() / denominator) if denominator else None
            ),
            "preview_absolute_intensity_difference": statistics(np.asarray(errors)),
            "photometric_metric_scope": "preview pixels with valid target sample and ROI; not analytic error",
            "visualization_field": "estimated" if model is not None else "identity_no_model",
        },
    )


def render_pair(
    reference: np.ndarray,
    target: np.ndarray,
    model: Any | None,
    output: Path,
    *,
    label: str,
    verdict: str,
    detail: str,
    width: int = 1280,
    mask: np.ndarray | None = None,
    diagnostic_only: bool = False,
    zoom: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ref, cor, diagnostics = warp_preview(reference, target, model, width, mask)
    Image.fromarray(ref).save(output / "image1_ref.png")
    name = "image1_def_cor.png" if model is not None else "image1_def_uncorrected.png"
    Image.fromarray(cor).save(output / name)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    frames = []
    for title, data in (("REFERENCE", ref), ("CORRECTED TARGET", cor)):
        if model is None and title == "CORRECTED TARGET":
            title = "UNCORRECTED TARGET - NO FITTED FIELD"
        if diagnostic_only and title == "CORRECTED TARGET":
            title = "CORRECTED TARGET - REJECTED PAIRS, DIAGNOSTIC ONLY"
        frame = Image.new("RGB", (ref.shape[1], ref.shape[0] + 84), "#101e2a")
        frame.paste(Image.fromarray(data).convert("RGB"), (0, 84))
        draw = ImageDraw.Draw(frame)
        draw.text((12, 8), title, font=font, fill="white")
        draw.text((12, 32), label, font=small, fill="#e5edf4")
        color = "#70dda4" if verdict == "PASS" else "#ffbf75"
        draw.text((12, 56), verdict + " | " + detail, font=small, fill=color)
        # One shared grayscale palette prevents temporal colour/quantization artefacts.
        frames.append(frame)
    palette_image = Image.new("RGB", (frames[0].width, frames[0].height * 2))
    palette_image.paste(frames[0], (0, 0))
    palette_image.paste(frames[1], (0, frames[0].height))
    palette = palette_image.quantize(colors=128)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(
        output / "comparison.gif",
        save_all=True,
        append_images=indexed[1:],
        duration=[900, 900],
        loop=0,
        disposal=2,
        optimize=False,
    )
    diagnostics.update(
        gif="comparison.gif",
        reference_png="image1_ref.png",
        corrected_png=name,
        diagnostic_only=diagnostic_only,
    )
    if zoom:
        diagnostics.update(
            render_zoom(
                reference,
                target,
                model,
                output,
                label=label,
                verdict=verdict,
                diagnostic_only=diagnostic_only,
            )
        )
    return diagnostics


def render_zoom(reference, target, model, output, *, label, verdict, diagnostic_only):
    """Fixed image-centre crop at native resolution; never selected by visual quality."""
    height, width = reference.shape
    side = min(512, width, height)
    x0, y0 = (width - side) // 2, (height - side) // 2
    yy, xx = np.meshgrid(np.arange(y0, y0 + side), np.arange(x0, x0 + side), indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    displaced = points + (model.evaluate(points) if model is not None else 0)
    sx, sy = displaced[:, 0].reshape(xx.shape), displaced[:, 1].reshape(yy.shape)
    valid = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sy >= 0)
    valid &= (sx <= width - 1) & (sy <= height - 1)
    bx = np.floor(np.clip(np.nan_to_num(sx), 0, width - 1)).astype(int)
    by = np.floor(np.clip(np.nan_to_num(sy), 0, height - 1)).astype(int)
    fx, fy = np.where(valid, sx - bx, 0), np.where(valid, sy - by, 0)
    ex, ey = np.minimum(bx + 1, width - 1), np.minimum(by + 1, height - 1)
    values = (
        target[by, bx] * (1 - fx) * (1 - fy)
        + target[by, ex] * fx * (1 - fy)
        + target[ey, bx] * (1 - fx) * fy
        + target[ey, ex] * fx * fy
    )
    values[~valid] = 0
    cropped = reference[y0 : y0 + side, x0 : x0 + side]
    corrected = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    Image.fromarray(cropped).save(output / "zoom_ref.png")
    Image.fromarray(corrected).save(output / "zoom_def_cor.png")
    frames = []
    for title, data in (("Reference", cropped), ("Corrected target", corrected)):
        frame = Image.new("RGB", (side, side + 76), "#101e2a")
        frame.paste(Image.fromarray(data).convert("RGB"), (0, 76))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 5), title + " | " + verdict, fill="white")
        draw.text((8, 23), "Native pixel crop; " + label.split("|")[1].strip(), fill="white")
        draw.text((8, 41), f"Source box: ({x0}, {y0}), {side} x {side}", fill="white")
        if model is None or diagnostic_only:
            draw.text((8, 58), "NO ACCEPTED FIELD - diagnostic only", fill="#ffbf75")
        frames.append(frame)
    stacked = Image.new("RGB", (side, 2 * (side + 76)))
    for i, frame in enumerate(frames):
        stacked.paste(frame, (0, i * (side + 76)))
    palette = stacked.quantize(colors=128)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(
        output / "zoom_comparison.gif",
        save_all=True,
        append_images=indexed[1:],
        duration=[900, 900],
        loop=0,
        disposal=2,
        optimize=False,
    )
    return {"zoom_gif": "zoom_comparison.gif", "zoom_source_box": [x0, y0, side, side]}
