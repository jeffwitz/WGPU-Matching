"""Measure whether image resizing preserves DIC speckle texture.

This is a diagnostic, not a matcher benchmark. It compares a real image crop
with synthetic textures at physical and training-like scales after the resize
used by the NDF pair matcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from wgpu_matching.dic_synthetic import generate_speckle


def correlation_summary(image: Image.Image) -> dict[str, object]:
    """Return contrast and radial half-correlation length for an image."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    gray -= gray.mean()
    height, width = gray.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.rfft2(gray * window)
    correlation = np.fft.fftshift(np.fft.irfft2(np.abs(spectrum) ** 2, s=gray.shape).real)
    center_y, center_x = height // 2, width // 2
    correlation /= max(float(correlation[center_y, center_x]), 1e-12)
    y, x = np.ogrid[:height, :width]
    radius = np.floor(np.hypot(y - center_y, x - center_x)).astype(np.int32)
    max_radius = min(height, width) // 4
    radial_sum = np.bincount(
        radius.ravel(), weights=correlation.ravel(), minlength=max_radius + 1
    )
    radial_count = np.bincount(radius.ravel(), minlength=max_radius + 1)
    radial = radial_sum[:max_radius] / np.maximum(radial_count[:max_radius], 1)
    crossing = np.flatnonzero(radial[1:] <= 0.5)
    half_correlation_px = None
    if crossing.size:
        index = int(crossing[0]) + 1
        previous, current = radial[index - 1], radial[index]
        fraction = (previous - 0.5) / max(previous - current, 1e-12)
        half_correlation_px = float(index - 1 + fraction)
    return {
        "image_size_px": [width, height],
        "gray_std_8bit": float(np.std(gray)),
        "radial_half_correlation_px": half_correlation_px,
        "radial_correlation_samples": {
            str(offset): float(radial[offset])
            for offset in (1, 2, 4, 8, 16, 32)
            if offset < len(radial)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-width", type=int, default=3840)
    parser.add_argument("--native-height", type=int, default=2160)
    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--model-height", type=int, default=108)
    parser.add_argument("--seed", type=int, default=12001)
    args = parser.parse_args()

    source = Image.open(args.reference_image).convert("L")
    left = max(0, (source.width - args.native_width) // 2)
    top = max(0, (source.height - args.native_height) // 2)
    native = source.crop((left, top, left + args.native_width, top + args.native_height))
    scales = sorted(
        {
            1,
            2,
            4,
            8,
            max(1, round(args.native_width / args.model_width)),
        }
    )
    cases: dict[str, object] = {
        "real_reference": {
            "source": str(args.reference_image),
            "crop_xy_px": [left, top],
            "resizes": {
                str(scale): correlation_summary(
                    native.resize(
                        (max(1, args.native_width // scale), max(1, args.native_height // scale)),
                        Image.Resampling.BOX,
                    )
                )
                for scale in scales
            },
        }
    }
    synthetic: dict[str, object] = {}
    for label, characteristic_size in (
        ("physical_like_7px_native", 7.0),
        ("matcher_training_like_140px_native", 7.0 * args.native_width / args.model_width),
    ):
        texture, metadata = generate_speckle(
            args.seed,
            args.native_width,
            args.native_height,
            characteristic_size_px=characteristic_size,
            coverage=0.31,
            blur_sigma_px=max(0.45 * characteristic_size / 3.5, 0.3),
            softness=0.612,
            noise_std=0.025,
        )
        image = Image.fromarray(np.uint8(np.clip(texture * 255, 0, 255)), mode="L")
        synthetic[label] = {
            "generator_metadata": metadata,
            "resizes": {
                str(scale): correlation_summary(
                    image.resize(
                        (max(1, args.native_width // scale), max(1, args.native_height // scale)),
                        Image.Resampling.BOX,
                    )
                )
                for scale in scales
            },
        }
    cases["synthetic"] = synthetic
    cases["protocol"] = {
        "native_raster_px": [args.native_width, args.native_height],
        "model_raster_px": [args.model_width, args.model_height],
        "model_resize_factor_x": args.native_width / args.model_width,
        "interpretation": "radial half-correlation length is an image-texture diagnostic, not a physical dot diameter or a matching accuracy metric",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
