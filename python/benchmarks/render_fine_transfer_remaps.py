"""Render native-image previews from accepted coarse and fine IC-GN states.

The renderer uses only solver acceptance and its six estimated parameters. It
does not read analytical truth, visibility labels, or correctness labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FAMILIES = {
    "translation": "translation_texture62051006_warp62091006",
    "rigid": "rigid_texture62051006_warp62091006",
    "affine": "affine_texture62051003_warp62091003",
    "heterogeneous": "heterogeneous_texture62051003_warp62091003",
    "discontinuous": "discontinuous_texture62051006_warp62091006",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remap_preview(
    target: np.ndarray,
    centres: np.ndarray,
    parameters: np.ndarray,
    accepted: np.ndarray,
    output_size: tuple[int, int],
    *,
    support_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-sample the target using accepted local affine solver states.

    Parameters follow the IC-GN order (u, du/dx, du/dy, v, dv/dx, dv/dy).
    Local states are evaluated at each preview pixel, then bilinearly blended
    on their regular centre grid. Unsupported preview pixels are neutral gray.
    """
    height, width = target.shape
    out_width, out_height = output_size
    xs = np.unique(centres[:, 0])
    ys = np.unique(centres[:, 1])
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("expected a two-dimensional regular centre grid")
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    if not np.allclose(np.diff(xs), dx) or not np.allclose(np.diff(ys), dy):
        raise ValueError("window centres are not on a regular grid")

    ix = np.rint((centres[:, 0] - xs[0]) / dx).astype(int)
    iy = np.rint((centres[:, 1] - ys[0]) / dy).astype(int)
    parameter_grid = np.full((len(ys), len(xs), 6), np.nan, dtype=np.float64)
    accepted_grid = np.zeros((len(ys), len(xs)), dtype=bool)
    parameter_grid[iy, ix] = parameters
    accepted_grid[iy, ix] = accepted

    # Preview sample positions map to native pixel-centre coordinates.
    x = (np.arange(out_width, dtype=np.float64) + 0.5) * width / out_width - 0.5
    y = (np.arange(out_height, dtype=np.float64) + 0.5) * height / out_height - 0.5
    xx, yy = np.meshgrid(x, y)
    gx = np.clip((xx - xs[0]) / dx, 0, len(xs) - 1)
    gy = np.clip((yy - ys[0]) / dy, 0, len(ys) - 1)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = np.minimum(x0 + 1, len(xs) - 1)
    y1 = np.minimum(y0 + 1, len(ys) - 1)
    tx, ty = gx - x0, gy - y0
    corners = (
        (y0, x0, (1 - tx) * (1 - ty)),
        (y0, x1, tx * (1 - ty)),
        (y1, x0, (1 - tx) * ty),
        (y1, x1, tx * ty),
    )

    displacement_x = np.zeros_like(xx)
    displacement_y = np.zeros_like(yy)
    support = np.zeros_like(xx)
    for cy, cx, weight in corners:
        valid = accepted_grid[cy, cx]
        p = parameter_grid[cy, cx]
        local_x = xx - xs[cx]
        local_y = yy - ys[cy]
        ux = p[..., 0] + p[..., 1] * local_x + p[..., 2] * local_y
        uy = p[..., 3] + p[..., 4] * local_x + p[..., 5] * local_y
        w = weight * valid
        displacement_x += np.nan_to_num(ux) * w
        displacement_y += np.nan_to_num(uy) * w
        support += w

    supported = support >= support_threshold
    displacement_x = np.divide(
        displacement_x, support, out=np.zeros_like(support), where=supported
    )
    displacement_y = np.divide(
        displacement_y, support, out=np.zeros_like(support), where=supported
    )
    sample_x = xx + displacement_x
    sample_y = yy + displacement_y
    in_bounds = (
        supported
        & (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    )

    # Bilinear image sampling in native resolution, performed only at preview
    # positions. Out-of-image and unsupported samples remain explicitly gray.
    sx = np.clip(np.nan_to_num(sample_x), 0, width - 1)
    sy = np.clip(np.nan_to_num(sample_y), 0, height - 1)
    xa, ya = np.floor(sx).astype(int), np.floor(sy).astype(int)
    xb, yb = np.minimum(xa + 1, width - 1), np.minimum(ya + 1, height - 1)
    fx, fy = sx - xa, sy - ya
    values = (
        target[ya, xa] * (1 - fx) * (1 - fy)
        + target[ya, xb] * fx * (1 - fy)
        + target[yb, xa] * (1 - fx) * fy
        + target[yb, xb] * fx * fy
    )
    image = np.full((out_height, out_width), 128, dtype=np.uint8)
    image[in_bounds] = np.clip(np.rint(values[in_bounds]), 0, 255).astype(np.uint8)
    return image, in_bounds


def labelled_frame(image: np.ndarray, label: str) -> Image.Image:
    rgb = Image.fromarray(image, mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    font = ImageFont.load_default(size=28)
    draw.rectangle((0, 0, rgb.width, 48), fill=(20, 25, 32))
    draw.text((16, 9), label, fill="white", font=font)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/seed_propagation_20261111/p3_fine_transfer_multifamily_independent"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "fine/remapped_images"
    output.mkdir(parents=True, exist_ok=True)
    image_root = root / "inputs/affine_ncc/solver_images_seed62000000"
    cases: dict[str, dict] = {}
    sheet_rows: list[tuple[str, list[Image.Image]]] = []

    for family, pair in FAMILIES.items():
        ref_path = image_root / f"{pair}_reference.png"
        target_path = image_root / f"{pair}_target.png"
        coarse_path = root / f"coarse/{pair}.npz"
        fine_path = root / f"fine/{pair}.npz"
        if not all(p.is_file() for p in (ref_path, target_path, coarse_path, fine_path)):
            raise FileNotFoundError(f"missing inputs for {family}: {pair}")
        reference = np.asarray(Image.open(ref_path).convert("L"))
        target = np.asarray(Image.open(target_path).convert("L"))
        if reference.shape != target.shape:
            raise ValueError(f"reference/target size mismatch in {pair}")
        preview_width = min(args.width, reference.shape[1])
        preview_height = round(reference.shape[0] * preview_width / reference.shape[1])
        reference_preview = np.asarray(
            Image.fromarray(reference).resize(
                (preview_width, preview_height), Image.Resampling.BILINEAR
            )
        )
        reference_preview_path = output / f"image1_ref_{family}.png"
        Image.fromarray(reference_preview).save(reference_preview_path, optimize=True)
        frames = [labelled_frame(reference_preview, f"{family}: reference")]
        counts = {}
        states = []
        for label, path in (("64 px IC-GN", coarse_path), ("32 px IC-GN", fine_path)):
            with np.load(path) as data:
                centres = data["centres"].copy()
                parameters = data["parameters"].copy()
                accepted = data["accepted"].astype(bool, copy=True)
            corrected, coverage = remap_preview(
                target,
                centres,
                parameters,
                accepted,
                (preview_width, preview_height),
            )
            stem = f"{family}_{'64px' if label.startswith('64') else '32px'}"
            image_path = output / f"image1_def_cor_{stem}.png"
            coverage_path = output / f"coverage_{stem}.png"
            Image.fromarray(corrected).save(image_path, optimize=True)
            Image.fromarray(coverage.astype(np.uint8) * 255).save(coverage_path, optimize=True)
            frames.append(labelled_frame(corrected, f"{family}: remapped from {label}"))
            counts[label] = int(accepted.sum())
            states.append(
                {
                    "label": label,
                    "accepted_windows": counts[label],
                    "preview_covered_pixels": int(coverage.sum()),
                    "preview_pixels": int(coverage.size),
                    "remapped_image": image_path.name,
                    "coverage_mask": coverage_path.name,
                    "state_sha256": sha256(path),
                }
            )
        # Downsample the animated view for reasonable documentation payloads;
        # the individual PNG remaps remain 1920 px wide by default.
        gif_frames = [
            frame.resize((min(960, frame.width), round(frame.height * min(960, frame.width) / frame.width)), Image.Resampling.BILINEAR)
            for frame in frames
        ]
        # 850 ms per frame, matching the existing flicker figures.
        gif_path = output / f"{family}_reference_and_remapped.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=[850, 850, 850],
            loop=0,
            optimize=True,
            disposal=2,
        )
        cases[family] = {
            "pair": pair,
            "native_size": [int(reference.shape[1]), int(reference.shape[0])],
            "preview_size": [preview_width, preview_height],
            "reference": str(ref_path),
            "target": str(target_path),
            "reference_sha256": sha256(ref_path),
            "target_sha256": sha256(target_path),
            "reference_preview": reference_preview_path.name,
            "accepted_windows": counts,
            "states": states,
            "gif": gif_path.name,
            "remap_method": (
                "bilinear interpolation of accepted local affine IC-GN states on the regular "
                "window-centre grid, followed by bilinear backward sampling of native target pixels"
            ),
            "unsupported_fill": "neutral gray (128); see corresponding coverage mask",
        }
        sheet_rows.append((family, frames))

    # Compact overview sheet: each row compares reference, 64 px and 32 px.
    tile_width = 640
    tile_height = round(tile_width * reference.shape[0] / reference.shape[1])
    title_height = 40
    sheet = Image.new("RGB", (tile_width * 3, (tile_height + title_height) * len(sheet_rows)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    for row_index, (family, frames) in enumerate(sheet_rows):
        top = row_index * (tile_height + title_height)
        draw.text((8, top + 8), family, fill="#111111", font=font)
        for column, frame in enumerate(frames):
            tile = frame.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
            sheet.paste(tile, (column * tile_width, top + title_height))
    sheet.save(output / "remapped_images_overview.png", optimize=True)

    manifest = {
        "description": (
            "Image remaps use accepted IC-GN states only; analytical truth, visible, correct, "
            "and support-crossing labels are not read by the renderer. These are visual dense "
            "interpolations of sparse window solutions, not independent pixelwise DIC measurements."
        ),
        "coordinate_convention": "native pixel-centre coordinates; half-pixel resize mapping",
        "parameter_order": ["u", "du/dx", "du/dy", "v", "dv/dx", "dv/dy"],
        "support_threshold": 0.5,
        "families": cases,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "families": list(cases), "overview": str(output / "remapped_images_overview.png")}, indent=2))


if __name__ == "__main__":
    main()
