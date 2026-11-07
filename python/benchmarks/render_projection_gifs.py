"""Render compact reference/corrected flicker GIFs for the projection report."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "projection"
DEFAULT_OUTPUT = RESULTS / "gifs"

CASES = (
    (
        "adrien_raco_affine",
        "dataset_run_adrien_visual_20261107",
        "adrien_000000_to_000075",
        "raco_aliked_lightglue",
        "affine",
    ),
    (
        "adrien_efficientloftr_affine",
        "dataset_run_adrien_visual_20261107",
        "adrien_000000_to_000075",
        "efficientloftr",
        "affine",
    ),
    (
        "adrien_raco_gaussian_rbf",
        "dataset_run_adrien_visual_20261107",
        "adrien_000000_to_000075",
        "raco_aliked_lightglue",
        "gaussian_rbf",
    ),
    (
        "adrien_efficientloftr_gaussian_rbf",
        "dataset_run_adrien_visual_20261107",
        "adrien_000000_to_000075",
        "efficientloftr",
        "gaussian_rbf",
    ),
    (
        "tensile_10pct_efficientloftr_affine_success",
        "dataset_run_traction_visual_20261107",
        "tensile_10pct",
        "efficientloftr",
        "affine",
    ),
    (
        "tensile_60pct_raco_affine_success",
        "dataset_run_traction_visual_20261107",
        "tensile_60pct",
        "raco_aliked_lightglue",
        "affine",
    ),
    (
        "tensile_60pct_efficientloftr_gaussian_rbf_failure",
        "dataset_run_traction_visual_20261107",
        "tensile_60pct",
        "efficientloftr",
        "gaussian_rbf",
    ),
)


def _small_palette_frame(path: Path, max_width: int) -> Image.Image:
    with Image.open(path) as source:
        frame = source.convert("L")
    if frame.width > max_width:
        height = round(frame.height * max_width / frame.width)
        frame = frame.resize((max_width, height), Image.Resampling.LANCZOS)
    return frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def render_gifs(output: Path, max_width: int = 1280, duration_ms: int = 850) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, run, case, model, basis in CASES:
        case_dir = RESULTS / run / "visualizations" / case / model / basis
        reference = _small_palette_frame(case_dir / "image1_ref.png", max_width)
        corrected = _small_palette_frame(case_dir / "image1_def_cor.png", max_width)
        if reference.size != corrected.size:
            raise ValueError(f"Mismatched reference/corrected sizes for {name}")
        destination = output / f"{name}.gif"
        reference.save(
            destination,
            save_all=True,
            append_images=[corrected],
            duration=[duration_ms, duration_ms],
            loop=0,
            optimize=True,
            disposal=2,
        )
        written.append(destination)
        print(f"{destination.relative_to(ROOT)}: {reference.width}x{reference.height}, 2 frames")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--duration-ms", type=int, default=850)
    args = parser.parse_args()
    if args.max_width < 1 or args.duration_ms < 1:
        parser.error("--max-width and --duration-ms must be positive")
    render_gifs(args.output.resolve(), args.max_width, args.duration_ms)


if __name__ == "__main__":
    main()
