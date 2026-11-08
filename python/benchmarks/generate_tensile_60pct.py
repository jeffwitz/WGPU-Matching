"""Generate a deterministic UHD tensile series extending the source model to 60%.

This is a local adaptation of
``DIC_FROM_H264/experiments/tensile_extensometer_4k/generate.py``. It preserves
the source texture, constitutive equations, inverse remap and border handling;
it changes the final engineering strain and writes selected PNGs instead of
modifying or regenerating files in the source checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import brentq

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY / "results/uniaxial/tensile_60pct"
WIDTH, HEIGHT, COUNT = 3840, 2160, 200
YIELD_MPA, YOUNG_MPA, NU, K_MPA, HARDENING_N = 235.0, 210000.0, 0.30, 450.0, 0.25
AREA0_MM2, SEED = 50.0, 20261012
SNAPSHOT_LEVELS = tuple(level / 100 for level in range(10, 61, 10))


def state(frame: int, max_engineering_strain: float) -> dict[str, float | int]:
    yield_strain = math.expm1(YIELD_MPA / YOUNG_MPA)
    if frame <= 40:
        engineering = yield_strain * frame / 40
    else:
        engineering = yield_strain + (max_engineering_strain - yield_strain) * (frame - 40) / (
            COUNT - 1 - 40
        )
    true_strain = math.log1p(engineering)
    if true_strain <= YIELD_MPA / YOUNG_MPA + 1e-14:
        stress, plastic = YOUNG_MPA * true_strain, 0.0
    else:
        plastic = brentq(
            lambda p: (YIELD_MPA + K_MPA * p**HARDENING_N) / YOUNG_MPA + p - true_strain,
            0.0,
            true_strain,
            xtol=1e-15,
        )
        stress = YIELD_MPA + K_MPA * plastic**HARDENING_N
    transverse_true = -NU * stress / YOUNG_MPA - plastic / 2
    stretch_x, stretch_y = math.exp(true_strain), math.exp(transverse_true)
    return {
        "frame": frame,
        "time_s": frame / 30,
        "engineering_axial": engineering,
        "true_axial": true_strain,
        "plastic_true": plastic,
        "stress_true_MPa": stress,
        "stretch_x": stretch_x,
        "stretch_y": stretch_y,
        "engineering_transverse": stretch_y - 1,
        "machine_force_N": stress * AREA0_MM2 * stretch_y * stretch_y,
    }


def master_texture() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((HEIGHT, WIDTH), dtype=np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.0)
    binary = (smooth > np.quantile(smooth, 0.58)).astype(np.float32)
    blurred = cv2.GaussianBlur(binary, (0, 0), 0.65)
    return np.clip(np.rint(205 - 160 * blurred), 16, 235).astype(np.uint8)


def render(base: np.ndarray, row: dict[str, float | int]) -> np.ndarray:
    height, width = base.shape
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    mx = np.broadcast_to(
        (x - width / 2) / float(row["stretch_x"]) + width / 2, (height, width)
    ).copy()
    my = np.broadcast_to(
        (y - height / 2) / float(row["stretch_y"]) + height / 2, (height, width)
    ).copy()
    return cv2.remap(
        base, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def write_contact_sheet(output: Path, selected: list[dict[str, float | int | str]]) -> None:
    cell_w, cell_h, label_h, gap = 640, 360, 40, 16
    columns = 3
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * gap, rows * (cell_h + label_h) + (rows + 1) * gap),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, item in enumerate(selected):
        x = gap + (index % columns) * (cell_w + gap)
        y = gap + (index // columns) * (cell_h + label_h + gap)
        with Image.open(output / str(item["file"])) as image:
            preview = (
                image.convert("L").resize((cell_w, cell_h), Image.Resampling.LANCZOS).convert("RGB")
            )
        sheet.paste(preview, (x, y))
        label = (
            f"{100 * float(item['actual_engineering_strain']):.1f}% engineering strain "
            f"- frame {int(item['frame']):03d}"
        )
        draw.text((x, y + cell_h + 8), label, fill=(20, 20, 20), font=font)
    sheet.save(output / "contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-engineering-strain", type=float, default=0.60)
    args = parser.parse_args()
    if not 0.10 < args.max_engineering_strain <= 1.0:
        parser.error("--max-engineering-strain must be greater than 0.10 and at most 1.0")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [state(frame, args.max_engineering_strain) for frame in range(COUNT)]
    truth_path = output / "machine_truth.csv"
    with truth_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    base = master_texture()
    reference_path = output / "reference.png"
    if not cv2.imwrite(str(reference_path), base):
        raise OSError(f"Could not write {reference_path}")

    selected: list[dict[str, float | int | str]] = []
    for level in SNAPSHOT_LEVELS:
        row = min(rows, key=lambda candidate: abs(float(candidate["engineering_axial"]) - level))
        image = render(base, row)
        label = round(level * 100)
        path = output / f"frame_{int(row['frame']):03d}_eng_{label:02d}pct.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
        selected.append(
            {
                "requested_engineering_strain": level,
                "file": path.name,
                "frame": int(row["frame"]),
                "actual_engineering_strain": float(row["engineering_axial"]),
                "stretch_x": float(row["stretch_x"]),
                "stretch_y": float(row["stretch_y"]),
            }
        )
        print(
            f"wrote {path.name}: {100 * float(row['engineering_axial']):.3f}% engineering strain",
            flush=True,
        )
        del image

    metadata = {
        "description": "Synthetic UHD uniaxial tensile images generated with the DIC_FROM_H264 source model.",
        "source_generator": "/home/jeff/CNRS/Code/DIC_CODEC/DIC_FROM_H264/experiments/tensile_extensometer_4k/generate.py",
        "source_modified": False,
        "width": WIDTH,
        "height": HEIGHT,
        "states": COUNT,
        "seed": SEED,
        "max_engineering_strain": args.max_engineering_strain,
        "material_model": {
            "young_modulus_MPa": YOUNG_MPA,
            "poisson_ratio": NU,
            "yield_stress_MPa": YIELD_MPA,
            "Ludwik_K_MPa": K_MPA,
            "Ludwik_n": HARDENING_N,
            "initial_area_mm2": AREA0_MM2,
        },
        "coordinate_map": {
            "center_px": [WIDTH / 2, HEIGHT / 2],
            "x_target": "cx + stretch_x * (x_reference - cx)",
            "y_target": "cy + stretch_y * (y_reference - cy)",
            "dx": "(stretch_x - 1) * (x_reference - cx)",
            "dy": "(stretch_y - 1) * (y_reference - cy)",
            "image_interpolation": "OpenCV INTER_LINEAR inverse remap, BORDER_REFLECT_101",
        },
        "extrapolation_note": (
            "The source experiment ends at 10% engineering strain. Values above 10% extrapolate "
            "the same constitutive equations and Ludwik hardening law; this is synthetic ground truth, "
            "not an experimentally validated material response."
        ),
        "truth_file": truth_path.name,
        "snapshots": selected,
        "contact_sheet": "contact_sheet.png",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_contact_sheet(output, selected)
    print(f"wrote {len(selected)} snapshots and {truth_path}")


if __name__ == "__main__":
    main()
