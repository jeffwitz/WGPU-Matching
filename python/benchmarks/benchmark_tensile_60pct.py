"""Evaluate Python matcher coverage against the 10%-to-60% tensile series."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from wgpu_matching import DeepMatchingOptions, HORGOptions, WGPUMatching, load_gray

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY / "results/uniaxial/tensile_60pct"
FIELDS = (
    "requested_percent",
    "actual_percent",
    "frame",
    "method",
    "visible_poi_count",
    "accepted_visible_count",
    "within_2px_count",
    "accepted_coverage_visible_percent",
    "accurate_coverage_visible_percent",
    "accepted_rmse_px",
    "accepted_median_px",
    "accepted_p95_px",
    "elapsed_s",
)


def visible_grid_seeds(stretch_x: float, stretch_y: float) -> tuple[tuple[float, float], ...]:
    patch_size, stride = 128, 512
    width, height = 3840, 2160
    columns = 1 + (width - patch_size) // stride
    rows = 1 + (height - patch_size) // stride
    points = []
    for gy in range(rows):
        for gx in range(columns):
            x = gx * stride + (patch_size - 1) / 2
            y = gy * stride + (patch_size - 1) / 2
            tx = 1920 + stretch_x * (x - 1920)
            ty = 1080 + stretch_y * (y - 1080)
            if 0 <= tx < width and 0 <= ty < height:
                points.append((x, y))
    return tuple(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = (args.output or dataset / "matcher_comparison.csv").resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    truth_rows = csv.DictReader(
        (dataset / manifest["truth_file"]).open(newline="", encoding="utf-8")
    )
    truth_by_frame = {int(row["frame"]): row for row in truth_rows}
    reference = load_gray(dataset / "reference.png")
    results = []

    for method in ("HORG", "DeepMatching-INRIA"):
        with WGPUMatching() as matcher:
            for snapshot in manifest["snapshots"]:
                frame = int(snapshot["frame"])
                truth = truth_by_frame[frame]
                target = load_gray(dataset / snapshot["file"])
                visible_seeds = visible_grid_seeds(
                    float(truth["stretch_x"]), float(truth["stretch_y"])
                )
                if method == "HORG":
                    options = HORGOptions(
                        patch_size=128,
                        stride=512,
                        reduction=4,
                        max_features=1024,
                        seed_points=visible_seeds,
                    )
                else:
                    options = DeepMatchingOptions(
                        mode="inria",
                        patch_size=128,
                        root_patch_size=512,
                        stride=512,
                        downscale_levels=2,
                        search_radius_px=256,
                        seed_points=visible_seeds,
                    )

                start = time.perf_counter()
                if method == "HORG":
                    field = matcher.match_horg(reference, target, options=options)
                else:
                    field = matcher.match_deepmatching(reference, target, options=options)
                elapsed = time.perf_counter() - start

                # Compare displacement at the reported source coordinate. DeepMatching
                # snaps each requested seed to its hierarchy's leaf location.
                errors = []
                visible_count = 0
                for match in field.matches:
                    seed_x = match.seed_x if match.seed_x is not None else match.x
                    seed_y = match.seed_y if match.seed_y is not None else match.y
                    target_seed_x = 1920 + float(truth["stretch_x"]) * (seed_x - 1920)
                    target_seed_y = 1080 + float(truth["stretch_y"]) * (seed_y - 1080)
                    visible = 0 <= target_seed_x < field.width and 0 <= target_seed_y < field.height
                    if not visible:
                        continue
                    visible_count += 1
                    if not match.usable:
                        continue
                    dx = (float(truth["stretch_x"]) - 1) * (match.x - 1920)
                    dy = (float(truth["stretch_y"]) - 1) * (match.y - 1080)
                    errors.append(math.hypot(match.dx - dx, match.dy - dy))
                errors.sort()
                row = {
                    "requested_percent": round(
                        100 * float(snapshot["requested_engineering_strain"])
                    ),
                    "actual_percent": f"{100 * float(truth['engineering_axial']):.6f}",
                    "frame": frame,
                    "method": method,
                    "visible_poi_count": visible_count,
                    "accepted_visible_count": len(errors),
                    "within_2px_count": sum(error <= 2 for error in errors),
                    "accepted_coverage_visible_percent": f"{100 * len(errors) / visible_count:.3f}"
                    if visible_count
                    else "",
                    "accurate_coverage_visible_percent": f"{100 * sum(error <= 2 for error in errors) / visible_count:.3f}"
                    if visible_count
                    else "",
                    "accepted_rmse_px": f"{math.sqrt(statistics.mean(error * error for error in errors)):.6f}"
                    if errors
                    else "",
                    "accepted_median_px": f"{statistics.median(errors):.6f}" if errors else "",
                    "accepted_p95_px": f"{errors[min(len(errors) - 1, math.ceil(0.95 * len(errors)) - 1)]:.6f}"
                    if errors
                    else "",
                    "elapsed_s": f"{elapsed:.6f}",
                }
                results.append(row)
                print(
                    f"{row['actual_percent']}% {method}: accepted={len(errors)}/{visible_count} visible POIs, "
                    f"within 2 px={row['within_2px_count']}, "
                    f"RMSE={row['accepted_rmse_px'] or 'n/a'} px, {elapsed:.3f} s",
                    flush=True,
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
