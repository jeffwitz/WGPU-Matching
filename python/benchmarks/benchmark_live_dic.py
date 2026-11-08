"""Benchmark HORG and INRIA-style DeepMatching on Live-DIC UHD frames.

Image loading, device creation, and adapter discovery are excluded from the
detector call timings. The first matcher call is recorded separately from
subsequent calls so pipeline creation/first-use costs remain visible.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from wgpu_matching import DeepMatchingOptions, HORGOptions, WGPUMatching, load_gray

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_DIC = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
DEFAULT_OUTPUT = REPOSITORY / "results/uniaxial/python_runtime_comparison.csv"
FRAMES = ("040", "100", "199")
ANALYTIC_FILES = {
    "040": "frame_040_horg_subpixel_analytic_error.csv",
    "100": "frame_100_horg_subpixel_analytic_error.csv",
    "199": "frame_199_horg_subpixel_analytic_error.csv",
}
FIELDS = (
    "frame",
    "method",
    "image_width",
    "image_height",
    "poi_count",
    "visible_poi_count",
    "accepted_count",
    "within_2px_count",
    "accepted_coverage_visible_percent",
    "accurate_coverage_visible_percent",
    "accepted_rmse_px",
    "accepted_median_error_px",
    "accepted_p95_error_px",
    "first_call_s",
    "repeat_median_s",
    "repeat_min_s",
    "repeat_max_s",
    "repeat_timings_s",
    "repeats",
    "adapter",
)


def analytic_truth(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    stretch_x = 1 + float(row["truth_dx"]) / (float(row["x"]) - 1920)
    stretch_y = 1 + float(row["truth_dy"]) / (float(row["y"]) - 1080)
    return stretch_x, stretch_y


def visible_grid_seeds(stretch_x: float, stretch_y: float, width: int, height: int):
    patch_size, stride = 128, 512
    points = []
    for gy in range(1 + (height - patch_size) // stride):
        for gx in range(1 + (width - patch_size) // stride):
            x = gx * stride + (patch_size - 1) / 2
            y = gy * stride + (patch_size - 1) / 2
            tx, ty = 1920 + stretch_x * (x - 1920), 1080 + stretch_y * (y - 1080)
            if 0 <= tx < width and 0 <= ty < height:
                points.append((x, y))
    return tuple(points)


def measure(matcher: WGPUMatching, method: str, reference, target, repeats: int, seed_points):
    if method == "HORG":

        def call():
            return matcher.match_horg(
                reference,
                target,
                options=HORGOptions(
                    patch_size=128,
                    stride=512,
                    reduction=4,
                    max_features=1024,
                    seed_points=seed_points,
                ),
            )
    else:

        def call():
            return matcher.match_deepmatching(
                reference,
                target,
                options=DeepMatchingOptions(
                    mode="inria",
                    patch_size=128,
                    root_patch_size=512,
                    stride=512,
                    downscale_levels=2,
                    search_radius_px=256,
                    seed_points=seed_points,
                ),
            )

    start = time.perf_counter()
    field = call()
    first_s = time.perf_counter() - start
    repeat_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        field = call()
        repeat_times.append(time.perf_counter() - start)
    return field, first_s, repeat_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dic", type=Path, default=DEFAULT_LIVE_DIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    rows = []
    for frame in FRAMES:
        reference = load_gray(args.live_dic / "reference.png")
        target = load_gray(args.live_dic / f"frame_{frame}.png")
        stretch_x, stretch_y = analytic_truth(
            REPOSITORY / "results/uniaxial" / ANALYTIC_FILES[frame]
        )
        seed_points = visible_grid_seeds(
            stretch_x, stretch_y, reference.shape[1], reference.shape[0]
        )
        for method in ("HORG", "DeepMatching-INRIA"):
            with WGPUMatching() as matcher:
                field, first_s, repeat_times = measure(
                    matcher,
                    "HORG" if method == "HORG" else "DeepMatching",
                    reference,
                    target,
                    args.repeats,
                    seed_points,
                )
                accepted = []
                visible_count = 0
                for match in field.matches:
                    seed_x = match.seed_x if match.seed_x is not None else match.x
                    seed_y = match.seed_y if match.seed_y is not None else match.y
                    target_x = 1920 + stretch_x * (seed_x - 1920)
                    target_y = 1080 + stretch_y * (seed_y - 1080)
                    if not (0 <= target_x < field.width and 0 <= target_y < field.height):
                        continue
                    visible_count += 1
                    if not match.usable:
                        continue
                    # Use the source coordinate where dx/dy are measured. DeepMatching
                    # snaps the requested grid seed to the hierarchy's leaf grid.
                    dx = (stretch_x - 1) * (match.x - 1920)
                    dy = (stretch_y - 1) * (match.y - 1080)
                    accepted.append(((match.dx - dx) ** 2 + (match.dy - dy) ** 2) ** 0.5)
                errors_sorted = sorted(accepted)
                rows.append(
                    {
                        "frame": frame,
                        "method": method,
                        "image_width": field.width,
                        "image_height": field.height,
                        "poi_count": len(field.matches),
                        "visible_poi_count": visible_count,
                        "accepted_count": len(accepted),
                        "within_2px_count": sum(error <= 2.0 for error in accepted),
                        "accepted_coverage_visible_percent": f"{100 * len(accepted) / visible_count:.3f}"
                        if visible_count
                        else "",
                        "accurate_coverage_visible_percent": f"{100 * sum(error <= 2.0 for error in accepted) / visible_count:.3f}"
                        if visible_count
                        else "",
                        "accepted_rmse_px": f"{statistics.mean(error * error for error in accepted) ** 0.5:.6f}"
                        if accepted
                        else "",
                        "accepted_median_error_px": f"{statistics.median(accepted):.6f}"
                        if accepted
                        else "",
                        "accepted_p95_error_px": f"{errors_sorted[min(len(errors_sorted) - 1, int(0.95 * len(errors_sorted)))]:.6f}"
                        if errors_sorted
                        else "",
                        "first_call_s": f"{first_s:.6f}",
                        "repeat_median_s": f"{statistics.median(repeat_times):.6f}",
                        "repeat_min_s": f"{min(repeat_times):.6f}",
                        "repeat_max_s": f"{max(repeat_times):.6f}",
                        "repeat_timings_s": ";".join(f"{value:.6f}" for value in repeat_times),
                        "repeats": args.repeats,
                        "adapter": matcher.device_name,
                    }
                )
                rmse_text = (
                    f"{statistics.mean(error * error for error in accepted) ** 0.5:.3f}px"
                    if accepted
                    else "n/a"
                )
                print(
                    f"{frame} {method}: first={first_s:.3f}s, "
                    f"repeat median={statistics.median(repeat_times):.3f}s, "
                    f"accepted={len(accepted)}/{visible_count} visible POIs, "
                    f"analytic RMSE={rmse_text}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
