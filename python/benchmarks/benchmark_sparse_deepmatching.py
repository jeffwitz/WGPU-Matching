"""Compare full-grid and seed-local INRIA-style DeepMatching on selected POIs."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from wgpu_matching import DeepMatchingOptions, WGPUMatching, load_gray

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_DIC = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
DEFAULT_OUTPUT = REPOSITORY / "results/uniaxial/sparse_deepmatching_study.csv"
ERROR_CSV = "frame_{frame}_horg_subpixel_analytic_error.csv"
FIELDS = (
    "frame", "requested_seed_count", "mode", "global_root_count", "evaluated_root_count",
    "root_fraction_percent", "estimated_descriptor_comparisons", "accepted_count",
    "within_2px_count", "accepted_rmse_px", "median_call_s", "timings_s",
    "same_accepted_as_dense", "same_displacement_as_dense",
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
            target_x = 1920 + stretch_x * (x - 1920)
            target_y = 1080 + stretch_y * (y - 1080)
            if 0 <= target_x < width and 0 <= target_y < height:
                points.append((x, y))
    return tuple(points)


def run_once(matcher, reference, target, seeds, mode):
    return matcher.match_deepmatching(
        reference,
        target,
        options=DeepMatchingOptions(
            mode=mode,
            patch_size=128,
            root_patch_size=512,
            stride=512,
            downscale_levels=2,
            search_radius_px=256,
            seed_points=seeds,
        ),
    )


def accepted_errors(field, stretch_x: float, stretch_y: float) -> list[float]:
    errors = []
    for match in field.matches:
        if not match.usable:
            continue
        truth_dx = (stretch_x - 1) * (match.x - 1920)
        truth_dy = (stretch_y - 1) * (match.y - 1080)
        errors.append(((match.dx - truth_dx) ** 2 + (match.dy - truth_dy) ** 2) ** 0.5)
    return errors


def quality_by_seed(field, stretch_x: float, stretch_y: float):
    results = {}
    for match in field.matches:
        key = (match.seed_x, match.seed_y)
        truth_dx = (stretch_x - 1) * (match.x - 1920)
        truth_dy = (stretch_y - 1) * (match.y - 1080)
        results[key] = (match.usable, match.dx, match.dy, ((match.dx - truth_dx) ** 2 + (match.dy - truth_dy) ** 2) ** 0.5)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dic", type=Path, default=DEFAULT_LIVE_DIC)
    parser.add_argument("--frame", choices=("040", "100", "199"), default="040")
    parser.add_argument("--seed-counts", default="1,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    try:
        seed_counts = sorted({int(value) for value in args.seed_counts.split(",")})
    except ValueError:
        parser.error("--seed-counts must be comma-separated positive integers")
    if not seed_counts or seed_counts[0] < 1:
        parser.error("--seed-counts must contain positive integers")

    reference = load_gray(args.live_dic / "reference.png")
    target = load_gray(args.live_dic / f"frame_{args.frame}.png")
    stretch_x, stretch_y = analytic_truth(REPOSITORY / "results/uniaxial" / ERROR_CSV.format(frame=args.frame))
    all_seeds = visible_grid_seeds(stretch_x, stretch_y, reference.shape[1], reference.shape[0])
    if seed_counts[-1] > len(all_seeds):
        parser.error(f"Largest seed count exceeds the {len(all_seeds)} visible grid seeds")

    rows = []
    with WGPUMatching() as matcher:
        print(f"Adapter: {matcher.device_name}; frame {args.frame}; visible grid seeds: {len(all_seeds)}")
        for seed_count in seed_counts:
            seeds = all_seeds[:seed_count]
            fields = {}
            timings = {}
            for mode in ("inria", "inria_sparse"):
                run_once(matcher, reference, target, seeds, mode)  # warm up this mode
                repeat_times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    fields[mode] = run_once(matcher, reference, target, seeds, mode)
                    repeat_times.append(time.perf_counter() - started)
                timings[mode] = repeat_times

            dense = quality_by_seed(fields["inria"], stretch_x, stretch_y)
            sparse = quality_by_seed(fields["inria_sparse"], stretch_x, stretch_y)
            shared = set(dense) & set(sparse)
            same_accepted = sum(dense[key][0] == sparse[key][0] for key in shared)
            shared_accepted = [key for key in shared if dense[key][0] and sparse[key][0]]
            same_displacement = sum(
                dense[key][1:3] == sparse[key][1:3] for key in shared_accepted
            )

            for mode in ("inria", "inria_sparse"):
                field = fields[mode]
                errors = accepted_errors(field, stretch_x, stretch_y)
                repeat_times = timings[mode]
                rows.append({
                    "frame": args.frame,
                    "requested_seed_count": seed_count,
                    "mode": mode,
                    "global_root_count": field.metadata["global_root_count"],
                    "evaluated_root_count": field.metadata["evaluated_root_count"],
                    "root_fraction_percent": f"{100 * field.metadata['evaluated_root_count'] / field.metadata['global_root_count']:.2f}",
                    "estimated_descriptor_comparisons": field.metadata["estimated_descriptor_comparisons"],
                    "accepted_count": len(errors),
                    "within_2px_count": sum(error <= 2 for error in errors),
                    "accepted_rmse_px": f"{statistics.mean(error * error for error in errors) ** 0.5:.6f}" if errors else "",
                    "median_call_s": f"{statistics.median(repeat_times):.6f}",
                    "timings_s": ";".join(f"{value:.6f}" for value in repeat_times),
                    "same_accepted_as_dense": f"{same_accepted}/{len(shared)}",
                    "same_displacement_as_dense": f"{same_displacement}/{len(shared_accepted)}" if shared_accepted else "n/a",
                })
            print(
                f"{seed_count} seeds: roots {fields['inria_sparse'].metadata['evaluated_root_count']}/"
                f"{fields['inria_sparse'].metadata['global_root_count']}; "
                f"dense {statistics.median(timings['inria']):.3f}s, "
                f"sparse {statistics.median(timings['inria_sparse']):.3f}s; "
                f"accepted parity {same_accepted}/{len(shared)}, "
                f"displacement parity {same_displacement}/{len(shared_accepted)}",
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
