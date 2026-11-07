"""Compare native-4K WGSL tensor and learned DIC detectors on analytic speckles."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from experiment_learned_detectors import keypoint_distribution, load_cases, repeatability
from wgpu_matching import DICDetectorOptions, WGPUMatching
from wgpu_matching.images import load_gray

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/features/dic_detector_native_4k_20261107.json"
DEFAULT_WEIGHTS = ROOT / "results/features/dic_detector_training_20261107/dic_detector_weights.npz"


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "median_s": float(statistics.median(values)),
        "p10_s": float(np.percentile(values, 10)),
        "p90_s": float(np.percentile(values, 90)),
        "all_s": values,
    }


def run_case(
    case: dict[str, Any],
    matcher: WGPUMatching,
    budgets: tuple[int, ...],
    stride: int,
    radius: int,
    repeats: int,
    learned_weights_path: Path | None,
) -> dict[str, Any]:
    reference = load_gray(case["reference"])
    target = load_gray(case["target"])
    if reference.shape != target.shape:
        raise ValueError(f"Mismatched image shapes for {case['name']}")
    height, width = reference.shape
    truth = case["truth"]
    row: dict[str, Any] = {
        "name": case["name"],
        "source": case["source"],
        "frame": case["frame"],
        "image_size_px": [width, height],
        "truth": {key: truth[key] for key in ("engineering_axial", "stretch_x", "stretch_y")},
        "budgets": {},
    }
    methods: dict[str, Path | None] = {"tensor": None}
    if learned_weights_path is not None:
        methods["learned"] = learned_weights_path
    for budget in budgets:
        settings = DICDetectorOptions(
            stride_px=stride,
            tensor_radius_px=radius,
            max_keypoints=budget,
        )
        budget_row: dict[str, Any] = {"requested_keypoints": budget, "methods": {}}
        for method_name, weights_path in methods.items():
            options = {"learned_weights_path": weights_path}
            matcher.detect_dic(reference, options=settings, **options)
            matcher.detect_dic(target, options=settings, **options)
            reference_times = []
            target_times = []
            result_reference = None
            result_target = None
            for _ in range(repeats):
                start = time.perf_counter()
                result_reference = matcher.detect_dic(reference, options=settings, **options)
                reference_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                result_target = matcher.detect_dic(target, options=settings, **options)
                target_times.append(time.perf_counter() - start)
            assert result_reference is not None and result_target is not None
            features_reference = result_reference.features
            features_target = result_target.features
            response = repeatability(
                features_reference.xy,
                features_target.xy,
                width,
                height,
                truth["stretch_x"],
                truth["stretch_y"],
            )
            pair_seconds = statistics.median(reference_times) + statistics.median(target_times)
            budget_row["methods"][method_name] = {
                "detector": result_reference.metadata["detector"],
                "reference_keypoints": len(features_reference.xy),
                "target_keypoints": len(features_target.xy),
                "reference_repeatability_distribution": keypoint_distribution(
                    features_reference.xy, width, height
                ),
                "target_repeatability_distribution": keypoint_distribution(
                    features_target.xy, width, height
                ),
                "detector_repeatability": response,
                "reference_detection": timing_summary(reference_times),
                "target_detection": timing_summary(target_times),
                "pair_detection_s_median": pair_seconds,
                "pair_fps": 1.0 / pair_seconds,
                "reference_dispatch_and_readback_ms": result_reference.metadata[
                    "dispatch_and_readback_ms"
                ],
                "target_dispatch_and_readback_ms": result_target.metadata[
                    "dispatch_and_readback_ms"
                ],
                "reference_eigenvalue_median": float(
                    np.median(result_reference.minimum_eigenvalues)
                )
                if len(result_reference.minimum_eigenvalues)
                else None,
                "reference_condition_p95": float(
                    np.percentile(result_reference.condition_numbers, 95)
                )
                if len(result_reference.condition_numbers)
                else None,
                "readback_bytes_per_image": result_reference.metadata["readback_bytes"],
                "settings": {
                    "stride_px": stride,
                    "tensor_radius_px": radius,
                    "minimum_eigenvalue": settings.minimum_eigenvalue,
                    "minimum_score": settings.minimum_score,
                    "quota_tiles": result_reference.metadata["quota_tiles"],
                },
            }
        row["budgets"][str(budget)] = budget_row
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--learned-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--progressive-root",
        type=Path,
        default=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile"),
    )
    parser.add_argument("--stress-root", type=Path, default=ROOT / "results/uniaxial/tensile_60pct")
    parser.add_argument("--cases", nargs="*", help="Case names; default runs every available case")
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--software-webgpu", action="store_true")
    args = parser.parse_args()
    if any(value < 1 for value in args.budgets) or args.repeats < 1:
        parser.error("budgets and repeats must be positive")
    if args.learned_weights is not None and not args.learned_weights.is_file():
        parser.error(f"DIC learned weights were not found: {args.learned_weights}")
    cases = load_cases(args.progressive_root, args.stress_root)
    if args.cases:
        cases = [case for case in cases if case["name"] in args.cases]
        missing = sorted(set(args.cases) - {case["name"] for case in cases})
        if missing:
            parser.error(f"Unknown or unavailable cases: {', '.join(missing)}")
    rows = []
    device_type = "auto" if args.software_webgpu else "gpu"
    with WGPUMatching(device_type=device_type, power_preference="low-power") as matcher:
        for case in cases:
            print(f"case={case['name']} adapter={matcher.device_name}", flush=True)
            rows.append(
                run_case(
                    case,
                    matcher,
                    tuple(sorted(set(args.budgets))),
                    args.stride,
                    args.radius,
                    args.repeats,
                    args.learned_weights,
                )
            )
        metadata = {
            "protocol": "native-resolution WGSL tensor and learned score detectors; analytic one-to-one mutual-nearest repeatability; image loading excluded; one warm-up per detector and budget",
            "adapter": matcher.device_name,
            "image_size_scope": "source images retained at their native dimensions; no resizing or tiling",
            "implementations": [
                "WGSL tensor response",
                "optional learned WGSL score network",
                "WGSL NMS and per-tile score quotas",
            ],
            "learned_weights_path": str(args.learned_weights) if args.learned_weights else None,
            "learned_weights_sha256": hashlib.sha256(args.learned_weights.read_bytes()).hexdigest()
            if args.learned_weights
            else None,
            "timing_repeats": args.repeats,
            "cases": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = args.output.with_suffix(".md")
    lines = [
        "# DIC-Detector: native-resolution WebGPU comparison",
        "",
        f"Adapter: {metadata['adapter']}. Inference runs on source images at native resolution. Medians use {args.repeats} calls after one warm-up. The full detector time includes texture upload, allocations, WGSL dispatches, and the small selected-point readback.",
        "",
        "| Case | Size | Detector | Budget | Points returned (ref/target) | Repeatability ≤1 px | ≤2 px | ≤5 px | Pair s | Pair FPS | Dispatch + readback ms/image | Readback bytes/image |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in rows:
        width, height = case["image_size_px"]
        for budget, budget_row in case["budgets"].items():
            for method, result in budget_row["methods"].items():
                rep = result["detector_repeatability"]
                lines.append(
                    f"| {case['name']} | {width}×{height} | {result['detector']} ({method}) | {budget} | "
                    f"{result['reference_keypoints']}/{result['target_keypoints']} | "
                    f"{rep['at_1px']}/{rep['denominator_min_visible']} | "
                    f"{rep['at_2px']}/{rep['denominator_min_visible']} | "
                    f"{rep['at_5px']}/{rep['denominator_min_visible']} | "
                    f"{result['pair_detection_s_median']:.3f} | {result['pair_fps']:.2f} | "
                    f"{result['target_dispatch_and_readback_ms']:.2f} | "
                    f"{result['readback_bytes_per_image']} |"
                )
    lines.extend(
        (
            "",
            "Per-point coordinates, confidence, conditioning, timing repeats, and analytic truth are in the JSON file.",
            "",
        )
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {report}", flush=True)


if __name__ == "__main__":
    main()
