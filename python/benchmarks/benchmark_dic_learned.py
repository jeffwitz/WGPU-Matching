"""Compare tensor and learned DIC score selection on held-out generated speckles."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from train_dic_detector import _repeatability
from wgpu_matching import DICDetectorOptions, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "results/features/dic_detector_training_20261107/dic_detector_weights.npz"
DEFAULT_OUTPUT = ROOT / "results/features/dic_detector_heldout_20261107.json"


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0, 1) * 255.0).astype(np.uint8)


def _timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(values) * 1000.0,
        "p10_ms": float(np.percentile(values, 10) * 1000.0),
        "p90_ms": float(np.percentile(values, 90) * 1000.0),
        "repeats_s": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--texture-seed-start", type=int, default=4_000_000)
    parser.add_argument("--warp-seed-start", type=int, default=5_000_000)
    parser.add_argument("--device-type", choices=("gpu", "cpu", "auto"), default="gpu")
    args = parser.parse_args()
    if min(args.pairs, args.image_size, args.budget, args.repeats) < 1:
        parser.error("pairs, image-size, budget, and repeats must be positive")
    if not args.weights.is_file():
        parser.error(f"DIC model weights were not found: {args.weights}")

    options = DICDetectorOptions(max_keypoints=args.budget)
    records: list[dict[str, Any]] = []
    elapsed: dict[str, list[float]] = {"tensor": [], "learned": []}
    aggregate = {
        method: {threshold: {"correct": 0, "denominator": 0} for threshold in (1, 2, 5)}
        for method in ("tensor", "learned")
    }
    aggregate_by_mode: dict[str, dict[str, dict[int, dict[str, int]]]] = {}
    elapsed_by_mode: dict[str, dict[str, list[float]]] = {}
    with WGPUMatching(device_type=args.device_type, power_preference="low-power") as matcher:
        for index in range(args.pairs):
            pair = generate_speckle_pair(
                args.texture_seed_start + index,
                args.warp_seed_start + index,
                width=args.image_size,
                height=args.image_size,
            )
            mode = str(pair.metadata["motion_mode"])
            aggregate_by_mode.setdefault(
                mode,
                {
                    method: {threshold: {"correct": 0, "denominator": 0} for threshold in (1, 2, 5)}
                    for method in ("tensor", "learned")
                },
            )
            elapsed_by_mode.setdefault(mode, {"tensor": [], "learned": []})
            reference = _uint8(pair.reference)
            target = _uint8(pair.target)
            methods = {
                "tensor": {"learned_weights_path": None},
                "learned": {"learned_weights_path": args.weights},
            }
            case_record: dict[str, Any] = {
                "index": index,
                "texture_seed": pair.metadata["texture_seed"],
                "warp_seed": pair.metadata["warp_seed"],
                "motion_mode": pair.metadata["motion_mode"],
                "visible_fraction": pair.metadata["target_visible_fraction"],
                "affine_matrix": pair.metadata["affine_matrix"],
                "translation_px": pair.metadata["translation_px"],
                "repeatability": {},
            }
            for name, kwargs in methods.items():
                matcher.detect_dic(reference, options=options, **kwargs)
                matcher.detect_dic(target, options=options, **kwargs)
                reference_times: list[float] = []
                target_times: list[float] = []
                reference_result = None
                target_result = None
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    reference_result = matcher.detect_dic(reference, options=options, **kwargs)
                    reference_times.append(time.perf_counter() - started)
                    started = time.perf_counter()
                    target_result = matcher.detect_dic(target, options=options, **kwargs)
                    target_times.append(time.perf_counter() - started)
                assert reference_result is not None and target_result is not None
                elapsed[name].append(
                    statistics.median(reference_times) + statistics.median(target_times)
                )
                elapsed_by_mode[mode][name].append(elapsed[name][-1])
                values = _repeatability(
                    reference_result.features.xy,
                    target_result.features.xy,
                    pair,
                )
                case_record["repeatability"][name] = {
                    "reference_keypoints": len(reference_result.features.xy),
                    "target_keypoints": len(target_result.features.xy),
                    "result": values,
                    "reference_time_ms": _timing_summary(reference_times),
                    "target_time_ms": _timing_summary(target_times),
                    "diagnostic": reference_result.metadata["score_interpretation"],
                }
                for threshold in (1, 2, 5):
                    aggregate[name][threshold]["correct"] += values[f"at_{threshold}px"]
                    aggregate[name][threshold]["denominator"] += values["denominator"]
                    aggregate_by_mode[mode][name][threshold]["correct"] += values[
                        f"at_{threshold}px"
                    ]
                    aggregate_by_mode[mode][name][threshold]["denominator"] += values["denominator"]
            records.append(case_record)
            print(
                f"pair {index + 1}/{args.pairs}: {pair.metadata['motion_mode']} "
                f"tensor={case_record['repeatability']['tensor']['result']['repeatability_2px']:.3f} "
                f"learned={case_record['repeatability']['learned']['result']['repeatability_2px']:.3f}",
                flush=True,
            )

        adapter = matcher.device_name

    summary = {
        method: {
            "pair_detection_s_median": statistics.median(times),
            "pair_fps": 1.0 / statistics.median(times),
            "pair_detection_s_repeats": times,
            "repeatability": {
                f"at_{threshold}px": {
                    "correct": counts["correct"],
                    "denominator": counts["denominator"],
                    "fraction": counts["correct"] / max(1, counts["denominator"]),
                }
                for threshold, counts in aggregate[method].items()
            },
        }
        for method, times in elapsed.items()
    }
    mode_summary = {
        mode: {
            method: {
                "pair_count": len(elapsed_by_mode[mode][method]),
                "pair_detection_s_median": statistics.median(elapsed_by_mode[mode][method]),
                "repeatability": {
                    f"at_{threshold}px": {
                        "correct": counts[threshold]["correct"],
                        "denominator": counts[threshold]["denominator"],
                        "fraction": counts[threshold]["correct"]
                        / max(1, counts[threshold]["denominator"]),
                    }
                    for threshold in (1, 2, 5)
                },
            }
            for method, counts in methods.items()
        }
        for mode, methods in aggregate_by_mode.items()
    }
    report = {
        "protocol": "paired held-out synthetic speckles; same texture warped by known affine/smooth field; native-resolution tensor and learned score networks; no descriptor matching or IC-GN",
        "adapter": adapter,
        "image_size_px": [args.image_size, args.image_size],
        "requested_budget": args.budget,
        "repeats_after_warmup": args.repeats,
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "seed_ranges": {
            "texture": [args.texture_seed_start, args.texture_seed_start + args.pairs - 1],
            "warp": [args.warp_seed_start, args.warp_seed_start + args.pairs - 1],
        },
        "summary": summary,
        "summary_by_motion_mode": mode_summary,
        "cases": records,
        "scientific_limit": "Keypoint repeatability is not match precision, local-affine coverage, or IC-GN convergence. No descriptor is used in this detector-only comparison.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    lines = [
        "# DIC-Detector held-out synthetic evaluation",
        "",
        f"Adapter: {adapter}; image size {args.image_size}×{args.image_size}; {args.pairs} held-out texture/warp pairs; budget {args.budget}; medians after {args.repeats} warmed repeats.",
        "",
        "| Detector | Repeatability ≤1 px | ≤2 px | ≤5 px | Pair detection median | Pair FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary.items():
        reps = values["repeatability"]
        lines.append(
            f"| {method} | {reps['at_1px']['correct']}/{reps['at_1px']['denominator']} "
            f"({reps['at_1px']['fraction']:.1%}) | "
            f"{reps['at_2px']['correct']}/{reps['at_2px']['denominator']} "
            f"({reps['at_2px']['fraction']:.1%}) | "
            f"{reps['at_5px']['correct']}/{reps['at_5px']['denominator']} "
            f"({reps['at_5px']['fraction']:.1%}) | "
            f"{values['pair_detection_s_median']:.4f} s | {values['pair_fps']:.2f} |"
        )
    lines.extend(
        (
            "",
            "This detector-only evaluation measures repeatability of selected locations. It does not use a descriptor and does not measure matching or IC-GN convergence.",
            "The breakdown below separates translation, rotation, affine, smooth heterogeneous, and discontinuous motion; a mode with few held-out pairs should be treated as exploratory.",
            "",
            "| Motion | Detector | Pairs | Repeatability ≤1 px | ≤2 px | ≤5 px | Median pair time |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for mode, methods in mode_summary.items():
        for method, values in methods.items():
            reps = values["repeatability"]
            lines.append(
                f"| {mode} | {method} | {values['pair_count']} | "
                f"{reps['at_1px']['correct']}/{reps['at_1px']['denominator']} "
                f"({reps['at_1px']['fraction']:.1%}) | "
                f"{reps['at_2px']['correct']}/{reps['at_2px']['denominator']} "
                f"({reps['at_2px']['fraction']:.1%}) | "
                f"{reps['at_5px']['correct']}/{reps['at_5px']['denominator']} "
                f"({reps['at_5px']['fraction']:.1%}) | "
                f"{values['pair_detection_s_median']:.4f} s |"
            )
    lines.extend(
        (
            "",
            "Per-case metrics, motion parameters, timing repeats, and checkpoint SHA-256 are in the adjacent JSON file.",
            "",
        )
    )
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {markdown}", flush=True)


if __name__ == "__main__":
    main()
