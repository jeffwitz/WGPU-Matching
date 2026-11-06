"""Evaluate the published RaCo-ALIKED-LightGlue+ ONNX pipeline on DIC images.

This optional experiment uses ONNX Runtime's native WebGPU execution provider
and the repository's WGSL local-affine RANSAC. It does not add ONNX Runtime to
the package dependencies. Input images are resized to the requested evaluation
size to make the model's high-resolution cost explicit and configurable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from wgpu_matching import WGPUMatching

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESSIVE = Path(
    "/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile"
)
DEFAULT_STRESS = REPOSITORY / "results/uniaxial/tensile_60pct"
DEFAULT_OUTPUT = REPOSITORY / "results/features/raco_aliked_lightglue_webgpu_study.json"
PROGRESSIVE_FRAMES = (40, 100, 199)
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
PATCH_SIZE = 128
SEED_STRIDE = 512


def load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(f"Expected a 3840x2160 image at {path}, got {image.size}.")
    rgb = np.asarray(image.resize((width, height)), dtype=np.float32) / 255.0
    padded_height = ((height + 31) // 32) * 32
    padded_width = ((width + 31) // 32) * 32
    return np.pad(rgb, ((0, padded_height - height), (0, padded_width - width), (0, 0)))


def read_truth(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["frame"]): {
                key: float(value) for key, value in row.items() if key != "frame"
            }
            for row in csv.DictReader(stream)
        }


def visible_seeds(stretch_x: float, stretch_y: float) -> list[tuple[float, float]]:
    centers = []
    offset = (PATCH_SIZE - 1) / 2
    for row in range(1 + (IMAGE_HEIGHT - PATCH_SIZE) // SEED_STRIDE):
        for column in range(1 + (IMAGE_WIDTH - PATCH_SIZE) // SEED_STRIDE):
            x = column * SEED_STRIDE + offset
            y = row * SEED_STRIDE + offset
            target_x = 1920 + stretch_x * (x - 1920)
            target_y = 1080 + stretch_y * (y - 1080)
            if 0 <= target_x < IMAGE_WIDTH and 0 <= target_y < IMAGE_HEIGHT:
                centers.append((x, y))
    return centers


def quantiles(errors: list[float]) -> dict[str, float | None]:
    if not errors:
        return {"median_px": None, "p95_px": None, "rmse_px": None}
    return {
        "median_px": float(statistics.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "rmse_px": float(np.mean(np.square(errors)) ** 0.5),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_pair(
    session: Any,
    matcher: WGPUMatching,
    reference: np.ndarray,
    target: np.ndarray,
    frame: int,
    truth: dict[str, float],
    width: int,
    height: int,
    repeats: int,
) -> dict[str, Any]:
    input_name = session.get_inputs()[0].name
    image_pair = np.stack((reference, target)).transpose(0, 3, 1, 2)
    scale_x = IMAGE_WIDTH / width
    scale_y = IMAGE_HEIGHT / height
    seeds = visible_seeds(truth["stretch_x"], truth["stretch_y"])

    def infer_and_fit() -> tuple[Any, Any, Any, Any, float, Any]:
        start = time.perf_counter()
        keypoints, matches, scores = session.run(None, {input_name: image_pair})
        pair_matches = matches[matches[:, 0] == 0]
        source = keypoints[0, pair_matches[:, 1].astype(np.int64)].astype(np.float32)
        destination = keypoints[1, pair_matches[:, 2].astype(np.int64)].astype(np.float32)
        source[:, 0] *= scale_x
        source[:, 1] *= scale_y
        destination[:, 0] *= scale_x
        destination[:, 1] *= scale_y
        fit = matcher.fit_local_affine(
            np.column_stack((source, destination)),
            np.asarray(seeds, dtype=np.float32),
            radius_px=256,
            threshold_px=3,
            min_inliers=6,
            min_candidates=8,
        )
        return keypoints, pair_matches, source, destination, time.perf_counter() - start, fit

    # Warm up ORT and compile the shared WGSL affine-fit pipelines.
    infer_and_fit()
    times_s = []
    last_fit = None
    last_source = None
    last_destination = None
    last_raw_matches = 0
    last_raw_errors: list[float] = []
    stretch_x = truth["stretch_x"]
    stretch_y = truth["stretch_y"]
    for _ in range(repeats):
        keypoints, pairs, source, destination, elapsed, fit = infer_and_fit()
        times_s.append(elapsed)
        last_fit = fit
        last_source, last_destination = source, destination
        last_raw_matches = len(pairs)
        expected = np.column_stack(
            (
                1920 + stretch_x * (source[:, 0] - 1920),
                1080 + stretch_y * (source[:, 1] - 1080),
            )
        )
        errors = np.linalg.norm(destination - expected, axis=1)
        visible = (
            (source[:, 0] >= 0)
            & (source[:, 0] < IMAGE_WIDTH)
            & (source[:, 1] >= 0)
            & (source[:, 1] < IMAGE_HEIGHT)
            & (expected[:, 0] >= 0)
            & (expected[:, 0] < IMAGE_WIDTH)
            & (expected[:, 1] >= 0)
            & (expected[:, 1] < IMAGE_HEIGHT)
        )
        last_raw_errors = errors[visible].tolist()

    assert last_fit is not None and last_source is not None and last_destination is not None
    per_seed = []
    for seed, match in zip(seeds, last_fit.matches, strict=True):
        x, y = seed
        exact_target = np.asarray(
            (1920 + stretch_x * (x - 1920), 1080 + stretch_y * (y - 1080))
        )
        predicted = np.asarray((match.x + match.dx, match.y + match.dy))
        error = float(np.linalg.norm(predicted - exact_target)) if match.usable else None
        per_seed.append(
            {
                "seed": [x, y],
                "target_analytic": exact_target.tolist(),
                "target_estimated": predicted.tolist() if match.usable else None,
                "error_px": error,
                "usable": match.usable,
                "inliers": match.inliers,
                "candidates": match.candidates,
                "inlier_fraction": match.inlier_fraction,
                "residual_px": match.residual,
            }
        )
    accepted_errors = [float(item["error_px"]) for item in per_seed if item["usable"]]
    return {
        "frame": frame,
        "engineering_axial": truth["engineering_axial"],
        "stretch_x": stretch_x,
        "stretch_y": stretch_y,
        "input_size_px": [width, height],
        "input_tensor_size_px": [image_pair.shape[3], image_pair.shape[2]],
        "requested_visible_seeds": len(seeds),
        "keypoints_per_image": int(keypoints.shape[1]),
        "raw_match_count": last_raw_matches,
        "raw_error": quantiles(last_raw_errors),
        "raw_matches_le2_px": sum(error <= 2 for error in last_raw_errors),
        "raw_matches_lt5_px": sum(error < 5 for error in last_raw_errors),
        "local_affine_accepted": len(accepted_errors),
        "local_affine_le2_px": sum(error <= 2 for error in accepted_errors),
        "local_affine_lt5_px": sum(error < 5 for error in accepted_errors),
        "local_affine_error": quantiles(accepted_errors),
        "warmup_excluded": True,
        "end_to_end_repeat_times_s": times_s,
        "end_to_end_median_s": float(statistics.median(times_s)),
        "fps": 1.0 / float(statistics.median(times_s)),
        "fit_gpu_compute_ms_last_repeat": last_fit.gpu_compute_ms,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--progressive-dir", type=Path, default=DEFAULT_PROGRESSIVE)
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--intra-op-threads", type=int, default=8)
    parser.add_argument("--provider", choices=("WebGpuExecutionProvider", "CPUExecutionProvider"), default="WebGpuExecutionProvider")
    args = parser.parse_args()
    if args.repeats < 1 or args.intra_op_threads < 1:
        parser.error("--repeats and --intra-op-threads must be positive")
    if args.width < 32 or args.height < 32:
        parser.error("--width and --height must both be at least 32 pixels")
    if not args.model.is_file():
        parser.error(f"ONNX model not found: {args.model}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        parser.error(f"Install optional ONNX Runtime to run this study: {exc}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.enable_profiling = True
    webgpu_devices = []
    if args.provider == "WebGpuExecutionProvider":
        try:
            import onnxruntime_ep_webgpu as webgpu_ep
        except ImportError as exc:
            parser.error(f"Install onnxruntime-ep-webgpu to run this study: {exc}")
        ort.register_execution_provider_library("raco_aliked_webgpu", webgpu_ep.get_library_path())
        webgpu_devices = [
            device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()
        ]
        if not webgpu_devices:
            parser.error("The WebGPU provider loaded, but no compatible adapter was discovered.")
        options.add_provider_for_devices(webgpu_devices, {"preferredLayout": "NHWC"})
    elif args.provider not in ort.get_available_providers():
        parser.error(f"Provider {args.provider!r} is unavailable: {ort.get_available_providers()}")

    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=None if args.provider == "WebGpuExecutionProvider" else [args.provider],
    )
    expected_names = ["images"]
    if [item.name for item in session.get_inputs()] != expected_names:
        parser.error(f"Unexpected model inputs: {[item.name for item in session.get_inputs()]}")

    progressive_truth = read_truth(args.progressive_dir / "machine_truth.csv")
    stress_truth = read_truth(args.stress_dir / "machine_truth.csv")
    progressive_reference = load_rgb(args.progressive_dir / "reference.png", args.width, args.height)
    stress_reference = load_rgb(args.stress_dir / "reference.png", args.width, args.height)
    report = {
        "protocol": {
            "method": "RaCo-ALIKED-LightGlue+ published end-to-end ONNX model",
            "model_sha256": sha256_file(args.model),
            "model_path": str(args.model),
            "model_source": "https://github.com/fabio-sim/LightGlue-ONNX/releases/tag/v3.0",
            "input_size_px": [args.width, args.height],
            "original_image_size_px": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "input_resize_scale": [IMAGE_WIDTH / args.width, IMAGE_HEIGHT / args.height],
            "padding": "bottom/right only to next multiple of 32",
            "seed_patch_size_px": PATCH_SIZE,
            "seed_stride_px": SEED_STRIDE,
            "local_affine": {
                "implementation": "repository HORG WGSL RANSAC",
                "radius_px": 256,
                "threshold_px": 3,
                "minimum_inliers": 6,
                "minimum_candidates": 8,
            },
            "warmup_calls": 1,
            "timed_repeats": args.repeats,
            "provider_requested": args.provider,
            "session_providers": session.get_providers(),
            "intra_op_threads": args.intra_op_threads,
            "timing_scope": "ONNX inference plus local affine fit; decode and resize excluded",
            "notes": [
                "This is an exploratory run at reduced image resolution, not a full-resolution speed comparison.",
                "Raw pairs and seed-level affine initializations are evaluated separately against analytic truth.",
                "WebGPU execution provider can delegate some operators to CPU; provider presence alone does not imply all-GPU execution.",
            ],
        },
        "results": [],
    }
    with WGPUMatching(device_type="gpu") as matcher:
        report["protocol"]["wgpu_adapter"] = str(matcher.adapter.info)
        for frame in PROGRESSIVE_FRAMES:
            report["results"].append(
                evaluate_pair(
                    session,
                    matcher,
                    progressive_reference,
                    load_rgb(args.progressive_dir / f"frame_{frame:03d}.png", args.width, args.height),
                    frame,
                    progressive_truth[frame],
                    args.width,
                    args.height,
                    args.repeats,
                )
            )
        report["results"].append(
            evaluate_pair(
                session,
                matcher,
                stress_reference,
                load_rgb(args.stress_dir / "frame_199_eng_60pct.png", args.width, args.height),
                199,
                stress_truth[199],
                args.width,
                args.height,
                args.repeats,
            )
        )

    profile_path = Path(session.end_profiling())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [
        event
        for event in profile
        if event.get("cat") == "Node" and event.get("ph") == "X"
    ]
    report["protocol"]["ort_profile_node_events_by_provider"] = {
        provider: sum(event.get("args", {}).get("provider") == provider for event in node_events)
        for provider in sorted(
            {event.get("args", {}).get("provider", "unknown") for event in node_events}
        )
    }
    os.unlink(profile_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in report["results"]:
        print(
            f"frame {item['frame']}: {item['local_affine_accepted']}/"
            f"{item['requested_visible_seeds']} accepted, "
            f"{item['local_affine_le2_px']} <=2 px, {item['local_affine_lt5_px']} <5 px, "
            f"{item['end_to_end_median_s']:.3f} s ({item['fps']:.3f} FPS)"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
