"""Evaluate the EfficientLoFTR ONNX export on analytic DIC tensile pairs.

This optional experiment uses ONNX Runtime's native WebGPU execution provider
and the repository's WGSL local-affine RANSAC. The ONNX model is an external
input; neither ONNX Runtime nor model weights are package dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from ort_profile import summarize_profile
from PIL import Image
from wgpu_matching import WGPUMatching

try:
    import resource
except ImportError:  # resource is unavailable on Windows.
    resource = None

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESSIVE = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
DEFAULT_STRESS = REPOSITORY / "results/uniaxial/tensile_60pct"
PROGRESSIVE_FRAMES = (40, 100, 199)
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
PATCH_SIZE = 128
SEED_STRIDE = 512


def load_gray(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(f"Expected a 3840x2160 image at {path}, got {image.size}.")
    gray = np.asarray(image.resize((width, height)), dtype=np.float32) / 255.0
    padded_height = ((height + 31) // 32) * 32
    padded_width = ((width + 31) // 32) * 32
    return np.pad(gray, ((0, padded_height - height), (0, padded_width - width)))


def read_truth(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["frame"]): {key: float(value) for key, value in row.items() if key != "frame"}
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
    input_names = {item.name for item in session.get_inputs()}
    if input_names != {"image0", "image1"}:
        raise ValueError(f"Expected model inputs image0/image1, got {sorted(input_names)}.")
    feeds = {"image0": reference[None, None], "image1": target[None, None]}
    scale_x = IMAGE_WIDTH / width
    scale_y = IMAGE_HEIGHT / height
    seeds = visible_seeds(truth["stretch_x"], truth["stretch_y"])

    def infer_and_fit() -> tuple[Any, Any, Any, Any, float, float, float]:
        start = time.perf_counter()
        source, destination, confidence = session.run(None, feeds)
        inference_s = time.perf_counter() - start
        fit_start = time.perf_counter()
        source = np.asarray(source, dtype=np.float32)
        destination = np.asarray(destination, dtype=np.float32)
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
        fit_s = time.perf_counter() - fit_start
        return (
            source,
            destination,
            np.asarray(confidence),
            fit,
            inference_s + fit_s,
            inference_s,
            fit_s,
        )

    infer_and_fit()  # Warm up ONNX Runtime and compile shared WGSL fit pipelines.
    times_s = []
    inference_times_s = []
    fit_times_s = []
    last_source = np.empty((0, 2), dtype=np.float32)
    last_destination = np.empty((0, 2), dtype=np.float32)
    last_confidence = np.empty((0,), dtype=np.float32)
    last_fit = None
    for _ in range(repeats):
        last_source, last_destination, last_confidence, last_fit, elapsed, inference_s, fit_s = (
            infer_and_fit()
        )
        times_s.append(elapsed)
        inference_times_s.append(inference_s)
        fit_times_s.append(fit_s)

    stretch_x = truth["stretch_x"]
    stretch_y = truth["stretch_y"]
    expected = np.column_stack(
        (
            1920 + stretch_x * (last_source[:, 0] - 1920),
            1080 + stretch_y * (last_source[:, 1] - 1080),
        )
    )
    errors = np.linalg.norm(last_destination - expected, axis=1)
    visible = (
        (last_source[:, 0] >= 0)
        & (last_source[:, 0] < IMAGE_WIDTH)
        & (last_source[:, 1] >= 0)
        & (last_source[:, 1] < IMAGE_HEIGHT)
        & (expected[:, 0] >= 0)
        & (expected[:, 0] < IMAGE_WIDTH)
        & (expected[:, 1] >= 0)
        & (expected[:, 1] < IMAGE_HEIGHT)
    )
    raw_errors = errors[visible].tolist()
    assert last_fit is not None
    per_seed = []
    for seed, match in zip(seeds, last_fit.matches, strict=True):
        x, y = seed
        exact_target = np.asarray((1920 + stretch_x * (x - 1920), 1080 + stretch_y * (y - 1080)))
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
        "input_tensor_size_px": [reference.shape[1], reference.shape[0]],
        "requested_visible_seeds": len(seeds),
        "raw_match_count": len(last_source),
        "raw_confidence_median": float(np.median(last_confidence))
        if len(last_confidence)
        else None,
        "raw_error": quantiles(raw_errors),
        "raw_matches_le2_px": sum(error <= 2 for error in raw_errors),
        "raw_matches_lt5_px": sum(error < 5 for error in raw_errors),
        "local_affine_accepted": len(accepted_errors),
        "local_affine_le2_px": sum(error <= 2 for error in accepted_errors),
        "local_affine_lt5_px": sum(error < 5 for error in accepted_errors),
        "local_affine_error": quantiles(accepted_errors),
        "warmup_excluded": True,
        "end_to_end_repeat_times_s": times_s,
        "end_to_end_median_s": float(statistics.median(times_s)),
        "ort_inference_repeat_times_s": inference_times_s,
        "ort_inference_median_s": float(statistics.median(inference_times_s)),
        "postprocess_and_affine_fit_repeat_times_s": fit_times_s,
        "postprocess_and_affine_fit_median_s": float(statistics.median(fit_times_s)),
        "fps": 1.0 / float(statistics.median(times_s)),
        "fit_gpu_compute_ms_last_repeat": last_fit.gpu_compute_ms,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--progressive-dir", type=Path, default=DEFAULT_PROGRESSIVE)
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS)
    parser.add_argument(
        "--output", type=Path, help="Output JSON path (default: unique dated run file)."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--intra-op-threads", type=int, default=8)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable ORT trace profiling (diagnostic; adds overhead).",
    )
    parser.add_argument(
        "--provider",
        choices=("WebGpuExecutionProvider", "CPUExecutionProvider"),
        default="WebGpuExecutionProvider",
    )
    args = parser.parse_args()
    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = REPOSITORY / "results/features" / f"efficientloftr_run_{stamp}.json"
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
    options.enable_profiling = args.profile
    if args.provider == "WebGpuExecutionProvider":
        try:
            import onnxruntime_ep_webgpu as webgpu_ep
        except ImportError as exc:
            parser.error(f"Install onnxruntime-ep-webgpu to run this study: {exc}")
        ort.register_execution_provider_library(
            "efficientloftr_webgpu", webgpu_ep.get_library_path()
        )
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()
        ]
        if not devices:
            parser.error("The WebGPU provider loaded, but no compatible adapter was discovered.")
        options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
    elif args.provider not in ort.get_available_providers():
        parser.error(f"Provider {args.provider!r} is unavailable: {ort.get_available_providers()}")

    session_start = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=None if args.provider == "WebGpuExecutionProvider" else [args.provider],
    )
    session_initialization_s = time.perf_counter() - session_start
    progressive_truth = read_truth(args.progressive_dir / "machine_truth.csv")
    stress_truth = read_truth(args.stress_dir / "machine_truth.csv")
    prep_start = time.perf_counter()
    progressive_reference = load_gray(
        args.progressive_dir / "reference.png", args.width, args.height
    )
    progressive_reference_prep_s = time.perf_counter() - prep_start
    prep_start = time.perf_counter()
    stress_reference = load_gray(args.stress_dir / "reference.png", args.width, args.height)
    stress_reference_prep_s = time.perf_counter() - prep_start
    report = {
        "protocol": {
            "method": "EfficientLoFTR outdoor model ONNX export",
            "model_sha256": sha256_file(args.model),
            "model_path": str(args.model),
            "model_source": "https://huggingface.co/SpatialHub/efficient-loftr-onnx",
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
            "ort_profiling_enabled": args.profile,
            "session_initialization_s": session_initialization_s,
            "reference_decode_resize_s": {
                "progressive": progressive_reference_prep_s,
                "stress_60pct": stress_reference_prep_s,
            },
            "provider_requested": args.provider,
            "session_providers": session.get_providers(),
            "intra_op_threads": args.intra_op_threads,
            "timing_scope": "ONNX inference plus local affine fit; decode and resize excluded",
            "notes": [
                "Exploratory run at reduced resolution; not a full-resolution performance result.",
                "This outdoor-scene model is evaluated on speckle images; domain mismatch is possible.",
                "Some operators may be delegated to CPU by ONNX Runtime's WebGPU execution provider.",
            ],
        },
        "results": [],
    }
    with WGPUMatching(device_type="gpu") as matcher:
        report["protocol"]["wgpu_adapter"] = str(matcher.adapter.info)
        for frame in PROGRESSIVE_FRAMES:
            prep_start = time.perf_counter()
            target = load_gray(
                args.progressive_dir / f"frame_{frame:03d}.png", args.width, args.height
            )
            target_prep_s = time.perf_counter() - prep_start
            result = evaluate_pair(
                session,
                matcher,
                progressive_reference,
                target,
                frame,
                progressive_truth[frame],
                args.width,
                args.height,
                args.repeats,
            )
            result["target_decode_resize_s"] = target_prep_s
            report["results"].append(result)
        prep_start = time.perf_counter()
        target = load_gray(args.stress_dir / "frame_199_eng_60pct.png", args.width, args.height)
        target_prep_s = time.perf_counter() - prep_start
        result = evaluate_pair(
            session,
            matcher,
            stress_reference,
            target,
            199,
            stress_truth[199],
            args.width,
            args.height,
            args.repeats,
        )
        result["target_decode_resize_s"] = target_prep_s
        report["results"].append(result)

    if args.profile:
        profile_path = Path(session.end_profiling())
        report["protocol"]["ort_diagnostics"] = summarize_profile(profile_path)
        os.unlink(profile_path)
    if resource is not None:
        rss_scale = 1 if sys.platform == "darwin" else 1024
        report["protocol"]["process_peak_rss_bytes"] = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
        )
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
