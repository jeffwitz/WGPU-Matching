"""Evaluate an exported XFeat ONNX matcher on analytic tensile pairs.

This is an optional research experiment. It does not add ONNX Runtime to the
package dependencies or alter the production matchers. Inputs are full-resolution
RGB tensors in [0, 1], matching the XFeat-ONNX reference preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from wgpu_matching import WGPUMatching

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESSIVE = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
DEFAULT_STRESS = REPOSITORY / "results/uniaxial/tensile_60pct"
DEFAULT_OUTPUT = REPOSITORY / "results/features/xfeat_onnx_tensile.json"
PROGRESSIVE_FRAMES = (40, 100, 199)
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
PATCH_SIZE = 128
SEED_STRIDE = 512
MATCH_RADII_PX = (64, 128, 256)
RANSAC_THRESHOLD_PX = 3.0
MIN_LOCAL_INLIERS = 6


def load_image(path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if gray.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Expected 3840x2160 image at {path}, got {gray.shape[::-1]}.")
    # XFeat-ONNX uses RGB/255. Replicated grayscale has the same channel order.
    return np.repeat(gray[None, None, :, :], 3, axis=1) / 255.0


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


def finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def evaluate_pair(
    session: Any,
    matcher: WGPUMatching,
    reference: np.ndarray,
    target: np.ndarray,
    frame: int,
    truth: dict[str, float],
    repeats: int,
) -> dict[str, Any]:
    input_names = [item.name for item in session.get_inputs()]
    if input_names != ["image0", "image1"]:
        raise ValueError(
            f"Expected XFeat-ONNX dense e2e input names image0/image1, got {input_names}."
        )
    feeds = {input_names[0]: reference, input_names[1]: target}
    start = time.perf_counter()
    outputs = session.run(None, feeds)
    first_call_s = time.perf_counter() - start
    run_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = session.run(None, feeds)
        run_times.append(time.perf_counter() - start)
    source, matched_target = (np.asarray(value).reshape(-1, 2) for value in outputs)
    if len(source) != len(matched_target):
        raise ValueError("XFeat returned different counts of reference and target coordinates.")

    stretch_x = truth["stretch_x"]
    stretch_y = truth["stretch_y"]
    expected_target = np.column_stack(
        (
            1920 + stretch_x * (source[:, 0] - 1920),
            1080 + stretch_y * (source[:, 1] - 1080),
        )
    )
    visible_match = (
        (source[:, 0] >= 0)
        & (source[:, 0] < IMAGE_WIDTH)
        & (source[:, 1] >= 0)
        & (source[:, 1] < IMAGE_HEIGHT)
        & (expected_target[:, 0] >= 0)
        & (expected_target[:, 0] < IMAGE_WIDTH)
        & (expected_target[:, 1] >= 0)
        & (expected_target[:, 1] < IMAGE_HEIGHT)
    )
    raw_error = np.linalg.norm(matched_target - expected_target, axis=1)[visible_match].tolist()

    seeds = visible_seeds(stretch_x, stretch_y)
    local_fits: dict[str, dict[str, Any]] = {}
    for radius in MATCH_RADII_PX:
        fit_start = time.perf_counter()
        affine = matcher.fit_local_affine(
            np.column_stack((source, matched_target)).astype(np.float32, copy=False),
            np.asarray(seeds, dtype=np.float32),
            radius_px=radius,
            threshold_px=RANSAC_THRESHOLD_PX,
            min_inliers=MIN_LOCAL_INLIERS,
        )
        records = []
        for center, match in zip(seeds, affine.matches, strict=True):
            true_target = np.array(
                (
                    1920 + stretch_x * (center[0] - 1920),
                    1080 + stretch_y * (center[1] - 1080),
                )
            )
            predicted = np.array((match.x + match.dx, match.y + match.dy))
            error = float(np.linalg.norm(predicted - true_target)) if match.inliers else None
            records.append(
                {
                    "center": [center[0], center[1]],
                    "predicted_target": predicted.tolist() if match.inliers else None,
                    "error_px": error,
                    "inliers": match.inliers,
                    "candidates": match.candidates,
                    "inlier_fraction": match.inlier_fraction,
                    "residual_px": finite_or_none(match.residual),
                    "usable": match.usable,
                }
            )
        accepted = [item for item in records if item["usable"]]
        accepted_errors = [float(item["error_px"]) for item in accepted]
        local_fits[str(radius)] = {
            "requested_visible_seeds": len(seeds),
            "fitted_count": len(accepted),
            "fitted_with_error_lt5_px": sum(error < 5 for error in accepted_errors),
            "fitted_with_error_le2_px": sum(error <= 2 for error in accepted_errors),
            "fit_error": quantiles(accepted_errors),
            "median_local_candidate_count": float(
                statistics.median(item["candidates"] for item in records)
            ),
            "fit_compute_ms": affine.gpu_compute_ms,
            "fit_call_wall_ms": (time.perf_counter() - fit_start) * 1000.0,
            "per_seed": records,
        }

    return {
        "frame": frame,
        "engineering_axial": truth["engineering_axial"],
        "stretch_x": stretch_x,
        "stretch_y": stretch_y,
        "visible_seeds": len(seeds),
        "match_count": len(source),
        "in_bounds_match_count": int(sum(visible_match)),
        "raw_pair_error_lt5_count": int(sum(error < 5 for error in raw_error)),
        "raw_pair_error_le2_count": int(sum(error <= 2 for error in raw_error)),
        "raw_pair_error": quantiles(raw_error),
        "first_call_s": first_call_s,
        "repeat_times_s": run_times,
        "repeat_median_s": float(statistics.median(run_times)),
        "local_affine": local_fits,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to XFeat-ONNX *_dense_e2e.onnx."
    )
    parser.add_argument("--progressive-dir", type=Path, default=DEFAULT_PROGRESSIVE)
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--intra-op-threads", type=int, default=8)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--wgpu-device-type", choices=("auto", "gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.repeats < 1 or args.intra_op_threads < 1:
        parser.error("--repeats and --intra-op-threads must be positive")
    if not args.model.is_file():
        parser.error(f"ONNX model not found: {args.model}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        parser.error(f"Install optional ONNX Runtime to run this study: {exc}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    options.inter_op_num_threads = 1
    provider_devices = []
    if args.provider == "WebGpuExecutionProvider":
        try:
            import onnxruntime_ep_webgpu as webgpu_ep
        except ImportError as exc:
            parser.error(f"Install the optional native WebGPU EP to run this study: {exc}")
        ort.register_execution_provider_library("xfeat_webgpu_ep", webgpu_ep.get_library_path())
        provider_devices = [
            device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()
        ]
        if not provider_devices:
            parser.error("The WebGPU EP loaded, but no WebGPU adapter was discovered.")
        options.add_provider_for_devices(provider_devices, {"preferredLayout": "NHWC"})
    elif args.provider not in ort.get_available_providers():
        parser.error(
            f"Provider {args.provider!r} unavailable; installed providers: {ort.get_available_providers()}"
        )
    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=None if args.provider == "WebGpuExecutionProvider" else [args.provider],
    )

    progressive_truth = read_truth(args.progressive_dir / "machine_truth.csv")
    stress_truth = read_truth(args.stress_dir / "machine_truth.csv")
    progressive_reference = load_image(args.progressive_dir / "reference.png")
    stress_reference = load_image(args.stress_dir / "reference.png")
    reports = []
    with WGPUMatching(device_type=args.wgpu_device_type) as matcher:
        for frame in PROGRESSIVE_FRAMES:
            reports.append(
                evaluate_pair(
                    session,
                    matcher,
                    progressive_reference,
                    load_image(args.progressive_dir / f"frame_{frame:03d}.png"),
                    frame,
                    progressive_truth[frame],
                    args.repeats,
                )
            )
        reports.append(
            evaluate_pair(
                session,
                matcher,
                stress_reference,
                load_image(args.stress_dir / "frame_199_eng_60pct.png"),
                199,
                stress_truth[199],
                args.repeats,
            )
        )

    result = {
        "experiment": "XFeat-ONNX dense end-to-end matching with local affine DIC initialization",
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "upstream": "https://github.com/DavideCatto/XFeat-ONNX/releases/tag/V1.0.0",
            "variant": "xfeat_dense_e2e.onnx",
            "license": "Apache-2.0 as declared by the ONNX export repository; upstream XFeat is Apache-2.0",
        },
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "provider_requested": args.provider,
            "provider_available": ort.get_available_providers(),
            "session_providers": session.get_providers(),
            "webgpu_devices": [
                {
                    "vendor_id": device.device.vendor_id,
                    "device_id": device.device.device_id,
                    "device_type": str(device.device.type),
                    "ep_vendor": device.ep_vendor,
                }
                for device in provider_devices
            ],
            "intra_op_threads": args.intra_op_threads,
            "wgpu_adapter": matcher.device_name,
            "repeats_after_first_call": args.repeats,
            "timed_region": "Inference and per-radius local-affine fit are timed separately; image decoding is excluded.",
        },
        "input": {
            "shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
            "range": "float32 [0,1]",
            "gray_conversion": "PIL grayscale then channel replication",
            "dic_window_px": PATCH_SIZE,
            "seed_stride_px": SEED_STRIDE,
            "local_affine_radii_px": list(MATCH_RADII_PX),
            "ransac_threshold_px": RANSAC_THRESHOLD_PX,
            "minimum_local_inliers": MIN_LOCAL_INLIERS,
            "visible_seed_rule": "only source centers with analytically in-bounds target centers are scored",
        },
        "data": {
            "progressive_dir": str(args.progressive_dir),
            "stress_dir": str(args.stress_dir),
            "stress_case": "synthetic global affine 60% tensile extrapolation; not a heterogeneous strain field",
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for report in reports:
        affine = report["local_affine"]["256"]
        print(
            f"frame {report['frame']}: {report['match_count']} pairs, "
            f"{report['repeat_median_s']:.3f}s; affine radius 256 px: "
            f"{affine['fitted_count']}/{affine['requested_visible_seeds']} fits, "
            f"{affine['fitted_with_error_lt5_px']} under 5 px"
        )


if __name__ == "__main__":
    main()
