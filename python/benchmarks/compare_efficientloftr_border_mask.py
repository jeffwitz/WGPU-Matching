"""Compare EfficientLoFTR before/after the factorized border-mask rewrite."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ort_profile import summarize_profile


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile")
DEFAULT_STRESS_DIR = REPOSITORY / "results/uniaxial/tensile_60pct"
_WEBGPU_EP_REGISTERED = False


def load_gray(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (3840, 2160):
        raise ValueError(f"Expected a 3840x2160 image at {path}, got {image.size}.")
    gray = np.asarray(image.resize((width, height)), dtype=np.float32) / 255.0
    padded_height = ((height + 31) // 32) * 32
    padded_width = ((width + 31) // 32) * 32
    return np.pad(gray, ((0, padded_height - height), (0, padded_width - width)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_session(model_path: Path, *, profile: bool = False, threads: int = 8) -> Any:
    global _WEBGPU_EP_REGISTERED
    try:
        import onnxruntime as ort
        import onnxruntime_ep_webgpu as webgpu_ep
    except ImportError as exc:
        raise RuntimeError(
            "Install onnxruntime and onnxruntime-ep-webgpu to run this study."
        ) from exc

    if not _WEBGPU_EP_REGISTERED:
        ort.register_execution_provider_library(
            "efficientloftr_border_mask_webgpu", webgpu_ep.get_library_path()
        )
        _WEBGPU_EP_REGISTERED = True
    devices = [
        device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()
    ]
    if not devices:
        raise RuntimeError("The WebGPU provider loaded, but no compatible adapter was discovered.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.enable_profiling = profile
    options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=None)
    return session


def model_metadata(model_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def model_graph_stats(model_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {"node_count": len(model.graph.node), "operator_counts": counts}


def _compare_outputs(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        return {"exact": False, "reason": "output_count_mismatch"}
    rows = []
    exact = True
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if left.shape != right.shape or left.dtype != right.dtype:
            rows.append(
                {
                    "index": index,
                    "reference_shape": list(left.shape),
                    "candidate_shape": list(right.shape),
                    "reference_dtype": str(left.dtype),
                    "candidate_dtype": str(right.dtype),
                    "exact": False,
                }
            )
            exact = False
            continue
        array_equal = bool(np.array_equal(left, right))
        exact &= array_equal
        max_abs = (
            float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            if left.size
            else 0.0
        )
        rows.append(
            {
                "index": index,
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "exact": array_equal,
                "max_absolute_difference": max_abs,
            }
        )
    return {"exact": exact, "outputs": rows}


def _profile_model(
    model_path: Path,
    image0: np.ndarray,
    image1: np.ndarray,
    *,
    threads: int,
) -> dict[str, Any]:
    session = create_session(model_path, profile=True, threads=threads)
    session.run(None, {"image0": image0[None, None], "image1": image1[None, None]})
    profile_path = Path(session.end_profiling())
    summary = summarize_profile(profile_path)
    profile_path.unlink(missing_ok=True)
    del session
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--optimized-model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS_DIR)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--profile", action="store_true", help="Profile one run per model and case."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("--repeats and --threads must be positive")
    for model_path in (args.baseline_model, args.optimized_model):
        if not model_path.is_file():
            parser.error(f"Model not found: {model_path}")

    cases = [
        (f"frame_{frame:03d}", args.image_dir / f"frame_{frame:03d}.png")
        for frame in (40, 100, 199)
    ]
    cases.append(("tensile_60pct", args.stress_dir / "frame_199_eng_60pct.png"))
    prepared = []
    reference_by_dir: dict[Path, np.ndarray] = {}
    for name, target_path in cases:
        reference_path = (
            args.stress_dir / "reference.png"
            if name == "tensile_60pct"
            else args.image_dir / "reference.png"
        )
        if reference_path not in reference_by_dir:
            reference_by_dir[reference_path] = load_gray(reference_path, args.width, args.height)
        prepared.append(
            (
                name,
                reference_by_dir[reference_path],
                load_gray(target_path, args.width, args.height),
            )
        )

    baseline = create_session(args.baseline_model, threads=args.threads)
    optimized = create_session(args.optimized_model, threads=args.threads)
    optimized_metadata = model_metadata(args.optimized_model)
    rewrite_metadata_json = optimized_metadata.get("com.wgpu_matching.efficientloftr_border_mask")
    rewrite_metadata = json.loads(rewrite_metadata_json) if rewrite_metadata_json else None
    report: dict[str, Any] = {
        "protocol": {
            "source_resolution_px": [3840, 2160],
            "model_input_px": [args.width, args.height],
            "padded_input_tensor_px": [prepared[0][1].shape[1], prepared[0][1].shape[0]],
            "provider": "ONNX Runtime native WebGPU EP",
            "preferred_layout": "NHWC",
            "threads": args.threads,
            "warmups_per_case": 1,
            "timed_repeats_per_case": args.repeats,
            "profile_enabled_for_timing": False,
            "timing_scope": "session.run including ORT input/output handling; image decode/resize excluded",
            "profiling_scope": "one extra diagnostic inference per model/case; event durations are not GPU timestamps",
            "baseline_model": str(args.baseline_model),
            "baseline_sha256": sha256_file(args.baseline_model),
            "baseline_model_bytes": args.baseline_model.stat().st_size,
            "baseline_graph": model_graph_stats(args.baseline_model),
            "optimized_model": str(args.optimized_model),
            "optimized_sha256": sha256_file(args.optimized_model),
            "optimized_model_bytes": args.optimized_model.stat().st_size,
            "optimized_graph": model_graph_stats(args.optimized_model),
            "optimized_model_metadata": optimized_metadata,
            "rewrite": rewrite_metadata,
            "runtime_versions": {
                "onnxruntime": importlib.metadata.version("onnxruntime"),
                "onnxruntime_ep_webgpu": importlib.metadata.version("onnxruntime-ep-webgpu"),
            },
            "session_providers": baseline.get_providers(),
        },
        "cases": [],
    }

    for case_name, image0, image1 in prepared:
        feeds = {"image0": image0[None, None], "image1": image1[None, None]}
        baseline.run(None, feeds)
        optimized.run(None, feeds)
        comparison = _compare_outputs(baseline.run(None, feeds), optimized.run(None, feeds))
        baseline_times: list[float] = []
        optimized_times: list[float] = []
        for repeat in range(args.repeats):
            pairs = [
                ("baseline", baseline, baseline_times),
                ("optimized", optimized, optimized_times),
            ]
            if repeat % 2:
                pairs.reverse()
            for _, session, times in pairs:
                start = time.perf_counter()
                session.run(None, feeds)
                times.append(time.perf_counter() - start)
        baseline_median = float(statistics.median(baseline_times))
        optimized_median = float(statistics.median(optimized_times))
        report["cases"].append(
            {
                "case": case_name,
                "output_parity": comparison,
                "baseline_times_s": baseline_times,
                "baseline_median_s": baseline_median,
                "optimized_times_s": optimized_times,
                "optimized_median_s": optimized_median,
                "speedup": baseline_median / optimized_median,
                "baseline_fps": 1.0 / baseline_median,
                "optimized_fps": 1.0 / optimized_median,
            }
        )

    if args.profile:
        report["diagnostic_profiles"] = {
            "baseline": [],
            "optimized": [],
        }
        for case_name, image0, image1 in prepared:
            for label, model_path in (
                ("baseline", args.baseline_model),
                ("optimized", args.optimized_model),
            ):
                report["diagnostic_profiles"][label].append(
                    {
                        "case": case_name,
                        "summary": _profile_model(
                            model_path,
                            image0,
                            image1,
                            threads=args.threads,
                        ),
                    }
                )

    del baseline, optimized
    gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in report["cases"]:
        print(
            f"{case['case']}: {case['baseline_median_s']:.3f}s -> "
            f"{case['optimized_median_s']:.3f}s ({case['speedup']:.2f}x), "
            f"exact outputs={case['output_parity']['exact']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
