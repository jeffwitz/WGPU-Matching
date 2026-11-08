"""Export DaD or DeDoDe v2 keypoint detection for ONNX/WebGPU experiments.

By default, the exported graph includes score normalization, detector-specific
selection, top-K, and coordinate recovery. It returns only keypoints and their
scores. A dense-score-map mode is available for diagnostics.

This is an optional research utility, not a dependency of ``wgpu_matching``.
It requires the official detector source and checkpoints, PyTorch, ONNX, and
optionally ONNX Runtime with its native WebGPU execution provider.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any

_REGISTERED_WEBGPU_LIBRARIES: set[str] = set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fuse_adjacent_conv_batch_norm(torch: Any, model: Any) -> int:
    """Fold eval-mode Conv2d + BatchNorm2d pairs in module containers."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    fused_count = 0
    for parent in list(model.modules()):
        if not isinstance(parent, (torch.nn.Sequential, torch.nn.ModuleList)):
            continue
        child_names = list(parent._modules)
        for left_name, right_name in itertools.pairwise(child_names):
            convolution = parent._modules[left_name]
            normalization = parent._modules[right_name]
            if isinstance(convolution, torch.nn.Conv2d) and isinstance(
                normalization, torch.nn.BatchNorm2d
            ):
                parent._modules[left_name] = fuse_conv_bn_eval(convolution, normalization)
                parent._modules[right_name] = torch.nn.Identity()
                fused_count += 1
    return fused_count


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("size must be written WIDTHxHEIGHT") from error
    if min(width, height) < 32 or width % 8 or height % 8:
        raise argparse.ArgumentTypeError("width and height must be >= 32 and divisible by 8")
    return width, height


def _create_runtime_session(
    model_path: Path, provider: str, *, profile: bool
) -> tuple[Any, dict[str, Any]]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install optional ONNX Runtime to execute the exported graph") from error

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = profile
    adapter: dict[str, Any] = {}
    if provider == "WebGpuExecutionProvider":
        try:
            import onnxruntime_ep_webgpu as webgpu_ep
        except ImportError as error:
            raise RuntimeError(
                "Install onnxruntime-ep-webgpu to use WebGpuExecutionProvider"
            ) from error
        if "wgpu_matching_detector_export" not in _REGISTERED_WEBGPU_LIBRARIES:
            ort.register_execution_provider_library(
                "wgpu_matching_detector_export", webgpu_ep.get_library_path()
            )
            _REGISTERED_WEBGPU_LIBRARIES.add("wgpu_matching_detector_export")
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()
        ]
        if not devices:
            raise RuntimeError("ONNX Runtime found no compatible WebGPU adapter")
        hardware = devices[0].device
        adapter = {
            "vendor_id": getattr(hardware, "vendor_id", None),
            "device_id": getattr(hardware, "device_id", None),
        }
        options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
        providers = None
    else:
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"ONNX Runtime provider {provider!r} is unavailable")
        providers = [provider]

    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    return session, {"onnxruntime_version": ort.__version__, "adapter": adapter}


def _profile_summary(profile_path: Path) -> dict[str, Any]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node" and event.get("ph") == "X"]
    cpu_nodes = [
        event for event in nodes if event.get("args", {}).get("provider") == "CPUExecutionProvider"
    ]
    by_provider: dict[str, int] = {}
    for event in nodes:
        provider = event.get("args", {}).get("provider", "unknown")
        by_provider[provider] = by_provider.get(provider, 0) + int(event.get("dur", 0))
    largest_cpu = sorted(cpu_nodes, key=lambda event: int(event.get("dur", 0)), reverse=True)[:20]
    return {
        "node_event_count": len(nodes),
        "node_duration_us_by_provider": by_provider,
        "cpu_node_event_count": len(cpu_nodes),
        "cpu_node_duration_us": sum(int(event.get("dur", 0)) for event in cpu_nodes),
        "largest_cpu_nodes": [
            {
                "name": event.get("name"),
                "op_type": event.get("args", {}).get("op_name"),
                "duration_us": int(event.get("dur", 0)),
                "output_type_shape": event.get("args", {}).get("output_type_shape"),
            }
            for event in largest_cpu
        ],
    }


def _nearest_point_metrics(
    reference: Any, candidate: Any, width: int, height: int
) -> dict[str, Any]:
    """Compare unordered keypoint sets in pixels without building a full matrix."""
    import numpy as np

    scale = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    def nearest(source: Any, target: Any) -> Any:
        chunks = []
        for start in range(0, len(source), 256):
            delta = (source[start : start + 256, None] - target[None, :]) * scale
            chunks.append(np.sqrt(np.sum(delta * delta, axis=-1)).min(axis=1))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    reference_to_candidate = nearest(reference, candidate)
    candidate_to_reference = nearest(candidate, reference)
    thresholds = (0.1, 0.5, 1.0, 2.0, 5.0)
    return {
        "metric": "one-way nearest-neighbour set distances; not a one-to-one assignment",
        "reference_to_candidate_count_within_px": {
            str(threshold): int(np.count_nonzero(reference_to_candidate <= threshold))
            for threshold in thresholds
        },
        "candidate_to_reference_count_within_px": {
            str(threshold): int(np.count_nonzero(candidate_to_reference <= threshold))
            for threshold in thresholds
        },
        "reference_to_candidate_median_px": float(np.median(reference_to_candidate))
        if len(reference_to_candidate)
        else None,
        "candidate_to_reference_median_px": float(np.median(candidate_to_reference))
        if len(candidate_to_reference)
        else None,
        "reference_to_candidate_max_px": float(reference_to_candidate.max())
        if len(reference_to_candidate)
        else None,
        "candidate_to_reference_max_px": float(candidate_to_reference.max())
        if len(candidate_to_reference)
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dad", "dedodev2"), required=True)
    parser.add_argument("--output-mode", choices=("keypoints", "scoremap"), default="keypoints")
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX model path")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--export-size", type=_parse_size, default=(128, 128))
    parser.add_argument("--test-size", type=_parse_size, default=(640, 360))
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--weights", type=Path, help="Optional explicit official checkpoint path")
    parser.add_argument(
        "--provider",
        choices=("WebGpuExecutionProvider", "CPUExecutionProvider"),
        default="WebGpuExecutionProvider",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.max_side < 64 or args.warmup < 0 or args.repeats < 1 or args.max_keypoints < 1:
        parser.error(
            "--max-side must be >=64, --warmup >=0, --repeats >=1, and --max-keypoints >=1"
        )
    if args.output_mode == "keypoints":
        export_width, export_height = args.export_size
        if args.max_keypoints > (export_width - 8) * (export_height - 8):
            parser.error("--max-keypoints exceeds the four-pixel-border export grid")

    try:
        import numpy as np
        import onnx
        import torch
    except ImportError as error:
        parser.error(f"Install optional research/export dependencies: {error}")

    from wgpu_matching.features import DaDDetector, DeDoDeV2Detector

    class DetectorOutputs(torch.nn.Module):
        """Tensor-only score-map or keypoint API over an upstream detector."""

        def __init__(self, detector: Any, method: str, output_name: str, count: int) -> None:
            super().__init__()
            self.model = detector.model
            self.method = method
            self.output_name = output_name
            self.count = count
            self.output_mode = args.output_mode

        def forward(self, image: Any) -> Any:
            raw = self.model.forward({"image": image})[self.output_name]
            if self.output_mode == "scoremap":
                return raw

            batch, channels, height, width = raw.shape
            probabilities = (
                raw.reshape(batch, channels * height * width)
                .softmax(dim=-1)
                .reshape(batch, channels, height * width)
                .sum(dim=1)
                .reshape(batch, height, width)
            )

            if self.method == "dad":
                if self.model.increase_coverage:
                    weights = (
                        -(
                            torch.linspace(
                                -2,
                                2,
                                steps=self.model.coverage_size,
                                device=image.device,
                            )
                            ** 2
                        )
                    ).exp()[None, None]
                    density_x = torch.nn.functional.conv2d(
                        (probabilities[:, None] + 1e-6) * 10000,
                        weights[..., None, :],
                        padding=(0, self.model.coverage_size // 2),
                    )
                    density = torch.nn.functional.conv2d(
                        density_x,
                        weights[..., None],
                        padding=(self.model.coverage_size // 2, 0),
                    )[:, 0]
                    probabilities = probabilities * (density + 1e-8) ** (-self.model.coverage_pow)
                probabilities = probabilities * (
                    probabilities
                    == torch.nn.functional.max_pool2d(
                        probabilities[:, None],
                        self.model.nms_size,
                        stride=1,
                        padding=self.model.nms_size // 2,
                    )[:, 0]
                )
                border = 4 if self.model.remove_borders else 0
                flat = probabilities[:, border : height - border, border : width - border]
                flat = flat.reshape(batch, -1)
                _, selected = torch.topk(flat, self.count, dim=1)
                interior_width = width - 2 * border
                x = selected % interior_width + border
                y = selected // interior_width + border
                image_indices = y * width + x
                scores = probabilities.reshape(batch, -1).gather(1, image_indices)

                if self.model.subpixel:
                    radius = self.model.nms_size // 2
                    if self.model.nms_size % 2 == 0 or border < radius:
                        raise ValueError(
                            "DaD export currently requires odd NMS and a border covering its support"
                        )
                    offsets_1d = (
                        torch.arange(self.model.nms_size, device=image.device, dtype=torch.int64)
                        - radius
                    )
                    offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
                    neighbor_indices = (
                        (y[:, :, None] + offset_y.reshape(1, 1, -1)) * width
                        + x[:, :, None]
                        + offset_x.reshape(1, 1, -1)
                    )
                    patches = (
                        raw.reshape(batch, -1)
                        .gather(1, neighbor_indices.reshape(batch, -1))
                        .reshape(batch, self.count, self.model.nms_size**2)
                        .transpose(1, 2)
                    )
                    patch_probs = (patches / self.model.subpixel_temp).softmax(dim=1)
                    offset_x = offset_x.reshape(-1).to(raw.dtype) * (2.0 / width)
                    offset_y = offset_y.reshape(-1).to(raw.dtype) * (2.0 / height)
                    delta_x = (patch_probs * offset_x[None, :, None]).sum(dim=1)
                    delta_y = (patch_probs * offset_y[None, :, None]).sum(dim=1)
                else:
                    delta_x = torch.zeros_like(x, dtype=raw.dtype)
                    delta_y = torch.zeros_like(y, dtype=raw.dtype)

                keypoints = torch.stack(
                    (
                        2.0 * (x.to(raw.dtype) + 0.5) / width - 1.0 + delta_x,
                        2.0 * (y.to(raw.dtype) + 0.5) / height - 1.0 + delta_y,
                    ),
                    dim=-1,
                )
                return keypoints, scores

            # DeDoDe v2 uses a Gaussian local-density correction, no NMS,
            # top-K sampling, and a four-pixel invalid border.
            weights = (-(torch.linspace(-2, 2, steps=51, device=image.device) ** 2)).exp()[
                None, None
            ]
            density_x = torch.nn.functional.conv2d(
                (probabilities[:, None] + 1e-6) * 10000,
                weights[..., None, :],
                padding=(0, 25),
            )
            density = torch.nn.functional.conv2d(density_x, weights[..., None], padding=(25, 0))[
                :, 0
            ]
            probabilities = probabilities * (density + 1e-8) ** -0.5
            border = 4 if self.model.remove_borders else 0
            flat = probabilities[:, border : height - border, border : width - border]
            flat = flat.reshape(batch, -1)
            scores, selected = torch.topk(flat, self.count, dim=1)
            interior_width = width - 2 * border
            x = selected % interior_width + border
            y = selected // interior_width + border
            keypoints = torch.stack(
                (
                    2.0 * (x.to(raw.dtype) + 0.5) / width - 1.0,
                    2.0 * (y.to(raw.dtype) + 0.5) / height - 1.0,
                ),
                dim=-1,
            )
            return keypoints, scores

    detector_type, output_name = (
        (DaDDetector, "scoremap") if args.method == "dad" else (DeDoDeV2Detector, "keypoint_logits")
    )
    detector = detector_type(max_side=args.max_side, device="cpu", weights_path=args.weights)
    model = (
        DetectorOutputs(detector, args.method, output_name, args.max_keypoints)
        .eval()
        .requires_grad_(False)
    )
    export_width, export_height = args.export_size
    torch.manual_seed(0)
    export_input = torch.randn(1, 3, export_height, export_width, dtype=torch.float32)

    with torch.inference_mode():
        reference_output = (
            model.model.forward({"image": export_input})[output_name].detach().cpu().numpy()
        )
    fused_batch_norm_count = 0
    if args.method == "dad":
        fused_batch_norm_count = _fuse_adjacent_conv_batch_norm(torch, model.model)
    with torch.inference_mode():
        torch_output = (
            model.model.forward({"image": export_input})[output_name].detach().cpu().numpy()
        )
    fusion_error = np.abs(torch_output - reference_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_names = ["scoremap"] if args.output_mode == "scoremap" else ["keypoints", "scores"]
    dynamic_axes = {"image": {2: "height", 3: "width"}}
    if args.output_mode == "scoremap":
        dynamic_axes["scoremap"] = {2: "height", 3: "width"}
    torch.onnx.export(
        model,
        export_input,
        str(args.output),
        input_names=["image"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    graph = onnx.load(str(args.output))
    custom_metadata = {
        "wgpu_matching.method": args.method,
        "wgpu_matching.output_mode": args.output_mode,
        "wgpu_matching.keypoint_capacity": (
            str(args.max_keypoints) if args.output_mode == "keypoints" else ""
        ),
        "wgpu_matching.max_side": str(args.max_side),
        "wgpu_matching.checkpoint_sha256": detector.metadata.get("weights_sha256") or "",
        "wgpu_matching.source_revision": detector.metadata.get("source_revision") or "",
    }
    for key, value in custom_metadata.items():
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(graph, str(args.output))
    onnx.checker.check_model(graph)
    operator_counts = collections.Counter(node.op_type for node in graph.graph.node)

    test_width, test_height = args.test_size
    test_input = torch.randn(1, 3, test_height, test_width, dtype=torch.float32)
    with torch.inference_mode():
        expected_output = model(test_input)
        if isinstance(expected_output, tuple):
            expected_outputs = [item.detach().cpu().numpy() for item in expected_output]
        else:
            expected_outputs = [expected_output.detach().cpu().numpy()]
    expected_outputs = [np.ascontiguousarray(item) for item in expected_outputs]
    torch_reference_keypoints = None
    torch_reference_scores = None
    if args.output_mode == "keypoints":
        with torch.inference_mode():
            official_keypoints = detector.model.detect(
                {"image": test_input}, num_keypoints=args.max_keypoints
            )
        official_score_name = "keypoint_probs" if args.method == "dad" else "confidence"
        torch_reference_keypoints = official_keypoints["keypoints"].detach().cpu().numpy()[0]
        torch_reference_scores = official_keypoints[official_score_name].detach().cpu().numpy()[0]
    feed = {"image": test_input.numpy()}

    session_start = time.perf_counter()
    timed_session, runtime = _create_runtime_session(args.output, args.provider, profile=False)
    session_initialization_s = time.perf_counter() - session_start
    for _ in range(args.warmup):
        timed_session.run(output_names, feed)
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        actual_outputs = timed_session.run(output_names, feed)
        durations.append(time.perf_counter() - start)
    if args.output_mode == "scoremap":
        numerical_error = np.abs(actual_outputs[0] - expected_outputs[0])
        numerical_parity: dict[str, Any] = {
            "max_abs": float(numerical_error.max()),
            "mean_abs": float(numerical_error.mean()),
            "rmse": float(np.sqrt(np.mean(np.square(actual_outputs[0] - expected_outputs[0])))),
        }
        torch_postprocess_parity = None
    else:
        actual_keypoints, actual_scores = actual_outputs
        expected_keypoints, expected_scores = expected_outputs
        numerical_parity = {
            "keypoint_set": _nearest_point_metrics(
                expected_keypoints[0], actual_keypoints[0], test_width, test_height
            ),
            "score_max_abs": float(np.abs(actual_scores - expected_scores).max()),
            "score_mean_abs": float(np.abs(actual_scores - expected_scores).mean()),
        }
        torch_postprocess_parity = {
            "keypoint_set": _nearest_point_metrics(
                torch_reference_keypoints,
                expected_keypoints[0],
                test_width,
                test_height,
            ),
            "score_max_abs": float(np.abs(torch_reference_scores - expected_scores[0]).max()),
        }

    profile_session, profile_runtime = _create_runtime_session(
        args.output, args.provider, profile=True
    )
    profile_session.run(output_names, feed)
    profile_path = Path(profile_session.end_profiling())
    profile_report = _profile_summary(profile_path)
    profile_output = args.profile_output or args.output.with_suffix(
        args.output.suffix + ".ort-profile.json"
    )
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile_output.write_bytes(profile_path.read_bytes())
    profile_path.unlink(missing_ok=True)

    report = {
        "method": args.method,
        "source": detector.metadata,
        "input_output": {
            "input_name": "image",
            "input_layout": "NCHW float32 normalized RGB",
            "output_mode": args.output_mode,
            "output_names": output_names,
            "output_meaning": (
                "dense detector logits/scores; keypoint selection is not included"
                if args.output_mode == "scoremap"
                else "normalized keypoint coordinates and detector confidence scores"
            ),
            "max_keypoints": args.max_keypoints if args.output_mode == "keypoints" else None,
            "batch_size": 1,
            "dynamic_spatial_axes": True,
            "required_input_multiple_px": 8,
            "export_shape_px": [export_width, export_height],
            "test_shape_px": [test_width, test_height],
        },
        "export": {
            "opset": 17,
            "fused_batch_norm_count": fused_batch_norm_count,
            "fusion_parity_max_abs": float(fusion_error.max()),
            "fusion_parity_mean_abs": float(fusion_error.mean()),
            "onnx_sha256": _sha256(args.output),
            "onnx_bytes": args.output.stat().st_size,
            "node_count": len(graph.graph.node),
            "operator_counts": dict(sorted(operator_counts.items())),
        },
        "runtime": {
            "provider_requested": args.provider,
            "providers_used_by_ort": timed_session.get_providers(),
            **runtime,
            "session_initialization_s": session_initialization_s,
            "warmup_calls": args.warmup,
            "timed_calls": args.repeats,
            "run_s": durations,
            "median_run_s": float(np.median(durations)),
            "output_bytes_returned_to_cpu": sum(int(output.nbytes) for output in actual_outputs),
            "session_run_returns_cpu_numpy": True,
            "profile_runtime": profile_runtime,
            "profile": profile_report,
            "profile_path": str(profile_output),
            "host_ram_and_gpu_peak_memory": "not measured",
        },
        "numerical_parity_torch_vs_onnx_runtime": numerical_parity,
        "custom_selection_vs_official_torch_detection": torch_postprocess_parity,
    }
    report_path = args.report_output or args.output.with_suffix(args.output.suffix + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Model: {args.output}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
