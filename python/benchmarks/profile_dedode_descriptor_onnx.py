"""Profile a fixed-shape DeDoDe-B descriptor ONNX graph on ONNX Runtime WebGPU."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats positive")

    import onnxruntime as ort
    import onnxruntime_ep_webgpu as webgpu_ep

    ort.register_execution_provider_library("dedode_profile", webgpu_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()]
    if not devices:
        raise RuntimeError("ONNX Runtime found no WebGPU adapter")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output.with_suffix(""))
    options.log_severity_level = 3
    options.add_provider_for_devices(devices, {"preferredLayout": "NHWC"})
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=None)
    shape = session.get_inputs()[0].shape
    keypoint_shape = session.get_inputs()[1].shape
    feeds = {
        "image": np.zeros(shape, dtype=np.float32),
        "keypoints": np.zeros(keypoint_shape, dtype=np.float32),
    }
    for _ in range(args.warmup):
        session.run(None, feeds)
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        session.run(None, feeds)
        durations.append(time.perf_counter() - start)
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text())
    operators = [event for event in events if event.get("cat") == "Node"]
    providers = Counter(event.get("args", {}).get("provider", "unknown") for event in operators)
    provider_durations_us = Counter()
    op_durations_us = Counter()
    for event in operators:
        args_event = event.get("args", {})
        provider_durations_us[args_event.get("provider", "unknown")] += event.get("dur", 0)
        op_durations_us[args_event.get("op_name", "unknown")] += event.get("dur", 0)
    output = {
        "model": str(args.model.resolve()),
        "model_bytes": args.model.stat().st_size,
        "providers": session.get_providers(),
        "adapter": {
            "device_id": devices[0].device.device_id,
            "type": str(devices[0].device.type),
            "vendor_id": int(devices[0].device.vendor_id),
            "vendor": str(devices[0].device.vendor),
            "metadata": devices[0].device.metadata,
        },
        "input_shapes": {item.name: item.shape for item in session.get_inputs()},
        "warmup_calls": args.warmup,
        "measured_calls": len(durations),
        "session_run_wall_s": durations,
        "session_run_median_wall_s": float(np.median(durations)),
        "operator_events": len(operators),
        "operator_events_by_provider": dict(providers),
        "operator_duration_ms_by_provider": {
            key: value / 1000 for key, value in provider_durations_us.items()
        },
        "top_20_operator_durations_ms": [
            {"operator": name, "duration_ms": duration / 1000}
            for name, duration in op_durations_us.most_common(20)
        ],
        "raw_profile": str(profile_path.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
