"""Compare CPU and WebGPU local NCC with warm-up and deterministic inputs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ndf_pair_model import refine_local_ncc
from wgpu_matching import WGPUMatching


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three measured repetitions are required")

    torch.manual_seed(20261109)
    height, width = 108, 192
    field_height, field_width = (height + 3) // 4, (width + 3) // 4
    reference = torch.rand((1, 1, height, width), dtype=torch.float32)
    target = torch.zeros_like(reference)
    target[:, :, 2:, 3:] = reference[:, :, :-2, :-3]
    initial = torch.zeros((1, field_height, field_width, 2), dtype=torch.float32)
    radius, patch_radius = 8, 5

    def time_cpu() -> tuple[float, torch.Tensor]:
        started = time.perf_counter()
        result = refine_local_ncc(
            reference,
            target,
            initial,
            search_radius=radius,
            patch_radius=patch_radius,
        )
        return time.perf_counter() - started, result.displacement_xy

    cpu_result = refine_local_ncc(
        reference, target, initial, search_radius=radius, patch_radius=patch_radius
    )
    cpu_times = []
    for _ in range(args.repeats + 1):
        duration, _ = time_cpu()
        cpu_times.append(duration)
    cpu_times = cpu_times[1:]

    with WGPUMatching(device_type=args.device) as matcher:
        def time_gpu(implementation: str):
            started = time.perf_counter()
            result = matcher.refine_ndf_local_ncc(
                reference[0, 0].numpy(),
                target[0, 0].numpy(),
                initial[0].numpy(),
                search_radius=radius,
                patch_radius=patch_radius,
                implementation=implementation,
            )
            return time.perf_counter() - started, result

        results = {}
        for implementation in ("parallel", "serial"):
            # First call warms pipelines and command execution; do not include
            # it in steady-state timing, but preserve it in the output.
            warmup_s, _ = time_gpu(implementation)
            times = []
            last = None
            for _ in range(args.repeats):
                duration, last = time_gpu(implementation)
                times.append(duration)
            assert last is not None
            cpu_valid = cpu_result.valid[0].numpy()
            common_valid = cpu_valid & last.valid
            error = np.linalg.norm(
                last.displacement_xy - cpu_result.displacement_xy[0].numpy(), axis=-1
            )
            common_error = error[common_valid]
            results[implementation] = {
                "warmup_s": warmup_s,
                "repeats_s": times,
                "median_s": statistics.median(times),
                "p90_s": float(np.percentile(times, 90)),
                "valid_fraction": float(np.mean(last.valid)),
                "validity_disagreement_fraction": float(np.mean(last.valid != cpu_valid)),
                "median_endpoint_error_px_on_common_valid": float(np.median(common_error)),
                "p99_endpoint_error_px_on_common_valid": float(np.percentile(common_error, 99)),
                "fraction_within_0_1px_on_common_valid": float(np.mean(common_error <= 0.1)),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "image_shape_yx": [height, width],
            "query_grid_shape_yx": [field_height, field_width],
            "query_count": field_height * field_width,
            "search_radius_px": radius,
            "patch_radius_px": patch_radius,
            "warmup_count_per_backend": 1,
            "measured_repeats": args.repeats,
            "images_and_flow_upload_included": True,
            "compact_output_readback_included": True,
            "adapter": matcher.device_name,
            "device_type": args.device,
            "cpu_reference": "PyTorch CPU local normalized cross-correlation",
        },
        "cpu": {
            "repeats_s": cpu_times,
            "median_s": statistics.median(cpu_times),
            "p90_s": float(np.percentile(cpu_times, 90)),
            "fraction_valid": float(cpu_result.valid.float().mean()),
            "median_displacement_xy_px": cpu_result.displacement_xy[0]
            .reshape(-1, 2)
            .median(dim=0)
            .values.tolist(),
        },
        "webgpu": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
