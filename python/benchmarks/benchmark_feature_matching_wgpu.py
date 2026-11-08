"""Compare WGSL reciprocal cosine matching with its bounded NumPy oracle."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from wgpu_matching import DescribedFeatures, FeatureSet, WGPUMatching
from wgpu_matching._gpu import make_pipeline, shader_source
from wgpu_matching.features import mutual_nearest_cosine, mutual_nearest_cosine_wgpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or args.dimensions < 1 or any(n < 1 for n in args.counts):
        parser.error("repeats must be >=2; counts and dimensions must be positive")

    report = {
        "adapter": None,
        "dimensions": args.dimensions,
        "repeats_after_warmup": args.repeats,
        "rows": [],
    }
    with WGPUMatching(device_type="gpu") as matcher:
        report["adapter"] = matcher.device_name
        module = matcher.device.create_shader_module(
            code=shader_source("feature-mutual-nearest.wgsl")
        )
        pipeline = make_pipeline(
            matcher.device, module, "nearest", "Mutual cosine nearest descriptor"
        )
        rng = np.random.default_rng(20261108)
        for count in args.counts:
            reference_desc = rng.normal(size=(count, args.dimensions)).astype(np.float32)
            target_desc = reference_desc + rng.normal(scale=0.1, size=reference_desc.shape).astype(
                np.float32
            )
            xy = rng.uniform(0, 2048, size=(count, 2)).astype(np.float32)
            reference = DescribedFeatures(FeatureSet(xy, np.ones(count)), reference_desc)
            target = DescribedFeatures(FeatureSet(xy + [1, -1], np.ones(count)), target_desc)
            expected = mutual_nearest_cosine(reference, target)
            warm = mutual_nearest_cosine_wgpu(matcher.device, reference, target, pipeline=pipeline)
            if not np.array_equal(warm.points, expected.points):
                raise RuntimeError(f"WGSL/NumPy parity failed at N={count}")
            gpu_times, cpu_times = [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                actual = mutual_nearest_cosine_wgpu(
                    matcher.device, reference, target, pipeline=pipeline
                )
                gpu_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                baseline = mutual_nearest_cosine(reference, target)
                cpu_times.append(time.perf_counter() - start)
                if not np.array_equal(actual.points, baseline.points):
                    raise RuntimeError(f"WGSL/NumPy parity failed at N={count}")
            report["rows"].append(
                {
                    "points_per_image": count,
                    "pairs": len(expected.points),
                    "wgpu_seconds": gpu_times,
                    "wgpu_median_seconds": statistics.median(gpu_times),
                    "numpy_seconds": cpu_times,
                    "numpy_median_seconds": statistics.median(cpu_times),
                    "same_correspondences": True,
                    "includes_upload_two_dispatches_and_small_readback": True,
                    "pipeline_cached_after_warmup": True,
                }
            )
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
