"""Paired end-to-end coarse-prefit diagnostic for PyTorch and ONNX WebGPU."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=146)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--diagnostic-profile", action="store_true",
                        help="profile only the ONNX warm-up; measured repetitions are unprofiled")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("at least one measured repetition is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("choose a new empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "graphs"
    qualifier = Path(__file__).with_name("qualify_ndf_pair_icgn.py")
    rows = []
    for rep in range(args.repetitions + 1):
        results = {}
        for backend in (("torch", "webgpu") if rep % 2 == 0 else ("webgpu", "torch")):
            folder = args.output_dir / f"run{rep}" / backend
            folder.mkdir(parents=True)
            cmd = [sys.executable, str(qualifier), "--weights", str(args.weights), "--live-dic-source",
                   str(args.live_dic_source), "--output", str(folder / "report.json"), "--window-diagnostics-dir",
                   str(folder / "windows"), "--external-reference", str(args.reference), "--external-target",
                   str(args.target), "--model-width", str(args.width), "--model-height", str(args.height),
                   "--families", "heterogeneous", "--cases-per-family", "1", "--contextual-features",
                   "--rotation-bank-steps", "9", "--rotation-bank-probe-query-budget", "256",
                   "--global-affine-prefit", "--field-residual", "none", "--retain-translation-affine",
                   "--local-ncc-radius", "8", "--local-ncc-patch-radius", "4", "--local-ncc-backend", "wgpu",
                   "--local-ncc-geometry", "translation", "--local-consensus-radius", "40",
                   "--local-consensus-threshold", "5", "--projection", "raw_bilinear", "--export-pre-ncc"]
            if backend == "webgpu":
                cmd.extend(("--ndf-onnx-webgpu-cache", str(cache)))
                if rep == 0 and args.diagnostic_profile:
                    cmd.append("--ndf-onnx-profile")
            start = time.perf_counter()
            with (folder / "run.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed = time.perf_counter() - start
            data = json.loads((folder / "report.json").read_text())
            files = list((folder / "windows").glob("*.npz"))
            if len(files) != 1:
                raise ValueError("expected exactly one output window archive")
            results[backend] = {"wall_s": elapsed, "case": data["cases"][0], "archive": str(files[0])}
        with np.load(results["torch"]["archive"]) as a, np.load(results["webgpu"]["archive"]) as b:
            arrays = {}
            for key in ("initial_parameters", "icgn_status", "icgn_parameters", "pre_ncc_initial_parameters"):
                left, right = a[key], b[key]
                finite = np.isfinite(left) & np.isfinite(right)
                arrays[key] = {"equal": bool(np.array_equal(left, right, equal_nan=True)),
                               "differing_values": int(np.count_nonzero(~np.isclose(left, right, rtol=0, atol=0,
                                                                                   equal_nan=True))),
                               "max_abs_finite": float(np.max(np.abs(left[finite] - right[finite]), initial=0))}
        rows.append({"warmup": rep == 0, "results": results, "arrays": arrays})
        print(f"Completed repetition {rep}: {results['torch']['wall_s']:.3f}/{results['webgpu']['wall_s']:.3f}s", flush=True)
    measured = [row for row in rows if not row["warmup"]]
    medians = {backend: statistics.median(row["results"][backend]["wall_s"] for row in measured)
               for backend in ("torch", "webgpu")}
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": rows, "median_process_s": medians}, indent=2) + "\n")
    print("Medians:", medians, flush=True)


if __name__ == "__main__":
    main()
