"""Compare separate and shared global-prefit scripts with exact output checks."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parity(separate, shared, seed):
    """Compare both initial fields and their IC-GN outputs window by window."""
    folder = f"windows_seed{seed}"
    separate_ncc = list((separate / "affine_ncc" / folder).glob("*.npz"))
    shared_ncc = list((shared / "affine_ncc" / folder).glob("*.npz"))
    if len(separate_ncc) != 1 or len(shared_ncc) != 1 or separate_ncc[0].name != shared_ncc[0].name:
        raise ValueError("expected matching single-pair archives")
    affine_path = separate / "affine_only" / folder / separate_ncc[0].name
    with np.load(affine_path) as affine, np.load(separate_ncc[0]) as ncc, np.load(shared_ncc[0]) as combined:
        pairs = {"affine_initial": (affine["initial_parameters"], combined["pre_ncc_initial_parameters"]),
                 "affine_status": (affine["icgn_status"], combined["pre_ncc_icgn_status"]),
                 "affine_result": (affine["icgn_parameters"], combined["pre_ncc_icgn_parameters"]),
                 "ncc_initial": (ncc["initial_parameters"], combined["initial_parameters"]),
                 "ncc_status": (ncc["icgn_status"], combined["icgn_status"]),
                 "ncc_result": (ncc["icgn_parameters"], combined["icgn_parameters"]),
                 "visibility": (ncc["visible"], combined["visible"]),
                 "truth": (ncc["truth_parameters"], combined["truth_parameters"])}
        return {key: {"exact": bool(np.array_equal(left, right, equal_nan=True)),
                      "differing_values": int(np.count_nonzero(~np.isclose(left, right, rtol=0, atol=0, equal_nan=True)))}
                for key, (left, right) in pairs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--frozen-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offset", type=int, default=25000000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("at least one measured repetition is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qualifier = Path(__file__).with_name("generate_p2_test_inputs.py")
    rows = []
    for repetition in range(args.repetitions + 1):
        elapsed = {}
        folders = {mode: args.output_dir / f"run{repetition}" / mode for mode in ("separate", "shared")}
        for mode in (("separate", "shared") if repetition % 2 == 0 else ("shared", "separate")):
            folder = folders[mode]
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"output already exists: {folder}; choose a new directory")
            folder.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(qualifier), "--frozen-parameters", str(args.frozen_parameters),
                   "--weights", str(args.weights), "--live-dic-source", str(args.live_dic_source),
                   "--output-dir", str(folder), "--seed-offsets", str(args.seed_offset),
                   "--families", "heterogeneous", "--cases-per-family", "1", "--retain-translation-affine"]
            if mode == "shared":
                cmd.append("--shared-prefit")
            start = time.perf_counter()
            with (folder / "run.log").open("w") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            elapsed[mode] = time.perf_counter() - start
        comparison = parity(folders["separate"], folders["shared"], args.seed_offset)
        rows.append({"warmup": repetition == 0, "seconds": elapsed, "parity": comparison})
        print(f"Completed repetition {repetition}: {elapsed}", flush=True)
    measured = [row for row in rows if not row["warmup"]]
    medians = {key: statistics.median(row["seconds"][key] for row in measured) for key in ("separate", "shared")}
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": rows, "medians_s": medians}, indent=2) + "\n")
    all_equal = all(all(v["exact"] for v in row["parity"].values()) for row in rows)
    lines = ["# Shared global prefit: paired script benchmark", "",
             ("One synthetic 3840 × 2160 heterogeneous pair; one warm-up and measured alternating-order repetitions. "
              "The scripts include image synthesis, process/model startup, truth/oracle controls, "
              "graphics processing unit (GPU) solve and disk output. "
              "The inverse-compositional Gauss–Newton (IC-GN) solver and normalized cross-correlation (NCC) "
              "use the same settings in both arms. "
              "These are prefit script times, not complete initializer or production latency."), "",
             f"Exact equality for both initial fields, solver results, status, visibility and truth: {all_equal}.",
             (f"Median separate / shared: {medians['separate']:.3f} / {medians['shared']:.3f} s; "
              f"ratio {medians['separate'] / medians['shared']:.2f}."), "",
             "| Run | Warm-up | Separate (s) | Shared (s) | Exact parity |", "|---:|---|---:|---:|---|"]
    for i, row in enumerate(rows):
        lines.append(f"| {i} | {row['warmup']} | {row['seconds']['separate']:.3f} | "
                     f"{row['seconds']['shared']:.3f} | {all(v['exact'] for v in row['parity'].values())} |")
    lines.extend(["", "The detailed JavaScript Object Notation (JSON) file records every comparison and all times. One pair does not establish general quality or runtime.", "",
                  "Reproduce in a new directory:", "", "```bash",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_shared_prefit.py \\",
                  f"  --weights {args.weights} --live-dic-source {args.live_dic_source} \\",
                  f"  --frozen-parameters {args.frozen_parameters} --output-dir {args.output_dir}_new \\",
                  f"  --seed-offset {args.seed_offset} --repetitions {args.repetitions}", "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))
    if not all_equal:
        raise SystemExit("shared-prefit parity failed")


if __name__ == "__main__":
    main()
