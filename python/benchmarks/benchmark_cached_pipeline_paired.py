"""Benchmark complete cached versus regenerated scripted pipelines in paired order."""

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
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--frozen-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offset", type=int, default=25000000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("at least one repetition is required")
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    scripts = Path(__file__).parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repetition in range(args.repetitions + 1):
        folders = {mode: args.output_dir / f"run{repetition}" / mode for mode in ("regenerated", "cached")}
        times, reports = {}, {}
        for mode in (("regenerated", "cached") if repetition % 2 == 0 else ("cached", "regenerated")):
            folder = folders[mode]
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"output already exists: {folder}; choose a new directory")
            folder.mkdir(parents=True, exist_ok=True)
            inputs, evaluation = folder / "inputs", folder / "evaluation"
            prepare = [sys.executable, str(scripts / "generate_p2_test_inputs.py"),
                       "--frozen-parameters", str(args.frozen_parameters), "--weights", str(args.weights),
                       "--live-dic-source", str(args.live_dic_source), "--output-dir", str(inputs),
                       "--seed-offsets", str(args.seed_offset), "--families", "heterogeneous", "--cases-per-family", "1",
                       "--retain-translation-affine", "--shared-prefit"]
            verify = [sys.executable, str(scripts / "qualify_image_seed_pilot.py"), "--archive", str(inputs),
                      "--live-dic-source", str(args.live_dic_source), "--output-dir", str(evaluation),
                      "--lazy-rotation-weights", str(args.weights), "--lazy-rotation-output", str(folder / "rotations"),
                      "--frozen-parameters", str(args.frozen_parameters), "--tau-u", str(frozen["tau_u"]),
                      "--tau-g", str(frozen["tau_g"]), "--sparse-stride", str(frozen["sparse_stride"]),
                      "--minimum-anchors", str(frozen["minimum_anchors"]), "--escalation", "--local-affine-escalation"]
            if mode == "cached":
                prepare.append("--export-solver-images")
                verify.append("--use-cached-solver-images")
            start = time.perf_counter()
            for label, command in (("prepare", prepare), ("verify", verify)):
                with (folder / f"{label}.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                if label == "prepare":
                    after_prepare = time.perf_counter()
            end = time.perf_counter()
            times[mode] = {"prepare_s": after_prepare - start, "verify_s": end - after_prepare, "full_s": end - start}
            reports[mode] = json.loads((evaluation / "summary.json").read_text())["rows"][0]
        left, right = (reports[x] for x in ("regenerated", "cached"))
        if left["pair"] != right["pair"]:
            raise ValueError("pair identities differ")
        with (np.load(folders["regenerated"] / "evaluation" / f"{left['pair']}.npz") as a,
              np.load(folders["cached"] / "evaluation" / f"{left['pair']}.npz") as b):
            keys = ("centres", "parameters", "accepted", "parent", "origin", "generation", "confirmed_seeds", "correct")
            parity = {key: bool(np.array_equal(a[key], b[key], equal_nan=True)) for key in keys}
        rows.append({"warmup": repetition == 0, "seconds": times, "parity": parity, "results": reports})
        print(f"Completed repetition {repetition}: {times}", flush=True)
    measured = [row for row in rows if not row["warmup"]]
    medians = {mode: {key: statistics.median(row["seconds"][mode][key] for row in measured)
                      for key in ("prepare_s", "verify_s", "full_s")} for mode in ("regenerated", "cached")}
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": rows, "medians_s": medians}, indent=2) + "\n")
    exact = all(all(row["parity"].values()) for row in rows)
    lines = ["# Paired complete scripted pipeline with cached solver pixels", "",
             ("One synthetic 3840 × 2160 pair, one warm-up and alternating-order measured repetitions. "
              "Each arm includes all image generation, matching, process startup, diagnostics and result writes. "
              "The cached arm also includes encoding, writing, hashing and reading both images. "
              "Times are not integrated production or graphics-processing-unit (GPU) resident latency."), "",
             (f"Median regenerated / cached full script: {medians['regenerated']['full_s']:.3f} / "
              f"{medians['cached']['full_s']:.3f} s. Exact state, status and provenance parity: {exact}."), "",
             "| Run | Warm-up | Regenerated (s) | Cached (s) | Exact parity |", "|---:|---|---:|---:|---|"]
    for index, row in enumerate(rows):
        lines.append(f"| {index} | {row['warmup']} | {row['seconds']['regenerated']['full_s']:.3f} | "
                     f"{row['seconds']['cached']['full_s']:.3f} | {all(row['parity'].values())} |")
    lines.extend(["", ("The saved JavaScript Object Notation (JSON) data contain stage times, quality and per-window "
                       "outputs. One pair does not establish performance or accuracy in general."), "",
                  "Reproduce in a new directory:", "", "```bash",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_cached_pipeline_paired.py \\",
                  f"  --weights {args.weights} --live-dic-source {args.live_dic_source} \\",
                  f"  --frozen-parameters {args.frozen_parameters} --output-dir {args.output_dir}_new \\",
                  f"  --seed-offset {args.seed_offset} --repetitions {args.repetitions}", "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))
    if not exact:
        raise SystemExit("cached-pipeline parity failed")


if __name__ == "__main__":
    main()
