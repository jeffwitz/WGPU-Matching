"""Compare serial and batched anchor confirmation on identical solver inputs."""

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
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--frozen-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--external-reference", type=Path)
    parser.add_argument("--external-target", type=Path)
    parser.add_argument("--roi-mask", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or bool(args.external_reference) != bool(args.external_target):
        parser.error("provide at least one repetition and either both external images or neither")
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot = Path(__file__).with_name("qualify_image_seed_pilot.py")
    rows = []
    for repetition in range(args.repetitions + 1):
        folders = {mode: args.output_dir / f"run{repetition}" / mode for mode in ("serial", "batched")}
        times, reports = {}, {}
        for mode in (("serial", "batched") if repetition % 2 == 0 else ("batched", "serial")):
            folder = folders[mode]
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"output already exists: {folder}; choose a new directory")
            folder.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(pilot), "--archive", str(args.inputs), "--live-dic-source", str(args.live_dic_source),
                   "--output-dir", str(folder), "--lazy-rotation-weights", str(args.weights),
                   "--lazy-rotation-output", str(folder / "rotations"), "--frozen-parameters", str(args.frozen_parameters),
                   "--tau-u", str(frozen["tau_u"]), "--tau-g", str(frozen["tau_g"]),
                   "--sparse-stride", str(frozen["sparse_stride"]), "--minimum-anchors", str(frozen["minimum_anchors"]),
                   "--escalation", "--local-affine-escalation", "--use-cached-solver-images"]
            for option in ("external_reference", "external_target", "roi_mask"):
                value = getattr(args, option)
                if value is not None:
                    cmd.extend(("--" + option.replace("_", "-"), str(value)))
            if mode == "batched":
                cmd.append("--batch-anchor-confirmation")
            start = time.perf_counter()
            with (folder / "run.log").open("w") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            times[mode] = time.perf_counter() - start
            reports[mode] = json.loads((folder / "summary.json").read_text())["rows"]
            if len(reports[mode]) != 1:
                raise ValueError("benchmark requires exactly one image pair")
            reports[mode] = reports[mode][0]
        left, right = reports["serial"], reports["batched"]
        if left["pair"] != right["pair"]:
            raise ValueError("pair identities differ")
        with (np.load(folders["serial"] / f"{left['pair']}.npz") as a,
              np.load(folders["batched"] / f"{left['pair']}.npz") as b):
            keys = ("centres", "parameters", "accepted", "parent", "origin", "generation", "confirmed_seeds", "correct")
            equal = {key: bool(np.array_equal(a[key], b[key], equal_nan=True)) for key in keys}
            mask_changes = int(np.count_nonzero(a["accepted"] != b["accepted"]))
            correct_changes = int(np.count_nonzero(a["correct"] != b["correct"]))
        rows.append({"warmup": repetition == 0, "seconds": times, "equal": equal,
                     "acceptance_changes": mask_changes, "correct_changes": correct_changes, "results": reports})
        print(f"Completed repetition {repetition}: {times}", flush=True)
    measured = [r for r in rows if not r["warmup"]]
    medians = {mode: {"full_s": statistics.median(r["seconds"][mode] for r in measured),
                      "verification_s": statistics.median(r["results"][mode]["wall_s_excluding_hypothesis_generation"]
                                                          for r in measured)} for mode in ("serial", "batched")}
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": rows, "medians": medians}, indent=2) + "\n")
    exact = all(all(r["equal"].values()) for r in rows)
    lines = ["# Serial and batched anchor confirmation", "",
             ("One image pair, one warm-up and alternating-order repetitions with identical cached solver pixels. "
              "Each run includes process startup, image-cache verification, graphics setup, solver work and output writes. "
              "The timers are wall-clock observations, not graphics processing unit (GPU) timestamps. The input cache and hypotheses were prepared "
              "beforehand; this is not full pipeline latency."), "",
             (f"Median verification time, serial / batched: {medians['serial']['verification_s']:.3f} / "
              f"{medians['batched']['verification_s']:.3f} s. Exact output parity across runs: {exact}."), "",
             "| Run | Warm-up | Serial verification (s) | Batched verification (s) | Calls serial / batched | Active windows serial / batched | Acceptance changes |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(rows):
        a, b = row["results"]["serial"], row["results"]["batched"]
        lines.append(f"| {index} | {row['warmup']} | {a['wall_s_excluding_hypothesis_generation']:.3f} | "
                     f"{b['wall_s_excluding_hypothesis_generation']:.3f} | "
                     f"{a['total_solver_calls_before_control']} / {b['total_solver_calls_before_control']} | "
                     f"{a['total_active_windows_before_control']} / {b['total_active_windows_before_control']} | "
                     f"{row['acceptance_changes']} |")
    sample = rows[-1]["results"]["serial"]
    if sample["truth_available"]:
        lines.extend(["", (f"Serial result: {sample['correct']} correct of {sample['visible']} visible; "
                      f"{sample['false_accepted']} incorrect accepted visible windows.")])
    else:
        lines.extend(["", (f"{sample['accepted']} windows were accepted. No analytical ground truth is available; "
                      "acceptance and numerical parity do not establish displacement accuracy.")])
    lines.extend(["", ("All saved parameters and provenance are compared in the JavaScript Object Notation (JSON) output. "
                       "Batching may perform speculative solves that the serial path would skip. "
                       "The method is experimental and requires a window-independent deterministic solver."), "",
                  "Reproduce in a new output directory:", "", "```bash",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_anchor_confirmation.py \\",
                  f"  --inputs {args.inputs} --weights {args.weights} --live-dic-source {args.live_dic_source} \\",
                  f"  --frozen-parameters {args.frozen_parameters} --output-dir {args.output_dir}_new \\",
                  "  " + " ".join("--" + option.replace("_", "-") + " " + str(getattr(args, option))
                                   for option in ("external_reference", "external_target", "roi_mask")
                                   if getattr(args, option) is not None) + f" --repetitions {args.repetitions}",
                  "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
