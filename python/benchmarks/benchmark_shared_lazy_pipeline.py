"""Measure shared-prefit plus deferred rotations on one complete scripted pair."""

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
    parser.add_argument("--reference-evaluation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offset", type=int, default=25000000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cache-solver-images", action="store_true",
                        help="export and verify exact solver pixels instead of regenerating the synthetic pair")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("at least one repetition is required")
    frozen = json.loads(args.frozen_parameters.read_text())["selected"]
    scripts = Path(__file__).parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repetition in range(args.repetitions + 1):
        folder = args.output_dir / f"run{repetition}"
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"output already exists: {folder}; choose a new directory")
        folder.mkdir()
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
        if args.cache_solver_images:
            prepare.append("--export-solver-images")
            verify.append("--use-cached-solver-images")
        start = time.perf_counter()
        for label, command in (("prepare", prepare), ("verify", verify)):
            with (folder / f"{label}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            if label == "prepare":
                after_prepare = time.perf_counter()
        end = time.perf_counter()
        run_row = json.loads((evaluation / "summary.json").read_text())["rows"][0]
        with np.load(args.reference_evaluation / f"{run_row['pair']}.npz") as old, np.load(evaluation / f"{run_row['pair']}.npz") as new:
            old_correct, new_correct = old["correct"].astype(bool), new["correct"].astype(bool)
            if not np.array_equal(old["centres"], new["centres"]):
                raise ValueError("reference and candidate window grids differ")
            lost = int((old_correct & ~new_correct).sum())
            gained = int((new_correct & ~old_correct).sum())
            old_accepted = old["accepted"].astype(bool)
            mask_changes = int(np.count_nonzero(old["accepted"] != new["accepted"]))
            preserved_parameters = bool(np.array_equal(old["parameters"][old_accepted],
                                                       new["parameters"][old_accepted]))
        rows.append({"warmup": repetition == 0, "pair": run_row["pair"], "prepare_s": after_prepare - start,
                     "verify_s": end - after_prepare, "full_s": end - start, "correct": run_row["correct"],
                     "visible": run_row["visible"], "false_accepted": run_row["false_accepted"],
                     "correct_lost_vs_eager": lost, "correct_gained_vs_eager": gained,
                     "accepted_mask_changes": mask_changes, "accepted_parameters_exact": preserved_parameters,
                     "rotations_requested": run_row["lazy_rotation_requested"]})
        print(f"Completed repetition {repetition}: {rows[-1]}", flush=True)
    measured = [row for row in rows if not row["warmup"]]
    medians = {key: statistics.median(row[key] for row in measured) for key in ("prepare_s", "verify_s", "full_s")}
    (args.output_dir / "summary.json").write_text(json.dumps({"cache_solver_images": args.cache_solver_images,
                                                               "runs": rows, "medians_s": medians}, indent=2) + "\n")
    lines = ["# Shared prefit and deferred rotations: full scripted pair", "",
             ("Synthetic 3840 × 2160 heterogeneous pair, one warm-up and repeated full scripted runs. "
              "Times include synthetic-image generation, model and process startup, truth/oracle diagnostics, "
              "graphics processing unit (GPU) solver and final result writes. With image caching enabled, "
              "image encoding, writing, fingerprinting and reading are included in these times. "
              "They do not establish integrated production "
              "or GPU-resident latency."), "",
             (f"Median prefit / verification / total: {medians['prepare_s']:.3f} / "
              f"{medians['verify_s']:.3f} / {medians['full_s']:.3f} s."), "",
             "| Run | Warm-up | Total (s) | Correct / visible | False accepted | Lost / gained vs eager | Rotations requested |",
             "|---:|---|---:|---:|---:|---:|---|"]
    for i, row in enumerate(rows):
        lines.append(f"| {i} | {row['warmup']} | {row['full_s']:.3f} | {row['correct']}/{row['visible']} | "
                     f"{row['false_accepted']} | {row['correct_lost_vs_eager']} / {row['correct_gained_vs_eager']} | "
                     f"{row['rotations_requested']} |")
    lines.extend(["", ("The eager reference was measured in an earlier run, not a contemporaneous paired timing. "
                       "The saved JavaScript Object Notation (JSON) file retains per-run stage times and outputs; "
                       "one case cannot establish general performance."), "",
                  "Reproduce in a new directory:", "", "```bash",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_shared_lazy_pipeline.py \\",
                  f"  --weights {args.weights} --live-dic-source {args.live_dic_source} \\",
                  f"  --frozen-parameters {args.frozen_parameters} --reference-evaluation {args.reference_evaluation} \\",
                  f"  --output-dir {args.output_dir}_new --seed-offset {args.seed_offset} --repetitions {args.repetitions}"
                  + (" --cache-solver-images" if args.cache_solver_images else ""),
                  "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
