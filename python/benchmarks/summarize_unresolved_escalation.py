"""Compare saved window-level outputs before and after unresolved-region escalation."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    old = {x["pair"]: x for x in json.loads((args.baseline / "summary.json").read_text())["rows"]}
    new = json.loads((args.experiment / "summary.json").read_text())["rows"]
    lines = ["# Unresolved-region local-affine escalation", "",
             ("The existing confirmed field is retained; local affine models are fitted only at "
              "unresolved valid centres. Ground truth is used for scoring only. These "
              "pairs were previously inspected and this is a post-hoc diagnostic, not a "
              "new independent qualification. The timings exclude image-hypothesis generation."), "",
             "| Pair | Correct before / after / oracle | False before / after | States preserved | Local fits / total | Diagnostic seconds before / after |",
             "|---|---:|---:|---|---:|---:|"]
    for row in new:
        name = row["pair"]
        before = old[name]
        with np.load(args.baseline / f"{name}.npz") as a, np.load(args.experiment / f"{name}.npz") as b:
            mask = a["accepted"].astype(bool)
            preserved = bool(np.all(b["accepted"][mask]) and np.array_equal(a["parameters"][mask], b["parameters"][mask]))
            total = len(mask)
        lines.append(f"| {name} | {before['correct']} / {row['correct']} / {row['oracle']} | "
                     f"{before['false_accepted']} / {row['false_accepted']} | {preserved} | "
                     f"{row['local_affine_fit_window_count_per_rotation']}/{total} | "
                     f"{before['wall_s_excluding_hypothesis_generation']:.3f} / "
                     f"{row['wall_s_excluding_hypothesis_generation']:.3f} |")
    lines.extend(["", ("Previously accepted means solver status, not proof of correctness. "
                       "The compact fit may change random sampling compared with full-grid fitting. "
                       "Diagnostic times are single runs and do not establish production performance. "
                       "Per-window results and escalation histories are in the saved NPZ and JSON files."), "",
                  "Reproduce this comparison:", "", "```bash",
                  "PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \\",
                  "  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \\",
                  "  --archive results/seed_propagation_20261111/p2_confirmatory_inputs \\",
                  f"  --output-dir {args.experiment} --sparse-stride 2 \\",
                  "  --rotation-manifest results/seed_propagation_20261111/p2_confirmatory/rotation_manifest.json \\",
                  "  --textures 23051001 23051003 23051015 24051003 24051011 \\",
                  "  --escalation --local-affine-escalation --escalate-unresolved --same-budget-control \\",
                  "  --frozen-parameters results/seed_propagation_20261111/p2_calibration/frozen_parameters.json",
                  "PYTHONPATH=python/src .venv/bin/python python/benchmarks/summarize_unresolved_escalation.py \\",
                  f"  --baseline {args.baseline} --experiment {args.experiment}", "```", ""])
    (args.experiment / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
