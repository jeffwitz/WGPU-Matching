"""Summarize paired support and interpolation experiments without selecting a production policy."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.directory / "summary.json").read_text())["cases"]
    configs = ((64, "linear"), (64, "cubic"), (32, "linear"), (32, "cubic"))
    names = ("64 linear", "64 cubic", "32 linear", "32 cubic")
    totals = []
    errors = []
    for subset, interpolation in configs:
        cases = [r for r in rows if r["subset"] == subset and r["interpolation"] == interpolation]
        if len(cases) != 3:
            raise ValueError("expected exactly three paired cases per configuration")
        keys = ("visible", "accepted_visible", "correct", "false", "original_bad", "original_bad_recovered",
                "original_bad_rejected", "original_bad_still_false", "original_good_lost",
                "original_oracle_good_lost", "original_oracle_good_gained", "accepted_invisible")
        totals.append({"subset": subset, "interpolation": interpolation,
                       **{k: sum(row[k] for row in cases) for k in keys}})
        arrays = []
        for case in cases:
            file = args.directory / f"{case['pair']}_s{subset}_{interpolation}.npz"
            with np.load(file) as data:
                values = data["error_px"][data["original_bad"].astype(bool) & data["status"].astype(bool)]
                arrays.extend(values.tolist())
        errors.append(arrays)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.boxplot(errors, tick_labels=names, showfliers=True)
    ax.axhline(2., color="#b12b35", linestyle="--", label="2 px scoring threshold")
    ax.set_ylabel("Absolute translation error at the original center (px)")
    ax.set_title("Oracle re-solves at the 57 originally incorrect centers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.directory / "errors_57_centers.png", dpi=160)
    plt.close(fig)
    (args.directory / "aggregated.json").write_text(json.dumps({"configurations": totals}, indent=2) + "\n")
    lines = ["# Shear-band support and interpolation diagnostic", "",
             ("Three synthetic 3840 × 2160 image pairs, peak displacement gradient 2.0. Every configuration "
              "starts from the exact analytical affine state at the same original 64-pixel-grid centers. "
              "The smaller support uses a 16-pixel grid only to address those identical centers; it does not "
              "change their number. These are oracle experiments, not a deployable initialization method. "
              "The frozen production settings and G3 gate have not been changed."), "",
             "| Support / interpolation | Correct / visible | False / accepted visible | Originally wrong recovered / 57 | Previously correct lost |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in totals:
        lines.append(f"| {row['subset']} px / {row['interpolation']} | {row['correct']}/{row['visible']} | "
                     f"{row['false']}/{row['accepted_visible']} | {row['original_bad_recovered']}/{row['original_bad']} | "
                     f"{row['original_good_lost']} |")
    lines.extend(["", ("For the linear 64-pixel analytical initialization, 323 windows are accepted yet wrong "
                       "among all visible original centers; the image-derived initializer previously accepted "
                       "only 57 of those errors. The exact analytical initialization is a diagnostic reference, "
                       "not an upper bound on convergence or a safety guarantee."), "",
                  ("Changing only interpolation recovers 2 of the 57 original errors. A 32-pixel support "
                   "recovers all 57 and yields no false accepted visible window on these three pairs. "
                   "This supports a finite-support/local-kinematics explanation for this case but does not "
                   "isolate every source of bias. It does not establish that a 32-pixel production pipeline is "
                   "safe: other 32-pixel development tests produced false acceptances, and these runs use "
                   "truth-provided initialization."), "", "![Oracle errors at originally wrong centers](errors_57_centers.png)", "",
                  ("The raw per-window results, per-pair counts and unwarmed solver wall times are saved in "
                   "`summary.json`, `aggregated.json` and the NPZ files. Unwarmed wall times are diagnostic "
                   "only; they are not production performance benchmarks. The protocol was fixed before "
                   "these runs in `PROTOCOL.md`."), "",
                  "Reproduce in a new directory:", "", "```bash",
                  "base=results/seed_propagation_20261111/p3_shear_band_frozen_protocol",
                  "out=results/seed_propagation_20261111/p3_shear_support_diagnostic_new",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/diagnose_shear_support.py \\",
                  "  --base \"$base\" --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src --output-dir \"$out\"",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_shear_support.py --directory \"$out\"",
                  "```", ""])
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
