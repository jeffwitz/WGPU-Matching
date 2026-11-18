"""Summarize the first fixed-parameter P3 motion-family qualification."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.evaluation / "summary.json").read_text())["rows"]
    families = ("rigid", "affine", "discontinuous")
    if any(sum(r["pair"].startswith(f + "_") for r in rows) != 10 for f in families):
        raise ValueError("expected ten pairs of each motion family")
    details = []
    for row in rows:
        with np.load(args.evaluation / f"{row['pair']}.npz") as data:
            centre = data["centres"]
            accepted = data["accepted"].astype(bool)
            visible = data["visible"].astype(bool)
            bad = accepted & visible & (data["error"] >= 2)
            side_data = {}
            if row["pair"].startswith("discontinuous_"):
                left = centre[:, 0] + 31.5 < 1920
                right = centre[:, 0] - 31.5 >= 1920
                for name, side in (("left", left), ("crossing", ~(left | right)), ("right", right)):
                    side_data[name] = {"accepted": int((accepted & visible & side).sum()),
                                       "false_accepted": int((bad & side).sum()),
                                       "confirmed_seeds": int((data["confirmed_seeds"].astype(bool) & side).sum())}
            details.append(row | {"discontinuity_regions": side_data})
    lines = ["# Initial P3 qualification: fixed-parameter motion families", "",
             ("The inverse-compositional Gauss–Newton (IC-GN) solver evaluates image-derived "
              "seeds and propagation. Analytic displacement and visibility are used only for "
              "scoring. False acceptance means a visible accepted centre with error at least 2 px. "
              "Interpolated values are not counted as solver successes."), "",
             "| Family | Correct / visible | Baseline correct | False / accepted | Pairs below baseline |",
             "|---|---:|---:|---:|---:|"]
    for family in families:
        group = [r for r in rows if r["pair"].startswith(family + "_")]
        lines.append(f"| {family} | {sum(r['correct'] for r in group)}/{sum(r['visible'] for r in group)} | "
                     f"{sum(r['baseline_correct'] for r in group)} | "
                     f"{sum(r['false_accepted'] for r in group)}/{sum(r['accepted'] for r in group)} | "
                     f"{sum(r['correct'] < r['baseline_correct'] for r in group)} |")
    discontinuous = [r for r in details if r["discontinuity_regions"]]
    crossing_accepted = sum(r["discontinuity_regions"]["crossing"]["accepted"] for r in discontinuous)
    both_sides = sum(all(r["discontinuity_regions"][side]["confirmed_seeds"] > 0
                         for side in ("left", "right")) for r in discontinuous)
    lines.extend(["", (f"Discontinuities: {both_sides}/{len(discontinuous)} pairs have confirmed seeds "
                      f"on each side. {crossing_accepted} accepted visible windows have support crossing "
                      "the material boundary. In this generator the boundary aligns with the 64 px grid; "
                      "no window support crosses it, so crossing-window reliability was not tested. "
                      "Centre error alone does not validate full support."), ""])
    lines.extend(["", ("G3 remains incomplete: the prescribed perturbations, traction series, smaller windows "
                  "and experimental images have not all been evaluated. The JSON contains per-pair "
                  "counts, including discontinuity sides; source and propagation maps are in the NPZ files."), ""])
    (args.evaluation / "p3_initial_summary.json").write_text(json.dumps({"pairs": details}, indent=2) + "\n")
    (args.evaluation / "p3_initial_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
