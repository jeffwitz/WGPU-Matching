"""Summarize fixed P3 perturbation cohorts and opening-crossing windows."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    names = ["shifted_opening", "doubled_noise", "photometric", "defocus"]
    if (args.directory / "camera_noise/qualification/evaluation/summary.json").is_file():
        names.append("camera_noise")
    if (args.directory / "periodic/qualification/evaluation/summary.json").is_file():
        names.append("periodic")
    groups = []
    for name in names:
        folder = args.directory / name / "qualification/evaluation"
        report = json.loads((folder / "summary.json").read_text())
        rows = report["rows"]
        if len(rows) != 5:
            raise ValueError(f"incomplete cohort {name}")
        crossing = {"visible": 0, "accepted": 0, "correct_lt2": 0, "false_accepted": 0,
                    "cross_boundary_parent_links": 0}
        for row in rows:
            if name != "shifted_opening":
                continue
            with np.load(folder / f"{row['pair']}.npz") as data:
                x = data["centres"][:, 0]
                left = x + 31.5 < 1932
                right = x - 31.5 >= 1932
                mask = ~(left | right) & data["visible"].astype(bool)
                accepted = data["accepted"].astype(bool) & mask
                crossing["visible"] += int(mask.sum())
                crossing["accepted"] += int(accepted.sum())
                crossing["correct_lt2"] += int((accepted & (data["error"] < 2)).sum())
                crossing["false_accepted"] += int((accepted & (data["error"] >= 2)).sum())
                parent = data["parent"]
                child = np.flatnonzero(data["accepted"].astype(bool) & (parent >= 0))
                crossing["cross_boundary_parent_links"] += int(
                    ((x[child] < 1932) != (x[parent[child]] < 1932)).sum()
                )
        groups.append({"name": name, "pairs": rows, "crossing": crossing})
    (args.directory / "summary.json").write_text(json.dumps({"cohorts": groups}, indent=2) + "\n")
    lines = ["# Fixed P3 perturbation cohorts", "",
             ("The inverse-compositional Gauss–Newton (IC-GN) solver uses the frozen image-only "
              "initialization settings. Analytic truth is used only after inference for scoring. "
              "Errors refer to centre displacement, not full-support physical validity."), "",
             "| Cohort | Correct / visible | Baseline correct | False / accepted | Pairs below baseline |",
             "|---|---:|---:|---:|---:|"]
    for group in groups:
        rows = group["pairs"]
        lines.append(f"| {group['name']} | {sum(r['correct'] for r in rows)}/{sum(r['visible'] for r in rows)} | "
                     f"{sum(r['baseline_correct'] for r in rows)} | {sum(r['false_accepted'] for r in rows)}/"
                     f"{sum(r['accepted'] for r in rows)} | {sum(r['correct'] < r['baseline_correct'] for r in rows)} |")
    seam = groups[0]["crossing"]
    no_acceptance = [group["name"] for group in groups if sum(r["accepted"] for r in group["pairs"]) == 0]
    lines.extend(["", (f"Shifted opening: {seam['visible']} analytically visible window centres have supports "
                       f"that cross the material boundary; {seam['accepted']} were accepted, of which "
                       f"{seam['correct_lt2']} have centre error <2 px. Even a correct centre does not "
                       "make the affine model physically valid across the discontinuity. "
                       f"There are {seam['cross_boundary_parent_links']} accepted propagation parent links "
                       "whose endpoints lie on opposite sides of the boundary."), "",
                  (f"Cohorts with no accepted windows: {', '.join(no_acceptance) if no_acceptance else 'none'}. "
                   "For these cohorts the false-acceptance rate is undefined; abstention does not validate rejection."), "",
                  ("G3 remains incomplete until the remaining prescribed cases are evaluated. See the "
                  "predeclared `p3_perturbation_protocol.md` and per-window outputs."), ""])
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
