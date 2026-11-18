"""Evaluate weighted estimates on qualified native-image P3 cases."""

import argparse
import json
from pathlib import Path

import numpy as np
from wgpu_matching.seed_propagation import weighted_neighbour_initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cases = json.loads((args.directory / "summary.json").read_text())["cases"]
    rows = []
    for case in cases:
        folder = args.directory / case["case"]
        path = folder / "qualification/evaluation" / f"{case['pair']}.npz"
        with np.load(path) as data:
            centres = data["centres"]
            shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))
            fill = weighted_neighbour_initialization(centres, shape, data["valid_roi"], data["parameters"], data["accepted"])
            visible = data["visible"].astype(bool)
        row = {"case": case["case"], "estimated_total": int(fill["available"].sum()),
               "estimated_visible": None, "correct_lt2": None, "error_ge2": None}
        error = np.full(len(centres), np.nan)
        if case["truth_available"]:
            texture = int(case["pair"].split("texture")[1].split("_")[0])
            seed = texture // 1000000 * 1000000
            path = folder / "inputs/affine_ncc" / f"windows_seed{seed}" / f"{case['pair']}.npz"
            with np.load(path) as data:
                error = np.linalg.norm(fill["parameters"][:, [0, 3]] - data["truth_parameters"][:, [0, 3]], axis=1)
            available = fill["available"] & visible
            row.update({"estimated_visible": int(available.sum()), "correct_lt2": int((available & (error < 2)).sum()),
                        "error_ge2": int((available & (error >= 2)).sum())})
        np.savez_compressed(folder / "weighted_estimates.npz", centres=centres, parameters=fill["parameters"],
                            available=fill["available"], support=fill["support"],
                            disagreement_px=fill["disagreement_px"], error_px=error)
        rows.append(row)
    (args.directory / "weighted_summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    lines = ["# Weighted estimates on native-image cases", "",
             ("Only solver-accepted neighbour states are sources. At least two neighbours and "
              "agreement within 8 px are required. The resulting values are interpolated estimates, "
              "not inverse-compositional Gauss–Newton (IC-GN) accepted measurements."), "",
             "| Case | Estimated total | Estimated visible | Correct <2 px | Error ≥2 px |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(value):
            return str(value) if value is not None else "not available"
        lines.append(f"| {row['case']} | {row['estimated_total']} | {fmt(row['estimated_visible'])} | "
                     f"{fmt(row['correct_lt2'])} | {fmt(row['error_ge2'])} |")
    lines.extend(["", ("Analytical truth is used only to score the tensile images. No accuracy "
                  "is inferred for the experimental pair. Estimates can cross unobserved "
                  "material boundaries; source masks should be supplied when available."), ""])
    (args.directory / "weighted_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
