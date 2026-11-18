"""Audit weighted unresolved estimates on archived discontinuity and shear fields."""

import argparse
import json
from pathlib import Path

import numpy as np
from wgpu_matching.seed_propagation import neighbours, weighted_neighbour_initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.archive.glob("seed*/*/windows/*.npz")):
        family = path.parent.parent.name
        if not (family == "discontinuous" or family.startswith("shear_band")):
            continue
        with np.load(path) as data:
            centres = data["centres_xy_native"]
            shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))
            visible = data["visible"].astype(bool)
            accepted = data["propagated_status"].astype(bool)
            fill = weighted_neighbour_initialization(centres, shape, np.ones(len(centres), bool),
                                                      data["propagated_parameters"], accepted)
            error = np.linalg.norm(fill["parameters"][:, [0, 3]]
                                   - data["truth_parameters"][:, [0, 3]], axis=1)
            eligible = fill["available"] & visible
            bad = np.flatnonzero(eligible & (error >= 2))
            source_error = data["propagated_displacement_error_px"]
            all_sources_correct = np.zeros(len(centres), dtype=bool)
            for index in np.flatnonzero(eligible):
                sources = [i for i in neighbours(shape, index) if accepted[i]]
                all_sources_correct[index] = all(visible[i] and source_error[i] < 2 for i in sources)
            rows.append({"pair": path.stem, "seed_group": path.parents[2].name, "family": family,
                         "visible_unresolved": int((visible & ~accepted).sum()),
                         "estimated_visible": int(eligible.sum()), "correct_lt2": int((eligible & (error < 2)).sum()),
                         "correct_lt5": int((eligible & (error < 5)).sum()),
                         "error_ge2": len(bad),
                         "error_ge2_with_all_sources_correct": int((eligible & (error >= 2) & all_sources_correct).sum()),
                         "max_error_px": float(error[eligible].max()) if eligible.any() else None,
                         "bad_windows": [{"index": int(i), "xy": centres[i].tolist(), "error_px": float(error[i])}
                                         for i in bad]})
            np.savez_compressed(args.output_dir / f"{path.parents[2].name}_{family}_{path.stem}.npz", centres=centres,
                                visible=visible, parameters=fill["parameters"], available=fill["available"],
                                support=fill["support"], error_px=error, all_sources_correct=all_sources_correct)
    if not rows:
        parser.error("no discontinuity or shear archives found")
    (args.output_dir / "summary.json").write_text(json.dumps({"pairs": rows}, indent=2) + "\n")
    lines = ["# Weighted estimates on discontinuities and shear bands", "",
             ("Only accepted inverse-compositional Gauss–Newton (IC-GN) neighbours are sources. "
              "Estimates use inverse-distance weights, at least two sources and an 8 px maximum "
              "displacement disagreement. They are not solver-accepted measurements. Truth is used "
              "only for scoring; these archived cases were previously inspected."), "",
             "| Family | Pairs | Unresolved visible | Estimated visible | Correct <2 px | Correct <5 px | Largest error (px) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for family in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == family]
        maximum = max((r["max_error_px"] for r in group if r["max_error_px"] is not None), default=None)
        lines.append(f"| {family} | {len(group)} | {sum(r['visible_unresolved'] for r in group)} | "
                     f"{sum(r['estimated_visible'] for r in group)} | {sum(r['correct_lt2'] for r in group)} | "
                     f"{sum(r['correct_lt5'] for r in group)} | {maximum:.2f} |" if maximum is not None
                     else f"| {family} | {len(group)} | {sum(r['visible_unresolved'] for r in group)} | 0 | 0 | 0 | — |")
    lines.extend(["", "Estimates with error ≥2 px despite every source being visible and correct to <2 px:", ""])
    for family in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == family]
        lines.append(f"- {family}: {sum(r['error_ge2_with_all_sources_correct'] for r in group)}")
    lines.extend(["", "Per-window errors and every estimate with error at least 2 px are in the saved JSON and NPZ outputs.",
                  "", "Reproduce:", "", "```bash",
                  "PYTHONPATH=python/src .venv/bin/python python/benchmarks/analyze_weighted_frontier_families.py \\",
                  f"  --archive {args.archive} --output-dir {args.output_dir}", "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
