"""Score snapshot-weighted estimates on archived image-only qualifications."""

import argparse
import json
from pathlib import Path

import numpy as np
from wgpu_matching.seed_propagation import neighbours, weighted_neighbour_initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-boundary-x", type=float)
    parser.add_argument("--oracle-region-boundary-x", type=float,
                        help="oracle-only control: prohibit interpolation across this known material boundary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.evaluation / "summary.json").read_text())
    rows = []
    for case in report["rows"]:
        name = case["pair"]
        texture = int(name.split("texture")[1].split("_")[0])
        seed = texture // 1000000 * 1000000
        truth_path = args.archive / "affine_ncc" / f"windows_seed{seed}" / f"{name}.npz"
        with np.load(args.evaluation / f"{name}.npz") as source, np.load(truth_path) as truth:
            centres = source["centres"]
            shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))
            accepted = source["accepted"].astype(bool)
            visible = source["visible"].astype(bool)
            labels = None if args.oracle_region_boundary_x is None else (
                centres[:, 0] >= args.oracle_region_boundary_x).astype(np.int32)
            fill = weighted_neighbour_initialization(centres, shape, np.ones(len(centres), bool),
                                                      source["parameters"], accepted, region_labels=labels)
            error = np.linalg.norm(fill["parameters"][:, [0, 3]]
                                   - truth["truth_parameters"][:, [0, 3]], axis=1)
            available = fill["available"] & visible
            correct = available & (error < 2)
            crossing_sources = np.zeros(len(centres), dtype=bool)
            if args.diagnostic_boundary_x is not None:
                for j in np.flatnonzero(available):
                    sources = [i for i in neighbours(shape, j) if accepted[i]]
                    crossing_sources[j] = any((centres[i, 0] < args.diagnostic_boundary_x)
                                              != (centres[j, 0] < args.diagnostic_boundary_x) for i in sources)
            rows.append({"pair": name, "visible_unresolved": int((visible & ~accepted).sum()),
                         "estimated_visible": int(available.sum()), "correct_lt2": int(correct.sum()),
                         "correct_lt5": int((available & (error < 5)).sum()),
                         "estimated_invisible": int((fill["available"] & ~visible).sum()),
                         "incorrect_ge5_with_cross_boundary_source": int((available & (error >= 5) & crossing_sources).sum()),
                         "median_error_px": float(np.median(error[available])) if available.any() else None,
                         "max_error_px": float(np.max(error[available])) if available.any() else None})
            np.savez_compressed(args.output_dir / f"{name}.npz", centres=centres,
                                parameters=fill["parameters"], available=fill["available"],
                                support=fill["support"], disagreement_px=fill["disagreement_px"],
                                visible=visible, error_px=error, correct_lt2=correct,
                                diagnostic_cross_boundary_source=crossing_sources)
    (args.output_dir / "summary.json").write_text(json.dumps({"protocol": {
        "min_neighbours": 2, "maximum_disagreement_px": 8, "weights": "inverse distance",
        "sources": "accepted solver states only", "estimates_are_solver_accepted": False,
        "oracle_region_boundary_x": args.oracle_region_boundary_x,
        "evaluation": str(args.evaluation), "archive": str(args.archive)}, "pairs": rows}, indent=2) + "\n")
    lines = ["# Weighted estimates for unresolved windows", "",
             ("Validated neighbouring inverse-compositional Gauss–Newton (IC-GN) states are transported "
              "to each unresolved centre and averaged with inverse-distance weights. At least two "
              "sources are required and candidate displacements must agree within 8 px. These "
              "estimates have not been accepted by the solver and are not used as new sources."), "",
             "| Family | Visible unresolved | Estimated visible | Correct <2 px | Correct <5 px | Estimated invisible |",
             "|---|---:|---:|---:|---:|---:|"]
    if args.oracle_region_boundary_x is not None:
        lines[2] = (f"ORACLE CONTROL: the analytically known material boundary at x = "
                    f"{args.oracle_region_boundary_x:g} px restricts which neighbours may be used. "
                    "This boundary is unavailable to the unknown-image pipeline. " + lines[2])
    for family in sorted({row["pair"].split("_")[0] for row in rows}):
        group = [row for row in rows if row["pair"].startswith(family + "_")]
        lines.append(f"| {family} | " + " | ".join(str(sum(row[k] for row in group)) for k in
                     ("visible_unresolved", "estimated_visible", "correct_lt2", "correct_lt5", "estimated_invisible")) + " |")
    lines.extend(["", ("The analytical field is used only for the reported errors. See `summary.json` "
                  "and the per-window NPZ archives for support counts, disagreement and errors."), ""])
    lines.extend([("The 8 px disagreement check is only a preliminary guard. It does not prove "
                   "that source windows share a material region and has not been qualified on "
                   "cracks or discontinuities. Estimated windows remain marked as interpolated."), ""])
    if args.diagnostic_boundary_x is not None:
        count = sum(row["incorrect_ge5_with_cross_boundary_source"] for row in rows)
        lines.extend([f"Post-hoc scoring at boundary x = {args.diagnostic_boundary_x:g} px finds {count} "
                      "estimates with error ≥5 px and at least one accepted source across the boundary. "
                      + ("This boundary also restricts sources in the oracle control."
                         if args.oracle_region_boundary_x is not None else
                         "The boundary is not used to select or alter estimates."), ""])
    lines.extend(["Reproduce from the repository root:", "", "```bash",
                  "PYTHONPATH=python/src .venv/bin/python python/benchmarks/analyze_weighted_seed_fill.py \\",
                  f"  --evaluation {args.evaluation} --archive {args.archive} \\",
                  f"  --output-dir {args.output_dir}"
                  + (f" --diagnostic-boundary-x {args.diagnostic_boundary_x:g}" if args.diagnostic_boundary_x is not None else "")
                  + (f" --oracle-region-boundary-x {args.oracle_region_boundary_x:g}" if args.oracle_region_boundary_x is not None else ""),
                  "```", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
