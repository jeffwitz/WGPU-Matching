"""Regenerate the compact IC-GN report from saved qualification JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--adapter", default="recorded in the acquisition run")
    args = parser.parse_args()
    raw = json.loads((args.qualification / "summary.json").read_text())
    aggregate = json.loads((args.qualification / "aggregate.json").read_text())
    families = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()

    def ratio(numerator: int, denominator: int) -> str:
        return f"{numerator}/{denominator}" if denominator else "—"

    lines = [
        "# Compact DIC descriptor requalification",
        "",
        "This report evaluates digital image correlation (DIC) initializations with the inverse compositional Gauss–Newton (IC-GN) solver, using the same detector, windows, matching consensus, affine and radial basis function (RBF) projections, and convergence limits as the preceding qualification. Six fixed fields are regression checks; independent texture and motion seeds are the test set.",
        "",
        f"Adapter: {args.adapter}. The compact convolutional neural network (CNN) weights have SHA-256 `{digest}`.",
        "",
        "Counts are correct final IC-GN displacements within 2 pixels / visible windows. A field-fit failure has no visible-window denominator and is shown as a dash, not as a measured zero success. Full-support scoring is a post-run subset requiring the 64 × 64 support to remain valid in the reference and mapped target; it does not change matching or solver inputs.",
        "",
        "## Independent test fields",
        "",
        "| Motion | Descriptor | Affine fields fitted | Affine correct / visible | Affine full-support correct / valid | RBF fields fitted | RBF correct / visible | RBF full-support correct / valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = aggregate["groups"]["independent_test"]
    for family in families:
        group = groups[family]
        for method, label in (("horg64", "HORG-64"), ("dic_cnn", "DIC-CNN hard-negative")):
            affine = group["methods"][method]["affine"]
            rbf = group["methods"][method]["gaussian_rbf"]
            lines.append(
                f"| {family} | {label} | {affine['measured_fields']}/{group['fields']} | "
                f"{ratio(affine['correct_windows'], affine['visible_windows'])} | "
                f"{ratio(affine['correct_windows_full_support'], affine['fully_valid_support_windows'])} | "
                f"{rbf['measured_fields']}/{group['fields']} | "
                f"{ratio(rbf['correct_windows'], rbf['visible_windows'])} | "
                f"{ratio(rbf['correct_windows_full_support'], rbf['fully_valid_support_windows'])} |"
            )
    lines.extend(
        [
            "",
            "A field is fitted only when the projection produced a usable model. Throughput is summed over the successfully fitted fields; it is not an all-field rate when any fit failed. Per-family timing and fit counts are in `aggregate.json`.",
            "",
            "| Motion | Descriptor | Affine correct windows/s, fitted fields only | RBF correct windows/s, fitted fields only |",
            "|---|---|---:|---:|",
        ]
    )
    for family in families:
        group = groups[family]
        for method, label in (("horg64", "HORG-64"), ("dic_cnn", "DIC-CNN hard-negative")):
            affine_rate = group["methods"][method]["affine"]["correct_windows_per_second"]
            rbf_rate = group["methods"][method]["gaussian_rbf"]["correct_windows_per_second"]
            affine_text = f"{affine_rate:.1f}" if affine_rate is not None else "—"
            rbf_text = f"{rbf_rate:.1f}" if rbf_rate is not None else "—"
            lines.append(f"| {family} | {label} | {affine_text} | {rbf_text} |")
    lines.extend(
        [
            "",
            "## Fixed regression fields",
            "",
            "| Case | Descriptor | Affine projection / correct windows | RBF projection / correct windows |",
            "|---|---|---:|---:|",
        ]
    )
    for row in raw:
        if row["split"] != "fixed_regression":
            continue
        for method, label in (("horg64", "HORG-64"), ("dic_cnn", "DIC-CNN hard-negative")):
            parts = []
            for basis in ("affine", "gaussian_rbf"):
                result = row[method].get(basis, {})
                solver = result.get("solver")
                if solver is None:
                    parts.append(result.get("error", "not run"))
                else:
                    parts.append(
                        f"fit; {solver.get('correct_converged_lt2px', 0)}/"
                        f"{solver.get('visible_windows', 0)}"
                    )
            lines.append(
                f"| {row['case']} `{row['texture_seed']}/{row['warp_seed']}` | {label} | {parts[0]} | {parts[1]} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "Hard-negative training improved retrieval for affine and discontinuous motions in the measured 4K retrieval pair, but did not improve rigid or heterogeneous retrieval. In independent IC-GN tests the descriptor still fails to produce a field for most affine and heterogeneous cases and for all independent rigid cases. It is not a general replacement for DeDoDe-B.",
            "",
            "For discontinuous fields, full-support counts exclude supports touching the invalid band. This is a post-hoc scoring subset and does not show that a smooth field model is safe across a physical discontinuity. The two historical failures `affine_8101_9101` and `heterogeneous_8102_9102` remain failures.",
            "",
            "The raw per-window output and generated image pairs remain in the scratch directory. Committed artifacts are compact summaries and this report.",
            "",
        ]
    )
    (args.qualification / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
