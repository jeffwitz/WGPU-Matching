"""Summarize fine-grid accepted and false accepted windows from saved archives."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("windows_seed*/*.npz")):
        with np.load(path) as data:
            visible = data["visible"].astype(bool)
            accepted = data["icgn_status"].astype(bool) & visible
            error = data["icgn_displacement_error_px"]
            rows.append({"pair": path.stem, "visible": int(visible.sum()),
                         "accepted": int(accepted.sum()), "correct": int((accepted & (error < 2)).sum()),
                         "false_accepted": int((accepted & (error >= 2)).sum()),
                         "false_indices": np.flatnonzero(accepted & (error >= 2)).tolist()})
    if not rows:
        parser.error("no window archives found")
    accepted = sum(r["accepted"] for r in rows)
    bad = sum(r["false_accepted"] for r in rows)
    (args.directory / "baseline_summary.json").write_text(json.dumps({"pairs": rows}, indent=2) + "\n")
    lines = ["# Direct 32 px window acceptance", "",
             ("The affine-plus-local normalized cross-correlation (NCC) initialization is followed by "
              "the inverse-compositional Gauss–Newton (IC-GN) solver with a 32 px subset and 16 px stride. "
              "These synthetic pairs were previously inspected; analytical visibility is used only for scoring."), "",
             f"Visible windows: {sum(r['visible'] for r in rows)}.",
             f"Correct: {sum(r['correct'] for r in rows)}.",
             f"False accepted: {bad}/{accepted} ({bad / max(accepted, 1):.4%}).", "",
             "Per-pair false-acceptance indices are in `baseline_summary.json`.", ""]
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
