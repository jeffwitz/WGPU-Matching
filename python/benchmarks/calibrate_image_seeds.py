"""Calibrate seed verification on development pairs and freeze its parameters."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Fixed before any held-out cases: deliberately small development-only grid.
    settings = [(4., .15, 4, 3), (8., .25, 4, 3), (12., .40, 4, 3),
                (8., .25, 2, 3), (8., .25, 4, 1), (8., .25, 4, 5)]
    results = []
    for tau_u, tau_g, stride, minimum in settings:
        directory = args.output_dir / f"u{tau_u:g}_g{tau_g:g}_s{stride}_n{minimum}"
        command = [sys.executable, str(Path(__file__).with_name("qualify_image_seed_pilot.py")),
                   "--archive", str(args.archive), "--rotation-manifest", str(args.rotation_manifest),
                   "--live-dic-source", str(args.live_dic_source), "--output-dir", str(directory),
                   "--sparse-stride", str(stride), "--tau-u", str(tau_u), "--tau-g", str(tau_g),
                   "--escalation", "--minimum-anchors", str(minimum)]
        if not (directory / "summary.json").exists():
            subprocess.run(command, check=True)
        data = json.loads((directory / "summary.json").read_text())
        rows = data["rows"]
        if len(rows) != 20:
            raise ValueError("calibration must use all 20 development pairs")
        accepted = sum(row["accepted"] for row in rows)
        bad = sum(row["false_accepted"] for row in rows)
        results.append({"tau_u": tau_u, "tau_g": tau_g, "sparse_stride": stride,
                        "minimum_anchors": minimum, "correct": sum(r["correct"] for r in rows),
                        "accepted": accepted, "false_accepted": bad,
                        "meets_error_limit": bad / max(accepted, 1) <= .001,
                        "development_report": str(directory / "summary.json")})
    eligible = [row for row in results if row["meets_error_limit"]]
    if not eligible:
        raise SystemExit("No development configuration meets the false acceptance limit")
    selected = max(eligible, key=lambda row: (row["correct"], row["sparse_stride"], -row["tau_u"]))
    frozen = {"purpose": "Frozen before independent test groups 21000000 and 22000000",
              "selection": "maximum development correct coverage with false accepted / accepted <= 0.001",
              "selected": selected, "candidates": results}
    (args.output_dir / "frozen_parameters.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
