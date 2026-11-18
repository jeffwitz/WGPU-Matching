"""Summarize external-image DIC qualifications without inventing missing truth."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.directory / "summary.json").read_text())["cases"]
    lines = ["# Native-image P3 qualification", "",
             ("The frozen image-only initializer and inverse-compositional Gauss–Newton (IC-GN) "
              "solver run on original pixels. For tensile cases, analytical centred affine truth comes "
              "from the archived dataset manifests. Experimental-image accuracy is not inferred."), "",
             "| Case | Native pixels | Correct / visible | Baseline correct | Oracle correct | False accepted | Accepted |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["truth_available"]:
            accuracy = f"{row['correct']}/{row['visible']}"
            baseline, oracle, bad = row["baseline_correct"], row["oracle"], row["false_accepted"]
        else:
            accuracy = baseline = oracle = bad = "not available"
        size = " × ".join(map(str, row["native_size"]))
        lines.append(f"| {row['case']} | {size} | {accuracy} | {baseline} | {oracle} | {bad} | {row['accepted']} |")
    lines.extend(["", ("The green and blue map in each case directory records accepted windows and "
                       "confirmed seeds. Per-window results are in that case's qualification directory. "
                       "These are scripted experimental runs, not production timing measurements."), ""])
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
