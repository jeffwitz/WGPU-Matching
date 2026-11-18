"""Compare separately generated affine/NCC archives with shared-prefit exports."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separate", required=True, type=Path)
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--seed-offset", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    folder = f"windows_seed{args.seed_offset}"
    shared_files = sorted((args.shared / "affine_ncc" / folder).glob("*.npz"))
    separate_files = sorted((args.separate / "affine_ncc" / folder).glob("*.npz"))
    if not shared_files or [x.name for x in shared_files] != [x.name for x in separate_files]:
        raise ValueError("shared and separate datasets must contain identical nonempty pair sets")
    rows = []
    for path in shared_files:
        with (np.load(args.separate / "affine_only" / folder / path.name) as affine,
              np.load(args.separate / "affine_ncc" / folder / path.name) as ncc,
              np.load(path) as shared):
            values = {"affine_initial": (affine["initial_parameters"], shared["pre_ncc_initial_parameters"]),
                      "affine_status": (affine["icgn_status"], shared["pre_ncc_icgn_status"]),
                      "affine_result": (affine["icgn_parameters"], shared["pre_ncc_icgn_parameters"]),
                      "ncc_initial": (ncc["initial_parameters"], shared["initial_parameters"]),
                      "ncc_status": (ncc["icgn_status"], shared["icgn_status"]),
                      "ncc_result": (ncc["icgn_parameters"], shared["icgn_parameters"]),
                      "visibility": (ncc["visible"], shared["visible"]),
                      "truth": (ncc["truth_parameters"], shared["truth_parameters"])}
            exact = {key: bool(np.array_equal(a, b, equal_nan=True)) for key, (a, b) in values.items()}
        rows.append({"pair": path.stem, "exact": exact})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pairs": rows}, indent=2) + "\n")
    if not all(all(row["exact"].values()) for row in rows):
        raise SystemExit("at least one comparison differs")
    print(f"Exact equality: {len(rows)} pairs")


if __name__ == "__main__":
    main()
