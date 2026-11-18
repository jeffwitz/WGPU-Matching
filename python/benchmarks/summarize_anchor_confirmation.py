"""Compare saved serial and batched anchor-confirmation campaigns by window."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--batched", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    serial = {r["pair"]: r for r in json.loads((args.serial / "summary.json").read_text())["rows"]}
    batched = {r["pair"]: r for r in json.loads((args.batched / "summary.json").read_text())["rows"]}
    if not serial or set(serial) != set(batched):
        raise ValueError("campaigns must contain the same nonempty pair set")
    rows = []
    for name in sorted(serial):
        a, b = serial[name], batched[name]
        with np.load(args.serial / f"{name}.npz") as left, np.load(args.batched / f"{name}.npz") as right:
            keys = ("centres", "parameters", "accepted", "parent", "origin", "generation", "confirmed_seeds", "correct")
            equality = {key: bool(np.array_equal(left[key], right[key], equal_nan=True)) for key in keys}
            old_correct, new_correct = left["correct"].astype(bool), right["correct"].astype(bool)
            lost = int((old_correct & ~new_correct).sum())
            gained = int((new_correct & ~old_correct).sum())
        rows.append({"pair": name, "visible": a["visible"], "correct_serial": a["correct"],
                     "correct_batched": b["correct"], "false_serial": a["false_accepted"],
                     "false_batched": b["false_accepted"], "lost": lost, "gained": gained,
                     "calls_serial": a["total_solver_calls_before_control"],
                     "calls_batched": b["total_solver_calls_before_control"],
                     "active_serial": a["total_active_windows_before_control"],
                     "active_batched": b["total_active_windows_before_control"], "exact": equality})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pairs": rows}, indent=2) + "\n")
    print(f"Compared {len(rows)} pairs; exact outputs: {sum(all(r['exact'].values()) for r in rows)}")


if __name__ == "__main__":
    main()
