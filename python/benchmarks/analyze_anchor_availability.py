"""Audit solver-accepted anchors in archived per-window qualifications."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def components(centres, visible, anchors):
    """Count anchors in each eight-connected component of the visible grid."""
    xs = np.unique(centres[:, 0])
    ys = np.unique(centres[:, 1])
    grid = {(int(np.searchsorted(xs, x)), int(np.searchsorted(ys, y))): i
            for i, (x, y) in enumerate(centres) if visible[i]}
    counts = []
    while grid:
        start = next(iter(grid))
        queue = deque([start])
        count = size = 0
        while queue:
            point = queue.popleft()
            idx = grid.pop(point, None)
            if idx is None:
                continue
            size += 1
            count += int(anchors[idx])
            x, y = point
            queue.extend((x + dx, y + dy) for dx in (-1, 0, 1)
                         for dy in (-1, 0, 1) if (dx or dy) and
                         (x + dx, y + dy) in grid)
        counts.append({"windows": size, "correct_anchors": count})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for path in sorted(args.campaign.glob("*/windows_seed*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            visible = data["visible"].astype(bool)
            accepted = data["icgn_status"].astype(bool)
            correct = data["icgn_correct_lt2px"].astype(bool) & visible
            row = {"pair": path.stem, "source": str(path),
                   "visible": int(visible.sum()),
                   "accepted": int((accepted & visible).sum()),
                   "accepted_invisible": int((accepted & ~visible).sum()),
                   "correct": int(correct.sum()),
                   "false_accepted": int((accepted & visible & ~correct).sum()),
                   "oracle": int((data["oracle_icgn_correct_lt2px"].astype(bool) & visible).sum()),
                   "components": components(data["centres_xy_native"], visible, correct)}
        results.setdefault(path.parent.parent.name, []).append(row)
    if not results:
        raise SystemExit("No per-window archives found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = {}
    for name, rows in results.items():
        totals[name] = {key: sum(row[key] for row in rows) for key in
                        ("visible", "accepted", "accepted_invisible", "correct", "false_accepted", "oracle")}
        totals[name]["median_correct"] = float(np.median([r["correct"] for r in rows]))
        totals[name]["pairs_at_most_one"] = sum(r["correct"] <= 1 for r in rows)
        totals[name]["pairs"] = len(rows)
    args.output.with_suffix(".json").write_text(json.dumps({"totals": totals, "pairs": results}, indent=2) + "\n")
    lines = ["# Archived anchor availability", "", "Inverse-compositional Gauss–Newton (IC-GN) acceptance and correctness are scored on analytically visible windows. Eight-connected components use the archived grid.", "", "| Variant | Pairs | Visible | Accepted | Correct | False accepted | Median correct | Pairs with at most one | Oracle |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, t in totals.items():
        lines.append(f"| {name} | {t['pairs']} | {t['visible']} | {t['accepted']} | {t['correct']} | {t['false_accepted']} | {t['median_correct']:g} | {t['pairs_at_most_one']} | {t['oracle']} |")
    lines += ["", "Per-pair and per-component counts, including accepted invisible windows, are in the accompanying JSON file.", "", "Reproduce from the repository root:", "", "```console", f"python python/benchmarks/analyze_anchor_availability.py {args.campaign} --output {args.output}", "```", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
