"""Summarize the frozen oracle-seeded propagation experiment by pair and draw."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    source = json.loads((args.directory / "summary.json").read_text())
    rows = source["rows"]
    expected = 20 * source["draws"] * 12
    if len(rows) != expected:
        raise ValueError(f"incomplete run: {len(rows)} rows, expected {expected}")
    lines = ["# Oracle-seeded propagation qualification", "",
             ("The inverse-compositional Gauss–Newton (IC-GN) solver runs on WebGPU. "
             "Analytical truth is used to choose the explicitly oracle-labelled seeds and score results. "
             "The direct-neighbour-first frontier uses two attempts per pass and fixed consistency "
             "limits of 8 pixels and 0.25 for the displacement-gradient Frobenius norm."), "",
             "| Anchors | Affine transport | Consistency | Correct / oracle | False accepted / accepted | Median passes | Control correct |",
             "|---:|---|---|---:|---:|---:|---:|"]
    groups = defaultdict(list)
    for row in rows:
        with np.load(args.directory / row["window_export"]) as data:
            # The oracle seeds are shared by both arms and are never in the frontier.
            shared = ((data["generation"] == 0) & data["correct"].astype(bool)
                      & ~data["control_status"].astype(bool))
            row["control_correct"] += int(shared.sum())
        groups[(row["k"], row["affine"], row["consistency"])].append(row)
    maps = args.directory / "frontier_maps"
    maps.mkdir(exist_ok=True)
    for row in groups[(4, True, True)]:
        with np.load(args.directory / row["window_export"]) as data:
            centres = data["centres"]
            xs, ys = np.unique(centres[:, 0]), np.unique(centres[:, 1])
            rgb = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)
            colours = np.full((len(centres), 3), (80, 80, 80), dtype=np.uint8)
            colours[data["reason"] == 1] = (220, 130, 0)
            colours[data["reason"] == 2] = (220, 0, 0)
            colours[data["reason"] == 3] = (170, 0, 170)
            colours[data["accepted"].astype(bool)] = (255, 255, 255)
            colours[data["correct"].astype(bool)] = (0, 180, 80)
            colours[data["generation"] == 0] = (0, 120, 255)
            colours[~data["visible"].astype(bool)] = (0, 0, 0)
            rgb[np.searchsorted(ys, centres[:, 1]), np.searchsorted(xs, centres[:, 0])] = colours
            image = Image.fromarray(rgb)
            image.resize((len(xs) * 8, len(ys) * 8), Image.Resampling.NEAREST).save(
                maps / f"{row['pair']}_draw{row['draw']}.png")
    for (k, affine, consistency), group in sorted(groups.items()):
        correct = sum(row["correct"] for row in group)
        oracle = sum(row["oracle"] for row in group)
        bad = sum(row["false_accepted"] for row in group)
        accepted = sum(row["accepted"] for row in group)
        control = sum(row["control_correct"] for row in group)
        lines.append(f"| {k} | {affine} | {consistency} | {correct}/{oracle} "
                     f"({correct / oracle:.2%}) | {bad}/{accepted} | "
                     f"{np.median([row['passes'] for row in group]):g} | {control} |")
    lines += ["", ("The no-propagation control repeats the same active-window schedules, starting "
              "each window from its own archived affine-plus-local-correlation initial state; it "
              "does not use the transported state. Both arms include the same correct oracle seeds."),
              "", "## Gate G1 by draw", ""]
    verdicts = []
    for draw in range(source["draws"]):
        selected = [r for r in groups[(4, True, True)] if r["draw"] == draw]
        correct = sum(r["correct"] for r in selected)
        oracle = sum(r["oracle"] for r in selected)
        count80 = sum(r["correct"] >= .8 * r["oracle"] for r in selected)
        bad = sum(r["false_accepted"] for r in selected)
        accepted = sum(r["accepted"] for r in selected)
        passed = correct >= 19231 and count80 >= 18 and bad / max(accepted, 1) <= .001
        verdicts.append(passed)
        verdict = "PASS" if passed else "REJECT" if correct < 16266 else "INDETERMINATE"
        lines.append(f"- Draw {draw}: {correct}/{oracle} correct, {count80}/20 pairs at "
                     f"80% of oracle, {bad}/{accepted} false accepted: "
                     f"{verdict}.")
    lines += ["", ("All five independent anchor draws must satisfy the acceptance criteria for "
              f"this conservative verdict: {'PASS' if all(verdicts) else 'NOT ESTABLISHED'}."), "",
              ("Per-window states, provenance, rejected-front reasons and matched control outputs "
              "are stored in the NPZ files; per-pair measurements and archive hashes are in JSON. "
              "The solver's private `_register_seeded_frame` adapter is an integration limitation."), "",
              "Reproduce with:", "", "```console",
              "PYTHONPATH=python/src .venv/bin/python python/benchmarks/qualify_seed_propagation.py \\",
              "  --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src \\",
              "  --archive results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109 \\",
              "  --output-dir results/seed_propagation_20261111/p1_oracle --draws 5",
              "```", ""]
    paired = defaultdict(list)
    for row in groups[(4, True, True)]:
        paired[row["pair"]].append(row)
    lines.extend(["## Paired comparison by image pair", "",
                  ("The difference is the mean coverage difference over the five seed draws; "
                  "bootstrap resampling uses image pairs as the independent units."), "",
                  "| Pair | Correct, mean | Control, mean | Visible | Mean difference (percentage points) |",
                  "|---|---:|---:|---:|---:|"])
    differences = []
    for name, group in sorted(paired.items()):
        visible = group[0]["visible"]
        correct = np.mean([r["correct"] for r in group])
        control = np.mean([r["control_correct"] for r in group])
        difference = 100 * (correct - control) / max(visible, 1)
        differences.append(difference)
        lines.append(f"| {name} | {correct:.1f} | {control:.1f} | {visible} | {difference:+.2f} |")
    rng = np.random.default_rng(20261111)
    differences = np.asarray(differences)
    samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    lines.extend(["", (f"Mean paired difference: {differences.mean():+.2f} points; "
                  f"95% pair bootstrap interval [{lo:+.2f}, {hi:+.2f}]. "
                  f"Improved/equal/regressed pairs: {int((differences > 0).sum())}/"
                  f"{int((differences == 0).sum())}/{int((differences < 0).sum())}. "
                  "Bootstrap seed: 20261111."), ""])
    lines.extend(["", ("The `frontier_maps` images show oracle anchors in blue, correctly "
                  "resolved visible windows in green, solver rejection in orange, "
                  "displacement inconsistency in red, gradient inconsistency in purple, "
                  "incorrect accepted windows in white, unattempted visible windows in gray "
                  "and invisible windows in black."), ""])
    lines.extend(["## Source fingerprints", "", "SHA-256 hashes of the current local sources:", ""])
    for source_path in (Path("python/benchmarks/qualify_seed_propagation.py"),
                        Path("python/src/wgpu_matching/seed_propagation.py"),
                        Path("python/src/wgpu_matching/dic_synthetic.py"),
                        Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src/live_dic/wgpu_backend.py")):
        if source_path.is_file():
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            lines.append(f"- `{source_path}`: `{digest}`")
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
