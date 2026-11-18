"""Compare deferred and eager rotation paths using saved per-window results."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager", required=True, type=Path)
    parser.add_argument("--lazy", required=True, type=Path)
    parser.add_argument("--new-pairs", action="store_true",
                        help="label data as newly generated under a previously fixed protocol")
    args = parser.parse_args()
    eager = {row["pair"]: row for row in json.loads((args.eager / "summary.json").read_text())["rows"]}
    lazy = json.loads((args.lazy / "summary.json").read_text())["rows"]
    out = []
    for row in lazy:
        name = row["pair"]
        before = eager[name]
        with np.load(args.eager / f"{name}.npz") as a, np.load(args.lazy / f"{name}.npz") as b:
            if not np.array_equal(a["centres"], b["centres"]):
                raise ValueError(f"window centres differ for {name}")
            old_correct, new_correct = a["correct"].astype(bool), b["correct"].astype(bool)
            lost = int((old_correct & ~new_correct).sum())
            gained = int((new_correct & ~old_correct).sum())
            accepted_changes = int(np.count_nonzero(a["accepted"] != b["accepted"]))
        out.append({"pair": name, "visible": row["visible"], "eager_correct": before["correct"],
                    "lazy_correct": row["correct"], "gained": gained, "lost": lost,
                    "accepted_mask_changes": accepted_changes,
                    "eager_false": before["false_accepted"], "lazy_false": row["false_accepted"],
                    "requested": row["lazy_rotation_requested"], "cache_hit": row["lazy_rotation_cache_hit"],
                    "generated": row["lazy_rotation_hypotheses_generated"]})
    (args.lazy / "comparison.json").write_text(json.dumps({"pairs": out}, indent=2) + "\n")
    scope = ("These pairs were newly generated under a fixed protocol; this small sample is not final qualification. "
             if args.new_pairs else "These previously inspected pairs are a post-hoc diagnostic. ")
    lines = ["# Deferred versus eager rotation hypotheses", "",
             (scope + "Both paths start from "
              "precomputed global affine and local normalized cross-correlation (NCC) inputs. "
              "This excludes their generation, the initial global image search and complete pipeline latency. "
              "Analytical truth is used only to score outputs."), "",
             (f"Pairs: {len(out)}; eager / deferred correct: {sum(x['eager_correct'] for x in out)} / "
             f"{sum(x['lazy_correct'] for x in out)}; lost / gained correct windows: "
             f"{sum(x['lost'] for x in out)} / {sum(x['gained'] for x in out)}."),
             f"Eager / deferred false accepted: {sum(x['eager_false'] for x in out)} / {sum(x['lazy_false'] for x in out)}.",
             (f"Pairs requesting extra rotations: {sum(x['requested'] for x in out)} / {len(out)}; "
             f"additional rotations generated: {sum(x['generated'] for x in out)}."), "",
             "| Pair | Visible | Correct eager / deferred | Correct lost / gained | False eager / deferred | Rotations requested |",
             "|---|---:|---:|---:|---:|---|"]
    for x in out:
        lines.append(f"| {x['pair']} | {x['visible']} | {x['eager_correct']} / {x['lazy_correct']} | "
                     f"{x['lost']} / {x['gained']} | {x['eager_false']} / {x['lazy_false']} | {x['requested']} |")
    lines.extend(["", ("Per-window arrays retain the location of all changed decisions. This experiment "
                       "does not establish general non-regression beyond these pairs or on real images. "
                       "See the five-pair report and fallback test for timing limits."), "",
                  "Reproduce this comparison of the saved outputs:", "", "```bash",
                  "PYTHONPATH=python/src .venv/bin/python python/benchmarks/summarize_lazy_rotation.py \\",
                  f"  --eager {args.eager} --lazy {args.lazy}" + (" --new-pairs" if args.new_pairs else ""), "```", ""])
    (args.lazy / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
