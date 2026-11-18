"""Summarize same-center fine-subset re-solves from image-verified coarse states."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gradient", type=float, default=2.0)
    parser.add_argument("--coarse-directory", type=Path)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=59000000)
    args = parser.parse_args()
    rows = json.loads((args.directory / "summary.json").read_text())["cases"]
    metrics = ("visible", "available_visible", "accepted_visible", "correct", "false", "accepted_invisible",
               "original_correct", "original_false", "original_bad_recovered", "original_bad_rejected",
               "original_bad_still_false", "original_good_lost", "newly_correct")
    totals = {key: sum(row[key] for row in rows) for key in metrics}
    incidents = []
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.3 * len(rows)), squeeze=False)
    colors = ListedColormap(["#d3d3d3", "#f5cf72", "#3c9d78", "#b72d34"])
    for n, row in enumerate(rows):
        path = args.directory / f"{row['pair']}.npz"
        coarse_path = (args.coarse_directory or (args.base / f"g{args.gradient}/evaluation")) / path.name
        truth_path = args.base / "exact_center_analysis" / f"g{args.gradient}" / path.name
        if not truth_path.exists() and args.gradient == 2.0:
            truth_path = args.base / "exact_center_analysis" / path.name
        with np.load(path) as data, np.load(coarse_path) as coarse, np.load(truth_path) as exact:
            xy = data["centres"]
            visible = data["visible"].astype(bool)
            truth = exact["truth_parameters"]
            initial_error = np.linalg.norm(data["initial"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
            fine_bad = visible & data["accepted"].astype(bool) & (data["error_px"] >= 2)
            coarse_error = np.linalg.norm(coarse["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
            coarse_bad = visible & coarse["accepted"].astype(bool) & (coarse_error >= 2)
            for i in np.flatnonzero(fine_bad):
                source = int(data["source"][i])
                incidents.append({"pair": row["pair"], "index": int(i), "center_xy": xy[i].tolist(),
                                  "error_px": float(data["error_px"][i]), "initial_error_px": float(initial_error[i]),
                                  "source_index": source, "source_center_xy": xy[source].tolist(),
                                  "source_error_px": float(coarse_error[source]),
                                  "source_visible_for_diagnostic": bool(visible[source]),
                                  "source_generation": int(coarse["generation"][source])})
            shape = (len(np.unique(xy[:, 1])), len(np.unique(xy[:, 0])))
            for col, (accepted, good, bad, title) in enumerate((
                (coarse["accepted"].astype(bool), data["original_good"].astype(bool), coarse_bad, "64 px original"),
                (data["accepted"].astype(bool), data["correct"].astype(bool), fine_bad, "32 px re-solve"),
            )):
                category = np.zeros(len(visible), np.uint8)
                category[visible] = 1
                category[visible & accepted & good] = 2
                category[bad] = 3
                ax = axes[n, col]
                ax.imshow(category.reshape(shape), cmap=colors, vmin=0, vmax=3, interpolation="nearest",
                          extent=(0, 3840, 2160, 0))
                ax.axhline(1080, color="#222", linewidth=.5, linestyle="--")
                ax.set_title(f"{row['pair'].split('_texture')[1]}: {title}")
                ax.set_xlabel("x (px)")
                ax.set_ylabel("y (px)")
    fig.suptitle("Exact-center scoring: gray invisible, yellow unresolved, green correct, red false")
    fig.tight_layout()
    fig.savefig(args.directory / "coarse_vs_fine.png", dpi=150)
    plt.close(fig)
    (args.directory / "aggregated.json").write_text(json.dumps({"totals": totals, "false_acceptances": incidents}, indent=2) + "\n")
    scope = ("These pairs were reserved before this run; this is a small independent test of the fixed transfer policy. "
             if args.independent else "These pairs were previously inspected; this is a post-hoc diagnostic. ")
    lines = ["# Image-initialized fine support at the original centers", "",
             (f"{len(rows)} synthetic 3840 × 2160 pairs with peak shear gradient {args.gradient}. "
              "Every fine state comes from the nearest accepted image-initialized 64-pixel state, transported "
              "without ground truth and re-solved with a 32-pixel subset on the same original center. "
              "The fine solver uses a 16-pixel grid solely to address those centers. " + scope
              + "The original frozen G3 protocol is unchanged."), "",
             "| Measurement | Original 64 px | Re-solved 32 px |", "| --- | ---: | ---: |",
             f"| Correct visible | {totals['original_correct']}/{totals['visible']} | {totals['correct']}/{totals['visible']} |",
             f"| False accepted visible | {totals['original_false']} | {totals['false']} / {totals['accepted_visible']} |",
             f"| Originally wrong, recovered | — | {totals['original_bad_recovered']}/{totals['original_false']} |",
             f"| Originally correct, lost | — | {totals['original_good_lost']} |", ""]
    for item in incidents:
        lines.append(f"A false acceptance is at {item['pair']}, center {item['center_xy']}: final error "
                     f"{item['error_px']:.3f} px, versus {item['initial_error_px']:.3f} px at the transported initial state. "
                     f"Its source at {item['source_center_xy']} was accepted and its source-center error was "
                     f"{item['source_error_px']:.3f} px. Source correctness is known only after the experiment.")
    lines.extend(["", ("These results concern only this specified shear-band family. Prior fine-subset tests "
                       "also found false acceptances. No automatic production switch, accuracy claim on real "
                       "images, or pass of the original G3 qualification follows from these data."), "",
                  "![Coarse and fine classifications](coarse_vs_fine.png)", "",
                  ("Raw per-window parameters and source indices are in NPZ files, counts are in `summary.json`, "
                   "and the false-acceptance details are in `aggregated.json`. The protocol was recorded in "
                   "`PROTOCOL.md` before this run. Unwarmed solver timings are diagnostic only."), "",
                  "Reproduce in a new directory:", "", "```bash",
                  f"base={args.base}",
                  f"out={args.directory}_new",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/diagnose_shear_fine_transfer.py \\",
                  ("  --base \"$base\" --live-dic-source /home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src "
                   f"--output-dir \"$out\" --seed-offset {args.seed_offset} --gradient {args.gradient}"
                   + (f" --coarse-directory {args.coarse_directory}" if args.coarse_directory else "")),
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_shear_fine_transfer.py \\",
                  f"  --base \"$base\" --directory \"$out\" --seed-offset {args.seed_offset} --gradient {args.gradient}"
                  + (" --independent" if args.independent else "")
                  + (f" --coarse-directory {args.coarse_directory}" if args.coarse_directory else ""), "```", ""])
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
