"""Score exploratory unresolved-region reseeding against the frozen image-only run."""

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
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.base / "g2.0/evaluation"
    candidate = args.experiment / "evaluation"
    metadata = {r["pair"]: r for r in json.loads((candidate / "summary.json").read_text())["rows"]}
    paths = sorted(candidate.glob("*.npz"))
    fig, axes = plt.subplots(len(paths), 2, figsize=(10, 3.3 * len(paths)), squeeze=False)
    cmap = ListedColormap(["#d3d3d3", "#f5cf72", "#3c9d78", "#b72d34"])
    rows, incidents = [], []
    for n, path in enumerate(paths):
        with (np.load(baseline / path.name) as a, np.load(path) as b,
              np.load(args.base / "exact_center_analysis/g2.0" / path.name) as exact):
            xy, truth, visible = a["centres"], exact["truth_parameters"], exact["visible"].astype(bool)
            scores = []
            for result in (a, b):
                accepted = result["accepted"].astype(bool)
                error = np.linalg.norm(result["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                scores.append((accepted, error, accepted & visible & (error < 2), accepted & visible & (error >= 2)))
            old, new = scores
            kept = bool(np.all(new[0][old[0]]) and np.array_equal(a["parameters"][old[0]], b["parameters"][old[0]]))
            row = {"pair": path.stem, "visible": int(visible.sum()), "baseline_correct": int(old[2].sum()),
                   "baseline_false": int(old[3].sum()), "candidate_correct": int(new[2].sum()),
                   "candidate_false": int(new[3].sum()), "candidate_accepted_visible": int((new[0] & visible).sum()),
                   "gained_correct": int((new[2] & ~old[2]).sum()), "lost_correct": int((old[2] & ~new[2]).sum()),
                   "baseline_accepted_preserved": kept, "escalation": metadata[path.stem]["escalation_history"],
                   "evaluation_wall_s_excluding_generation": metadata[path.stem]["wall_s_including_lazy_rotation"]}
            rows.append(row)
            for i in np.flatnonzero(new[3]):
                incidents.append({"pair": path.stem, "index": int(i), "centre_xy": xy[i].tolist(),
                                  "error_px": float(new[1][i]), "source_stage": int(b["source_stage"][i]),
                                  "oracle_good": bool(exact["oracle_status"][i] and exact["oracle_error_px"][i] < 2)})
            shape = (len(np.unique(xy[:, 1])), len(np.unique(xy[:, 0])))
            for col, (score, title) in enumerate(((old, "frozen image-only"), (new, "exploratory reseeding"))):
                category = np.zeros(len(xy), np.uint8)
                category[visible] = 1
                category[score[2]] = 2
                category[score[3]] = 3
                ax = axes[n, col]
                ax.imshow(category.reshape(shape), cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                          extent=(0, 3840, 2160, 0))
                ax.axhline(1080, color="#222", linewidth=.5, linestyle="--")
                ax.set_title(f"{path.stem.split('_texture')[1]}: {title}")
                ax.set_xlabel("x (px)")
                ax.set_ylabel("y (px)")
    fig.suptitle("Exact-center scoring: gray invisible, yellow unresolved, green correct, red false")
    fig.tight_layout()
    fig.savefig(args.experiment / "reseed_comparison.png", dpi=150)
    plt.close(fig)
    (args.experiment / "comparison.json").write_text(json.dumps({"cases": rows, "false_acceptances": incidents}, indent=2) + "\n")
    lines = ["# Exploratory reseeding of unresolved shear-band regions", "",
             ("These two pairs were selected after inspecting their poor coverage. The run eagerly provides "
              "the three leading image-ranked rotations and enables local-affine unresolved-region escalation. "
              "Truth is used only to score the outputs. The comparison is exploratory and does not isolate "
              "the effects of the larger hypothesis set and of escalation."), "",
             "| Pair | Correct baseline → exploratory | False baseline → exploratory | Accepted visible | Baseline accepted preserved |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['pair']} | {row['baseline_correct']} → {row['candidate_correct']} | "
                     f"{row['baseline_false']} → {row['candidate_false']} | {row['candidate_accepted_visible']} | "
                     f"{row['baseline_accepted_preserved']} |")
    oracle_correct_false = sum(item["oracle_good"] for item in incidents)
    manifest = json.loads((args.experiment / "rotations/manifest.json").read_text())
    rotation_time = sum(item["subprocess_wall_s"] or 0.0 for item in manifest)
    lines.extend(["", (f"Among {len(incidents)} visible false acceptances, {oracle_correct_false} are correct "
                       "after exact analytical affine oracle initialization. This does not prove the cause. "
                       "`comparison.json` retains every false window and the escalation stages. Timings "
                       "exclude earlier rotation-hypothesis generation and are not full-pipeline benchmarks."), "",
                  (f"The six rotation-generation subprocesses took {rotation_time:.2f} s in total, including "
                   "startup, synthetic image generation and output. This is a single unwarmed scripted "
                   "measurement, not GPU kernel time or production latency."), "",
                  ("A subsequent, also post-hoc 32-pixel re-solve is reported in `fine_transfer/README.md`. "
                   "Neither exploratory result changes the frozen G3 failure."), "",
                  "![Reseeding comparison](reseed_comparison.png)", "",
                  "Reproduce using the existing five-pair input archive and a new output directory:", "", "```bash",
                  f"base={args.base}",
                  f"out={args.experiment}_new",
                  "weights=results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt",
                  "solver=/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src",
                  "frozen=results/seed_propagation_20261111/p2_calibration/frozen_parameters.json",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/generate_rotation_seed_hypotheses.py \\",
                  "  --ranking-reports \"$base/g2.0/inputs/affine_ncc/combined_seed61000000.json\" --family shear_band \\",
                  "  --textures 61051005 61051007 --weights \"$weights\" --live-dic-source \"$solver\" --output-dir \"$out/rotations\" \\",
                  "  --model-width 192 --model-height 108 --shear-band-peak-gradient 2.0",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/qualify_image_seed_pilot.py \\",
                  "  --archive \"$base/g2.0/inputs\" --live-dic-source \"$solver\" --output-dir \"$out/evaluation\" \\",
                  "  --rotation-manifest \"$out/rotations/manifest.json\" --frozen-parameters \"$frozen\" \\",
                  "  --textures 61051005 61051007 --sparse-stride 2 --tau-u 8 --tau-g .25 --minimum-anchors 3 \\",
                  "  --shear-band-peak-gradient 2.0 --escalation --local-affine-escalation --escalate-unresolved \\",
                  "  --batch-anchor-confirmation --same-budget-control --use-cached-solver-images",
                  "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/summarize_shear_reseeding.py \\",
                  "  --base \"$base\" --experiment \"$out\"", "```", ""])
    (args.experiment / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
