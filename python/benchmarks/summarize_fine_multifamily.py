"""Paired family-level scoring for image-initialized fine subsets at coarse centers."""

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
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--seed", type=int, default=62001111)
    args = parser.parse_args()
    rows = json.loads((args.directory / "summary.json").read_text())["cases"]
    for row in rows:
        path = args.archive / "affine_ncc" / f"windows_seed{args.seed_offset}" / f"{row['pair']}.npz"
        with np.load(path) as data:
            row["zero_64"] = int(data["zero_icgn_correct_lt2px"].sum())
            row["oracle_64"] = int(data["oracle_icgn_correct_lt2px"].sum())
            row["one_pass_64"] = int(data["icgn_correct_lt2px"].sum())
        prior = json.loads((args.coarse / f"{row['pair']}.json").read_text())
        if prior["control_correct"] is None:
            raise ValueError("same-budget control is missing")
        row["same_budget_control_64"] = int(prior["control_correct"])
    families = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")
    keys = ("visible", "coarse_correct", "coarse_false", "coarse_accepted_visible", "available_visible",
            "fine_correct", "fine_false", "fine_accepted_visible", "accepted_invisible", "coarse_bad_recovered",
            "coarse_good_lost", "new_correct", "fine_accepted_crossing", "visible_opposite_source", "visible_source_crossing",
            "zero_64", "oracle_64", "one_pass_64", "same_budget_control_64")
    groups, incidents = [], []
    fig, axes = plt.subplots(len(families), 2, figsize=(10, 3.2 * len(families)), squeeze=False)
    cmap = ListedColormap(["#d3d3d3", "#f5cf72", "#3c9d78", "#b72d34"])
    for row_id, family in enumerate(families):
        cases = [row for row in rows if row["family"] == family]
        if len(cases) != 5:
            raise ValueError(f"expected five cases for {family}, got {len(cases)}")
        totals = {key: sum(row[key] for row in cases) for key in keys}
        random = np.random.default_rng(args.seed)
        draws = random.integers(0, len(cases), size=(10000, len(cases)))
        gains = np.array([row["fine_correct"] - row["coarse_correct"] for row in cases], float)
        visible = np.array([row["visible"] for row in cases], float)
        paired = gains[draws].sum(axis=1) / visible[draws].sum(axis=1)
        group = {"family": family, **totals, "pairs": len(cases),
                 "paired_gain_bootstrap_95pct": np.quantile(paired, [.025, .975]).tolist(),
                 "fine_false_fraction": totals["fine_false"] / totals["fine_accepted_visible"]
                 if totals["fine_accepted_visible"] else None}
        fine_translation_errors, fine_gradient_errors = [], []
        common_coarse_translation, common_fine_translation = [], []
        common_coarse_grad, common_fine_grad = [], []
        # Choosing a worst case for a map is diagnostic, not model selection.
        shown = max(cases, key=lambda row: (row["fine_false"], row["coarse_good_lost"], -row["fine_correct"]))
        for row in cases:
            with (np.load(args.directory / f"{row['pair']}.npz") as data,
                  np.load(args.coarse / f"{row['pair']}.npz") as coarse):
                valid = data["visible"].astype(bool)
                accepted = data["accepted"].astype(bool)
                bad = valid & accepted & (data["error_px"] >= 2)
                good = data["correct"].astype(bool)
                common = good & data["prior_correct"].astype(bool)
                gradient_columns = [1, 2, 4, 5]
                fine_translation_errors.extend(data["error_px"][good].tolist())
                common_coarse_translation.extend(data["prior_error_px"][common].tolist())
                common_fine_translation.extend(data["error_px"][common].tolist())
                fine_gradient_errors.extend(np.linalg.norm(
                    data["parameters"][good][:, gradient_columns] - data["truth"][good][:, gradient_columns], axis=1).tolist())
                common_coarse_grad.extend(np.linalg.norm(
                    coarse["parameters"][common][:, gradient_columns] - data["truth"][common][:, gradient_columns], axis=1).tolist())
                common_fine_grad.extend(np.linalg.norm(
                    data["parameters"][common][:, gradient_columns] - data["truth"][common][:, gradient_columns], axis=1).tolist())
                for index in np.flatnonzero(bad):
                    source = int(data["source"][index])
                    incidents.append({"family": family, "pair": row["pair"], "index": int(index),
                                      "center_xy": data["centres"][index].tolist(),
                                      "error_px": float(data["error_px"][index]), "source": source,
                                      "source_center_xy": data["centres"][source].tolist(),
                                      "source_error_px": float(data["prior_error_px"][source]),
                                      "support_crossing": bool(data["support_crossing"][index]),
                                      "source_crossing": bool(data["source_crossing"][index]),
                                      "opposite_source": bool(data["opposite_source"][index])})
                if row["pair"] != shown["pair"]:
                    continue
                xy = data["centres"]
                shape = (len(np.unique(xy[:, 1])), len(np.unique(xy[:, 0])))
                for col, (status, good, error, label) in enumerate((
                    (data["prior_accepted"].astype(bool), data["prior_correct"].astype(bool), data["prior_error_px"], "64 px"),
                    (accepted, data["correct"].astype(bool), data["error_px"], "32 px"),
                )):
                    category = np.zeros(len(xy), np.uint8)
                    category[valid] = 1
                    category[good] = 2
                    category[valid & status & (error >= 2)] = 3
                    ax = axes[row_id, col]
                    ax.imshow(category.reshape(shape), cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                              extent=(0, 3840, 2160, 0))
                    if family == "discontinuous":
                        ax.axvline(1920, color="#222", linewidth=.5, linestyle="--")
                    ax.set_title(f"{family} {row['pair'].split('_texture')[1]}: {label}")
                    ax.set_xlabel("x (px)")
                    ax.set_ylabel("y (px)")
        def stats(values):
            a = np.asarray(values, dtype=float)
            return {"n": len(a), "median": float(np.median(a)) if len(a) else None,
                    "rmse": float(np.sqrt(np.mean(a * a))) if len(a) else None}

        group["fine_translation_error_px_on_correct"] = stats(fine_translation_errors)
        group["fine_gradient_frobenius_error_on_correct"] = stats(fine_gradient_errors)
        group["coarse_displacement_error_px_on_both_correct"] = stats(common_coarse_translation)
        group["fine_displacement_error_px_on_both_correct"] = stats(common_fine_translation)
        group["coarse_gradient_error_on_both_correct"] = stats(common_coarse_grad)
        group["fine_gradient_error_on_both_correct"] = stats(common_fine_grad)
        groups.append(group)
    fig.suptitle("Diagnostic worst-case per family: gray invisible, yellow unresolved, green correct, red false")
    fig.tight_layout()
    fig.savefig(args.directory / "representative_maps.png", dpi=140)
    plt.close(fig)
    (args.directory / "analysis.json").write_text(json.dumps({"groups": groups, "bootstrap_seed": args.seed,
                                                              "false_accepted_windows": incidents}, indent=2) + "\n")
    lines = ["# Independent multifamily transfer to 32-pixel subsets", "",
             ("Five predeclared new pairs per family, native 3840 × 2160 pixels. Each fine solve uses a "
              "nearest accepted 64-pixel image-derived state at the same requested center; truth is used "
              "only for scoring. Window positions and thresholds are frozen. This evaluates the transfer "
              "on the original 64-grid centers, not all centers of a dense 32/16 field."), "",
             "| Family | 64 px correct / visible | 32 px correct / visible | Fine false / accepted visible | Lost 64 correct | Paired gain CI |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        lo, hi = g["paired_gain_bootstrap_95pct"]
        fraction = "undefined" if g["fine_false_fraction"] is None else f"{100 * g['fine_false_fraction']:.2f}%"
        lines.append(f"| {g['family']} | {g['coarse_correct']}/{g['visible']} | {g['fine_correct']}/{g['visible']} | "
                     f"{g['fine_false']}/{g['fine_accepted_visible']} ({fraction}) | {g['coarse_good_lost']} | "
                     f"[{100 * lo:.2f}; {100 * hi:.2f}] points |")
    lines.extend(["", "| Family | Zero 64 | One pass 64 | Same-budget control 64 | Analytical oracle 64 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for g in groups:
        lines.append(f"| {g['family']} | {g['zero_64']} | {g['one_pass_64']} | "
                     f"{g['same_budget_control_64']} | {g['oracle_64']} |")
    lines.extend(["", "| Family | Fine displacement median / RMSE (px) | Displacement RMSE, coarse → fine on common correct | Fine gradient median / RMSE | Gradient RMSE, coarse → fine on common correct |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for g in groups:
        displacement = g["fine_translation_error_px_on_correct"]
        gradient = g["fine_gradient_frobenius_error_on_correct"]
        coarse_gradient = g["coarse_gradient_error_on_both_correct"]
        fine_gradient = g["fine_gradient_error_on_both_correct"]
        coarse_displacement = g["coarse_displacement_error_px_on_both_correct"]
        fine_displacement = g["fine_displacement_error_px_on_both_correct"]
        def fmt(value):
            return "—" if value is None else f"{value:.4f}"
        lines.append(f"| {g['family']} | {fmt(displacement['median'])} / {fmt(displacement['rmse'])} | "
                     f"{fmt(coarse_displacement['rmse'])} → {fmt(fine_displacement['rmse'])} | "
                     f"{fmt(gradient['median'])} / {fmt(gradient['rmse'])} | "
                     f"{fmt(coarse_gradient['rmse'])} → {fmt(fine_gradient['rmse'])} |")
    lines.extend(["", ("Gradient error is the dimensionless Frobenius norm of the difference between the "
                       "four displacement-gradient components. The last column uses only windows correct "
                       "in both methods; it does not establish a production-quality threshold. The smaller "
                       "support increases the gradient error on this cohort despite recovering additional centers."), ""])
    lines.extend(["", "| Pair | 64 correct | 32 correct | Visible | 32 false | Lost 64 correct |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(f"| {row['pair']} | {row['coarse_correct']} | {row['fine_correct']} | {row['visible']} | "
                     f"{row['fine_false']} | {row['coarse_good_lost']} |")
    lines.extend(["", (f"Intervals resample whole image pairs 10,000 times with seed {args.seed}; only five "
                       "pairs per family are available, so the intervals do not establish broad generalization. "
                       "Different families can reuse the same texture seeds; they are not independent across families. "
                       "Invisible accepted windows are counted separately in `analysis.json`. G3 from the "
                       "original 64-pixel protocol remains failed. This qualification does not measure "
                       "complete-pipeline execution time."), ""])
    opening = next(g for g in groups if g["family"] == "discontinuous")
    lines.extend([(f"For discontinuous opening, {opening['fine_accepted_crossing']} fine accepted supports cross "
                  f"the analytical boundary; {opening['visible_opposite_source']} visible initializations use "
                  f"a source on the opposite side, and {opening['visible_source_crossing']} use a source whose "
                  "64-pixel support crosses it. These labels are evaluated after inference, never used for selection. "
                  "The boundary is aligned with this grid; shifted cracks remain unqualified."), "",
                  ("`summary.json` gives each pair's counts and `analysis.json` lists every visible false "
                   "acceptance and its source. The figure selects the worst case per family for diagnosis "
                   "after scoring; it is not used to select the algorithm. Reproduction commands are in "
                   "`../PROTOCOL.md`."), "",
                  "![Representative diagnostic maps](representative_maps.png)", ""])
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
