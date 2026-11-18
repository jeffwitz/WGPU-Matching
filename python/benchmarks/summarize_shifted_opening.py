"""Report fine-transfer accuracy and post-hoc physical support near a shifted crack."""

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
    parser.add_argument("--boundary", type=float, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    args = parser.parse_args()
    rows = json.loads((args.directory / "summary.json").read_text())["cases"]
    if len(rows) != 5 or any(row["family"] != "discontinuous" for row in rows):
        raise ValueError("expected five discontinuous-opening pairs")
    for row in rows:
        path = args.archive / "affine_ncc" / f"windows_seed{args.seed_offset}" / f"{row['pair']}.npz"
        with np.load(path) as data:
            row["zero_64"] = int(data["zero_icgn_correct_lt2px"].sum())
            row["one_pass_64"] = int(data["icgn_correct_lt2px"].sum())
            row["oracle_64"] = int(data["oracle_icgn_correct_lt2px"].sum())
        control = json.loads((args.coarse / f"{row['pair']}.json").read_text())["control_correct"]
        if control is None:
            raise ValueError("missing same-budget control")
        row["same_budget_control_64"] = int(control)
    keys = ("visible", "coarse_correct", "coarse_false", "coarse_accepted_visible", "fine_correct", "fine_false",
            "fine_accepted_visible", "available_visible", "accepted_invisible", "coarse_good_lost",
            "coarse_bad_recovered", "new_correct", "fine_accepted_crossing", "visible_opposite_source", "visible_source_crossing",
            "zero_64", "one_pass_64", "oracle_64", "same_budget_control_64")
    totals = {key: sum(row[key] for row in rows) for key in keys}
    incidents, support_incidents = [], []
    accepted_with_crossing_source = 0
    correct_with_crossing_source = 0
    paired_coarse_gradient, paired_fine_gradient = [], []
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.1 * len(rows)), squeeze=False)
    cmap = ListedColormap(["#d3d3d3", "#f5cf72", "#3c9d78", "#b72d34", "#8552a1"])
    for row_id, row in enumerate(rows):
        with (np.load(args.directory / f"{row['pair']}.npz") as data,
              np.load(args.coarse / f"{row['pair']}.npz") as coarse):
            xy, visible = data["centres"], data["visible"].astype(bool)
            accepted = data["accepted"].astype(bool)
            bad = accepted & visible & (data["error_px"] >= 2)
            common = data["correct"].astype(bool) & data["prior_correct"].astype(bool)
            columns = [1, 2, 4, 5]
            paired_coarse_gradient.extend(np.linalg.norm(
                coarse["parameters"][common][:, columns] - data["truth"][common][:, columns], axis=1).tolist())
            paired_fine_gradient.extend(np.linalg.norm(
                data["parameters"][common][:, columns] - data["truth"][common][:, columns], axis=1).tolist())
            crossing = accepted & data["support_crossing"].astype(bool)
            crossing_source = visible & data["available"].astype(bool) & data["source_crossing"].astype(bool)
            accepted_with_crossing_source += int((crossing_source & accepted).sum())
            correct_with_crossing_source += int((crossing_source & data["correct"].astype(bool)).sum())
            for index in np.flatnonzero(bad):
                source = int(data["source"][index])
                incidents.append({"pair": row["pair"], "center_xy": xy[index].tolist(), "error_px": float(data["error_px"][index]),
                                  "source_center_xy": xy[source].tolist(), "source_error_px": float(data["prior_error_px"][source]),
                                  "support_crossing": bool(data["support_crossing"][index]),
                                  "opposite_source": bool(data["opposite_source"][index])})
            for index in np.flatnonzero(crossing):
                support_incidents.append({"pair": row["pair"], "center_xy": xy[index].tolist(),
                                          "center_visible": bool(visible[index]),
                                          "error_px": float(data["error_px"][index])})
            shape = (len(np.unique(xy[:, 1])), len(np.unique(xy[:, 0])))
            for col, (accepted_i, good_i, error_i, title) in enumerate((
                (data["prior_accepted"].astype(bool), data["prior_correct"].astype(bool), data["prior_error_px"], "64 px"),
                (accepted, data["correct"].astype(bool), data["error_px"], "32 px"),
            )):
                category = np.zeros(len(xy), np.uint8)
                category[visible] = 1
                category[good_i] = 2
                category[visible & accepted_i & (error_i >= 2)] = 3
                if col == 1:
                    category[crossing] = 4
                ax = axes[row_id, col]
                ax.imshow(category.reshape(shape), cmap=cmap, vmin=0, vmax=4, interpolation="nearest",
                          extent=(0, 3840, 2160, 0))
                ax.axvline(args.boundary, color="#222", linestyle="--", linewidth=.6)
                ax.set_title(f"{row['pair'].split('_texture')[1]}: {title}")
                ax.set_xlabel("x (px)")
                ax.set_ylabel("y (px)")
    fig.suptitle("Gray invisible; yellow unresolved; green correct; red wrong; purple accepted crossing support")
    fig.tight_layout()
    fig.savefig(args.directory / "shifted_support_maps.png", dpi=140)
    plt.close(fig)
    totals["accepted_with_crossing_source_visible"] = accepted_with_crossing_source
    totals["correct_with_crossing_source_visible"] = correct_with_crossing_source
    totals["gradient_frobenius_rmse_coarse_common"] = (float(np.sqrt(np.mean(np.square(paired_coarse_gradient))))
                                                        if paired_coarse_gradient else None)
    totals["gradient_frobenius_rmse_fine_common"] = (float(np.sqrt(np.mean(np.square(paired_fine_gradient))))
                                                      if paired_fine_gradient else None)
    coarse_gradient_text = (f"{totals['gradient_frobenius_rmse_coarse_common']:.5f}"
                            if paired_coarse_gradient else "undefined")
    fine_gradient_text = (f"{totals['gradient_frobenius_rmse_fine_common']:.5f}"
                          if paired_fine_gradient else "undefined")
    rng = np.random.default_rng(63001111)
    draws = rng.integers(0, len(rows), size=(10000, len(rows)))
    gains = np.array([row["fine_correct"] - row["coarse_correct"] for row in rows], dtype=float)
    visible_count = np.array([row["visible"] for row in rows], dtype=float)
    totals["paired_gain_bootstrap_95pct"] = np.quantile(gains[draws].sum(axis=1) / visible_count[draws].sum(axis=1),
                                                        [.025, .975]).tolist()
    (args.directory / "analysis.json").write_text(json.dumps({"totals": totals, "false_acceptances": incidents,
                                                               "accepted_crossing_supports": support_incidents}, indent=2) + "\n")
    lines = ["# Independent shifted-opening transfer", "",
             (f"Five new 3840 × 2160 synthetic pairs, with the analytical opening boundary at x = {args.boundary:g} px. "
              "The boundary, exact truth and visibility are used only after inference. The 32-pixel subset is "
              "re-solved at each available original 64-grid center from the nearest accepted image-derived "
              "64-pixel state. This is not a dense 32/16-grid evaluation."), "",
             "| Metric | Original 64 px | Re-solved 32 px |", "| --- | ---: | ---: |",
             f"| Correct visible | {totals['coarse_correct']}/{totals['visible']} | {totals['fine_correct']}/{totals['visible']} |",
             (f"| False accepted visible | {totals['coarse_false']}/{totals['coarse_accepted_visible']} | "
              f"{totals['fine_false']}/{totals['fine_accepted_visible']} |"),
             f"| Original correct lost | — | {totals['coarse_good_lost']} |",
             f"| Originally wrong recovered | — | {totals['coarse_bad_recovered']} |", "",
             (f"On these pairs, the zero-initialization 64-pixel control yields {totals['zero_64']} correct, "
              f"the original one-pass prefit {totals['one_pass_64']}, the equal-work 64-pixel control "
              f"{totals['same_budget_control_64']}, and the analytical 64-pixel oracle {totals['oracle_64']}. "
              "The oracle is not a strict upper bound."), "",
             (f"On windows correct with both methods, displacement-gradient Frobenius RMSE is "
              f"{coarse_gradient_text} for 64 pixels and {fine_gradient_text} for 32 pixels; this dimensionless "
              "statistic combines all four gradient components."), "",
             (f"The fine solver accepts {totals['fine_accepted_crossing']} supports that cross the true opening "
              f"boundary; {totals['visible_opposite_source']} visible initializations use an accepted source "
              f"from the opposite side, and {totals['visible_source_crossing']} use a 64-pixel source support "
              f"that crosses the boundary. Of those, {accepted_with_crossing_source} are accepted after the fine "
              f"solve, including {correct_with_crossing_source} with a correct center displacement. "
              "Crossing-support counts include accepted centers whether visible "
              "or not; a correct center displacement does not validate a crossing support."), "",
             ("`analysis.json` enumerates every visible false acceptance and every accepted fine support crossing "
              "the boundary, including source provenance. The small synthetic test does not establish "
              "robustness near arbitrary cracks or accuracy on real images; the original G3 failure remains."), "",
             "![Shifted opening maps](shifted_support_maps.png)", ""]
    lines.extend(["| Pair | Coarse correct | Fine correct | Visible | Fine false | Accepted crossing fine support |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(f"| {row['pair']} | {row['coarse_correct']} | {row['fine_correct']} | {row['visible']} | "
                     f"{row['fine_false']} | {row['fine_accepted_crossing']} |")
    lo, hi = totals["paired_gain_bootstrap_95pct"]
    lines.extend(["", (f"The paired coverage-gain bootstrap interval is [{100 * lo:.2f}; {100 * hi:.2f}] "
                       "percentage points, using 10,000 pair resamples and seed 63001111. With only five pairs "
                       "this does not establish broad statistical generalization. Reproduction commands are "
                       "in `../PROTOCOL.md`."), ""])
    (args.directory / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
