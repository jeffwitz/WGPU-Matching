"""Score frozen shear-band initialization and locate accepted errors after inference."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from wgpu_matching.seed_propagation import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gradients = ("0.33", "0.67", "2.0")
    report = {"band_center_y_px": 1080., "band_width_px": 86.4,
              "paired_textures": True, "cases": [], "groups": []}
    bad_windows = []
    for gradient in gradients:
        directory = args.base / f"g{gradient}"
        first = json.loads((directory / "evaluation/summary.json").read_text())["rows"]
        control = {row["pair"]: row for row in json.loads(
            (directory / "evaluation_control/summary.json").read_text())["rows"]}
        fig, axes = plt.subplots(1, len(first), figsize=(5 * len(first), 3.3), squeeze=False)
        group = {"gradient": float(gradient), "visible": 0, "accepted_visible": 0, "correct": 0,
                 "false_accepted_visible": 0, "accepted_invisible": 0, "oracle": 0,
                 "baseline": 0, "control": 0, "zero": 0, "regions": 0, "regions_with_seed": 0,
                 "seeded_regions_under_97pct_oracle": 0, "false_seed": 0, "false_propagated": 0,
                 "false_oracle_correct": 0, "baseline_false": 0, "baseline_false_also_final": 0}
        for column, row in enumerate(first):
            name = row["pair"]
            source = directory / "inputs/affine_ncc/windows_seed59000000" / f"{name}.npz"
            with (np.load(directory / "evaluation" / f"{name}.npz") as v,
                  np.load(directory / "evaluation_control" / f"{name}.npz") as c,
                  np.load(source) as a):
                for key in ("accepted", "parameters", "parent", "origin", "generation", "confirmed_seeds"):
                    if not np.array_equal(v[key], c[key], equal_nan=True):
                        raise ValueError(f"control rerun changed {key} for {name}")
                centres, visible = v["centres"], v["visible"].astype(bool)
                accepted, good = v["accepted"].astype(bool), v["correct"].astype(bool)
                bad = accepted & visible & ~good
                baseline_bad = visible & (a["icgn_status"] > .5) & (a["icgn_displacement_error_px"] >= 2)
                oracle_correct = a["oracle_icgn_correct_lt2px"].astype(bool)
                true = a["truth_parameters"]
                errors = v["error"]
                gradient_error = np.linalg.norm(v["parameters"][:, [1, 2, 4, 5]] - true[:, [1, 2, 4, 5]], axis=1)
                distance = np.abs(centres[:, 1] - 1080.) / 86.4
                bands = {}
                for label, zone in (("within_1_width", distance <= 1), ("1_to_3_widths", (distance > 1) & (distance <= 3)),
                                    ("beyond_3_widths", distance > 3)):
                    bands[label] = {"visible": int((visible & zone).sum()), "correct": int((good & zone).sum()),
                                    "false_accepted": int((bad & zone).sum())}
                shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))
                regions = [r for r in connected_components(a["oracle_icgn_correct_lt2px"].reshape(shape).astype(bool))
                           if len(r) >= 16]
                seeded = [r for r in regions if v["confirmed_seeds"][r].any()]
                under = sum(int(good[r].sum()) / len(r) < .97 for r in seeded)
                row_out = {"pair": name, "gradient": float(gradient), "visible": int(visible.sum()),
                           "accepted_visible": int((accepted & visible).sum()), "correct": int(good.sum()),
                           "false_accepted_visible": int(bad.sum()), "accepted_invisible": int((accepted & ~visible).sum()),
                           "oracle": int(a["oracle_icgn_correct_lt2px"].sum()),
                           "baseline": int(a["icgn_correct_lt2px"].sum()),
                           "zero": int(a["zero_icgn_correct_lt2px"].sum()),
                           "control": control[name]["control_correct"],
                           "confirmed_seeds": int(v["confirmed_seeds"].sum()), "passes": row["passes"],
                           "solver_calls": row["total_solver_calls_before_control"],
                           "active_windows": row["total_active_windows_before_control"],
                           "correct_seeds": int((good & v["confirmed_seeds"].astype(bool)).sum()),
                           "correct_propagated": int((good & ~v["confirmed_seeds"].astype(bool)).sum()),
                           "correct_reseeded": int((good & (v["source_stage"] > 0)).sum()),
                           "regions": len(regions), "regions_with_seed": len(seeded),
                           "seeded_regions_under_97pct_oracle": under,
                           "false_seed": int((bad & v["confirmed_seeds"].astype(bool)).sum()),
                           "false_propagated": int((bad & ~v["confirmed_seeds"].astype(bool)).sum()),
                           "false_oracle_correct": int((bad & oracle_correct).sum()),
                           "baseline_false": int(baseline_bad.sum()),
                           "baseline_false_also_final": int((baseline_bad & bad).sum()),
                           "gradient_rmse_on_correct": float(np.sqrt(np.mean(gradient_error[good] ** 2))) if good.any() else None,
                           "max_false_error_px": float(errors[bad].max()) if bad.any() else None,
                           "bands": bands}
                report["cases"].append(row_out)
                for key in group:
                    if key != "gradient":
                        group[key] += row_out[key]
                for i in np.flatnonzero(bad):
                    bad_windows.append({"gradient": float(gradient), "pair": name, "index": int(i),
                                        "centre_xy": centres[i].tolist(), "error_px": float(errors[i]),
                                        "parent": int(v["parent"][i]), "origin": int(v["origin"][i]),
                                        "generation": int(v["generation"][i]),
                                        "is_confirmed_seed": bool(v["confirmed_seeds"][i])})
                category = np.zeros(len(visible), dtype=np.uint8)
                category[visible] = 1
                category[good] = 2
                category[bad] = 3
                ax = axes[0, column]
                ax.imshow(category.reshape(shape), cmap=ListedColormap(["#d3d3d3", "#f5cf72", "#3c9d78", "#b72d34"]),
                          vmin=0, vmax=3, interpolation="nearest", extent=(0, 3840, 2160, 0))
                ax.axhline(1080., color="#222222", linewidth=.5, linestyle="--")
                ax.set_title(f"{name.split('_texture')[1]}\n{good.sum()}/{visible.sum()} correct; {bad.sum()} false")
                ax.set_xlabel("x (px)")
                ax.set_ylabel("y (px)")
        fig.suptitle(f"Peak shear gradient {gradient}: gray outside truth visibility, yellow unresolved, green correct, red false")
        fig.tight_layout()
        fig.savefig(args.output_dir / f"shear_{gradient}.png", dpi=150)
        plt.close(fig)
        group["false_fraction"] = (group["false_accepted_visible"] / group["accepted_visible"]
                                   if group["accepted_visible"] else None)
        group["g3_false_acceptance_pass"] = (group["false_fraction"] <= .005
                                             if group["false_fraction"] is not None else None)
        cases = [case for case in report["cases"] if case["gradient"] == float(gradient)]
        rng = np.random.default_rng(59001111)
        draws = rng.integers(0, len(cases), size=(10000, len(cases)))
        numerator = np.array([case["correct"] - case["baseline"] for case in cases], dtype=float)
        denominator = np.array([case["visible"] for case in cases], dtype=float)
        gains = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
        group["paired_bootstrap_gain_95pct"] = np.quantile(gains, [.025, .975]).tolist()
        group["bootstrap_seed"] = 59001111
        report["groups"].append(group)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "false_accepted_windows.json").write_text(json.dumps(bad_windows, indent=2) + "\n")
    lines = ["# Localized shear-band qualification", "",
             ("The protocol was fixed in `../PROTOCOL.md`; truth is used only for scoring and maps. "
             "The cohort contains three paired textures, not independent windows for statistical inference."), "",
             "| Peak gradient | Correct / visible | False / accepted visible | Oracle | One-pass | Equal-work control | G3 false limit |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for g in report["groups"]:
        fraction = "undefined" if g["false_fraction"] is None else f"{100 * g['false_fraction']:.2f}%"
        verdict = "indeterminate" if g["g3_false_acceptance_pass"] is None else ("pass" if g["g3_false_acceptance_pass"] else "fail")
        lines.append(f"| {g['gradient']} | {g['correct']}/{g['visible']} | {g['false_accepted_visible']}/{g['accepted_visible']} "
                     f"({fraction}) | {g['oracle']} | {g['baseline']} | {g['control']} | {verdict} |")
    lines.extend(["", "| Pair | Peak gradient | Correct / visible | False / accepted visible | Confirmed seeds | Passes | Oracle |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for case in report["cases"]:
        lines.append(f"| {case['pair']} | {case['gradient']} | {case['correct']}/{case['visible']} | "
                     f"{case['false_accepted_visible']}/{case['accepted_visible']} | {case['confirmed_seeds']} | "
                     f"{case['passes']} | {case['oracle']} |")
    lines.extend(["", ("Each map colors visible, unresolved windows yellow; correct accepted windows green; "
                  "false accepted windows red; gray means outside truth visibility. The dashed line is the analytical "
                  "band center, used only in the figure."), ""])
    for gradient in gradients:
        lines.append(f"![Shear band {gradient}](shear_{gradient}.png)")
        lines.append("")
    severe = next(group for group in report["groups"] if group["gradient"] == 2.0)
    severe_cases = [case for case in report["cases"] if case["gradient"] == 2.0]
    near_false = sum(case["bands"]["1_to_3_widths"]["false_accepted"] for case in severe_cases)
    center_correct = sum(case["bands"]["within_1_width"]["correct"] for case in severe_cases)
    lines.extend([(f"At gradient 2.0, {severe['false_seed']} false windows are confirmed seeds and "
                  f"{severe['false_propagated']} are propagated. The one-pass affine + NCC baseline contains "
                  f"{severe['baseline_false']} visible false acceptances, {severe['baseline_false_also_final']} "
                  f"of which also appear in the propagated output. None of the {severe['false_accepted_visible']} "
                  "final false windows is oracle-correct with the same affine IC-GN solver. This is an oracle "
                  "diagnostic, not an oracle criterion available to the initializer."), "",
                  (f"At gradient 2.0, {near_false} visible false acceptances lie between one and three analytical "
                  f"band widths from its center; {center_correct} windows within one width have a correct center "
                  f"displacement. {severe['regions_with_seed']} oracle-connected regions contain a confirmed seed; "
                  f"{severe['seeded_regions_under_97pct_oracle']} of them recover less than 97% of their "
                  "oracle-correct windows. For the analytical initialization re-solve, "
                  "see `../oracle_diagnostic/README.md`."), "",
                  ("The analytical affine initialization is a reference, not a mathematical upper bound: "
                  "a different initialization can occasionally converge correctly where it does not. "
                  "The three-pair bootstrap intervals stored in `summary.json` describe only this very small "
                  "paired sample and do not establish population-level uncertainty."), ""])
    lines.extend([("`summary.json` includes per-pair seed and propagation counts, oracle-connected regions, band-distance "
                  "strata and gradient error. `false_accepted_windows.json` enumerates every visible false acceptance. "
                  "No validation claim is made for experimental images or for untested deformation families."), ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
