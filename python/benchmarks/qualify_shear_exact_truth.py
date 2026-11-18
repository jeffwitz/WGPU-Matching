"""Rescore frozen shear-band outputs and oracle solves at exact pixel centers."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images
from wgpu_matching.dic_synthetic import _motion_parameters
from wgpu_matching.seed_propagation import connected_components


def exact_shear_truth(centres, warp_seed, gradient, width=3840, height=2160):
    """Evaluate the generator's known shear map and derivative at float centers."""
    motion = _motion_parameters(np.random.default_rng(warp_seed), width, height, "shear_band", gradient, .5)
    xy = np.asarray(centres, dtype=np.float64)
    z = (xy[:, 1] - motion["shear_band_center_y_px"]) / motion["shear_band_width_px"]
    truth = np.zeros((len(xy), 6), dtype=np.float64)
    truth[:, 0] = motion["translation"][0] + gradient * motion["shear_band_width_px"] * np.tanh(z)
    truth[:, 2] = gradient / np.cosh(z) ** 2
    truth[:, 3] = motion["translation"][1]
    target = xy + truth[:, [0, 3]]
    visible = ((target[:, 0] >= 0) & (target[:, 0] <= width - 1)
               & (target[:, 1] >= 0) & (target[:, 1] <= height - 1))
    return truth, visible


def score_shear(parameters, status, truth, visible):
    parameters = np.asarray(parameters).reshape(-1, 6)
    accepted = np.asarray(status).reshape(-1) > .5
    error = np.linalg.norm(parameters[:, [0, 3]] - truth[:, [0, 3]], axis=1)
    return accepted, error, accepted & visible & (error < 2)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=59000000)
    parser.add_argument("--gradients", nargs="+", type=float, default=(.33, .67, 2.))
    parser.add_argument("--control-directory-name", default="evaluation_control_exact_truth")
    parser.add_argument("--baseline-rounded", action="store_true",
                        help="record that the source archive used rounded-pixel truth")
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=64, subset_stride=64,
                      interpolation="linear", motion_model="affine", outlier_policy="disabled")
    rows, bad_windows = [], []
    try:
        for gradient in args.gradients:
            folder = args.base / f"g{gradient}"
            output_group = args.output_dir / f"g{gradient}"
            output_group.mkdir(parents=True, exist_ok=True)
            plots = sorted((folder / "evaluation").glob("*.npz"))
            fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 3.3), squeeze=False)
            for col, path in enumerate(plots):
                warp = int(path.stem.rsplit("warp", 1)[1])
                source = folder / f"inputs/affine_ncc/windows_seed{args.seed_offset}" / path.name
                reference, target = read_solver_images(folder / f"inputs/affine_ncc/solver_images_seed{args.seed_offset}", source)
                with (np.load(path) as result, np.load(source) as original,
                      np.load(folder / args.control_directory_name / path.name) as control):
                    if not np.array_equal(result["parameters"], control["parameters"], equal_nan=True):
                        raise ValueError("same-budget rerun changed initialization")
                    centres = result["centres"]
                    truth, visible = exact_shear_truth(centres, warp, gradient)
                    solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
                    active = np.ones(len(truth), np.uint32)
                    state = np.zeros((len(truth), 8), np.float32)
                    state[:, :6] = truth
                    state[:, 6] = 1
                    oracle = solver._register_seeded_frame(target, state, active)
                    zero_state = np.zeros_like(state)
                    zero_state[:, 6] = 1
                    zero = solver._register_seeded_frame(target, zero_state, active)

                    final_status, final_error, correct = score_shear(result["parameters"], result["accepted"], truth, visible)
                    oracle_status, oracle_error, oracle_good = score_shear(oracle.parameters, oracle.status, truth, visible)
                    _, _, zero_good = score_shear(zero.parameters, zero.status, truth, visible)
                    _, _, baseline_good = score_shear(original["icgn_parameters"], original["icgn_status"], truth, visible)
                    _, _, control_good = score_shear(control["control_parameters"], control["control_status"], truth, visible)
                    bad = visible & final_status & (final_error >= 2)
                    original_bad = result["visible"].astype(bool) & result["accepted"].astype(bool) & (result["error"] >= 2)
                    d = np.abs(centres[:, 1] - 1080.) / 86.4
                    shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))
                    regions = [r for r in connected_components(oracle_good.reshape(shape)) if len(r) >= 16]
                    seeded = [r for r in regions if result["confirmed_seeds"][r].any()]
                    entry = {"pair": path.stem, "gradient": gradient, "visible": int(visible.sum()),
                             "correct": int(correct.sum()), "accepted_visible": int((visible & final_status).sum()),
                             "false": int(bad.sum()), "accepted_invisible": int((~visible & final_status).sum()),
                             "rounded_truth_false": int(original_bad.sum()), "oracle": int(oracle_good.sum()),
                             "baseline": int(baseline_good.sum()), "control": int(control_good.sum()),
                             "zero": int(zero_good.sum()), "confirmed_seeds": int(result["confirmed_seeds"].sum()),
                             "false_seed": int((bad & result["confirmed_seeds"].astype(bool)).sum()),
                             "false_propagated": int((bad & ~result["confirmed_seeds"].astype(bool)).sum()),
                             "false_oracle_accepted": int((bad & oracle_status).sum()),
                             "false_oracle_correct": int((bad & oracle_good).sum()),
                             "false_between_1_and_3_widths": int((bad & (d > 1) & (d <= 3)).sum()),
                             "oracle_regions": len(regions), "seeded_oracle_regions": len(seeded),
                             "seeded_regions_below_97pct": int(sum(correct[r].sum() / len(r) < .97 for r in seeded))}
                    rows.append(entry)
                    for i in np.flatnonzero(bad):
                        bad_windows.append({"pair": path.stem, "gradient": gradient, "index": int(i),
                                            "centre_xy": centres[i].tolist(), "error_px": float(final_error[i]),
                                            "oracle_accepted": bool(oracle_status[i]), "oracle_error_px": float(oracle_error[i]),
                                            "generation": int(result["generation"][i])})
                    np.savez_compressed(output_group / path.name, truth_parameters=truth, visible=visible,
                                        final_error_px=final_error, oracle_parameters=np.asarray(oracle.parameters),
                                        oracle_status=oracle_status, oracle_error_px=oracle_error, correct=correct)
                    category = np.zeros(len(visible), np.uint8)
                    category[visible] = 1
                    category[correct] = 2
                    category[bad] = 3
                    ax = axes[0, col]
                    ax.imshow(category.reshape(shape), cmap=ListedColormap(["#d3d3d3", "#f5cf72", "#3c9d78", "#b72d34"]),
                              vmin=0, vmax=3, interpolation="nearest", extent=(0, 3840, 2160, 0))
                    ax.axhline(1080, color="#222222", linewidth=.5, linestyle="--")
                    ax.set_title(f"{path.stem.split('_texture')[1]}\n{correct.sum()}/{visible.sum()} correct; {bad.sum()} false")
                    ax.set_xlabel("x (px)")
                    ax.set_ylabel("y (px)")
            fig.suptitle(f"Exact-center truth, peak gradient {gradient}: gray invisible, yellow unresolved, green correct, red false")
            fig.tight_layout()
            fig.savefig(args.output_dir / f"shear_exact_{gradient}.png", dpi=150)
            plt.close(fig)
    finally:
        solver.close()
    (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (args.output_dir / "false_accepted_windows.json").write_text(json.dumps(bad_windows, indent=2) + "\n")
    background = ("This report supersedes `../analysis/README.md`: its previous truth was sampled at "
                  "rounded integer pixels instead of the actual half-pixel window centers. "
                  if args.baseline_rounded else "This report uses analytical truth at the exact window centers. ")
    lines = ["# Shear-band qualification with exact-center analytical truth", "",
             (background + "Inference outputs, frozen thresholds and solver settings are unchanged. The synthetic shear "
              "map and its Jacobian are now evaluated analytically at each exact center; visibility is "
              "recomputed from that map. Oracle and zero-initialization controls were re-solved on the GPU."), "",
             "| Peak gradient | Correct / visible | False / accepted visible | Exact oracle | One pass | Same-budget control | G3 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for gradient in args.gradients:
        selected = [r for r in rows if r["gradient"] == gradient]
        totals = {k: sum(r[k] for r in selected) for k in
                  ("visible", "correct", "accepted_visible", "false", "oracle", "baseline", "control")}
        fraction = totals["false"] / totals["accepted_visible"] if totals["accepted_visible"] else None
        verdict = "indeterminate" if fraction is None else ("pass" if fraction <= .005 else "fail")
        pct = "undefined" if fraction is None else f"{100 * fraction:.2f}%"
        lines.append(f"| {gradient} | {totals['correct']}/{totals['visible']} | "
                     f"{totals['false']}/{totals['accepted_visible']} ({pct}) | {totals['oracle']} | "
                     f"{totals['baseline']} | {totals['control']} | {verdict} |")
        print(gradient, totals)
    lines.extend(["", "| Pair | Peak gradient | Correct / visible | False / accepted visible | Oracle |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(f"| {row['pair']} | {row['gradient']} | {row['correct']}/{row['visible']} | "
                     f"{row['false']}/{row['accepted_visible']} | {row['oracle']} |")
    severe = [r for r in rows if r["gradient"] == 2.]
    if severe:
        false_count = sum(r["false"] for r in severe)
        oracle_bad = sum(r["false_oracle_accepted"] - r["false_oracle_correct"] for r in severe)
        lines.extend(["", (f"At peak gradient 2.0, {false_count} windows are falsely accepted; "
                          f"{oracle_bad} of them are also accepted and remain incorrect after re-solving "
                          "from exact analytical local affine parameters. This oracle is a diagnostic only. "
                          "The fixed-size affine subset under a curved field is a plausible source of bias, "
                          "but this observation alone does not establish its cause.")])
    lines.extend(["", ("The textures are paired across gradients when multiple gradients are requested. "
                   "This small synthetic cohort does not "
                   "establish performance on experimental images or other deformation families. "
                   "The exact oracle is a reference, not a mathematical upper bound."), ""])
    for gradient in args.gradients:
        lines.extend([f"![Exact-center shear band {gradient}](shear_exact_{gradient}.png)", ""])
    lines.extend([("`summary.json` records per-pair values and region counts; `false_accepted_windows.json` "
                   "lists each false acceptance. NPZ files retain exact truth and re-solved oracle outputs. "
                   "The protocol and reproduction commands are in `../PROTOCOL.md`."), ""])
    (args.output_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
