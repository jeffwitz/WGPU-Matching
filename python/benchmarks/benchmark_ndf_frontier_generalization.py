"""Qualify frontier propagation on held-out synthetic motion families."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected component masks without an optional image library."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for start_y, start_x in np.argwhere(mask):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        points = []
        while stack:
            y, x = stack.pop()
            points.append(y * width + x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height and 0 <= nx < width
                        and mask[ny, nx] and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        components.append(np.asarray(points, dtype=np.int64))
    return components


def _summarize_case(case: dict) -> dict:
    with np.load(case["window_diagnostics"]) as data:
        visible = data["visible"].astype(bool)
        oracle_flat = data["oracle_icgn_correct_lt2px"].astype(bool) & visible
        seeds = data["propagated_seed_mask"].astype(bool)
        accepted = data["propagated_status"].astype(bool)
        correct = data["propagated_correct_lt2px"].astype(bool) & visible
        centres = data["centres_xy_native"]
        xs = np.unique(centres[:, 0])
        ys = np.unique(centres[:, 1])
        grid_x = np.searchsorted(xs, centres[:, 0])
        grid_y = np.searchsorted(ys, centres[:, 1])
        oracle = np.zeros((len(ys), len(xs)), dtype=bool)
        oracle[grid_y, grid_x] = oracle_flat
        regions = []
        for indices in _components8(oracle):
            size = int(indices.size)
            if size < 16:
                continue
            region_grid = np.zeros_like(oracle)
            region_grid.ravel()[indices] = True
            region_flat = region_grid[grid_y, grid_x]
            regions.append({
                "oracle_correct_windows": size,
                "initial_status_seeds": int((region_flat & seeds).sum()),
                "propagated_correct": int((region_flat & correct).sum()),
                "propagated_oracle_fraction": float((region_flat & correct).sum() / size),
                "seeded": bool((region_flat & seeds).any()),
            })
        errors = data["propagated_displacement_error_px"]
        bad_indices = np.flatnonzero(accepted & visible & (errors >= 2))
        return {
            "texture_seed": case["texture_seed"],
            "warp_seed": case["warp_seed"],
            "visible": int(visible.sum()),
            "oracle_correct": int(oracle_flat.sum()),
            "one_shot_correct": int((data["icgn_correct_lt2px"].astype(bool) & visible).sum()),
            "seed_count": int(seeds.sum()),
            "seeded": bool(seeds.any()),
            "propagated_correct": int(correct.sum()),
            "propagated_oracle_fraction": float(correct.sum() / max(oracle.sum(), 1)),
            "accepted_error_ge_2px": len(bad_indices),
            "bad_accepted_windows": [
                {"index": int(index), "centre_xy": centres[index].tolist(),
                 "error_px": float(errors[index]), "ring": int(data["propagated_ring"][index])}
                for index in bad_indices
            ],
            "accepted_invisible": int((accepted & ~visible).sum()),
            "rings": int(data["propagated_ring"].max(initial=0)),
            "ring_attempts_and_acceptances": case.get("propagation_rings", []),
            "extra_windows": int((accepted & ~seeds).sum()),
            "propagation_s": case["propagation_wall_s"],
            "oracle_regions_min16": regions,
            "all_oracle_regions_seeded": all(region["seeded"] for region in regions),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=(18000000, 19000000))
    parser.add_argument("--cases-per-family", type=int, default=5)
    parser.add_argument("--families", nargs="+", choices=("rigid", "affine", "discontinuous"),
                        default=("rigid", "affine", "discontinuous"))
    parser.add_argument("--shear-gradients", nargs="+", type=float, default=(0.33, 0.67, 2.0))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    qualifier = Path(__file__).with_name("qualify_ndf_pair_icgn.py")
    group_reports = []
    for seed in args.seed_offsets:
        groups = [(family, None) for family in args.families]
        groups.extend(("shear_band", gradient) for gradient in args.shear_gradients)
        for family, gradient in groups:
            name = family if gradient is None else f"shear_band_g{gradient:g}"
            directory = out / f"seed{seed}" / name
            directory.mkdir(parents=True, exist_ok=True)
            report_path = directory / "qualification.json"
            command = [
                sys.executable, str(qualifier),
                "--weights", str(args.weights.resolve()),
                "--live-dic-source", str(args.live_dic_source.resolve()),
                "--output", str(report_path),
                "--window-diagnostics-dir", str(directory / "windows"),
                "--native-scale", "20", "--model-width", "192", "--model-height", "108",
                "--speckle-size", "0.35", "--families", family,
                "--cases-per-family", str(args.cases_per_family), "--case-start-index", "0",
                "--seed-offset", str(seed), "--contextual-features",
                "--rotation-bank-steps", "9", "--rotation-bank-probe-query-budget", "256",
                "--global-affine-prefit", "--field-residual", "none",
                "--local-ncc-radius", "8", "--local-ncc-patch-radius", "4",
                "--local-ncc-backend", "wgpu", "--local-ncc-geometry", "translation",
                "--local-consensus-radius", "40", "--local-consensus-threshold", "5",
                "--projection", "raw_bilinear", "--propagation", "frontier",
            ]
            if gradient is not None:
                command.extend(("--shear-band-peak-gradient", str(gradient)))
            if not report_path.exists():
                with (directory / "run.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = [_summarize_case(case) for case in report["cases"]]
            group_reports.append({
                "seed_offset": seed,
                "family": family,
                "shear_band_peak_gradient": gradient,
                "pairs": rows,
                "no_seed_pairs": [
                    {"texture_seed": row["texture_seed"], "oracle_correct": row["oracle_correct"],
                     "oracle_regions": len(row["oracle_regions_min16"])}
                    for row in rows if not row["seeded"]
                ],
                "no_seed_pairs_with_oracle_support": sum(
                    not row["seeded"] and row["oracle_correct"] > 0 for row in rows
                ),
                "seeded_pairs": sum(row["seeded"] for row in rows),
                "mean_propagated_oracle_fraction_seeded": float(np.mean([
                    row["propagated_oracle_fraction"] for row in rows if row["seeded"]
                ])) if any(row["seeded"] for row in rows) else None,
                "pairs_passing_97pct": sum(
                    row["seeded"] and row["propagated_oracle_fraction"] >= 0.97 for row in rows
                ),
                "pairs_all_regions_seeded": sum(row["all_oracle_regions_seeded"] for row in rows),
                "accepted_total": sum(row["seed_count"] + row["extra_windows"] for row in rows),
                "accepted_error_ge_2px": sum(row["accepted_error_ge_2px"] for row in rows),
                "accepted_error_fraction": sum(row["accepted_error_ge_2px"] for row in rows)
                    / max(sum(row["seed_count"] + row["extra_windows"] for row in rows), 1),
                "gate_g2_pass": sum(row["accepted_error_ge_2px"] for row in rows)
                    / max(sum(row["seed_count"] + row["extra_windows"] for row in rows), 1) <= 0.001,
                "seeded_regions_below_97pct": sum(
                    region["seeded"] and region["propagated_oracle_fraction"] < 0.97
                    for row in rows for region in row["oracle_regions_min16"]
                ),
            })

    result = {
        "protocol": {
            "families": "held-out rigid, affine, discontinuous, and analytic shear-band fields",
            "seed_offsets": args.seed_offsets,
            "pairs_per_group": args.cases_per_family,
            "baseline": "no NDF field residual; local normalized cross-correlation and image-derived affine prefit",
            "propagation_seed_rule": "IC-GN status only; oracle and visibility are used only for scoring",
            "region_definition": "8-connected components of oracle-correct windows, retaining components of at least 16 windows",
            "pass_rule": "each retained oracle region contains at least one first-pass accepted seed",
            "solver": "WebGPU inverse-compositional Gauss-Newton, unchanged 64 px subset, stride 64, 30 iterations, 1e-3 tolerance, ZNCC minimum 0.65",
            "real_image_claim": "none; synthetic families only",
        },
        "groups": group_reports,
    }
    report_path = out / "summary.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Held-out frontier propagation generalization",
        "",
        "Digital image correlation (DIC) is initialized by the learned neural deformation field (NDF). Inverse-compositional Gauss-Newton (IC-GN) is the local solver. Normalized cross-correlation (NCC), here scored as zero-mean normalized cross-correlation (ZNCC), supplies the local photometric refinement. The solver's accepted status alone defines seeds. Oracle fields are used only for scoring.",
        "",
        "| Seed group | Family | Peak shear gradient | Seeded pairs | Pairs at >=97% oracle | Seeded oracle regions below 97% | Pairs with every oracle region seeded | G2 bad accepted / total |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in group_reports:
        lines.append(
            f"| {group['seed_offset']} | {group['family']} | {group['shear_band_peak_gradient'] or '—'} "
            f"| {group['seeded_pairs']}/{len(group['pairs'])} | {group['pairs_passing_97pct']}/{len(group['pairs'])} "
            f"| {group['seeded_regions_below_97pct']} | {group['pairs_all_regions_seeded']}/{len(group['pairs'])} "
            f"| {group['accepted_error_ge_2px']}/{group['accepted_total']} "
            f"({'PASS' if group['gate_g2_pass'] else 'FAIL'}) |"
        )
    bad_rows = [
        (group, pair, item)
        for group in group_reports for pair in group["pairs"]
        for item in pair["bad_accepted_windows"]
    ]
    if bad_rows:
        lines.extend(["", "Accepted visible windows with displacement error >= 2 px:", ""])
        lines.extend([
            f"- Seed {group['seed_offset']}, {group['family']} {group['shear_band_peak_gradient'] or ''}, "
            f"texture {pair['texture_seed']}, window {item['index']} at "
            f"({item['centre_xy'][0]:.1f}, {item['centre_xy'][1]:.1f}) px: "
            f"{item['error_px']:.3f} px at ring {item['ring']}."
            for group, pair, item in bad_rows
        ])
    else:
        lines.extend(["", "No accepted visible windows exceeded the 2 px error limit in these held-out groups."])
    lines.extend([
        "", "Per-pair stopping rings, error counts, region seed maps, timing, and source exports are retained in `summary.json` and each group's `windows/` directory.",
        "", "A seedless pair or oracle region without a seed is outside propagation's reach. The exact seedless pairs and their oracle support are listed in each group in `summary.json`; they are plausible decoder targets, though this does not establish that a learned field model would recover them. Per-ring attempt and acceptance counts identify where propagation stops.",
        "", "Reproduce the full campaign with:", "",
        "```console",
        "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_frontier_generalization.py \\",
        "  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \\",
        "  --live-dic-source /path/to/Live-DIC/python/src \\",
        "  --output-dir results/ndf_dic_20261109/frontier_generalization_20261109",
        "```",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
