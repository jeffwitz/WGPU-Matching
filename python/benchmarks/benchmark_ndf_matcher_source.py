"""Compare learned, HORG, and DeepMatching coarse affine sources through NCC and IC-GN."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _merge_reports(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    merged = dict(reports[0])
    merged["cases"] = [case for report in reports for case in report["cases"]]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--learned-baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=(16000000, 17000000))
    parser.add_argument("--cases-per-seed", type=int, default=10)
    parser.add_argument("--methods", nargs="+", choices=("horg", "deepmatching"), default=("horg", "deepmatching"))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    qualifier = Path(__file__).with_name("qualify_ndf_pair_icgn.py")
    method_reports = {
        "learned": _merge_reports([
            args.learned_baseline_dir / "affine_ncc" / f"heterogeneous_seed{seed}.json"
            for seed in args.seed_offsets
        ])
    }
    for method in args.methods:
        method_dir = out / method
        method_dir.mkdir(parents=True, exist_ok=True)
        if args.summarize_only:
            method_reports[method] = _merge_reports([
                method_dir / f"heterogeneous_seed{seed}.json" for seed in args.seed_offsets
            ])
            continue
        report_paths = []
        for seed in args.seed_offsets:
            report_path = method_dir / f"heterogeneous_seed{seed}.json"
            report_paths.append(report_path)
            command = [
                sys.executable, str(qualifier),
                "--weights", str(args.weights.resolve()),
                "--live-dic-source", str(args.live_dic_source.resolve()),
                "--output", str(report_path),
                "--window-diagnostics-dir", str(method_dir / f"windows_seed{seed}"),
                "--native-scale", "20", "--model-width", "192", "--model-height", "108",
                "--speckle-size", "0.35", "--families", "heterogeneous",
                "--cases-per-family", str(args.cases_per_seed), "--seed-offset", str(seed),
                "--contextual-features", "--coarse-matcher", method,
                "--global-affine-prefit", "--field-residual", "none",
                "--local-ncc-radius", "8", "--local-ncc-patch-radius", "4",
                "--local-ncc-backend", "wgpu", "--local-ncc-geometry", "translation",
                "--local-consensus-radius", "40", "--local-consensus-threshold", "5",
                "--projection", "raw_bilinear", "--propagation", "frontier",
            ]
            with (method_dir / f"heterogeneous_seed{seed}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        method_reports[method] = _merge_reports(report_paths)

    cases = {}
    for method, report in method_reports.items():
        for case in report["cases"]:
            key = f"{case['texture_seed']}_{case['warp_seed']}"
            cases.setdefault(key, {})[method] = case
    if any(set(row) != set(method_reports) for row in cases.values()):
        raise ValueError("matcher methods did not produce a fully paired case set")

    summary = {}
    details = {}
    for method, report in method_reports.items():
        per_pair = []
        for case in report["cases"]:
            with np.load(case["window_diagnostics"]) as data:
                accepted = data["propagated_status"].astype(bool)
                visible = data["visible"].astype(bool)
                correct = data["propagated_correct_lt2px"].astype(bool) & visible
                seeds = data["propagated_seed_mask"].astype(bool)
                oracle = data["oracle_icgn_correct_lt2px"].astype(bool) & visible
                erroneous = accepted & (data["propagated_displacement_error_px"] >= 2.0)
                per_pair.append({
                    "texture_seed": case["texture_seed"],
                    "warp_seed": case["warp_seed"],
                    "seeds": int(seeds.sum()),
                    "one_shot_correct": int((data["icgn_correct_lt2px"].astype(bool) & visible).sum()),
                    "propagated_correct": int(correct.sum()),
                    "oracle": int(oracle.sum()),
                    "accepted_error_ge_2px": int(erroneous.sum()),
                    "bad_windows": [
                        {"index": int(i), "xy": data["centres_xy_native"][i].tolist(),
                         "error_px": float(data["propagated_displacement_error_px"][i])}
                        for i in np.flatnonzero(erroneous)
                    ],
                    "rings": int(data["propagated_ring"].max(initial=0)),
                    "extra_windows": int((accepted & ~seeds).sum()),
                    "accepted_invisible": int((accepted & ~visible).sum()),
                    "one_shot_s": case["icgn_s"],
                    "propagation_s": case["propagation_wall_s"],
                    "pipeline_s": case["inference_s_cpu"] + case["field_projection_s_cpu"] + case["icgn_s"] + case["propagation_wall_s"],
                    "coarse_matcher": case.get("coarse_matcher_diagnostics"),
                    "affine_hypothesis_available": (
                        bool(case["coarse_matcher_diagnostics"].get("affine_fit_available"))
                        if case.get("coarse_matcher_diagnostics") is not None
                        else bool(case.get("global_affine_prefit_used"))
                    ),
                })
        visible_total = sum(row["oracle"] for row in per_pair)
        summary[method] = {
            "pairs": len(per_pair),
            "seeded_pairs": sum(row["seeds"] > 0 for row in per_pair),
            "matcher_failure_pairs": sum(
                not row["affine_hypothesis_available"]
                for row in per_pair
            ),
            "pairs_with_affine_hypothesis": sum(
                row["affine_hypothesis_available"] for row in per_pair
            ),
            "one_shot_correct": sum(row["one_shot_correct"] for row in per_pair),
            "propagated_correct": sum(row["propagated_correct"] for row in per_pair),
            "oracle_correct": visible_total,
            "post_propagation_coverage_over_visible": sum(row["propagated_correct"] for row in per_pair)
                / max(sum(case["visible_windows"] for case in report["cases"]), 1),
            "seeded_pair_oracle_fraction": sum(row["propagated_correct"] for row in per_pair if row["seeds"])
                / max(sum(row["oracle"] for row in per_pair if row["seeds"]), 1),
            "accepted_error_ge_2px": sum(row["accepted_error_ge_2px"] for row in per_pair),
            "accepted_invisible": sum(row["accepted_invisible"] for row in per_pair),
            "median_pipeline_s": float(np.median([row["pipeline_s"] for row in per_pair])),
            "median_extra_windows": float(np.median([row["extra_windows"] for row in per_pair])),
        }
        details[method] = per_pair
    result = {
        "protocol": {
            "input": "192x108 model raster from native 3840x2160 synthetic images; IC-GN on native pixels",
            "baseline": "same local NCC, image-derived global affine prefit, raw bilinear projection, and status-only frontier propagation",
            "matcher-specific settings": {
                "HORG": "WGSL keypoints/descriptors with 32 px support, 16 px stride, reduction 1",
                "DeepMatching": "WGSL INRIA hierarchy, 32 px patch, 64 px root, 32 px stride, one downscale level",
            },
            "geometric_fit": "same robust affine threshold 5 model px, minimum 24 inliers",
            "truth_used_to_select_matches_or_hypotheses": False,
        },
        "summary": summary,
        "pairs": details,
    }
    path = out / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    labels = {"learned": "Learned coarse matcher", "horg": "HORG", "deepmatching": "DeepMatching"}
    lines = [
        "# Coarse matcher source ablation with NCC and IC-GN propagation", "",
        "Normalized cross-correlation (NCC) is the local photometric search. Inverse-compositional Gauss-Newton (IC-GN) is the window solver. Histogram of oriented and rotated gradients (HORG) is compared with DeepMatching and the learned coarse matcher. Truth is used only after selection, for scoring. All methods use the same NCC search, robust affine-fit acceptance threshold, IC-GN settings, and status-only frontier propagation. Matcher extraction settings are method-specific and listed in `summary.json`.", "",
        "| Source | Pairs with affine | Matcher failures | Pairs with seeds | One-pass correct | Correct after propagation | Oracle correct | Seeded-pair oracle fraction | Median pipeline (s) | Bad accepted | Invisible accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.items():
        lines.append(
            f"| {labels[method]} | {row['pairs_with_affine_hypothesis']}/{row['pairs']} "
            f"| {row['matcher_failure_pairs']} | {row['seeded_pairs']}/{row['pairs']} | {row['one_shot_correct']:,} "
            f"| {row['propagated_correct']:,} | {row['oracle_correct']:,} | {row['seeded_pair_oracle_fraction']:.2%} "
            f"| {row['median_pipeline_s']:.3f} | {row['accepted_error_ge_2px']} | {row['accepted_invisible']} |"
        )
    lines.extend([
        "", "Per-pair counts, affine fit support, bad-acceptance coordinates, and timings are in `summary.json`.",
        "", "Reproduce the full comparison with:", "", "```console",
        "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_matcher_source.py \\",
        "  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \\",
        "  --live-dic-source /path/to/Live-DIC/python/src \\",
        "  --learned-baseline-dir results/ndf_dic_20261109/causal_ablation_frontier_corrected_20261109 \\",
        "  --output-dir results/ndf_dic_20261109/matcher_source_frontier_20261109",
        "```",
    ])
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
