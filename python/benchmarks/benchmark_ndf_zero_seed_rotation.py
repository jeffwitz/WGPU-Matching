"""Verify top-ranked image-only rotations on a fixed sparse IC-GN subset."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-report", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subset-budget", type=int, default=128)
    parser.add_argument("--n-hypotheses", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.subset_budget <= 128:
        parser.error("--subset-budget must be between 1 and 128")
    if args.n_hypotheses < 1:
        parser.error("--n-hypotheses must be positive")

    ranking = json.loads(args.ranking_report.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    qualifier = Path(__file__).with_name("qualify_ndf_pair_icgn.py")
    pairs = []
    for texture_seed in (17051005, 17051007):
        case = next(row for row in ranking["cases"] if row["texture_seed"] == texture_seed)
        top = sorted(
            case["rotation_bank_candidate_scores"],
            key=lambda candidate: (candidate["ranking_score"], candidate["consensus_support"]),
            reverse=True,
        )[: args.n_hypotheses]
        # The generated pair indexing is part of the deterministic benchmark protocol.
        sample = (texture_seed - 51000 - 17000000 - 1) // 2
        hypothesis_rows = []
        for rank, candidate in enumerate(top, start=1):
            angle = float(candidate["angle_rad"])
            path = output_dir / f"texture{texture_seed}_rank{rank}.json"
            command = [
                sys.executable, str(qualifier),
                "--weights", str(args.weights.resolve()),
                "--live-dic-source", str(args.live_dic_source.resolve()),
                "--output", str(path),
                "--native-scale", "20", "--model-width", "192", "--model-height", "108",
                "--speckle-size", "0.35", "--families", "heterogeneous",
                "--cases-per-family", "1", "--case-start-index", str(sample),
                "--seed-offset", "17000000", "--contextual-features",
                "--rotation-bank-steps", "9", "--rotation-bank-probe-query-budget", "256",
                "--global-affine-prefit", "--field-residual", "none",
                "--local-ncc-radius", "8", "--local-ncc-patch-radius", "4",
                "--local-ncc-backend", "wgpu", "--local-ncc-geometry", "translation",
                "--local-consensus-radius", "40", "--local-consensus-threshold", "5",
                "--projection", "raw_bilinear", "--forced-rotation-angle-rad", str(angle),
                "--solver-window-budget", str(args.subset_budget),
            ]
            started = time.perf_counter()
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            wall_s = time.perf_counter() - started
            result = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
            if result["texture_seed"] != texture_seed or result["warp_seed"] != texture_seed + 40000:
                raise ValueError("fixed-subset run did not regenerate the requested synthetic pair")
            hypothesis_rows.append({
                "rank_by_image_only_score": rank,
                "angle_rad": angle,
                "rotation_ranking_score": float(candidate["ranking_score"]),
                "rotation_consensus_support": int(candidate["consensus_support"]),
                "solver_accepted_on_fixed_subset": int(result["solver_accepted_active"]),
                "subset_size": int(result["solver_active_window_count"]),
                "qualifier_wall_s": wall_s,
                "qualifier_report": str(path),
            })
        selected = max(
            hypothesis_rows,
            key=lambda row: row["solver_accepted_on_fixed_subset"],
        )
        pairs.append({
            "texture_seed": texture_seed,
            "warp_seed": texture_seed + 40000,
            "hypotheses": hypothesis_rows,
            "selected_by_solver_status_count": selected,
        })
    report = {
        "protocol": {
            "description": "Top hypotheses are taken from the existing image-only ranking; selection uses accepted IC-GN statuses on a fixed evenly spaced sparse subset only.",
            "truth_or_visibility_used_for_hypothesis_selection": False,
            "subset_budget": args.subset_budget,
            "maximum_attempts_per_hypothesis": args.n_hypotheses,
            "local_ncc": "enabled, search radius 8 model pixels",
            "field_residual": "none",
            "solver": "WebGPU IC-GN, affine, 64 px subset and stride, 30 iterations, tolerance 1e-3, ZNCC minimum 0.65",
        },
        "pairs": pairs,
    }
    output = output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Image-only rotation verification for zero-seed pairs",
        "",
        "Inverse-compositional Gauss-Newton (IC-GN) status is the only selection criterion after taking the three highest image-derived rotation scores. Each angle is evaluated on the same fixed 128-window subset with no learned nonlinear field residual and local normalized cross-correlation (NCC). Truth and visibility are used only for post-selection diagnostics.",
        "",
        "| Pair | Rank | Hypothesis angle (degrees) | Image score | Accepted IC-GN seeds / 128 | Selected |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for pair in pairs:
        selected_rank = pair["selected_by_solver_status_count"]["rank_by_image_only_score"]
        for hypothesis in pair["hypotheses"]:
            lines.append(
                f"| {pair['texture_seed']} | {hypothesis['rank_by_image_only_score']} "
                f"| {math.degrees(hypothesis['angle_rad']):.1f} "
                f"| {hypothesis['rotation_ranking_score']:.5f} "
                f"| {hypothesis['solver_accepted_on_fixed_subset']}/{hypothesis['subset_size']} "
                f"| {'yes' if hypothesis['rank_by_image_only_score'] == selected_rank else 'no'} |"
            )
    lines.extend([
        "", "All attempted hypotheses, selected angles, report paths, and timing are in `summary.json`. A tie is resolved by the existing image-score rank order; the solver-status count itself is the selection value.",
        "", "Reproduce with:", "", "```console",
        "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_zero_seed_rotation.py \\",
        "  --ranking-report results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109/affine_ncc/heterogeneous_seed17000000.json \\",
        "  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \\",
        "  --live-dic-source /path/to/Live-DIC/python/src \\",
        "  --output-dir results/ndf_dic_20261109/zero_seed_rotation_top3_20261109",
        "```",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
