"""Summarize paired IC-GN results for single- and multi-scale NDF models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_cases(paths: list[Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    cases: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            key = (
                str(case["mode"]),
                int(case["texture_seed"]),
                int(case["warp_seed"]),
            )
            if key in cases:
                raise ValueError(f"Duplicate paired case {key}: {path}")
            cases[key] = case
    return cases


def summarize(
    control: dict[tuple[str, int, int], dict[str, Any]],
    multiscale: dict[tuple[str, int, int], dict[str, Any]],
    *,
    control_label: str = "Control",
    candidate_label: str = "Candidate",
) -> dict[str, Any]:
    if control.keys() != multiscale.keys():
        missing_control = sorted(multiscale.keys() - control.keys())
        missing_multiscale = sorted(control.keys() - multiscale.keys())
        raise ValueError(
            f"Unpaired inputs; missing control={missing_control}, "
            f"missing multi-scale={missing_multiscale}"
        )

    paired = []
    families: dict[str, Any] = {}
    for key in sorted(control):
        baseline = control[key]
        candidate = multiscale[key]
        paired.append(
            {
                "mode": key[0],
                "texture_seed": key[1],
                "warp_seed": key[2],
                "visible_windows": int(baseline["visible_windows"]),
                "control_correct": int(baseline["correct_converged_lt2px"]),
                "multiscale_correct": int(candidate["correct_converged_lt2px"]),
                "delta_correct": int(candidate["correct_converged_lt2px"])
                - int(baseline["correct_converged_lt2px"]),
                "control_seconds": float(baseline["inference_s_cpu"])
                + float(baseline["icgn_s"]),
                "multiscale_seconds": float(candidate["inference_s_cpu"])
                + float(candidate["icgn_s"]),
            }
        )

    for mode in sorted({row["mode"] for row in paired}):
        rows = [row for row in paired if row["mode"] == mode]
        visible = sum(row["visible_windows"] for row in rows)
        control_correct = sum(row["control_correct"] for row in rows)
        multiscale_correct = sum(row["multiscale_correct"] for row in rows)
        control_seconds = sum(row["control_seconds"] for row in rows)
        multiscale_seconds = sum(row["multiscale_seconds"] for row in rows)
        deltas = [row["delta_correct"] for row in rows]
        families[mode] = {
            "pairs": len(rows),
            "visible_windows": visible,
            "control_correct": control_correct,
            "multiscale_correct": multiscale_correct,
            "control_coverage": control_correct / visible if visible else 0.0,
            "multiscale_coverage": multiscale_correct / visible if visible else 0.0,
            "delta_correct": multiscale_correct - control_correct,
            "positive_pairs": sum(delta > 0 for delta in deltas),
            "equal_pairs": sum(delta == 0 for delta in deltas),
            "negative_pairs": sum(delta < 0 for delta in deltas),
            "control_seconds_sum": control_seconds,
            "multiscale_seconds_sum": multiscale_seconds,
            "control_correct_per_second": control_correct / control_seconds
            if control_seconds
            else 0.0,
            "multiscale_correct_per_second": multiscale_correct
            / multiscale_seconds
            if multiscale_seconds
            else 0.0,
        }

    return {
        "protocol": {
            "comparison": f"paired {control_label} versus {candidate_label}",
            "control_label": control_label,
            "candidate_label": candidate_label,
            "timed_pipeline": "CPU model inference plus WebGPU IC-GN; input rasters are native 3840x2160, model input 192x108",
            "correctness": "IC-GN displacement error below 2 native pixels",
            "seeds": sorted({key[1] for key in control}),
        },
        "families": families,
        "paired_cases": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", type=Path, required=True)
    parser.add_argument("--multiscale", nargs="+", type=Path, required=True)
    parser.add_argument("--control-label", default="Control")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    result = summarize(
        load_cases(args.control),
        load_cases(args.multiscale),
        control_label=args.control_label,
        candidate_label=args.candidate_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for mode, values in result["families"].items():
        print(
            f"{mode}: {args.control_label} "
            f"{values['control_correct']}/{values['visible_windows']} "
            f"({values['control_coverage']:.2%}), {args.candidate_label} "
            f"{values['multiscale_correct']}/{values['visible_windows']} "
            f"({values['multiscale_coverage']:.2%}), "
            f"delta={values['delta_correct']:+d}, "
            f"paired changes +/=/−={values['positive_pairs']}/"
            f"{values['equal_pairs']}/{values['negative_pairs']}, "
            f"throughput {values['control_correct_per_second']:.0f} → "
            f"{values['multiscale_correct_per_second']:.0f} correct/s"
        )
    print(f"Wrote {args.output}")
    if args.figure:
        write_figure(result, args.figure)


def write_figure(result: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, (coverage_ax, delta_ax) = plt.subplots(
        1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [0.9, 1.5]}
    )
    methods = [
        result["protocol"]["control_label"],
        result["protocol"]["candidate_label"],
    ]
    colors = ["#536779", "#167D9A"]
    modes = ["translation", "heterogeneous"]
    pair_count = result["families"][modes[0]]["pairs"]
    x = np.arange(len(modes))
    width = 0.34
    for index, (label, key, color) in enumerate(
        zip(methods, ("control_coverage", "multiscale_coverage"), colors)
    ):
        values = [result["families"][mode][key] * 100 for mode in modes]
        bars = coverage_ax.bar(x + (index - 0.5) * width, values, width, label=label, color=color)
        for bar, mode in zip(bars, modes):
            family = result["families"][mode]
            count_key = "control_correct" if index == 0 else "multiscale_correct"
            count = family[count_key]
            total = family["visible_windows"]
            coverage_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{100 * count / total:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )
    coverage_ax.set_xticks(x, ["Translation", "Heterogeneous"])
    coverage_ax.set_ylabel("Correct IC-GN windows (%)")
    coverage_ax.set_ylim(0, 100)
    coverage_ax.set_title(f"Coverage over {pair_count} held-out pairs")
    coverage_ax.grid(axis="y", alpha=0.22)
    coverage_ax.legend(frameon=False, fontsize=8, loc="upper right")

    for position, mode in enumerate(modes):
        rows = [row for row in result["paired_cases"] if row["mode"] == mode]
        deltas = [row["delta_correct"] for row in rows]
        offset = -0.18 if position == 0 else 0.18
        delta_ax.scatter(
            np.full(len(deltas), position + offset),
            deltas,
            s=27,
            alpha=0.78,
            color=colors[position],
            edgecolors="white",
            linewidths=0.45,
            label=f"{mode.title()} ({sum(d > 0 for d in deltas)} gains, "
            f"{sum(d == 0 for d in deltas)} equal, {sum(d < 0 for d in deltas)} losses)",
        )
        mean_delta = float(np.mean(deltas))
        delta_ax.hlines(
            mean_delta,
            position - 0.32,
            position + 0.32,
            color=colors[position],
            linewidth=2.5,
        )
        delta_ax.text(
            position,
            mean_delta + 6.0,
            f" mean {mean_delta:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[position],
        )
    delta_ax.axhline(0, color="#333333", linewidth=0.9)
    delta_ax.set_xticks(x, ["Translation", "Heterogeneous"])
    delta_ax.set_ylabel("Change in correct windows per pair")
    delta_ax.set_title(
        f"Paired change: {result['protocol']['candidate_label']} minus "
        f"{result['protocol']['control_label']}"
    )
    delta_ax.grid(axis="y", alpha=0.22)
    delta_ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        "Paired independent IC-GN qualification",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        "Native 3840 × 2160 sources; learned matcher input 192 × 108; "
        "correct = final displacement error < 2 px",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
