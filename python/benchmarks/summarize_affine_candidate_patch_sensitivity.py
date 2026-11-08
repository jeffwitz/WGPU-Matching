"""Summarize the translation-free affine candidate-patch experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METHODS = {
    "zncc": "ZNCC",
    "horg64": "HORG-64",
    "dic_cnn": "DIC-Descriptor CNN",
}


def condition_rows(payload: dict, parameter: str, delta: float) -> list[tuple[dict, dict]]:
    return [
        (case, row)
        for case in payload["results"]
        for row in case["conditions"]
        if row.get("parameter") == parameter and row.get("delta") == delta
    ]


def aggregate_retrieval(rows: list[tuple[dict, dict]], method: str) -> dict[str, int]:
    totals = {
        "queries": 0,
        "repeatable": 0,
        "top1": 0,
        "top5": 0,
        "top20": 0,
        "accepted": 0,
        "correct_accepted": 0,
        "false_accepted": 0,
    }
    for _case, row in rows:
        values = row["retrieval"][method]
        totals["queries"] += values["queries"]
        totals["repeatable"] += values["repeatable_le5px"]
        totals["top1"] += values["top1_correct_le5px"]
        totals["top5"] += values["top5_retrieval"]
        totals["top20"] += values["top20_retrieval"]
        consensus = row["gpu_affine_consensus"][method]
        totals["accepted"] += consensus.get("accepted_by_model", 0)
        totals["correct_accepted"] += consensus.get("correct_accepted_le5px", 0)
        totals["false_accepted"] += consensus.get("false_accepted", 0)
    return totals


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}%" if denominator else "n/a"


def render_sensitivity_svg(payload: dict) -> str:
    axes = [
        ("angle_deg", "Rotation error (degrees)", (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)),
        ("stretch_x_percent", "X stretch error (percent)", (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)),
        ("stretch_y_percent", "Y stretch error (percent)", (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)),
        ("shear", "Shear error", (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)),
    ]
    colors = {"top1": "#1769aa", "top5": "#e87500", "top20": "#168447"}
    labels = {"top1": "Top-1", "top5": "Top-5", "top20": "Top-20"}
    width, height = 1120, 760
    panel_w, panel_h = 500, 300
    lefts, tops = (70, 610), (105, 430)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">DIC descriptor retrieval sensitivity to Jacobian error</title>",
        "<desc id=\"desc\">Top-1, top-5 and top-20 retrieval fractions for the compact DIC descriptor across independent rotation, x stretch, y stretch and shear errors. Three 4K synthetic global affine pairs are pooled.</desc>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="560" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">Retrieval sensitivity to one-axis Jacobian error</text>',
    ]
    for index, (metric, color) in enumerate(colors.items()):
        x = 390 + index * 130
        out.extend(
            [
                f'<line x1="{x}" y1="65" x2="{x + 28}" y2="65" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 36}" y="70" font-family="sans-serif" font-size="14">{labels[metric]}</text>',
            ]
        )
    for panel_index, (parameter, x_label, deltas) in enumerate(axes):
        origin_x, origin_y = lefts[panel_index % 2], tops[panel_index // 2]
        plot_x, plot_y = origin_x + 55, origin_y + 20
        plot_w, plot_h = panel_w - 80, panel_h - 78
        out.append(
            f'<text x="{origin_x + panel_w / 2}" y="{origin_y}" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="700">{x_label}</text>'
        )
        for tick in range(6):
            fraction = tick / 5
            y = plot_y + plot_h * (1 - fraction)
            out.extend(
                [
                    f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#d9dfe5" stroke-width="1"/>',
                    f'<text x="{plot_x - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{int(fraction * 100)}%</text>',
                ]
            )
        out.extend(
            [
                f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#34495e"/>',
                f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#34495e"/>',
            ]
        )
        for tick_index, delta in enumerate(deltas):
            x = plot_x + plot_w * tick_index / (len(deltas) - 1)
            if parameter == "angle_deg":
                label = f"{delta:+.0f}"
            elif parameter == "shear":
                label = f"{delta:+.2f}"
            else:
                label = f"{delta:+.0f}"
            out.append(
                f'<text x="{x:.1f}" y="{plot_y + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>'
            )
        for metric, color in colors.items():
            field = {"top1": "top1", "top5": "top5", "top20": "top20"}[metric]
            values: list[float | None] = []
            for delta in deltas:
                aggregate = aggregate_retrieval(
                    condition_rows(payload, parameter, delta), "dic_cnn"
                )
                values.append(
                    aggregate[field] / aggregate["repeatable"]
                    if aggregate["repeatable"]
                    else None
                )
            points = [
                (
                    plot_x + plot_w * point_index / (len(deltas) - 1),
                    plot_y + plot_h * (1.0 - value),
                )
                for point_index, value in enumerate(values)
                if value is not None
            ]
            if len(points) > 1:
                coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                out.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
                )
            for x, y in points:
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>'
                )
    out.append(
        '<text x="20" y="390" transform="rotate(-90 20 390)" text-anchor="middle" font-family="sans-serif" font-size="14">Correct retrieval among repeatable queries</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_report(payload: dict) -> str:
    protocol = payload["protocol"]
    cases = payload["results"]
    exact = condition_rows(payload, "exact", 0.0)
    all_repeatable = sum(
        case["baseline_retrieval"]["dic_cnn"]["repeatable_le5px"] for case in cases
    )
    all_queries = sum(case["source_query_count"] for case in cases)
    candidate_counts = [case["target_candidate_count"] for case in cases]

    lines = [
        "# Translation-free affine-patch retrieval sensitivity",
        "",
        "## Result",
        "",
        "On three independent 3840 × 2160 synthetic digital image correlation (DIC) pairs, applying the exact local Jacobian to every target candidate patch—without using the true translation to construct those patches—raised the learned compact descriptor's top-1 correct retrieval from **196/1002 to 823/1002 repeatable queries**. Its top-five retrieval rose from **366/1002 to 901/1002**. A WebGPU (the Web Graphics API for graphics processing units (GPUs)) affine consensus retained 775 correct estimates and accepted no false estimates on these globally affine fields.",
        "",
        "This isolates geometric support mismatch as a major cause of the earlier descriptor failure. It is an oracle diagnostic: the Jacobian (the derivative of the target coordinate with respect to the reference coordinate) is supplied from synthetic truth, so the result is not an initializer for unknown images and does not establish inverse compositional Gauss–Newton convergence.",
        "",
        "## Protocol",
        "",
        f"- Graphics processing unit (GPU) adapter: {protocol['webgpu_adapter']}; Intel's Lunar Lake (LNL) graphics platform.",
        f"- Image size: {protocol['image_size_px'][0]} × {protocol['image_size_px'][1]} pixels (ultra-high-definition (UHD), or 4K); three texture/warp seed pairs.",
        f"- Per pair: {protocol['query_budget']} source queries and target candidate budget {protocol['candidate_budget']}; actual candidate counts: {', '.join(map(str, candidate_counts))}.",
        f"- Detector: {protocol['detector']}.",
        f"- Repeatable denominator: {all_repeatable}/{all_queries} source queries had a detected target candidate within 5 pixels under analytic truth.",
        "- The same detector-selected candidate coordinates were retained for every Jacobian hypothesis. Candidate selection used detector validity and image-border support only; no analytic visibility or occlusion mask was read.",
        "- For each unknown target candidate position y, the sampled support was `I1(y + F_hat ξ)`. The true translation was absent from support construction. Analytic displacement was used only to label retrieval correctness after scoring.",
        "- The tested descriptors were zero-normalized cross correlation (ZNCC), histogram of oriented and rotated gradients (HORG-64), and a compact convolutional neural network (CNN) descriptor trained for DIC. Their patch support was 64 × 64 pixels.",
        "- The perturbation sweep changed one output-frame Jacobian parameter at a time: rotation, x/y stretch, or shear. Both signs were included.",
        "- Top-1/top-5/top-20 mean that at least one repeatable candidate appears in the first 1/5/20 similarity ranks. The geometric stage fits the top-1 pairs with random sample consensus (RANSAC) on WebGPU, then counts estimates whose analytic displacement error is at most 5 pixels.",
        "",
        "## Baseline versus exact Jacobian",
        "",
        "Counts below use the 1,002 queries with a target candidate within 5 pixels as the retrieval denominator. The all-query population is 1,536.",
        "",
        "| Descriptor | Baseline top-1 | Baseline top-5 | Exact-Jacobian top-1 | Exact-Jacobian top-5 | Exact-Jacobian top-20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in METHODS.items():
        baseline = {
            key: sum(case["baseline_retrieval"][method][field] for case in cases)
            for key, field in (("top1", "top1_correct_le5px"), ("top5", "top5_retrieval"), ("top20", "top20_retrieval"))
        }
        compensated = aggregate_retrieval(exact, method)
        lines.append(
            f"| {label} | {baseline['top1']}/{all_repeatable} ({pct(baseline['top1'], all_repeatable)}) | "
            f"{baseline['top5']}/{all_repeatable} ({pct(baseline['top5'], all_repeatable)}) | "
            f"{compensated['top1']}/{compensated['repeatable']} ({pct(compensated['top1'], compensated['repeatable'])}) | "
            f"{compensated['top5']}/{compensated['repeatable']} ({pct(compensated['top5'], compensated['repeatable'])}) | "
            f"{compensated['top20']}/{compensated['repeatable']} ({pct(compensated['top20'], compensated['repeatable'])}) |"
        )
    cnn_exact = aggregate_retrieval(exact, "dic_cnn")
    lines.extend(
        [
            "",
            f"For the exact-Jacobian CNN arm, WebGPU RANSAC accepted {cnn_exact['correct_accepted']}/{cnn_exact['accepted']} points within 5 pixels, including {cnn_exact['false_accepted']} false acceptances. This is a global-affine synthetic case and should not be extrapolated to heterogeneous fields.",
            "",
            "## Sensitivity to Jacobian error",
            "",
            "![Top-1, top-5, and top-20 retrieval sensitivity for independent Jacobian errors.](jacobian_sensitivity.svg)",
            "",
            "The table reports pooled CNN retrieval among the same repeatable queries (n=1,002). Candidate support validity can vary slightly with the hypothesized Jacobian because transformed patches near the image border are invalidated; candidate coordinates themselves do not change.",
            "",
            "| Perturbed parameter | Error | Top-1 | Top-5 | Top-20 | RANSAC correct / accepted | False accepted |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    condition_keys = {
        (row["parameter"], row["delta"])
        for case in cases
        for row in case["conditions"]
        if row.get("parameter") != "exact" and "retrieval" in row
    }
    axis_order = {"angle_deg": 0, "stretch_x_percent": 1, "stretch_y_percent": 2, "shear": 3}
    for parameter, delta in sorted(
        condition_keys, key=lambda item: (axis_order[item[0]], item[1])
    ):
        rows = condition_rows(payload, parameter, delta)
        aggregate = aggregate_retrieval(rows, "dic_cnn")
        denominator = aggregate["repeatable"]
        parameter_label = parameter.replace("_", " ")
        if parameter == "shear":
            error = f"{delta:+.2f}"
        elif parameter == "angle_deg":
            error = f"{delta:+.0f}°"
        else:
            error = f"{delta:+.0f}%"
        lines.append(
            f"| {parameter_label} | {error} | {aggregate['top1']}/{denominator} ({pct(aggregate['top1'], denominator)}) | "
            f"{aggregate['top5']}/{denominator} ({pct(aggregate['top5'], denominator)}) | "
            f"{aggregate['top20']}/{denominator} ({pct(aggregate['top20'], denominator)}) | "
            f"{aggregate['correct_accepted']}/{aggregate['accepted']} | {aggregate['false_accepted']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: one-axis errors up to 5° rotation, 10% axial stretch, or 0.10 shear retain about 80% top-1 retrieval. At ±10° rotation, ±20% stretch, or ±0.20 shear, top-1 falls to roughly 55–68%, while top-5 remains higher. This supports testing a coarse-to-fine hypothesis strategy, but it does not set a bank spacing: simultaneous errors, non-affine variation, and image-derived hypothesis quality have not been tested.",
            "",
            "### Per-texture exact-Jacobian result",
            "",
            "| Texture seed | Queries | Target candidates | Repeatable | Baseline CNN top-1 | Exact-Jacobian CNN top-1 | Top-5 | RANSAC correct / accepted | False accepted |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case, row in exact:
        base = case["baseline_retrieval"]["dic_cnn"]
        result = row["retrieval"]["dic_cnn"]
        fit = row["gpu_affine_consensus"]["dic_cnn"]
        lines.append(
            f"| {case['texture_seed']} | {case['source_query_count']} | {case['target_candidate_count']} | "
            f"{result['repeatable_le5px']} | {base['top1_correct_le5px']} | {result['top1_correct_le5px']} | "
            f"{result['top5_retrieval']} | {fit['correct_accepted_le5px']}/{fit['accepted_by_model']} | {fit['false_accepted']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime and implementation limits",
            "",
            "| Stage | Median per texture | What is included |",
            "|---|---:|---|",
        ]
    )
    exact_rows = [row for _, row in exact]
    timing_rows = [
        ("Paired-image DIC detection", [case["detection_s"] for case in cases], "Both native 4K images; WebGPU detector"),
        ("Affine patch sampling", [row["affine_patch_sampling_s"] for row in exact_rows], "Candidate patch resampling; currently CPU-side"),
        ("Descriptor atlas", [row["descriptor_atlas_s"] for row in exact_rows], "Atlas build plus target ZNCC/HORG/CNN descriptor computation"),
        ("Affine RANSAC", [row["gpu_affine_consensus"]["dic_cnn"]["ransac_s"] for row in exact_rows], "CNN top-1 pairs; WebGPU fit and readback"),
    ]
    for stage, values, description in timing_rows:
        lines.append(f"| {stage} | {statistics.median(values):.3f} s | {description} |")
    lines.extend(
        [
            "",
            "These are stage timings, not an end-to-end production latency. Similarity-matrix multiplication and ranking run in NumPy on the central processing unit (CPU) and are not separately timed. Each affine hypothesis currently requires candidate patch resampling and descriptor recomputation; the measured descriptor-atlas stage alone is about 1.87 seconds per hypothesis for this workload. A dense bank is therefore not practical with this prototype. The atlas is a diagnostic batching device and is larger than the input image; it is not the intended production data path. DeDoDe (Detect, Don’t Describe — Describe, Don’t Detect)-B is omitted from this support-level sweep because its current descriptor API consumes a fixed-shape dense image map; the previous full-image oracle report retains that descriptor comparison.",
            "",
            "## Conclusion",
            "",
            "The translation-free oracle test confirms that explicit affine support compensation can restore the learned descriptor's global retrieval ranking. Moderate Jacobian error is tolerated in these synthetic global-affine cases, but performance degrades at larger one-axis errors. The next implementation should not be a dense all-image bank using this atlas path. The evidence favors a small set of image-derived coarse hypotheses followed by local refinement, or a learned/analytic transformation of compact descriptors. Before selecting a bank spacing, test combined parameter errors and heterogeneous fields, then profile a candidate-centred WebGPU sampler that avoids image-sized atlases.",
            "",
            "## Reproduction",
            "",
            "Run from the repository root after installing the recorded descriptor weights:",
            "",
            "```console",
            "PYTHONPATH=python/benchmarks:python/src python3 \\",
            "  python/benchmarks/benchmark_affine_candidate_patch_sensitivity.py \\",
            "  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \\",
            "  --output results/descriptors/affine_candidate_sensitivity_20261108/oracle_sensitivity_4k_8192.json \\",
            "  --width 3840 --height 2160 --candidate-budget 8192 \\",
            "  --global-queries 512 --textures 3 --families affine --sensitivity",
            "```",
            "",
            f"The raw JavaScript Object Notation (JSON) records per-texture candidate counts, timing stages, all signed perturbations, the WebGPU adapter, model-weight 256-bit Secure Hash Algorithm (SHA-256) fingerprint `{protocol['cnn_weights_sha256']}`, and every retrieval and consensus count. The diagnostic uses synthetic truth by design; the query and target candidates are not filtered using analytical visibility masks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(payload), encoding="utf-8")
    args.output.with_name("jacobian_sensitivity.svg").write_text(
        render_sensitivity_svg(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
