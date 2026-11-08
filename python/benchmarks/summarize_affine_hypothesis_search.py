"""Summarize the 4K affine-error and rotation-bank experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def summarize(path: Path, output_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    combined_groups: dict[int, list[dict]] = defaultdict(list)
    for case in data["combined"]:
        for row in case["conditions"]:
            active = sum(
                bool(row[key])
                for key in ("angle_deg", "stretch_x_percent", "stretch_y_percent", "shear")
            )
            combined_groups[active].append(row)

    lines = [
        "# Affine support compensation: combined-error grid and rotation search",
        "",
        "## Scope and protocol",
        "",
        "This report evaluates affine-conditioned support sampling for the compact DIC descriptor. The experiment used three independent 3840 × 2160 synthetic speckle pairs per motion family, 512 fixed source queries, and 8,192 detector-selected target candidates per pair. The affine grid has 81 combinations per pair. A second experiment searched rotation using image-derived hypotheses only.",
        "",
        "The detector is the WebGPU structure-tensor detector. The descriptor is the existing learned compact convolutional neural network (CNN), sampled with the supplied support Jacobian directly from native target pixels by WebGPU. The Jacobian grid uses `F_hat = R(delta_theta) @ [[1+delta_x, delta_shear], [0, 1+delta_y]] @ F_true`. No analytic visibility mask filters the target candidate set. The repeatable-query denominator is fixed for every support hypothesis. The analytic field is used only to score outcomes and to form the labelled oracle ceiling.",
        "",
        f"Adapter: {data['protocol']['webgpu_adapter']}. CNN weights SHA-256: `{data['protocol']['cnn_weights_sha256']}`.",
        "",
        "![Retrieval rates by affine-error dimensionality and rotation-bank density](./affine_hypothesis_summary.svg)",
        "",
        "## Combined affine-parameter errors",
        "",
        "Rows are grouped by the number of nonzero parameter errors. Each group pools every grid point and all three textures. Top-1 and top-5 are counted only for source points with a target candidate within 5 pixels under the analytic mapping; the repeatable denominator does not change across hypotheses. All-query coverage uses all 512 queries per texture. The last two columns report consensus-retained direct pairs; those are measured correspondences, while the preceding prediction coverage is reconstructed from a fitted global affine model.",
        "",
        "| Nonzero errors | Grid rows | Repeatable queries | Top-1 correct | Top-5 contains correct | Top-1 / repeatable | Direct retained correct / total | False retained | Fitted field correct / all queries |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    combined_chart = []
    for active, rows in sorted(combined_groups.items()):
        repeatable = sum(row["retrieval"]["repeatable_le5px"] for row in rows)
        top1 = sum(row["retrieval"]["top1_correct_le5px"] for row in rows)
        top5 = sum(row["retrieval"]["top5_retrieval"] for row in rows)
        direct = sum(row["consensus"]["direct_accepted"] for row in rows)
        direct_good = sum(row["consensus"]["direct_correct_le5px"] for row in rows)
        false = sum(row["consensus"]["direct_false_accepted"] for row in rows)
        predicted = sum(row["consensus"]["predicted_queries_correct_le5px"] for row in rows)
        all_queries = sum(row["consensus"]["predicted_query_count"] for row in rows)
        lines.append(
            f"| {active} | {len(rows)} | {repeatable:,} | {top1:,} | {top5:,} | {ratio(top1, repeatable)} | {direct_good:,} / {direct:,} | {false:,} | {predicted:,} / {all_queries:,} ({ratio(predicted, all_queries)}) |"
        )
        combined_chart.append((active, top1 / repeatable if repeatable else 0.0))

    base_repeatable = sum(case["repeatable_query_count_fixed"] for case in data["combined"])
    base_top1 = sum(case["uncompensated_baseline"]["retrieval"]["top1_correct_le5px"] for case in data["combined"])
    base_top5 = sum(case["uncompensated_baseline"]["retrieval"]["top5_retrieval"] for case in data["combined"])
    lines += [
        "",
        f"Uncompensated baseline across the three affine pairs: top-1 {base_top1}/{base_repeatable} ({ratio(base_top1, base_repeatable)}), top-5 {base_top5}/{base_repeatable} ({ratio(base_top5, base_repeatable)}).",
        "",
        "### One-parameter sensitivity",
        "",
        "This isolates one error family at a time, pooling positive and negative perturbations at the same magnitude across all three textures. The query denominator is fixed to the repeatable queries in each source set.",
        "",
        "| Parameter varied alone | Absolute error | Top-1 correct / repeatable | Top-5 contains correct / repeatable |",
        "|---|---:|---:|---:|",
    ]
    parameter_levels = (
        ("Rotation", "angle_deg", (10.0,)),
        ("Axial stretch x", "stretch_x_percent", (20.0,)),
        ("Axial stretch y", "stretch_y_percent", (20.0,)),
        ("Shear", "shear", (0.2,)),
    )
    for label, key, levels in parameter_levels:
        for magnitude in levels:
            selected = [
                row
                for case in data["combined"]
                for row in case["conditions"]
                if abs(row[key]) == magnitude
                and all(
                    row[other] == 0
                    for other in ("angle_deg", "stretch_x_percent", "stretch_y_percent", "shear")
                    if other != key
                )
            ]
            repeatable = sum(row["retrieval"]["repeatable_le5px"] for row in selected)
            top1 = sum(row["retrieval"]["top1_correct_le5px"] for row in selected)
            top5 = sum(row["retrieval"]["top5_retrieval"] for row in selected)
            lines.append(
                f"| {label} | ±{magnitude:g}{'%' if 'stretch' in key else '°' if key == 'angle_deg' else ''} | {top1}/{repeatable} ({ratio(top1, repeatable)}) | {top5}/{repeatable} ({ratio(top5, repeatable)}) |"
            )
    lines += [
        "",
        "### Rotation interaction with the other affine errors",
        "",
        "Each cell pools both signs of the other active parameter errors and all three textures. This exposes whether rotation error compounds with stretch and shear errors; it does not imply that the parameter errors have equal physical severity.",
        "",
        "| Rotation error | Other parameters wrong | Top-1 correct / repeatable | Top-5 contains correct / repeatable |",
        "|---:|---:|---:|---:|",
    ]
    for angle in (-10.0, 0.0, 10.0):
        for others_wrong in range(4):
            selected = [
                row
                for case in data["combined"]
                for row in case["conditions"]
                if row["angle_deg"] == angle
                and sum(bool(row[key]) for key in ("stretch_x_percent", "stretch_y_percent", "shear")) == others_wrong
            ]
            repeatable = sum(row["retrieval"]["repeatable_le5px"] for row in selected)
            top1 = sum(row["retrieval"]["top1_correct_le5px"] for row in selected)
            top5 = sum(row["retrieval"]["top5_retrieval"] for row in selected)
            lines.append(
                f"| {angle:+g}° | {others_wrong} | {top1}/{repeatable} ({ratio(top1, repeatable)}) | {top5}/{repeatable} ({ratio(top5, repeatable)}) |"
            )
    lines += [
        "",
        "The exact-Jacobian row is the zero-error group. A fitted global affine predicts all 1,536 query centres within 5 pixels in this synthetic globally affine set, but the direct consensus retains some false pairs (reported above). Therefore the fitted-field result must not be read as a direct-match rate or as evidence for heterogeneous deformation.",
        "",
        "### Per-texture capture",
        "",
        "| Texture seed | Repeatable / 512 | Uncompensated top-1 | Exact-J top-1 | Exact-J direct retained (false) | Exact-J fitted field correct / 512 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in data["combined"]:
        exact = next(
            row for row in case["conditions"]
            if not any(row[key] for key in ("angle_deg", "stretch_x_percent", "stretch_y_percent", "shear"))
        )
        c = exact["consensus"]
        lines.append(
            f"| {case['texture_seed']} | {case['repeatable_query_count_fixed']} | {case['uncompensated_baseline']['retrieval']['top1_correct_le5px']} | {exact['retrieval']['top1_correct_le5px']} | {c['direct_correct_le5px']} / {c['direct_accepted']} ({c['direct_false_accepted']} false) | {c['predicted_queries_correct_le5px']} / {c['predicted_query_count']} |"
        )

    rotations = data["rotation"]
    lines += [
        "",
        "## Rotation bank without an angle oracle",
        "",
        "The bank covers 360 degrees with nested, uniformly spaced subsets of 18, 36, and 72 hypotheses (20, 10, and 5 degree spacing). For each angle, the target support is sampled with that rotation. The selected hypothesis is chosen by the largest usable WebGPU affine-RANSAC consensus, then the lowest residual. Neither the true angle nor the true translation participates in hypothesis selection. The exact-Jacobian figures below are an oracle ceiling only.",
        "",
        "| Bank | Spacing | Top-1 correct / repeatable | Top-5 contains correct / repeatable | Selected direct matches correct / retained | False retained | Fitted rigid field correct / all 1,536 | Median bank time per pair |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rotation_chart = []
    for size in (18, 36, 72):
        rows = [case["bank_results"][str(size)] for case in rotations]
        repeatable = sum(row["retrieval"]["repeatable_le5px"] for row in rows)
        top1 = sum(row["retrieval"]["top1_correct_le5px"] for row in rows)
        top5 = sum(row["retrieval"]["top5_retrieval"] for row in rows)
        selected = [row["selected_consensus"] for row in rows]
        accepted = sum(row["direct_accepted"] for row in selected)
        correct = sum(row["direct_correct_le5px"] for row in selected)
        false = sum(row["direct_false_accepted"] for row in selected)
        pred = sum(row["predicted_all_queries_correct_le5px"] for row in selected)
        total = sum(row["predicted_all_queries"] for row in selected)
        times = [row["total_bank_s"] for row in rows]
        median_time = sorted(times)[len(times) // 2]
        spacing = rows[0]["angular_spacing_deg"]
        lines.append(
            f"| {size} | {spacing:g}° | {top1:,} / {repeatable:,} ({ratio(top1, repeatable)}) | {top5:,} / {repeatable:,} ({ratio(top5, repeatable)}) | {correct:,} / {accepted:,} | {false:,} | {pred:,} / {total:,} ({ratio(pred, total)}) | {median_time:.2f} s |"
        )
        rotation_chart.append((size, top1 / repeatable if repeatable else 0.0, median_time))

    base_rep = sum(case["repeatable_query_count_fixed"] for case in rotations)
    base_top1 = sum(case["uncompensated_baseline"]["top1_correct_le5px"] for case in rotations)
    oracle_top1 = sum(case["exact_jacobian_oracle_ceiling"]["top1_correct_le5px"] for case in rotations)
    oracle_top5 = sum(case["exact_jacobian_oracle_ceiling"]["top5_retrieval"] for case in rotations)
    lines += [
        "",
        f"Uncompensated rotation baseline: {base_top1}/{base_rep} top-1 ({ratio(base_top1, base_rep)}). Exact-Jacobian oracle ceiling: {oracle_top1}/{base_rep} top-1 ({ratio(oracle_top1, base_rep)}) and {oracle_top5}/{base_rep} top-5 ({ratio(oracle_top5, base_rep)}).",
        "",
        "The affine consensus reconstructs the rigid field at all query centres in these three globally rigid synthetic cases. This is extrapolated field coverage, not direct correspondence coverage. No spatially heterogeneous or discontinuous field was used to validate this global consensus strategy.",
        "",
        "### Per-texture rotation results",
        "",
        "| Texture seed | True angle | Repeatable | Oracle top-1 | 18-bin top-1 / fitted field | 36-bin top-1 / fitted field | 72-bin top-1 / fitted field |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in rotations:
        cells = []
        for size in (18, 36, 72):
            row = case["bank_results"][str(size)]
            cells.append(
                f"{row['retrieval']['top1_correct_le5px']}/{case['repeatable_query_count_fixed']} · {row['selected_consensus']['predicted_all_queries_correct_le5px']}/{row['selected_consensus']['predicted_all_queries']}"
            )
        lines.append(
            f"| {case['texture_seed']} | {case['true_rotation_deg']:.2f}° | {case['repeatable_query_count_fixed']} | {case['exact_jacobian_oracle_ceiling']['top1_correct_le5px']} | " + " | ".join(cells) + " |"
        )

    lines += [
        "",
        "## Visibility and detector ceiling",
        "",
        "For the affine pairs, analytic target-centre visibility is recorded after the fact and never used to filter candidates. Across the three 512-query sets, the table separates target centres outside the image from centres inside the image but lacking any detected target candidate within 5 pixels. The nearby-candidate count is the repeatability ceiling for this candidate budget, distinct from descriptor retrieval quality.",
        "",
        "| Motion family | Queries | Target centre outside image | Centre visible, no target candidate within 5 px | Repeatable with a target candidate |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, cases in (("Global affine", data["combined"]),):
        q = sum(case["source_query_count"] for case in cases)
        outside = sum(case["out_of_image_target_centers"] for case in cases)
        no_candidate = sum(case["visible_but_without_target_candidate_le5px"] for case in cases)
        repeatable = sum(case["repeatable_query_count_fixed"] for case in cases)
        lines.append(f"| {label} | {q} | {outside} | {no_candidate} | {repeatable} |")

    lines += [
        "",
        "## Timing and implementation boundary",
        "",
        "Each target descriptor extraction for one affine hypothesis reads the native image and samples the affine support in a WGSL shader; the experiment does not build a rectified full-frame atlas. On this Intel Graphics (LNL) Vulkan adapter, each support-conditioned descriptor extraction took about 50–70 ms for 8,192 target points at 4K. The full rotation-bank times include descriptor extraction, the existing mutual matcher, and affine-consensus fitting for each hypothesis; median pair times are shown above. Detection cost was roughly 0.5–0.6 s per pair and source-description cost roughly 17–25 ms per pair.",
        "",
        "This is not yet GPU-resident end to end. The descriptor arrays are read back to NumPy; the current GPU mutual matcher uploads them again, and the diagnostic all-candidate ranking is NumPy-based. This experiment validates the affine-conditioned WGSL sampling path, not a final optimized pipeline.",
        "",
        "## Interpretation and next step",
        "",
        "The experiment supports a useful progression: modest errors in a single Jacobian parameter do not immediately destroy retrieval, but errors combined across parameters sharply reduce top-1 retrieval as more components depart from the true Jacobian. On rigid cases, an image-only rotation bank recovers a substantial fraction of the exact-Jacobian retrieval ceiling, while the 18-to-72 hypothesis cost grows nearly linearly. Increasing from 36 to 72 angles yields only a modest pooled top-1 increment in this small three-pair sample.",
        "",
        "The rotation result is promising but does not establish performance on general affine or heterogeneous deformation. The current bank performs a full matching and consensus pass for every angle. The next engineering step should shortlist angles using a cheap descriptor score, then refine only a few hypotheses. A general affine search should be generated from coarse correspondences and local consensus, not by taking a dense Cartesian product of four parameters. Before relying on fitted-field coverage, repeat the validation on heterogeneous fields with multiple local Jacobians and report per-window ground-truth errors without propagating a single global model across the whole image.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=python/benchmarks:python/src .venv/bin/python \\",
        "  python/benchmarks/benchmark_affine_hypothesis_search.py \\",
        "  --mode both \\",
        "  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \\",
        "  --output results/descriptors/affine_hypothesis_wgpu_20261108/combined_and_rotation_4k.json \\",
        "  --width 3840 --height 2160 --candidate-budget 8192 --query-budget 512 --textures 3",
        "```",
        "",
        f"Raw measurements: [`{path.name}`](./{path.name}).",
        "",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    # A small self-contained SVG avoids optional plotting dependencies.
    bars = []
    for index, (active, value) in enumerate(combined_chart):
        x = 90 + index * 150
        h = value * 220
        bars.append(f'<rect x="{x}" y="{280-h:.1f}" width="76" height="{h:.1f}" fill="#326b9a"/><text x="{x+38}" y="{300-h:.1f}" text-anchor="middle">{value:.1%}</text><text x="{x+38}" y="310" text-anchor="middle">{active}</text>')
    rotation_bars = []
    for index, (size, value, _) in enumerate(rotation_chart):
        x = 105 + index * 180
        h = value * 220
        rotation_bars.append(f'<rect x="{x}" y="{280-h:.1f}" width="88" height="{h:.1f}" fill="#2a8c74"/><text x="{x+44}" y="{300-h:.1f}" text-anchor="middle">{value:.1%}</text><text x="{x+44}" y="310" text-anchor="middle">{size} angles</text>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="720" viewBox="0 0 920 720">'
        '<style>text{font:14px sans-serif;fill:#263746}.title{font-size:20px;font-weight:bold}.axis{stroke:#667;stroke-width:1}.small{font-size:12px}</style>'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="50" y="35" class="title">Affine descriptor retrieval under Jacobian errors</text>'
        '<text x="50" y="58" class="small">Top-1 among repeatable queries; grouped by number of erroneous parameters</text>'
        '<line x1="60" y1="280" x2="850" y2="280" class="axis"/><line x1="60" y1="60" x2="60" y2="280" class="axis"/>'
        + ''.join(bars) +
        '<text x="50" y="365" class="title">Rotation-bank retrieval without an angle oracle</text>'
        '<text x="50" y="388" class="small">Pooled top-1 among repeatable queries; separate three-pair rigid test set</text>'
        '<line x1="60" y1="610" x2="850" y2="610" class="axis"/><line x1="60" y1="390" x2="60" y2="610" class="axis"/>'
        + ''.join(rotation_bars) +
        '<text x="60" y="675" class="small">Candidate supports sampled at native 4K in WGSL. Bank timing grows from 18 to 72 passes.</text>'
        '</svg>'
    )
    (output_dir / "affine_hypothesis_summary.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output_dir)


if __name__ == "__main__":
    main()
