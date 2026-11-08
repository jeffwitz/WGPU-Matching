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
        "The bank covers 360 degrees with nested, uniformly spaced subsets of 18, 36, and 72 hypotheses (20, 10, and 5 degree spacing). For each angle, the target support is sampled with that rotation. The per-query max-bank retrieval metric keeps each query's best score across all angles, so different queries can implicitly use different angles; it is an upper-bound diagnostic, not one selected-bank output. The usable WebGPU affine-RANSAC consensus selects a single angle using the largest inlier count, then lowest residual. Its selected-angle top-1 column reports descriptor retrieval at that one angle. Neither the true angle nor true translation participates in this selection. The exact-Jacobian figures below are an oracle ceiling only.",
        "",
        "| Bank | Spacing | Per-query max-bank top-1 / repeatable | Per-query max-bank top-5 / repeatable | Selected-angle top-1 / repeatable | Selected direct matches correct / retained | False retained | Fitted rigid field correct / all 1,536 | Median bank time per pair |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rotation_chart = []
    for size in (18, 36, 72):
        rows = [case["bank_results"][str(size)] for case in rotations]
        repeatable = sum(row["retrieval"]["repeatable_le5px"] for row in rows)
        top1 = sum(row["retrieval"]["top1_correct_le5px"] for row in rows)
        top5 = sum(row["retrieval"]["top5_retrieval"] for row in rows)
        selected_top1 = sum(row["selected_angle_retrieval"]["top1_correct_le5px"] for row in rows)
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
            f"| {size} | {spacing:g}° | {top1:,} / {repeatable:,} ({ratio(top1, repeatable)}) | {top5:,} / {repeatable:,} ({ratio(top5, repeatable)}) | {selected_top1:,} / {repeatable:,} ({ratio(selected_top1, repeatable)}) | {correct:,} / {accepted:,} | {false:,} | {pred:,} / {total:,} ({ratio(pred, total)}) | {median_time:.2f} s |"
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

    coarse_cases = data.get("coarse_to_fine", [])
    if coarse_cases:
        direct = sum(case["selected_full_consensus"].get("direct_correct_le5px", 0) for case in coarse_cases)
        accepted = sum(case["selected_full_consensus"].get("direct_accepted", 0) for case in coarse_cases)
        false = sum(case["selected_full_consensus"].get("direct_false_accepted", 0) for case in coarse_cases)
        repeatable = sum(case["repeatable_queries"] for case in coarse_cases)
        predicted = sum(case["selected_full_consensus"].get("predicted_queries_correct_le5px", 0) for case in coarse_cases)
        predicted_total = sum(case["selected_full_consensus"].get("predicted_query_count", 0) for case in coarse_cases)
        search_times = [case["rotation_search_s"] for case in coarse_cases]
        pipeline_times = [case["timing_s"]["total_pipeline"] for case in coarse_cases]
        lines += [
            "",
            "## Coarse-to-fine rotation search",
            "",
            "This pipeline selects geometry at three stages: 18 coarse angles on 128 spatially distributed source queries and 2,048 spatially distributed target candidates; local 5-degree refinement around the two strongest coarse consensuses; then full verification of the two strongest remaining angles on 512 queries and 8,192 candidates. Each native image and the CNN weights are uploaded once; point-set sessions reuse them and reuse their convolution workspaces across Jacobian hypotheses. Descriptor vectors remain on the device through WGSL reciprocal matching; only compact reciprocal indices and scores are read before geometric consensus. Selection uses observed image matches and geometric consensus; analytic motion is consulted only for scoring and reported errors.",
            "",
            "| Texture seed | Repeatable / queries | Selected angle | Angle error | Direct correct / retained | False retained | Fitted field correct / 512 | Search only | Pipeline including detection and source description |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for case in coarse_cases:
            consensus = case["selected_full_consensus"]
            lines.append(
                f"| {case['texture_seed']} | {case['repeatable_queries']} / {case['queries']} | {case['selected_angle_deg']:.1f}° | {case['selected_angle_error_deg']:+.2f}° | {consensus.get('direct_correct_le5px', 0)} / {consensus.get('direct_accepted', 0)} | {consensus.get('direct_false_accepted', 0)} | {consensus.get('predicted_queries_correct_le5px', 0)} / {consensus.get('predicted_query_count', 0)} | {case['rotation_search_s']:.3f} s | {case['timing_s']['total_pipeline']:.3f} s |"
            )
        med_search = sorted(search_times)[len(search_times) // 2]
        med_pipeline = sorted(pipeline_times)[len(pipeline_times) // 2]
        lines += [
            "",
            f"Pooled coarse-to-fine direct recovery is {direct}/{repeatable} repeatable queries ({ratio(direct, repeatable)}), with {direct}/{accepted} retained matches correct and {false} false acceptances. A global rigid consensus predicts {predicted}/{predicted_total} query centres within 5 pixels in these globally rigid cases; this is extrapolated field output, not direct coverage. Median search-only time is {med_search:.3f} s and median total time including both-image detection and source descriptor preparation is {med_pipeline:.3f} s.",
            "",
            "The comparison uses the same three textures and candidate/query budgets as the full rotation bank. Search-only timing includes target image/model preparation, coarse/refinement sessions, descriptor extraction, matching, and consensus, but excludes feature detection and source description. Total time adds both-image detection and source-description stages. Neither time includes synthetic-image generation or external file decoding. Each texture seed was run once in this pass; the reported median is across three different pairs, not repeated calls on one pair.",
        ]

        previous = data.get("coarse_to_fine_host_reference", [])
        if previous:
            previous_by_seed = {case["texture_seed"]: case for case in previous}
            paired = [case for case in coarse_cases if case["texture_seed"] in previous_by_seed]
            previous_search = [previous_by_seed[case["texture_seed"]]["rotation_search_s"] for case in paired]
            previous_total = [previous_by_seed[case["texture_seed"]]["timing_s"]["total_pipeline"] for case in paired]
            prior_search = sorted(previous_search)[len(previous_search) // 2]
            prior_total = sorted(previous_total)[len(previous_total) // 2]
            search_change = (med_search / prior_search - 1.0) * 100.0
            total_change = (med_pipeline / prior_total - 1.0) * 100.0
            same_direct = all(
                case["selected_full_consensus"].get("direct_correct_le5px")
                == previous_by_seed[case["texture_seed"]]["selected_full_consensus"].get("direct_correct_le5px")
                and case["selected_full_consensus"].get("direct_false_accepted")
                == previous_by_seed[case["texture_seed"]]["selected_full_consensus"].get("direct_false_accepted")
                for case in paired
            )
            lines += [
                "",
                "### Resident versus host descriptor path",
                "",
                f"Against the earlier coarse-to-fine run on the same seeds and settings, median search time changed from {prior_search:.3f} s to {med_search:.3f} s ({search_change:+.1f}%), while complete-pair time changed from {prior_total:.3f} s to {med_pipeline:.3f} s ({total_change:+.1f}%). Direct accepted-match counts and false-accept counts are identical for all paired cases: {same_direct}. This isolates descriptor residency and reusable image/model/work buffers; it does not compare coarse-to-fine against an exhaustive bank as if their work were equivalent.",
                "",
                "The exhaustive 36-angle result is a separate quality reference: it evaluates full query/candidate sets at every angle, while coarse-to-fine uses smaller sets for hypothesis screening and full-verifies only the short final list.",
            ]

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
        "In the exhaustive rotation-bank diagnostic, descriptor arrays are still read back to NumPy for its per-query max-bank ranking, and the ordinary matcher path uploads them again. The separate coarse-to-fine path avoids that round trip: native images, model weights, convolution workspaces and descriptor vectors remain on WebGPU through reciprocal matching. Compact indices and scores still cross to the host for geometric consensus selection. This is device-resident image/descriptor extraction and reciprocal matching, not an end-to-end device-only pipeline.",
        "",
        "## Interpretation and next step",
        "",
        "The experiment supports a useful progression: modest errors in a single Jacobian parameter do not immediately destroy retrieval, but errors combined across parameters sharply reduce top-1 retrieval as more components depart from the true Jacobian. On rigid cases, an image-only rotation bank recovers a substantial fraction of the exact-Jacobian retrieval ceiling, while the 18-to-72 hypothesis cost grows nearly linearly. Increasing from 36 to 72 angles yields only a modest pooled top-1 increment in this small three-pair sample.",
        "",
        "The rotation result is promising but does not establish performance on general affine or heterogeneous deformation, and it does not measure IC-GN convergence. The coarse-to-fine run selects the rigid rotation without an angle oracle, but its final global affine model is tested only on globally rigid synthetic fields. The next step is to replace one global model with competing local models on heterogeneous fields and report direct observations separately from local interpolation and extrapolation. Do not propagate a model across a physical discontinuity without independent evidence.",
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
        "Supplemental raw measurements are `rotation_bank_reuse_4k.json` and `coarse_to_fine_4k.json`.",
        "",
    ]
    if data.get("coarse_to_fine_host_reference"):
        lines += [
            "### Reproduce the resident comparison",
            "",
            "```bash",
            "PYTHONPATH=python/benchmarks:python/src .venv/bin/python \\",
            "  python/benchmarks/benchmark_affine_hypothesis_search.py \\",
            "  --mode coarse --coarse-queries 128 --coarse-candidates 2048 \\",
            "  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \\",
            "  --output results/descriptors/affine_hypothesis_wgpu_20261108/coarse_to_fine_4k_resident.json \\",
            "  --width 3840 --height 2160 --candidate-budget 8192 --query-budget 512 --textures 3",
            "PYTHONPATH=python/benchmarks:python/src .venv/bin/python \\",
            "  python/benchmarks/summarize_affine_hypothesis_search.py \\",
            "  results/descriptors/affine_hypothesis_wgpu_20261108/affine_search_4k_optimized.json \\",
            "  --coarse-to-fine-resident results/descriptors/affine_hypothesis_wgpu_20261108/coarse_to_fine_4k_resident.json \\",
            "  --output-dir results/descriptors/affine_hypothesis_resident_20261108",
            "```",
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
    parser.add_argument("--coarse-to-fine-resident", type=Path)
    parser.add_argument("--coarse-to-fine-host-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if args.coarse_to_fine_resident:
        resident = json.loads(args.coarse_to_fine_resident.read_text(encoding="utf-8"))
        if not args.coarse_to_fine_host_reference:
            data["coarse_to_fine_host_reference"] = data.get("coarse_to_fine", [])
        data["coarse_to_fine"] = resident["coarse_to_fine"]
        data["protocol"]["rotation_bank_residency"] = resident["protocol"][
            "rotation_bank_residency"
        ]
    if args.coarse_to_fine_host_reference:
        reference = json.loads(args.coarse_to_fine_host_reference.read_text(encoding="utf-8"))
        data["coarse_to_fine_host_reference"] = reference["coarse_to_fine"]
    merged_input = args.output_dir / "affine_search_4k_resident.json"
    if args.coarse_to_fine_resident or args.coarse_to_fine_host_reference:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        merged_input.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        args.input = merged_input
    summarize(args.input, args.output_dir)


if __name__ == "__main__":
    main()
