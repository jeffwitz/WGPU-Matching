"""Run a paired ablation of image-derived affine prefit, NDF residual, and NCC."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

VARIANTS = {
    "affine_only": {"field": "none", "ncc": 0},
    "affine_ncc": {"field": "none", "ncc": 8},
    "affine_ndf": {"field": "ndf", "ncc": 0},
    "affine_ndf_ncc": {"field": "ndf", "ncc": 8},
}


def _load_window_files(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    loaded = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as data:
            loaded[path.name] = {name: data[name].copy() for name in data.files}
    return loaded


def _window_and_region_metrics(windows: dict[str, np.ndarray], width: int, height: int):
    visible = windows["visible"].astype(bool)
    correct = windows["icgn_correct_lt2px"].astype(bool) & visible
    initial = windows["initial_parameters"].astype(np.float64)
    truth = windows["truth_parameters"].astype(np.float64)
    delta = initial - truth
    translation_error = np.linalg.norm(delta[:, [0, 3]], axis=1)
    gradient_error = np.linalg.norm(delta[:, [1, 2, 4, 5]], axis=1)
    centres = windows["centres_xy_native"]
    x_bin = np.clip((centres[:, 0] / width * 3).astype(int), 0, 2)
    y_bin = np.clip((centres[:, 1] / height * 3).astype(int), 0, 2)
    regions = {}
    for row in range(3):
        for col in range(3):
            region = visible & (x_bin == col) & (y_bin == row)
            regions[f"r{row + 1}c{col + 1}"] = {
                "visible": int(region.sum()),
                "correct": int((correct & region).sum()),
            }
    return {
        "visible": int(visible.sum()),
        "correct": int(correct.sum()),
        "coverage": float(correct.sum() / max(visible.sum(), 1)),
        "initial_translation_rmse_px": float(np.sqrt(np.mean(translation_error[visible] ** 2))),
        "initial_translation_median_px": float(np.median(translation_error[visible])),
        "initial_gradient_rmse": float(np.sqrt(np.mean(gradient_error[visible] ** 2))),
        "initial_parameter_rmse": {
            name: float(np.sqrt(np.mean(delta[visible, index] ** 2)))
            for index, name in enumerate(("u", "du_dx", "du_dy", "v", "dv_dx", "dv_dy"))
        },
        "regions_3x3": regions,
        "correct_mask": correct,
        "visible_mask": visible,
        "centres_xy_native": centres,
    }


def _paired_recovery(candidate: dict, reference: dict, width: int, height: int):
    valid = candidate["visible_mask"] & reference["visible_mask"]
    candidate_only = candidate["correct_mask"] & ~reference["correct_mask"] & valid
    reference_only = reference["correct_mask"] & ~candidate["correct_mask"] & valid
    centres = candidate["centres_xy_native"]
    x_bin = np.clip((centres[:, 0] / width * 3).astype(int), 0, 2)
    y_bin = np.clip((centres[:, 1] / height * 3).astype(int), 0, 2)
    regions = {}
    for row in range(3):
        for col in range(3):
            region = valid & (x_bin == col) & (y_bin == row)
            regions[f"r{row + 1}c{col + 1}"] = {
                "candidate_only": int((candidate_only & region).sum()),
                "reference_only": int((reference_only & region).sum()),
            }
    return {
        "candidate_only_correct_windows": int(candidate_only.sum()),
        "reference_only_correct_windows": int(reference_only.sum()),
        "both_correct_windows": int((candidate["correct_mask"] & reference["correct_mask"] & valid).sum()),
        "regions_3x3": regions,
    }


def _bootstrap_pair_mean(values: np.ndarray, *, seed: int = 20261109, rounds: int = 10000):
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(rounds, len(values)))
    means = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _make_report(output_dir: Path, variant_runs: dict, windows_by_variant: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = variant_runs["affine_only"]
    pair_rows = {}
    width, height = reference["protocol"]["generated_native_px"]
    common_pairs = {}
    for variant, report in variant_runs.items():
        common_pairs[variant] = {
            f"{case['texture_seed']}_{case['warp_seed']}": case
            for case in report["cases"]
        }
    reference_cases = common_pairs["affine_only"]
    for variant, cases in common_pairs.items():
        if set(cases) != set(reference_cases):
            raise ValueError(f"{variant} does not contain the same generated pairs as affine_only")
        for key, base_case in reference_cases.items():
            case = cases[key]
            if case["rotation_bank_selected_angle_rad"] != base_case["rotation_bank_selected_angle_rad"]:
                raise ValueError(f"rotation selection changed across ablations for pair {key}")
            base_affine = base_case["global_affine_prefit_coefficients"]
            affine = case["global_affine_prefit_coefficients"]
            if (base_affine is None) != (affine is None) or (
                base_affine is not None and not np.allclose(base_affine, affine, atol=1e-10, rtol=0)
            ):
                raise ValueError(f"image-derived affine prefit changed across ablations for pair {key}")
    for variant, report in variant_runs.items():
        case_map = {
            f"{case['texture_seed']}_{case['warp_seed']}": case
            for case in report["cases"]
        }
        for key, case in case_map.items():
            pair_rows.setdefault(key, {})[variant] = {
                "case": case,
                "windows": _window_and_region_metrics(
                    windows_by_variant[variant][Path(case["window_diagnostics"]).name], width, height
                ),
            }
    for key, variants in pair_rows.items():
        base_windows = variants["affine_only"]["windows"]
        for variant, data in variants.items():
            windows = data["windows"]
            if not np.array_equal(windows["visible_mask"], base_windows["visible_mask"]):
                raise ValueError(f"visibility mask changed across ablations for pair {key}, variant {variant}")
            if not np.array_equal(windows["centres_xy_native"], base_windows["centres_xy_native"]):
                raise ValueError(f"DIC window centers changed across ablations for pair {key}, variant {variant}")
        base = variants["affine_only"]["windows"]
        for candidate in ("affine_ncc", "affine_ndf", "affine_ndf_ncc"):
            variants[candidate]["exclusive_recovery_vs_affine_only"] = _paired_recovery(
                variants[candidate]["windows"], base, width, height
            )
        variants["affine_ndf_ncc"]["conditional_recovery_vs_affine_ncc"] = _paired_recovery(
            variants["affine_ndf_ncc"]["windows"],
            variants["affine_ncc"]["windows"], width, height,
        )
        variants["affine_ndf_ncc"]["conditional_recovery_vs_affine_ndf"] = _paired_recovery(
            variants["affine_ndf_ncc"]["windows"],
            variants["affine_ndf"]["windows"], width, height,
        )
    summary = {}
    for variant in VARIANTS:
        cases = [value[variant] for value in pair_rows.values()]
        visible = sum(row["windows"]["visible"] for row in cases)
        correct = sum(row["windows"]["correct"] for row in cases)
        summary[variant] = {
            "pairs": len(cases),
            "visible_windows": visible,
            "correct_windows": correct,
            "coverage": correct / max(visible, 1),
            "pair_median_coverage": float(np.median([row["windows"]["coverage"] for row in cases])),
            "initial_translation_rmse_px_pair_balanced": float(np.sqrt(np.mean([
                row["windows"]["initial_translation_rmse_px"] ** 2 for row in cases
            ]))),
            "initial_gradient_rmse_pair_median": float(np.median([
                row["windows"]["initial_gradient_rmse"] for row in cases
            ])),
            "initial_parameter_rmse_pair_balanced": {
                name: float(np.sqrt(np.mean([
                    row["windows"]["initial_parameter_rmse"][name] ** 2 for row in cases
                ])))
                for name in ("u", "du_dx", "du_dy", "v", "dv_dx", "dv_dy")
            },
            "initial_translation_median_pair_median_px": float(np.median([
                row["windows"]["initial_translation_median_px"] for row in cases
            ])),
            "correct_windows_per_second_median": float(np.median([
                row["case"]["correct_windows_per_second"] for row in cases
            ])),
            "pipeline_seconds_median": float(np.median([
                row["case"]["inference_s_cpu"]
                + row["case"]["field_projection_s_cpu"]
                + row["case"]["icgn_s"] for row in cases
            ])),
            "oracle_correct_windows": sum(
                row["case"]["analytic_oracle_correct_converged_lt2px"] for row in cases
            ),
            "oracle_coverage": sum(
                row["case"]["analytic_oracle_correct_converged_lt2px"] for row in cases
            ) / max(visible, 1),
            "spatial_regions_3x3": {
                region: {
                    "visible": sum(row["windows"]["regions_3x3"][region]["visible"] for row in cases),
                    "correct": sum(row["windows"]["regions_3x3"][region]["correct"] for row in cases),
                }
                for region in [f"r{row}c{col}" for row in range(1, 4) for col in range(1, 4)]
            },
        }
        for region in summary[variant]["spatial_regions_3x3"].values():
            region["coverage"] = region["correct"] / max(region["visible"], 1)
        if variant != "affine_only":
            recovery = [row["exclusive_recovery_vs_affine_only"] for row in cases]
            summary[variant]["exclusive_recovery_vs_affine_only"] = {
                "candidate_only": sum(item["candidate_only_correct_windows"] for item in recovery),
                "reference_only": sum(item["reference_only_correct_windows"] for item in recovery),
            }
    for candidate, reference_name in (
        ("affine_ndf_ncc", "affine_ncc"),
        ("affine_ndf_ncc", "affine_ndf"),
    ):
        summary_key = f"recovery_vs_{reference_name}"
        pair_key = f"conditional_recovery_vs_{reference_name}"
        recovery = [pair_rows[pair][candidate][pair_key] for pair in pair_rows]
        summary[candidate][summary_key] = {
            "candidate_only": sum(item["candidate_only_correct_windows"] for item in recovery),
            "reference_only": sum(item["reference_only_correct_windows"] for item in recovery),
        }

    contrasts = {}
    for candidate in ("affine_ncc", "affine_ndf", "affine_ndf_ncc"):
        paired = np.asarray([
            pair_rows[key][candidate]["windows"]["coverage"]
            - pair_rows[key]["affine_only"]["windows"]["coverage"]
            for key in sorted(pair_rows)
        ])
        contrasts[f"{candidate}_minus_affine_only"] = {
            "candidate": candidate,
            "reference": "affine_only",
            "mean_pair_coverage_difference": float(paired.mean()),
            "median_pair_coverage_difference": float(np.median(paired)),
            "bootstrap_95_percentile_ci": _bootstrap_pair_mean(paired),
            "pairs_improved": int((paired > 0).sum()),
            "pairs_tied": int((paired == 0).sum()),
            "pairs_regressed": int((paired < 0).sum()),
        }
    for candidate, reference_name in (
        ("affine_ndf_ncc", "affine_ncc"),
        ("affine_ndf_ncc", "affine_ndf"),
    ):
        paired = np.asarray([
            pair_rows[key][candidate]["windows"]["coverage"]
            - pair_rows[key][reference_name]["windows"]["coverage"]
            for key in sorted(pair_rows)
        ])
        label = "NDF_given_NCC" if reference_name == "affine_ncc" else "NCC_given_NDF"
        contrasts[label] = {
            "candidate": candidate,
            "reference": reference_name,
            "mean_pair_coverage_difference": float(paired.mean()),
            "bootstrap_95_percentile_ci": _bootstrap_pair_mean(paired, seed=20261111 if label == "NDF_given_NCC" else 20261112),
            "pairs_improved": int((paired > 0).sum()),
            "pairs_tied": int((paired == 0).sum()),
            "pairs_regressed": int((paired < 0).sum()),
        }
    interaction = np.asarray([
        pair_rows[key]["affine_ndf_ncc"]["windows"]["coverage"]
        - pair_rows[key]["affine_ndf"]["windows"]["coverage"]
        - pair_rows[key]["affine_ncc"]["windows"]["coverage"]
        + pair_rows[key]["affine_only"]["windows"]["coverage"]
        for key in sorted(pair_rows)
    ])
    contrasts["ndf_ncc_interaction"] = {
        "mean_pair_coverage_difference": float(interaction.mean()),
        "bootstrap_95_percentile_ci": _bootstrap_pair_mean(interaction, seed=20261110),
    }
    for item in contrasts.values():
        candidate_name, reference_name = item.get("candidate"), item.get("reference")
        if candidate_name is None:
            continue
        item["spatial_regions_3x3"] = {}
        for region in [f"r{row}c{col}" for row in range(1, 4) for col in range(1, 4)]:
            candidate_correct = sum(
                pair_rows[key][candidate_name]["windows"]["regions_3x3"][region]["correct"]
                for key in pair_rows
            )
            reference_correct = sum(
                pair_rows[key][reference_name]["windows"]["regions_3x3"][region]["correct"]
                for key in pair_rows
            )
            visible_region = sum(
                pair_rows[key][candidate_name]["windows"]["regions_3x3"][region]["visible"]
                for key in pair_rows
            )
            item["spatial_regions_3x3"][region] = {
                "visible": visible_region,
                "candidate_correct": candidate_correct,
                "reference_correct": reference_correct,
                "candidate_minus_reference_coverage": (
                    candidate_correct - reference_correct
                ) / max(visible_region, 1),
            }
    report = {
        "protocol": {
            "description": (
                "Paired causal ablation. All variants reuse the same model-derived rotation choice "
                "and image-derived global affine fit; only the nonlinear NDF residual and local NCC vary."
            ),
            "limitations": [
                "Affine-only still depends on the learned coarse matcher to choose rotation and estimate the image-derived affine map.",
                "The learned multiscale spatial decoder is not implemented and is not represented by this experiment.",
                "Oracle results are per-window analytic initializations and are not available at inference.",
                "The images are generated at 3840 x 2160, but matching, affine prefit and local NCC use the 192 x 108 model raster; only the IC-GN solve uses the native-resolution images.",
            ],
            "generated_native_px": reference["protocol"]["generated_native_px"],
            "model_input_px": reference["protocol"]["model_input_px"],
            "scale_xy": reference["protocol"]["scale_xy"],
            "motion_family": "heterogeneous",
            "generated_pair_count": len(pair_rows),
            "rotation_bank_steps": reference["protocol"]["rotation_bank_steps"],
            "rotation_bank_probe_query_budget": reference["protocol"]["rotation_bank_probe_query_budget"],
            "local_ncc_search_radius_model_px": reference["protocol"]["local_ncc_search_radius_px"],
            "local_ncc_patch_radius_model_px": reference["protocol"]["local_ncc_patch_radius_px"],
            "global_affine_prefit": reference["protocol"]["global_affine_prefit"],
            "field_projection": reference["protocol"]["field_projection"],
            "training_weights": reference["protocol"]["training_weights"],
            "solver": reference["protocol"]["solver"],
        },
        "summary": summary,
        "contrasts_paired_by_generated_image_pair": contrasts,
        "pairs": {
            key: {
                variant: {
                    "initial_parameters_and_window_results": {
                        name: metric for name, metric in value["windows"].items()
                        if name not in {"correct_mask", "visible_mask", "centres_xy_native"}
                    },
                    "exclusive_recovery_vs_affine_only": value.get("exclusive_recovery_vs_affine_only"),
                    "conditional_recovery_vs_affine_ncc": value.get("conditional_recovery_vs_affine_ncc"),
                    "conditional_recovery_vs_affine_ndf": value.get("conditional_recovery_vs_affine_ndf"),
                    "run_metrics": {
                        "selected_rotation_rad": value["case"]["rotation_bank_selected_angle_rad"],
                        "affine_prefit_used": value["case"]["global_affine_prefit_used"],
                        "affine_prefit_inliers": value["case"]["global_affine_prefit_inliers"],
                        "initial_translation_rmse_px": value["case"]["initial_translation_rmse_px"],
                        "initial_gradient_rmse": value["case"]["initial_gradient_rmse"],
                        "correct_windows_per_second": value["case"]["correct_windows_per_second"],
                    },
                }
                for variant, value in variants.items()
            }
            for key, variants in pair_rows.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# NDF causal ablation: image-derived affine, nonlinear residual and NCC",
        "",
        "All variants use paired heterogeneous synthetic image pairs, the same rotation bank, the same coarse matcher, the same image-derived affine prefit, and identical WebGPU IC-GN settings.",
        "The affine-only label means no NDF nonlinear residual after the image-derived affine. It still uses the learned coarse matcher to select rotation and estimate that affine.",
        "The generated images are native 3840 × 2160, but matching, affine prefit, and local NCC use the 192 × 108 model raster (20× reduction); only IC-GN uses native pixels. This is not a native-resolution matching benchmark.",
        "",
        "## Pooled results",
        "",
        "| Variant | Correct / visible | Coverage | Pair median | Init translation RMSE (px) | Init gradient RMSE (median) | Median pipeline (s) | Correct windows/s | Oracle correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "affine_only": "Image-derived affine only",
        "affine_ncc": "Affine + local NCC",
        "affine_ndf": "Affine + NDF residual",
        "affine_ndf_ncc": "Affine + NDF + local NCC",
    }
    for name, values in summary.items():
        lines.append(
            f"| {labels[name]} | {values['correct_windows']:,}/{values['visible_windows']:,} "
            f"| {values['coverage']:.2%} | {values['pair_median_coverage']:.2%} "
            f"| {values['initial_translation_rmse_px_pair_balanced']:.2f} | "
            f"{values['initial_gradient_rmse_pair_median']:.4f} | {values['pipeline_seconds_median']:.3f} "
            f"| {values['correct_windows_per_second_median']:.0f} | {values['oracle_correct_windows']:,} |"
        )
    lines.extend(["", "Exclusive recovery relative to affine-only:", ""])
    for name in ("affine_ncc", "affine_ndf", "affine_ndf_ncc"):
        recovery = summary[name]["exclusive_recovery_vs_affine_only"]
        lines.append(
            f"- {labels[name]}: {recovery['candidate_only']:,} windows recovered only by this variant; "
            f"{recovery['reference_only']:,} affine-only successes lost."
        )
    full_ncc_recovery = summary["affine_ndf_ncc"]["recovery_vs_affine_ncc"]
    lines.append(
        f"- Adding the NDF residual when NCC is already enabled: "
        f"{full_ncc_recovery['candidate_only']:,} extra windows, "
        f"{full_ncc_recovery['reference_only']:,} NCC-only windows lost."
    )
    lines.extend(["", "## Paired changes relative to affine-only", "", "Coverage differences are percentage points; intervals resample the 20 image pairs, not individual windows.", "", "| Contrast | Mean coverage change | Pair bootstrap 95% interval | Better / tied / worse pairs |", "|---|---:|---:|---:|"])
    for name in ("affine_ncc_minus_affine_only", "affine_ndf_minus_affine_only", "affine_ndf_ncc_minus_affine_only"):
        item = contrasts[name]
        lo, hi = item["bootstrap_95_percentile_ci"]
        lines.append(
            f"| {labels[name.removesuffix('_minus_affine_only')]} | "
            f"{item['mean_pair_coverage_difference']:+.2%} | [{lo:+.2%}, {hi:+.2%}] | "
            f"{item['pairs_improved']} / {item['pairs_tied']} / {item['pairs_regressed']} |"
        )
    lines.extend([
        "",
        "Conditional contrasts:",
        "",
    ])
    for name, label in (("NDF_given_NCC", "Add NDF when NCC is enabled"), ("NCC_given_NDF", "Add NCC when NDF is enabled")):
        item = contrasts[name]
        lo, hi = item["bootstrap_95_percentile_ci"]
        lines.append(
            f"- {label}: {item['mean_pair_coverage_difference'] * 100:+.2f} percentage points "
            f"(95% pair-bootstrap interval {lo * 100:+.2f} to {hi * 100:+.2f}); "
            f"{item['pairs_improved']} pairs improve, {item['pairs_tied']} are tied, "
            f"{item['pairs_regressed']} regress."
        )
    interaction = contrasts["ndf_ncc_interaction"]
    lo, hi = interaction["bootstrap_95_percentile_ci"]
    lines.append(
        f"- NDF–NCC interaction: {interaction['mean_pair_coverage_difference'] * 100:+.2f} "
        f"percentage points (95% pair-bootstrap interval {lo * 100:+.2f} to {hi * 100:+.2f})."
    )
    lines.extend([
        "",
        "## Initial affine-parameter errors",
        "",
        "Pair-balanced root mean square error over visible centres; `u` and `v` are native-pixel displacements, and the other four entries are displacement gradients.",
        "",
        "| Variant | u (px) | du/dx | du/dy | v (px) | dv/dx | dv/dy | Median translation error (px) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, values in summary.items():
        params = values["initial_parameter_rmse_pair_balanced"]
        lines.append(
            f"| {labels[name]} | {params['u']:.2f} | {params['du_dx']:.4f} | "
            f"{params['du_dy']:.4f} | {params['v']:.2f} | {params['dv_dx']:.4f} | "
            f"{params['dv_dy']:.4f} | {values['initial_translation_median_pair_median_px']:.2f} |"
        )
    lines.extend([
        "",
        "The JSON retains each pair’s six initial affine parameters, per-window IC-GN outcomes, windows recovered or lost versus affine-only, and a 3 × 3 spatial breakdown. The oracle count is a diagnostic ceiling, not an available pipeline.",
        "",
        "Conditional gains from adding the NDF residual to NCC, and adding NCC to the NDF residual, plus their interaction, are reported in `summary.json`.",
    ])
    spatial = contrasts["NDF_given_NCC"]["spatial_regions_3x3"]
    lines.extend([
        "",
        "## Spatial change when adding the NDF residual to affine + NCC",
        "",
        "| Region | Visible windows | NCC correct | NCC + NDF correct | Coverage change |",
        "|---|---:|---:|---:|---:|",
    ])
    for region, item in spatial.items():
        lines.append(
            f"| {region} | {item['visible']:,} | {item['reference_correct']:,} "
            f"| {item['candidate_correct']:,} | {item['candidate_minus_reference_coverage']:+.2%} |"
        )
    lines.extend([
        "",
        "## Reproduction",
        "",
        "Set `live_dic_source` to the `python/src` directory of the Live-DIC checkout used by this repository’s IC-GN adapter, then run:",
        "",
        "```console",
        "python python/benchmarks/benchmark_ndf_causal_ablation.py \\",
        "  --weights results/ndf_dic_20261109/p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt \\",
        "  --live-dic-source \"$live_dic_source\" \\",
        "  --output-dir results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109",
        "```",
        "",
        "The runner uses seed offsets 16,000,000 and 17,000,000, ten heterogeneous pairs per offset, and checks that all variants retain the same selected orientation, affine map, visibility mask, and DIC window centres.",
    ])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=(16000000, 17000000))
    parser.add_argument("--cases-per-seed", type=int, default=10)
    parser.add_argument("--native-scale", type=int, default=20)
    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--model-height", type=int, default=108)
    parser.add_argument("--speckle-size", type=float, default=0.35)
    parser.add_argument("--rotation-bank-steps", type=int, default=9)
    parser.add_argument("--probe-query-budget", type=int, default=256)
    parser.add_argument("--local-ncc-backend", choices=("cpu", "wgpu"), default="wgpu")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild summary.json and README.md from completed raw runs without rerunning inference",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        variant_reports = {}
        window_data = {}
        for variant in VARIANTS:
            variant_dir = args.output_dir / variant
            reports = [
                json.loads((variant_dir / f"heterogeneous_seed{seed}.json").read_text(encoding="utf-8"))
                for seed in args.seed_offsets
            ]
            merged = reports[0]
            for report in reports[1:]:
                merged["cases"].extend(report["cases"])
            variant_reports[variant] = merged
            window_data[variant] = {}
            for report in reports:
                window_dir = Path(report["cases"][0]["window_diagnostics"]).parent
                window_data[variant].update(_load_window_files(window_dir))
        _make_report(args.output_dir, variant_reports, window_data)
        print(f"summarized existing runs in {args.output_dir}", flush=True)
        return
    script = Path(__file__).with_name("qualify_ndf_pair_icgn.py")
    variant_reports = {}
    window_data = {}
    for variant, config in VARIANTS.items():
        variant_dir = args.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for seed_offset in args.seed_offsets:
            report_path = variant_dir / f"heterogeneous_seed{seed_offset}.json"
            diagnostic_dir = variant_dir / f"windows_seed{seed_offset}"
            command = [
                sys.executable, str(script),
                "--weights", str(args.weights.resolve()),
                "--live-dic-source", str(args.live_dic_source.resolve()),
                "--output", str(report_path),
                "--window-diagnostics-dir", str(diagnostic_dir),
                "--native-scale", str(args.native_scale),
                "--model-width", str(args.model_width),
                "--model-height", str(args.model_height),
                "--speckle-size", str(args.speckle_size),
                "--families", "heterogeneous",
                "--cases-per-family", str(args.cases_per_seed),
                "--seed-offset", str(seed_offset),
                "--contextual-features",
                "--rotation-bank-steps", str(args.rotation_bank_steps),
                "--rotation-bank-probe-query-budget", str(args.probe_query_budget),
                "--global-affine-prefit",
                "--field-residual", config["field"],
                "--local-ncc-radius", str(config["ncc"]),
                "--local-ncc-patch-radius", "4",
                "--local-ncc-backend", args.local_ncc_backend,
                "--local-ncc-geometry", "translation",
                "--local-consensus-radius", "40",
                "--local-consensus-threshold", "5",
                "--projection", "raw_bilinear",
            ]
            log_path = variant_dir / f"heterogeneous_seed{seed_offset}.log"
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        merged = reports[0]
        for report in reports[1:]:
            merged["cases"].extend(report["cases"])
        variant_reports[variant] = merged
        window_data[variant] = {}
        for report in reports:
            window_dir = Path(report["cases"][0]["window_diagnostics"]).parent
            window_data[variant].update(_load_window_files(window_dir))
        print(f"completed {variant}: {len(merged['cases'])} paired image pairs", flush=True)
    _make_report(args.output_dir, variant_reports, window_data)
    print(f"wrote {args.output_dir / 'README.md'} and summary.json", flush=True)


if __name__ == "__main__":
    main()
