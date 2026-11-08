"""Plot independent compact-descriptor IC-GN qualification results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = {
    "DeDoDe-B": ("dic_detector_dedode_b", None),
    "HORG-64": ("dic_detector_horg64", None),
    "DIC CNN": ("dic_detector_patch_cnn_descriptor", None),
    "DeepMatching": ("baselines", "deepmatching"),
    "Native HORG": ("baselines", "horg"),
    "EfficientLoFTR": ("baselines", "efficientloftr"),
}
FAMILIES = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")


def pipeline_result(case: dict, label: str) -> dict:
    key, nested = METHODS[label]
    result = case[key][nested] if nested else case[key]
    return result


def fit_result(case: dict, label: str) -> dict:
    result = pipeline_result(case, label)
    if METHODS[label][1]:
        return result.get("basis_results", {}).get("gaussian_rbf", {})
    return result.get("gaussian_rbf", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        parser.error(f"Install the optional research extra to plot results: {error}")

    all_cases = json.loads(args.summary.read_text(encoding="utf-8"))
    independent = [case for case in all_cases if case["texture_seed"] >= 970_000_000]
    colors = dict(zip(METHODS, plt.cm.tab10.colors, strict=False))
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.2))

    # Correct IC-GN convergence across the complete visible-window population.
    coverage_axis = axes[0]
    x = np.arange(len(FAMILIES), dtype=float)
    width = 0.12
    for method_index, method in enumerate(METHODS):
        fractions = []
        for family in FAMILIES:
            cases = [
                case
                for case in independent
                if case["case"] == family and "analytic_local_affine_oracle" in case["dic_detector_patch_cnn_descriptor"]
            ]
            visible = correct = 0
            for case in cases:
                oracle = case["dic_detector_patch_cnn_descriptor"]["analytic_local_affine_oracle"]
                visible += oracle["visible_windows"]
                metrics = fit_result(case, method).get("solver", {})
                correct += metrics.get("correct_converged_lt2px", 0)
            fractions.append(100.0 * correct / visible if visible else 0.0)
        offsets = (method_index - (len(METHODS) - 1) / 2) * width
        coverage_axis.bar(x + offsets, fractions, width, color=colors[method], label=method)
    oracle_fractions = []
    for family in FAMILIES:
        cases = [
            case
            for case in independent
            if case["case"] == family
            and "analytic_local_affine_oracle" in case["dic_detector_patch_cnn_descriptor"]
        ]
        visible = sum(
            case["dic_detector_patch_cnn_descriptor"]["analytic_local_affine_oracle"][
                "visible_windows"
            ]
            for case in cases
        )
        correct = sum(
            case["dic_detector_patch_cnn_descriptor"]["analytic_local_affine_oracle"][
                "correct_converged_lt2px"
            ]
            for case in cases
        )
        oracle_fractions.append(100.0 * correct / visible if visible else 0.0)
    coverage_axis.plot(
        x,
        oracle_fractions,
        color="black",
        marker="D",
        linestyle="--",
        linewidth=1.2,
        markersize=4,
        label="Analytic oracle",
    )
    coverage_axis.set_xticks(x, [family.capitalize() for family in FAMILIES], rotation=18)
    coverage_axis.set_ylabel("Correctly converged visible windows (%)")
    coverage_axis.set_title("IC-GN coverage")
    coverage_axis.set_ylim(0, 105)
    coverage_axis.grid(axis="y", alpha=0.25)

    # Median per-field final displacement RMSE among solver-accepted windows.
    error_axis = axes[1]
    markers = ("o", "s", "^", "v", "D", "P")
    jitter = np.linspace(-0.1, 0.1, len(METHODS))
    for method_index, method in enumerate(METHODS):
        medians = []
        for family in FAMILIES:
            values = []
            for case in independent:
                if case["case"] != family:
                    continue
                solver = fit_result(case, method).get("solver", {})
                value = solver.get("accepted_displacement_rmse_px")
                if value is not None and np.isfinite(value):
                    values.append(value)
            medians.append(float(np.median(values)) if values else np.nan)
        error_axis.scatter(
            x + jitter[method_index],
            medians,
            marker=markers[method_index],
            s=34,
            color=colors[method],
            label=method,
        )
    error_axis.set_xticks(x, [family.capitalize() for family in FAMILIES], rotation=18)
    error_axis.set_ylabel("Median final displacement RMSE (px)")
    error_axis.set_title("Error after IC-GN among accepted fields")
    error_axis.grid(alpha=0.25)
    error_axis.set_ylim(bottom=0)

    # Every method completed a valid field fit in all ten discontinuous cases.
    speed_axis = axes[2]
    family = "discontinuous"
    for method in METHODS:
        cases = [case for case in independent if case["case"] == family]
        visible = sum(
            case["dic_detector_patch_cnn_descriptor"]["analytic_local_affine_oracle"][
                "visible_windows"
            ]
            for case in cases
        )
        correct = 0
        elapsed = 0.0
        fitted_fields = 0
        for case in cases:
            fitted = fit_result(case, method)
            solver = fitted.get("solver", {})
            correct += solver.get("correct_converged_lt2px", 0)
            if "solver" in fitted:
                fitted_fields += 1
                elapsed += fitted.get("full_pipeline_s", fitted.get("end_to_end_s", 0.0))
            elif METHODS[method][1]:
                pipeline = pipeline_result(case, method)
                elapsed += pipeline.get("extraction_s", 0.0) + pipeline.get(
                    "reference_preparation_s", 0.0
                )
            else:
                pipeline = pipeline_result(case, method)
                elapsed += sum(
                    pipeline.get(name, 0.0)
                    for name in (
                        "reference_preparation_s",
                        "detection_pair_s",
                        "description_pair_s",
                        "matching_and_consensus_s",
                    )
                )
        coverage = 100.0 * correct / visible if visible else 0.0
        throughput = correct / elapsed if elapsed else 0.0
        speed_axis.scatter(throughput, coverage, s=58, color=colors[method])
        label = method if fitted_fields == len(cases) else f"{method}*"
        speed_axis.annotate(
            label, (throughput, coverage), xytext=(5, 4), textcoords="offset points", fontsize=8
        )
    speed_axis.set_xlabel("Correct windows per second")
    speed_axis.set_ylabel("Correct coverage (%)")
    speed_axis.set_title("Discontinuous fields: coverage / speed")
    speed_axis.grid(alpha=0.25)
    figure.suptitle("Compact speckle descriptors: independent IC-GN qualification", fontsize=14)
    figure.legend(
        *coverage_axis.get_legend_handles_labels(),
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=8,
    )
    figure.text(
        0.5,
        0.015,
        "Ten independent synthetic fields per family; correct means final displacement error < 2 px. "
        "In speed plot, * marks upper-bound throughput where failed fits lack their attempt duration.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(top=0.83, bottom=0.22, wspace=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
