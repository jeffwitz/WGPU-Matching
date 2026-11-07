"""Combine the 60% shared-descriptor detector budget sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOW = ROOT / "results/features/detector_common_descriptor_60pct_budgets_low_20261107.json"
HIGH = ROOT / "results/features/detector_common_descriptor_60pct_budgets_20261107.json"
OUTPUT = ROOT / "results/features/detector_budget_sweep_60pct_20261107.md"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _only_case(report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    if len(cases) != 1 or cases[0]["name"] != "synthetic_60pct":
        raise ValueError("Budget sweep input must contain only the synthetic_60pct case")
    return cases[0]


def _error(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def build() -> str:
    low, high = read(LOW), read(HIGH)
    low_case, high_case = _only_case(low), _only_case(high)
    if low_case["reference_path"] != high_case["reference_path"]:
        raise ValueError("Low/high budget runs use different reference images")
    if low_case["target_path"] != high_case["target_path"]:
        raise ValueError("Low/high budget runs use different target images")
    if low_case["visible_dic_seeds"] != high_case["visible_dic_seeds"]:
        raise ValueError("Low/high budget runs have different visible DIC seed counts")
    if set(low_case["detectors"]) != set(high_case["detectors"]):
        raise ValueError("Low/high budget runs use different detector sets")
    for detector in low_case["detectors"]:
        low_sha = low_case["detectors"][detector]["detector_metadata"].get("weights_sha256")
        high_sha = high_case["detectors"][detector]["detector_metadata"].get("weights_sha256")
        if low_sha != high_sha:
            raise ValueError(f"Detector weights differ between budget runs: {detector}")

    lines = [
        "# Shared-descriptor detector budget sweep at 60% tensile strain",
        "",
        "This experiment covers the same 3840×2160 synthetic global-affine 60% tensile pair at budgets 512, 1024, 2048, 4096, and 8192. Each method receives a candidate pool twice the requested budget; the table uses global top-K from that returned pool. The complete reports also include spatial-quota selection.",
        "",
        f"All {low_case['visible_dic_seeds']} analytic seed centres are visible. Learned DaD/DeDoDe inference, DeDoDe-B description, and mutual-cosine matching run on CPU; HORG/DIC detection and affine RANSAC run on WGSL through {low['adapter']}. These timings include CPU descriptor/matching and WGSL RANSAC and are not GPU-resident pipeline times. DIC's detector candidate pool is produced with its internal 16×9 spatial quotas, so global top-K is not over the full response map.",
        "",
        "| Detector | Budget | Raw mutual matches | Raw correct <5 px | Accepted affine | Correct ≤2 px | Correct <5 px | Accepted RMSE/P95 px | Total s | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for detector in low_case["detectors"]:
        low_rows = low_case["detectors"][detector]["budgets"]
        high_rows = high_case["detectors"][detector]["budgets"]
        for budget in (512, 1024, 2048, 4096, 8192):
            source = low_rows if budget <= 1024 else high_rows
            values = source[str(budget)]["global_topk"]
            accepted = values["accepted_affine_seeds"]
            rmse = _error(values["accepted_affine_error_rmse_px"])
            p95 = _error(values.get("accepted_affine_error_p95_px"))
            lines.append(
                f"| {detector} | {budget} | {values['raw_mutual_matches']} | "
                f"{values['raw_correct_under_5px']} | {accepted}/16 | "
                f"{values['affine_seed_correct_under_2px']}/16 | "
                f"{values['affine_seed_correct_under_5px']}/16 | {rmse}/{p95} | "
                f"{values['total_pair_s_median']:.3f} | {values['pair_fps']:.3f} |"
            )
    lines.extend(
        (
            "",
            "DaD goes from 9/16 accepted seeds at budget 512 to 16/16 by 1024; at 8192 it has 0.46 px accepted RMSE and 0.81 px P95. DeDoDe v2-L improves from 3/16 at 2048 to 16/16 at 8192, but takes about 17.14 s/pair. The discontinuity-aware DIC score model + DeDoDe-B reaches 16/16 accepted, 16/16 within 2 px, and 16/16 within 5 px from budget 4096; at 8192 its accepted RMSE/P95 is 0.38/0.61 px at 7.01 s/pair. HORG/SIFT goes from 1/16 at 512 to 2/16 from 1024 onward, and the tensor-only DIC baseline reaches 7/16 accepted at 8192 with 5 correct under 5 px.",
            "",
            "These results show the value of increasing candidate count for this particular analytically affine stress image; they do not establish performance on heterogeneous strain, experimental speckles, cracks, or IC-GN convergence. The descriptor and mutual matching stages are CPU references, and the keypoint repeatability denominators vary with the selected points. Do not treat accepted coverage alone as correctness; the analytic-error columns show both.",
            "",
            "Full low-budget data: `detector_common_descriptor_60pct_budgets_low_20261107.json`. Full high-budget data: `detector_common_descriptor_60pct_budgets_20261107.json`.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
