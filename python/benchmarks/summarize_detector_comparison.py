"""Build a concise report from the detector and historical matcher benchmarks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "results/features/detector_common_descriptor_20261107.json"
DEEP = ROOT / "results/features/deepmatching_same_protocol.json"
XFEAT = ROOT / "results/features/xfeat_onnx_tensile.json"
EFFICIENT_LOFTR = ROOT / "results/features/efficientloftr_webgpu_study.json"
TENSILE_60 = ROOT / "results/uniaxial/tensile_60pct/three_method_browser_60.json"
DIC_DISCONTINUOUS = ROOT / "results/features/dic_detector_discontinuous_training_20261107.md"
OUTPUT = ROOT / "results/features/detector_method_comparison_20261107.md"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_errors(values: dict[str, Any]) -> list[float]:
    return [
        float(row["analytic_error_px"])
        for row in values["per_seed_affine"]
        if row["usable"] and row["analytic_error_px"] is not None
    ]


def _format_error(values: dict[str, Any]) -> str:
    errors = _accepted_errors(values)
    if not errors:
        return "—"
    errors_array = sorted(errors)
    rmse = (sum(error * error for error in errors_array) / len(errors_array)) ** 0.5
    p95 = errors_array[min(len(errors_array) - 1, math.ceil(0.95 * len(errors_array)) - 1)]
    return f"{rmse:.2f}/{p95:.2f}"


def build() -> str:
    common = read(COMMON)
    lines = [
        "# Detector and matcher comparison",
        "",
        "This report combines the shared-descriptor detector experiment with separate historical full-pipeline results. The protocols are intentionally kept distinct: the first controls the descriptor and local-affine acceptance criteria, while the second reports each matcher as previously benchmarked.",
        "",
        "## Shared descriptor and affine RANSAC",
        "",
        f"The common protocol uses official DaD and DeDoDe v2-L detector weights at a {common['model_max_side_px']} px maximum side, DeDoDe-B descriptors, mutual cosine nearest neighbours, and the same WGSL local-affine RANSAC settings. DaD/DeDoDe detector inference, DeDoDe-B description, and cosine matching ran on PyTorch/CPU; HORG/DIC point detection and affine RANSAC ran on WebGPU. These are reference results, not a GPU-resident pipeline. The DIC learned detector's internal 16×9 quota pool is followed by global top-K or a second-stage quota selection; the global top-K figure is therefore not over all NMS peaks.",
        "",
        f"Adapter: {common['adapter']}; PyTorch CPU threads: {common['torch_num_threads']}; steady-state repeats: {common['steady_state_repeats']}; candidate pool: {common['candidate_multiplier']}× each requested budget. Error cells are accepted affine-seed RMSE/P95 in pixels. This four-case run uses the first DIC learned checkpoint ({common['cases'][0]['detectors']['DIC-Learned-WGSL']['detector_metadata']['weights_sha256']}); the dedicated five-budget stress sweep uses the discontinuity-aware checkpoint and is reported separately.",
        "",
        "| Case | Detector | Selection | Budget | Repeatable ≤2 px | Raw matches <5 px | Accepted affine | Correct <5 px | RMSE/P95 px | Detect s | Describe s | Match s | RANSAC ms | Total s | FPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in common["cases"]:
        for name, detector in case["detectors"].items():
            selections = detector["budgets"].get("1024", {})
            for selection, values in selections.items():
                lines.append(
                    f"| {case['name']} | {name} | {selection} | 1024 | "
                    f"{values['detector_repeatability']['at_2px']}/"
                    f"{values['detector_repeatability']['denominator_min_visible']} | "
                    f"{values['raw_correct_under_5px']}/{values['raw_mutual_matches']} | "
                    f"{values['accepted_affine_seeds']}/{case['visible_dic_seeds']} | "
                    f"{values['affine_seed_correct_under_5px']}/"
                    f"{case['visible_dic_seeds']} | {_format_error(values)} | "
                    f"{values['detection_pair_s']:.3f} | {values['descriptor_pair_s']:.3f} | "
                    f"{values['descriptor_matching_s']:.3f} | "
                    f"{values['wgsl_ransac_compute_ms_median']:.2f} | "
                    f"{values['total_pair_s_median']:.3f} | {values['pair_fps']:.3f} |"
                )

    deep = read(DEEP)
    deep_60 = next(row for row in deep["reports"] if row["case"] == "tensile_60pct")
    xfeat = read(XFEAT)
    xfeat_60 = next(row for row in xfeat["reports"] if row["visible_seeds"] == 16)
    xfeat_affine = xfeat_60["local_affine"]["256"]
    loftr = read(EFFICIENT_LOFTR)
    loftr_60 = next(row for row in loftr["results"] if row["requested_visible_seeds"] == 16)
    browser_60 = read(TENSILE_60)
    horg_60 = next(row for row in browser_60["methods"] if row["method"] == "HORG")
    dad_common = common["cases"][-1]["detectors"]
    lines.extend(
        (
            "",
            "## 60% tensile stress test: complete-pipeline context",
            "",
            "All 16 seeds are analytically visible in the deformed image. The common-descriptor entries use budget 1024 and the global top-K-from-candidate-pool selection. Historical matchers retain their own published parameter and timing scopes; use this table to compare behavior, not to claim an identical implementation path.",
            "",
            "| Pipeline | Accepted | Accepted correct <5 px | Accepted ≤2 px | Time | Timing scope |",
            "|---|---:|---:|---:|---:|---|",
            f"| DeepMatching INRIA | {deep_60['accepted']}/16 | {deep_60['correct_lt5']}/16 | {deep_60['correct_le2']}/16 | {deep_60['median_s']:.3f} s | WebGPU native-resolution full matcher |",
            f"| HORG | {horg_60['acceptedCount']}/16 | {horg_60['repairableAcceptedCountLt5Px']}/16 | {horg_60['accurateAcceptedCount']}/16 | {horg_60['medianElapsedMs'] / 1000:.3f} s | Browser WebGPU full matcher |",
            f"| XFeat | {xfeat_affine['fitted_count']}/16 | {xfeat_affine['fitted_with_error_lt5_px']}/16 | {xfeat_affine['fitted_with_error_le2_px']}/16 | {xfeat_60['repeat_median_s']:.3f} s | CPU ONNX inference only; affine fit excluded |",
            f"| EfficientLoFTR | {loftr_60['local_affine_accepted']}/16 | {loftr_60['local_affine_lt5_px']}/16 | {loftr_60['local_affine_le2_px']}/16 | {loftr_60['end_to_end_median_s']:.3f} s | ONNX inference plus local affine; 640×360 input |",
        )
    )
    for name, detector in dad_common.items():
        values = detector["budgets"]["1024"]["global_topk"]
        detector_names = {
            "DaD": "DaD + DeDoDe-B",
            "DeDoDe-v2-L": "DeDoDe v2-L + DeDoDe-B",
            "DIC-Tensor-WGSL": "DIC tensor WGSL + DeDoDe-B",
            "DIC-Learned-WGSL": "DIC learned WGSL + DeDoDe-B",
            "HORG-SIFT-common-DeDoDe-B": "HORG/SIFT + DeDoDe-B",
        }
        timing_scope = (
            "CPU detector/descriptor/mutual match plus WGSL RANSAC"
            if "WGSL" not in name and name != "HORG-SIFT-common-DeDoDe-B"
            else "WGSL detection/RANSAC; DeDoDe-B and mutual matching on CPU"
        )
        lines.append(
            f"| {detector_names[name]} | {values['accepted_affine_seeds']}/16 | "
            f"{values['affine_seed_correct_under_5px']}/16 | "
            f"{values['affine_seed_correct_under_2px']}/16 | "
            f"{values['total_pair_s_median']:.3f} s | {timing_scope} |"
        )
    lines.extend(
        (
            "",
            "DeepMatching's accepted-set RMSE at 60% is 147.4 px despite five accepted vectors below 5 px, so its 8/16 repairable coverage must not be read as 8/16 trustworthy acceptances. The four-case DaD row reaches 16/16 affine seeds under 5 px in this one synthetic, globally affine stress case, but costs about 11.17 s/pair in its CPU reference path. At budget 1024, the first DIC learned checkpoint yields 5/16 correct under 5 px with global top-K (6/16 with spatial quotas), versus 0/16 for the tensor baseline; the separate budget sweep evaluates the later discontinuity-aware checkpoint. None of these figures measures actual IC-GN convergence.",
            "",
            "## Artifacts and limitations",
            "",
            "* The complete per-seed common-descriptor data and model hashes are in `detector_common_descriptor_20261107.json`.",
            "* Native 4K DIC detector latency and score parity are in `dic_detector_native_4k_20261107.md` and `dic_detector_wgsl_parity_4k_20261107.json`.",
            "* Held-out 256×256 synthetic repeatability is in `dic_detector_heldout_20261107.md`; this is detector-only and does not use descriptors.",
            "* A discontinuity-aware checkpoint and a paired detector-only comparison are in `dic_detector_discontinuous_training_20261107.md`; this does not establish matching or IC-GN performance.",
            "* The shared-descriptor budget sweep from 512 to 8192 on the 60% case is in `detector_budget_sweep_60pct_20261107.md`; it uses the discontinuity-aware DIC score checkpoint, not the first checkpoint above.",
            "* The 60% tensile cases are synthetic global affine extrapolations. Smooth heterogeneous and piecewise-discontinuous motions have only small held-out synthetic samples; experimental cracks, localization bands, finite-element fields, and downstream IC-GN convergence remain unqualified.",
            "* DaD and DeDoDe checkpoints were trained for natural-image local-feature benchmarks. DaD + DeDoDe-B is an experimental combination, not an officially released end-to-end model.",
            "",
        )
    )
    if DIC_DISCONTINUOUS.is_file():
        lines.extend(
            (
                "",
                "The discontinuity-aware held-out study is summarized separately because its metric is keypoint repeatability, not matched DIC-window coverage. It compares the earlier and updated score weights on identical texture/warp pairs and reports motion-family breakdowns.",
            )
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
