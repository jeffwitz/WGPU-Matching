"""Run the real IC-GN qualification for HORG-64 and one compact CNN checkpoint.

Uses the same six fixed regression cases and deterministic independent motion
seeds as ``qualify_heterogeneous_icgn.py``. Raw per-window debug arrays are
written to a scratch directory; the compact summary is written to ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from geometric_consensus import LocalConsensus
from qualify_heterogeneous_icgn import grid_centres, json_safe, run_case, seed_state
from wgpu_matching import DICDetectorOptions, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--cnn-descriptor", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--independent-eval-per-family", type=int, default=10)
    parser.add_argument("--seed", type=int, default=970_000_000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument(
        "--live-dic-source",
        type=Path,
        default=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src"),
    )
    args = parser.parse_args()
    if args.independent_eval_per_family < 0:
        parser.error("independent evaluation count must not be negative")
    if not args.cnn_descriptor.is_file() or not args.detector_weights.is_file():
        parser.error("CNN descriptor and detector weight files must exist")
    if not args.live_dic_source.is_dir():
        parser.error(f"IC-GN source is missing: {args.live_dic_source}")
    sys.path.insert(0, str(args.live_dic_source))
    from live_dic import ICGNWGPU

    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    fixed_cases = [
        ("affine", 8101, 9101),
        ("heterogeneous", 8102, 9102),
        ("heterogeneous", 8103, 9103),
        ("discontinuous", 8104, 9104),
        ("discontinuous", 8105, 9105),
        ("rigid", 8106, 9106),
    ]
    families = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")
    independent = []
    for family_index, family in enumerate(families):
        for sample_index in range(args.independent_eval_per_family):
            seed = args.seed + family_index * 1_000_003 + sample_index * 10_007
            independent.append((family, seed, seed + 4_000_000_001))
    scenarios = fixed_cases + independent
    results = []
    with WGPUMatching(device_type="gpu") as matcher:
        print(f"adapter: {matcher.device_name}", flush=True)
        solver = ICGNWGPU(
            device_type="gpu",
            prefer_subgroups=False,
            subset_size=64,
            subset_stride=64,
            interpolation="linear",
            motion_model="affine",
            outlier_policy="disabled",
        )
        consensus = LocalConsensus(matcher, radius=192.0, threshold=3.0)
        warm_pair = generate_speckle_pair(8811, 9911, args.width, args.height, motion_mode="affine")
        warm_reference = np.rint(warm_pair.reference * 255).astype(np.uint8)
        warm_target = np.rint(warm_pair.target * 255).astype(np.uint8)
        solver.set_reference(warm_reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
        warm_features = [
            matcher.detect_dic(
                image,
                options=DICDetectorOptions(max_keypoints=512),
                learned_weights_path=args.detector_weights,
            ).features
            for image in (warm_reference, warm_target)
        ]
        warm_horg = [
            matcher.describe_horg_keypoints(image, feature, reduction=4)
            for image, feature in zip((warm_reference, warm_target), warm_features, strict=True)
        ]
        warm_cnn = [
            matcher.describe_dic_cnn_keypoints(
                image, feature, weights_path=args.cnn_descriptor
            )
            for image, feature in zip((warm_reference, warm_target), warm_features, strict=True)
        ]
        warm_points = grid_centres(args.width, args.height, 64, 64)
        matcher.match_features(warm_horg[0], warm_horg[1], warm_points, fit_local_models=False)
        matcher.match_features(warm_cnn[0], warm_cnn[1], warm_points, fit_local_models=False)
        warm_active = np.ones(len(warm_points), dtype=np.uint32)
        solver._register_seeded_frame(
            warm_target,
            seed_state(np.zeros((len(warm_points), 6), dtype=np.float32), warm_active),
            warm_active,
        )
        for index, (mode, texture_seed, warp_seed) in enumerate(scenarios, start=1):
            pair = generate_speckle_pair(
                texture_seed, warp_seed, args.width, args.height, motion_mode=mode
            )
            reference = np.rint(pair.reference * 255).astype(np.uint8)
            prep_start = time.perf_counter()
            solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
            reference_preparation_s = time.perf_counter() - prep_start
            row = {
                "case_index": index,
                "case": mode,
                "texture_seed": texture_seed,
                "warp_seed": warp_seed,
                "split": "fixed_regression" if index <= len(fixed_cases) else "independent_test",
            }
            row["horg64"] = run_case(
                matcher,
                None,
                consensus,
                solver,
                mode,
                texture_seed,
                warp_seed,
                args.scratch / "horg64",
                (args.width, args.height),
                args.detector_weights,
                reference_preparation_s,
                descriptor_variant="horg64",
            )
            row["dic_cnn"] = run_case(
                matcher,
                None,
                consensus,
                solver,
                mode,
                texture_seed,
                warp_seed,
                args.scratch / "dic_cnn",
                (args.width, args.height),
                args.detector_weights,
                reference_preparation_s,
                descriptor_variant="dic_cnn_descriptor",
                cnn_descriptor_weights=args.cnn_descriptor,
            )
            results.append(row)
            print(
                f"{index}/{len(scenarios)} {mode} {texture_seed}/{warp_seed}: "
                f"HORG and DIC-CNN qualified",
                flush=True,
            )
        solver.close()

    safe = json_safe(results)
    digest = hashlib.sha256(args.cnn_descriptor.read_bytes()).hexdigest()
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(safe, indent=2, allow_nan=False) + "\n")
    summary = {"cases": len(safe), "groups": {}}
    for split in ("fixed_regression", "independent_test"):
        group = {}
        for family in families:
            selected = [
                row for row in safe if row["case"] == family and row["split"] == split
            ]
            family_summary = {"fields": len(selected), "methods": {}}
            for method in ("horg64", "dic_cnn"):
                family_summary["methods"][method] = {}
                for basis in ("affine", "gaussian_rbf"):
                    valid_cases = [
                        row[method].get(basis)
                        for row in selected
                        if isinstance(row[method].get(basis), dict)
                    ]
                    measured = [item for item in valid_cases if "solver" in item]
                    correct = sum(
                        item["solver"].get("correct_converged_lt2px", 0) for item in measured
                    )
                    visible = sum(item["solver"].get("visible_windows", 0) for item in measured)
                    support_correct = sum(
                        item.get("solver_on_fully_valid_support_windows", {}).get(
                            "correct_converged_lt2px", 0
                        )
                        for item in measured
                    )
                    support_visible = sum(
                        item.get("solver_on_fully_valid_support_windows", {}).get(
                            "visible_windows", 0
                        )
                        for item in measured
                    )
                    elapsed = sum(item.get("full_pipeline_s", 0.0) for item in measured)
                    family_summary["methods"][method][basis] = {
                        "measured_fields": len(measured),
                        "fields_with_fit_error": len(valid_cases) - len(measured),
                        "correct_windows": int(correct),
                        "visible_windows": int(visible),
                        "correct_windows_full_support": int(support_correct),
                        "fully_valid_support_windows": int(support_visible),
                        "pipeline_time_s": float(elapsed),
                        "correct_windows_per_second": float(correct / elapsed)
                        if elapsed > 0
                        else None,
                    }
            group[family] = family_summary
        summary["groups"][split] = group
    (args.output / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    report_lines = [
        "# Compact DIC descriptor requalification",
        "",
        "This report evaluates the existing inverse compositional Gauss–Newton (IC-GN) solver using the same point detector, windows, matching consensus, affine and radial basis function (RBF) field projections, and convergence limits as the preceding qualification. The six fixed cases are regression checks; independent texture and motion seeds are the test set.",
        "",
        f"Adapter: {matcher.device_name}. Image size: {args.width} × {args.height}. CNN weights SHA-256: `{digest}`.",
        "",
        "Counts are correct IC-GN windows within 2 pixels / visible windows. ‘Full support’ restricts evaluation after the run to windows whose 64 × 64 support is valid in the reference and mapped target masks; it does not change matching or solver inputs.",
        "",
        "## Independent test fields",
        "",
        "| Motion | Descriptor | Affine fits | Correct / visible | Full-support correct / valid | RBF fits | Correct / visible | Full-support correct / valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def ratio_or_dash(correct: int, total: int) -> str:
        return f"{correct}/{total}" if total else "—"

    for family in families:
        group = summary["groups"]["independent_test"][family]
        for method, label in (("horg64", "HORG-64"), ("dic_cnn", "DIC-CNN hard-negative")):
            affine = group["methods"][method]["affine"]
            rbf = group["methods"][method]["gaussian_rbf"]
            n_fields = group["fields"]
            report_lines.append(
                f"| {family} | {label} | {affine['measured_fields']}/{n_fields} | "
                f"{ratio_or_dash(affine['correct_windows'], affine['visible_windows'])} | "
                f"{ratio_or_dash(affine['correct_windows_full_support'], affine['fully_valid_support_windows'])} | "
                f"{rbf['measured_fields']}/{n_fields} | "
                f"{ratio_or_dash(rbf['correct_windows'], rbf['visible_windows'])} | "
                f"{ratio_or_dash(rbf['correct_windows_full_support'], rbf['fully_valid_support_windows'])} |"
            )
    report_lines.extend(
        [
            "",
            "A fit is counted only when the field projection returns a usable model. A dash means there were no visible windows to score because the fit failed; it is not a measured zero-success rate. Full-support counts are a post-run subset and may have a different denominator.",
            "",
            "Correct-window throughput, summed over fields, is recorded in `aggregate.json`; it is omitted from the table when a family has failed field fits so the partial-field rate is not mistaken for an all-field rate.",
        ]
    )
    report_lines.extend(
        [
            "",
            "## Fixed failure regressions",
            "",
            "| Case | Descriptor | Affine fit status | RBF fit status |",
            "|---|---|---:|---:|",
        ]
    )
    for row in safe:
        if row["split"] != "fixed_regression":
            continue
        for method, label in (("horg64", "HORG-64"), ("dic_cnn", "DIC-CNN hard-negative")):
            affine = row[method].get("affine", {})
            rbf = row[method].get("gaussian_rbf", {})
            affine_result = affine.get("solver", {}).get("correct_converged_lt2px")
            rbf_result = rbf.get("solver", {}).get("correct_converged_lt2px")
            report_lines.append(
                f"| {row['case']} `{row['texture_seed']}/{row['warp_seed']}` | {label} | {affine_result if affine_result is not None else affine.get('error', 'not run')} | {rbf_result if rbf_result is not None else rbf.get('error', 'not run')} |"
            )
    report_lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "The descriptor is not accepted as a general replacement for DeDoDe-B. The benchmark separates detector repeatability, descriptor retrieval, local consensus, field projection, and IC-GN convergence. For discontinuous fields, the full-support counts exclude supports that touch the invalid band; they do not imply that the field model is safe across a physical discontinuity.",
            "",
            "Raw per-window data and generated images are reproducible from the seeds and remain in the scratch directory. The committed artifacts are the compact JSON summaries and this report.",
            "",
        ]
    )
    (args.output / "README.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"saved qualification to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
