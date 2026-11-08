"""Generate the printed evidence chapter from the validated complete campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qualify_all_initializers import LABELS
from report_qualification import case_title

ROOT = Path(__file__).resolve().parents[2]


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def list_table(title, headers, records, widths=None):
    lines = [f".. list-table:: {title}", "   :header-rows: 1"]
    if widths:
        lines.append("   :widths: " + " ".join(map(str, widths)))
    lines += ["", "   * - " + headers[0]]
    lines += ["     - " + str(v) for v in headers[1:]]
    for row in records:
        lines += ["   * - " + str(row[0])]
        lines += ["     - " + str(v) for v in row[1:]]
    return lines + [""]


def prepare(run):
    validation = json.loads((run / "validation.json").read_text())
    assert validation["status"].startswith("complete"), "Validate the campaign first"
    manifest = json.loads((run / "manifest.json").read_text())
    rows = json.loads((run / "summary.json").read_text())["rows"]
    environment = json.loads((run / "environment.json").read_text())
    relative = run.resolve().relative_to(ROOT)
    online = "../_generated_results/" + relative.relative_to("results").as_posix()
    lines = [
        ":orphan:",
        "",
        "Complete initializer qualification: measurements and visual evidence",
        "====================================================================",
        "",
        (
            "This chapter replaces the historical context tables for the current comparison. "
            "It is generated from the complete re-run and its independent artifact audit. "
            "All historical reports remain intact. The full web galleries contain every "
            "reference/corrected GIF, including failures; print shows static native-pixel crops."
        ),
        "",
        (
            "The measured pipelines include DeepMatching, HORG, CPM-HOG, XFeat, EfficientLoFTR, "
            "RaCo-ALIKED-LightGlue+, DaD and DeDoDe v2-L with DeDoDe-B, tensor and learned "
            "DIC detection with DeDoDe-B, and HORG/SIFT detection with the same descriptor."
        ),
        "",
        "What passing means",
        "------------------",
        "",
        (
            "A pass requires every analytically visible requested centre to have a finite "
            "displacement estimate with error strictly below 5 pixels. Exactly 5 pixels fails. "
            "Correct counts at most 2 pixels, conditional RMSE, P95, maximum error and false "
            "estimates are separate measurements. Passing assesses centre translation only, "
            "not an affine-gradient tolerance. No IC-GN convergence is measured. These "
            "are smooth field initializers inferred from verified direct matches, not "
            "independent image measurements at every seed."
        ),
        "",
        (
            "The primary affine basis was declared before this campaign. The secondary "
            "Gaussian-RBF request selects between affine and Gaussian candidates using "
            "spatial cross-validation of observed pairs. Analytic truth only defines visible "
            "evaluation seeds and scores errors afterward. It never selects pairs or a basis."
        ),
        "",
        (
            "The same local WGSL affine RANSAC verifies observations for all pipelines: "
            "256-pixel radius, 3-pixel residual threshold, at least 6 inliers from 8 candidates "
            "and an inlier fraction of 0.2. A second WGSL kernel retains original observed "
            "pairs supported by accepted local models. Large sets use balanced source-space "
            "quotas to meet the projector's 8192-pair capacity. All original pairs remain "
            "downloadable; no model-predicted endpoint is counted as a new observation."
        ),
        "",
        (
            "Homogeneous traction is exactly representable by a global affine field. "
            "This prior can recover more centres than the original local initializer. "
            "It does not establish performance on heterogeneous deformation or cracks. "
            "The raw-pair ablation documents what happens when common geometric verification "
            "is omitted. See :doc:`../explanation/field-projection` for the objective and limits."
        ),
        "",
        "Timing and execution scope",
        "--------------------------",
        "",
        (
            "Each affine latency is the median of three complete calls after a per-case warm-up, "
            "including decoded-image preprocessing, extraction, geometric verification, field "
            "fit and patch initialization. Device/model creation, decoding, serialization and "
            "rendering are excluded. Both images are processed each time; no prepared-reference "
            "cache is used. Minimum and maximum are the observed three-call range, not a "
            "statistical confidence interval. Automatic-basis fit time is measured once on "
            "the last pair set and has no separately measured complete-pipeline frame rate."
        ),
        "",
        (
            "Native tensile images are 3840 by 2160; Adrien is 7920 by 6004. Learned matching "
            "models use bounded reduced inputs with dimensions recorded in the source result. "
            "Tensor and learned DIC detection use native pixels, but the common DeDoDe-B "
            "descriptor and mutual matcher are CPU reference stages. ONNX outputs return host "
            "arrays before the independent wgpu-py context. This is not a pipeline resident "
            "on GPU from end to end, and it does not perform native-resolution photometric refinement."
        ),
        "",
        (
            "DeepMatching uses a 1024-original-pixel search radius and all hierarchy roots here. "
            "Its timing is not a regression comparison against the historical default-radius or "
            "sparse-root protocols. Matcher defaults were preserved; benchmark settings are explicit."
        ),
        "",
        (
            "The figures sample native target pixels through the estimated forward field with "
            "bilinear interpolation. Black indicates out-of-image samples, including expected "
            "loss of field of view under stretch; black margins alone do not establish failure. Source resize "
            "coordinates follow (x + 0.5) times scale minus 0.5. Central crops are fixed in "
            "advance and retain native pixel resolution; their location is never selected by error."
        ),
        "",
        "Run created: " + manifest["created_utc"] + ".",
        "",
        "Adapter: "
        + manifest["adapter"]["device"].replace("(LNL)", "(Lunar Lake)")
        + "; backend "
        + manifest["adapter"]["backend_type"]
        + "; driver "
        + manifest["adapter"]["description"]
        + ".",
        "",
        "CPU: "
        + environment["cpu_model"].replace("(R)", "").replace("(TM)", "")
        + "; numerical stages configured for eight threads.",
        "",
        "Protocol fingerprint: ``" + manifest["protocol_sha256"] + "``.",
        "",
        (
            f"The :doc:`complete online report <{online}/index>` contains all stage timings, "
            "models, thresholds, file hashes, per-seed parameters, raw-pair rejection reasons "
            "and every animation. The independent audit decoded all animations and recomputed "
            "analytic metric counts from the exported patch parameters."
        ),
        "",
        "Qualification at 60 percent traction",
        "------------------------------------",
        "",
    ]
    stress = [r for r in rows if r["case"] == "tensile_60pct" and r["basis_requested"] == "affine"]
    records = []
    for row in stress:
        a = row["assessment"]
        records.append(
            (
                LABELS[row["method"]],
                "Pass" if a["pass_below_5px"] else "Fail",
                f"{a['correct_le2']}/{a['requested']}",
                f"{a['correct_lt5']}/{a['requested']}",
                number(a.get("errors_px", {}).get("rmse")),
                number(row.get("affine_pipeline_median_s"), 3),
            )
        )
    lines += list_table(
        "Primary affine field; displacement errors in original pixels",
        ["Pipeline", "Verdict", "At most 2", "Below 5", "RMSE", "Seconds"],
        records,
        [34, 10, 12, 12, 14, 18],
    )
    for plot, caption in (
        (
            "correct_coverage_by_strain",
            "Correct coverage below 5 pixels at every tensile strain; labels are correct/requested counts.",
        ),
        (
            "stress_60_accuracy_and_latency",
            "Accuracy and measured complete-call latency at 60 percent. The observed timing range is not a confidence interval.",
        ),
    ):
        image = relative / "figures" / (plot + ".png")
        lines += [
            f".. figure:: ../../{image.as_posix()}",
            "   :width: 100%",
            "",
            "   " + caption,
            "",
        ]
    for case in manifest["protocol"]["cases"]:
        title = case_title(case)
        lines += [title, "-" * len(title), "", "Case identifier: ``" + case + "``.", ""]
        image = next(v for v in manifest["protocol"]["images"] if v["case"] == case)
        if image["analytic_truth"] is not None:
            truth = image["analytic_truth"]
            lines += [
                (
                    f"Native images: {image['size'][0]} by {image['size'][1]} pixels. "
                    f"Prescribed stretch factors: x = {truth['stretch_x']!r}, y = {truth['stretch_y']!r}; "
                    f"axial engineering strain {100 * (truth['stretch_x'] - 1):.6f} percent. "
                    "The analytical map scales coordinates about (width/2, height/2)."
                ),
                "",
            ]
        if case.startswith("adrien"):
            lines += [
                (
                    "No analytic truth is available for this experimental pair. No accuracy "
                    "pass is assigned. Residuals, visual alignment and photometric differences "
                    "are diagnostics only. Each method's synthetic 60-percent status appears "
                    "in the separate table above."
                ),
                "",
            ]
        else:
            lines += [
                (
                    "Counts use all visible requested centres. Missing estimates reduce coverage; "
                    "RMSE, median, P95 and maximum are conditional on finite estimated centres."
                ),
                "",
            ]
        af = [r for r in rows if r["case"] == case and r["basis_requested"] == "affine"]
        nonconverged = [
            LABELS[r["method"]]
            for r in af
            if r["basis_effective"]
            and not (r.get("diagnostics") or {}).get("irls_converged", False)
        ]
        lines += [
            "Affine IRLS fits without numerical convergence: "
            + (", ".join(nonconverged) if nonconverged else "none among the returned fields")
            + ". Accuracy and numerical convergence are separate checks.",
            "",
        ]
        count_records = []
        for row in af:
            a = row["assessment"]
            count_records.append(
                (
                    LABELS[row["method"]],
                    f"{a['estimated']}/{a['requested']}",
                    a.get("correct_le2") if a.get("correct_le2") is not None else "—",
                    a.get("correct_lt5") if a.get("correct_lt5") is not None else "—",
                    a.get("false_estimates_ge5")
                    if a.get("false_estimates_ge5") is not None
                    else "—",
                    number(a.get("correct_coverage"), 3),
                    number(a.get("conditional_precision"), 3),
                )
            )
        lines += list_table(
            "All requested centres versus finite estimates",
            ["Pipeline", "Estimated", "At most 2", "Below 5", "False ≥5", "Coverage", "Precision"],
            count_records,
            [31, 12, 11, 11, 11, 12, 12],
        )
        records = []
        for row in af:
            a = row["assessment"]
            e = a.get("errors_px", {})
            records.append(
                (
                    LABELS[row["method"]],
                    f"{a['estimated']}/{a['requested']}",
                    "—" if a["correct_lt5"] is None else str(a["correct_lt5"]),
                    number(e.get("rmse")),
                    number(e.get("median")),
                    number(e.get("p95")),
                    number(e.get("maximum")),
                )
            )
        lines += list_table(
            "Affine field accuracy, pixels",
            ["Pipeline", "Estimated", "Below 5", "RMSE", "Median", "P95", "Max"],
            records,
            [32, 12, 11, 11, 11, 11, 12],
        )
        records = []
        for row in af:
            times = row.get("affine_pipeline_wall_s", [])
            pairs = row.get("direct_pairs", {})
            records.append(
                (
                    LABELS[row["method"]],
                    number(row.get("affine_pipeline_median_s"), 3),
                    number(min(times) if times else None, 3),
                    number(max(times) if times else None, 3),
                    f"{pairs.get('retained_count', '—')}/{pairs.get('raw_count', '—')}",
                )
            )
        lines += list_table(
            "Repeated complete affine calls; observations after common verification",
            ["Pipeline", "Median s", "Min s", "Max s", "Verified/raw"],
            records,
            [36, 15, 15, 15, 19],
        )
        automatic = [
            r for r in rows if r["case"] == case and r["basis_requested"] == "gaussian_rbf"
        ]
        records = []
        for row in automatic:
            a = row["assessment"]
            e = a.get("errors_px", {})
            records.append(
                (
                    LABELS[row["method"]],
                    row["basis_effective"] or "none",
                    "—" if a["correct_lt5"] is None else f"{a['correct_lt5']}/{a['requested']}",
                    number(e.get("rmse")),
                    number(e.get("maximum")),
                    str((row.get("diagnostics") or {}).get("irls_converged", "—")),
                )
            )
        lines += list_table(
            "Automatic request: actual selected model and analytic errors",
            ["Pipeline", "Effective", "Below 5", "RMSE", "Max", "Converged"],
            records,
            [32, 18, 12, 12, 12, 14],
        )
        lines += [
            f":doc:`All methods, both requests and animations for this case <{online}/case_{case}>`.",
            "",
        ]
    lines += [
        "Native-pixel visual checks",
        "--------------------------",
        "",
        (
            "The following fixed central crops alternate in the web edition. Print places the "
            "unchanged reference on the left and corrected target on the right. All methods "
            "are shown at 60 percent and for Adrien, including failures. The main galleries "
            "also retain whole-image views and both basis requests."
        ),
        "",
    ]
    for case in ("tensile_60pct", "adrien_000000_to_000075"):
        for row in rows:
            if row["case"] != case or row["basis_requested"] != "affine":
                continue
            folder = run / case / row["method"] / "affine"
            images = [folder / "zoom_ref.png", folder / "zoom_def_cor.png"]
            if not all(p.is_file() for p in images):
                continue
            status = (
                "No analytic truth"
                if row["assessment"]["pass_below_5px"] is None
                else "Pass"
                if row["assessment"]["pass_below_5px"]
                else "Fail"
            )
            title = LABELS[row["method"]] + " / " + case_title(case)
            lines += [
                ".. raw:: latex",
                "",
                r"   \Needspace{115mm}",
                "",
                title,
                "~" * len(title),
                "",
                status + ".",
                "",
            ]
            if row["visualizations"].get("diagnostic_only"):
                lines += [
                    (
                        "No accepted field: the corrected image is a diagnostic fit of rejected "
                        "proposals and supplies no usable initialization."
                    ),
                    "",
                ]
            correction_label = (
                "rejected-proposal diagnostic"
                if row["visualizations"].get("diagnostic_only")
                else "estimated correction"
            )
            caption = f"{LABELS[row['method']]} / {case_title(case)}. {status}. Reference left; {correction_label} right."
            lines += [
                ".. list-table:: " + caption,
                "   :widths: 50 50",
                "",
            ]
            for i, image in enumerate(images):
                rel = image.resolve().relative_to(ROOT)
                directive = "   * - " if i == 0 else "     - "
                lines += [
                    directive + f".. image:: ../../{rel.as_posix()}",
                    "          :width: 100%",
                ]
            lines += [""]
    lines += [
        "Evidence that remains open",
        "--------------------------",
        "",
        (
            "These measurements qualify the reported homogeneous traction cases and provide "
            "experimental visual comparisons. They do not measure IC-GN convergence, crack "
            "handling or accuracy on heterogeneous physical deformation. A successful global "
            "affine projection is not proof that all requested regions were directly matched. "
            "The learned DIC score is trained on a structure-tensor proxy and consistency, "
            "not on actual convergence labels. Peak graphics and process memory were not measured."
        ),
        "",
    ]
    (ROOT / "docs/pdf/evidence.rst").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    prepare(args.run)


if __name__ == "__main__":
    main()
