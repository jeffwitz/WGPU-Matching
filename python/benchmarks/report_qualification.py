"""Build the complete themed benchmark gallery from a qualification run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from qualify_all_initializers import LABELS


def case_title(name):
    if name.startswith("progressive_"):
        return "Live-DIC frame " + name.rsplit("_", 1)[1]
    if name.startswith("tensile_"):
        return "Homogeneous traction: " + name.split("_")[1].replace("pct", " percent")
    return "Adrien: reference 000000 and target 000075"


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def verdict(row):
    value = row["assessment"]["pass_below_5px"]
    if value is None:
        return "Not measurable: no analytic truth"
    return "Pass" if value else "Fail"


def error_summary(row):
    return row.get("assessment", {}).get("errors_px", {})


def first_stage(row):
    return row.get("affine_pipeline_median_s")


def link(directory, name):
    path = directory / name
    return path.as_posix()


def table(rows, root, case):
    lines = [
        "| Method | Requested basis → effective | Verdict | Estimated / visible | Correct ≤2 px | Correct <5 px | RMSE / P95 / max (px) | Verified / upstream / raw pairs | IRLS converged |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        assess, errors = row["assessment"], error_summary(row)
        raw = row.get("direct_pairs", {})
        lines.append(
            f"| {LABELS[row['method']]} | {row['basis_requested']} → {row.get('basis_effective') or 'none'} "
            f"| {verdict(row)} | {assess['estimated']} / {assess['requested']} "
            f"| {assess.get('correct_le2') if assess.get('correct_le2') is not None else '—'} "
            f"| {assess.get('correct_lt5') if assess.get('correct_lt5') is not None else '—'} "
            f"| {number(errors.get('rmse'))} / {number(errors.get('p95'))} / {number(errors.get('maximum'))} "
            f"| {raw.get('retained_count', '—')} / {raw.get('upstream_retained_count', '—')} / {raw.get('raw_count', '—')} "
            f"| {(row.get('diagnostics') or {}).get('irls_converged', '—')} |"
        )
    return lines


def build_report(root: Path):
    run = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    rows = run["rows"]
    case_names = manifest["protocol"]["cases"]
    methods = manifest["protocol"]["methods"]
    by_case = {name: [r for r in rows if r["case"] == name] for name in case_names}
    stress = {(r["method"], r["basis_requested"]): r for r in by_case.get("tensile_60pct", [])}
    lines = [
        "# Current initializer qualification and complete visual benchmark",
        "",
        (
            f"Run created: {manifest['created_utc']}. This report is generated from measured results. "
            "Historical studies remain available separately and do not supply missing current measurements."
        ),
        "",
        (
            "Jump to the [60-percent table](#qualification-results-60) or the "
            "[complete case galleries](#qualification-case-galleries)."
        ),
        "",
        (
            "Read the [scientific interpretation and failure analysis](interpretation.md) alongside the "
            "tables. It explains detector cost, geometric filtering, the affine prior and the cases where "
            "automatic radial bases reduce correct coverage."
        ),
        "",
        "## What counts as passing 60%?",
        "",
        (
            "A projection passes only when every analytically visible requested DIC centre has a finite "
            "estimated displacement with Euclidean error strictly below 5 pixels. Correct counts at ≤2 pixels, "
            "conditional error and false estimates are reported independently. A failed row can still recover "
            "some windows; its coverage remains in the table. Numerical IRLS convergence is a separate flag, "
            "not substituted for analytic accuracy. The pass threshold concerns centre translation; "
            "the six patch parameters are exported, but it is not a gradient-accuracy or downstream "
            "IC-GN convergence criterion."
        ),
        "",
        (
            "The primary tensile basis is affine, as chosen before this run. The secondary request selects "
            "between affine and Gaussian RBF using spatial cross-validation of observed pairs. Analytic truth "
            "never selects a fit, a basis or correspondences."
        ),
        "",
        "## Pipeline labels and metric terminology",
        "",
        "- Native baselines: DeepMatching with the INRIA hierarchy, HORG and CPM-HOG.",
        "- Learned matching pipelines: XFeat, EfficientLoFTR and RaCo-ALIKED-LightGlue+.",
        (
            "- Common-descriptor experiments: DaD and DeDoDe v2-L detectors, plus tensor and learned "
            "DIC detectors, all paired with DeDoDe-B. The HORG/SIFT detector control uses that same descriptor."
        ),
        "",
        (
            "GPU computations use WGSL or ONNX/WebGPU as recorded below; the common descriptor and "
            "its mutual matcher remain CPU reference stages. These labels identify different pipelines, "
            "not interchangeable detector-only experiments."
        ),
        "",
        (
            "Errors are in original-image pixels. RMSE, median error, P95 and maximum error summarize "
            "estimated centres only; correct coverage uses all visible requested centres. Definitions "
            "are given here before their short table labels so they do not expand inside narrow cells."
        ),
        "",
        "(qualification-results-60)=",
        "",
        "## All methods at 60% traction",
        "",
    ]
    lines += table(by_case.get("tensile_60pct", []), root, "tensile_60pct")
    not_converged = [
        r
        for r in rows
        if r.get("basis_effective")
        and not (r.get("diagnostics") or {}).get("irls_converged", False)
    ]
    lines += [
        "",
        "## Numerical convergence is a separate result",
        "",
        (
            f"Among {sum(bool(r.get('basis_effective')) for r in rows)} returned fields "
            f"(counting both requests), {len(not_converged)} did not meet the IRLS weight-change "
            "criterion within the configured iteration budget. Their accuracy counts are retained, "
            "but an accuracy pass must not be read as a numerically converged or production-qualified "
            "fit. No iteration budget or threshold was adjusted from analytic test errors. See each "
            "case's final weight change and full convergence diagnostics."
        ),
        "",
    ]
    lines += [
        "",
        (
            "Every method and requested basis has a reference/corrected-image GIF, including failed "
            "initializations. Open the 60% gallery below. If no accepted field exists, a reconstruction from "
            "rejected proposals is explicitly labelled diagnostic only; otherwise the uncorrected target "
            "is shown. Neither fallback enters the pass count."
        ),
        "",
        "![Correct coverage at every tensile strain](figures/correct_coverage_by_strain.png)",
        "",
        "![60-percent accuracy and complete affine-call latency](figures/stress_60_accuracy_and_latency.png)",
        "",
        "## Complete measured affine-pipeline timing",
        "",
        (
            "Each timed trial includes preprocessing of already decoded grayscale images, detector/descriptor "
            "or native matching, observed-pair export, affine field fitting, and patch initialization. "
            "Initialization of the device/models, image decoding, warm-up, data serialization and visualization "
            "are excluded. These are actual whole-call wall times, not sums of independently measured medians. "
            "The final measured pair set supplies the accuracy and figures. Models are not reference-cached."
        ),
        "",
        "| Case | Method | Repeats | Median / min / max (s) | Pair calls/s | Correct affine windows <5 px | Correct windows per second |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["basis_requested"] != "affine":
            continue
        times = row.get("affine_pipeline_wall_s", [])
        good = row["assessment"].get("correct_lt5")
        median = float(np.median(times)) if times else None
        lines.append(
            f"| {row['case']} | {LABELS[row['method']]} | {len(times)} "
            f"| {number(median, 3)} / {number(min(times) if times else None, 3)} / "
            f"{number(max(times) if times else None, 3)} | {number(1 / median if median else None, 2)} | {good if good is not None else '—'} "
            f"| {number(good / median if median and good is not None else None, 2)} |"
        )
    lines += [
        "",
        (
            "DeepMatching uses a 1024-original-pixel search radius and the complete hierarchy in this "
            "campaign. Its latency must not be compared as a speed regression against the historical "
            "roughly 1.3-second default-radius run or the sparse-root studies. The requested workload and "
            "output stage differ; no default matcher setting was changed."
        ),
        "",
        (
            "Automatic RBF fitting is measured once on the final correspondence set. Its fit/patch duration "
            "is shown in the individual galleries. It is not given an independently measured full-pipeline "
            "FPS. Peak graphics memory was not measured."
        ),
        "",
        "(qualification-case-galleries)=",
        "",
        "## Every case, including failures",
        "",
    ]
    for name in case_names:
        lines.append(f"- [{case_title(name)}: statistics and all GIFs](case_{name}.md)")
    lines += [
        "",
        "## Input resolution and backend boundaries",
        "",
        (
            "The source is UHD for every tensile row; the following are working resolutions, not "
            "independent claims of native-resolution detection. The common point budget is 8192; "
            "native HORG retains its existing 1024-feature setting. These are full-pipeline comparisons, "
            "not detector-only comparisons with a controlled descriptor."
        ),
        "",
        "| Pipeline | Tensile working input | Main execution boundary |",
        "|---|---|---|",
        "| DeepMatching | 960 × 540 descriptors from 3840 × 2160; search radius 1024 original pixels | Native WGSL hierarchy; host raw-pair export |",
        "| HORG | Reduction 4, 1024 features | Native WGSL detection/description/matching |",
        "| CPM-HOG | Reduction 4; 16-pixel work-grid stride; search radius 1024 original pixels | Native WGSL descriptor/search |",
        "| XFeat | 3840 × 2160 | ONNX/WebGPU; output pairs return to CPU |",
        "| EfficientLoFTR | 640 × 360 content, padded to 640 × 384 | Optimized ONNX/WebGPU; output pairs return to CPU |",
        "| RaCo-ALIKED-LightGlue+ | 640 × 360 content, padded to 640 × 384 | Optimized ONNX/WebGPU; output pairs return to CPU |",
        "| DaD and DeDoDe v2-L + B | 640 × 360 detector; 1024 × 576 common descriptor | ONNX/WebGPU detector; CPU description/matching |",
        "| Tensor and learned DIC + B | 3840 × 2160 detector; 1024 × 576 common descriptor | Native WGSL scoring; CPU description/matching |",
        "| HORG/SIFT + B | Reduction 4 detector; 1024 × 576 common descriptor | Native WGSL detector; CPU description/matching |",
        "",
        (
            "All rows then use the same WGSL local geometric verification and final field solver, with "
            "CPU robust-scale pilots and automatic model selection. Adrien input shapes are recorded "
            "individually; EfficientLoFTR uses its dynamic graph rather than the specialized tensile graph."
        ),
        "",
        "## Shared protocol and execution scope",
        "",
        (
            "- Tensor and learned DIC detection run on native image pixels through WGSL. "
            "DaD and DeDoDe v2 detection use ONNX/WebGPU at maximum side 640. All common-description "
            "variants use the same DeDoDe-B descriptor on CPU at the recorded maximum side and the same "
            "mutual cosine matcher, with no analytic correspondence filtering."
        ),
        (
            "- Native HORG, CPM-HOG and DeepMatching run through wgpu-py. Projection uses actual HORG "
            "descriptor pairs, CPM work-grid measurements and DeepMatching atom matches, retaining native "
            "rejection masks. Fitted CPM seed endpoints are never reused as direct observations."
        ),
        (
            "- Before projection, the same existing local affine RANSAC verifies observed pairs for every "
            "pipeline: 256-pixel neighbourhood, 3-pixel residual threshold, at least 6 inliers among at least "
            "8 candidates and an inlier fraction of at least 0.2. A WGSL kernel retains original measured "
            "pairs supported by a usable local model. It never replaces them by model predictions. The "
            "complete extraction timing includes this verification. These geometric assumptions are local; "
            "they do not establish accuracy in localization bands or across discontinuities."
        ),
        (
            "- The current projection solver accepts at most 8192 pairs. Larger verified sets are reduced "
            "deterministically using balanced 16 by 9 source-space quotas and uniform selection within cells. "
            "No score or analytic error enters this capacity reduction. Full pre-reduction observations "
            "remain downloadable and model provenance records the exact retained identifiers."
        ),
        (
            "- EfficientLoFTR and RaCo-ALIKED-LightGlue+ use the current optimized ONNX graphs where their "
            "input shapes permit. A specialized 640 by 384 graph is not used on the differently shaped Adrien "
            "input. The selected file and SHA-256 digest are recorded."
        ),
        (
            "- Tensor coordinates are restored by the inverse pixel-centre mapping "
            "(x+0.5) × native/content − 0.5. Inputs use bicubic resizing; output figures use bilinear sampling "
            "of the native target. Bottom/right zero padding does not change the coordinate origin."
        ),
        (
            "- Images for tensile cases are 3840 by 2160 pixels; Adrien is 7920 by 6004. XFeat uses native "
            "tensile input, and is bounded to maximum side 3840 for Adrien. Its input dimensions are recorded."
        ),
        (
            "- The same 128-pixel DIC windows and 512-pixel seed-grid spacing are used. Tensile centres must "
            "remain analytically visible in the target. The Adrien mask restricts source and target match "
            "endpoints and reference seed windows (at least 95 percent mask coverage); no analytic truth "
            "exists for its physical deformation. The mask filters endpoints after extraction and does not "
            "refill point budgets. Whole-image displays extrapolate the field outside the ROI and measured support."
        ),
        (
            "- Raw-pair counts refer to each pipeline's exported proposal stage, not all possible "
            "descriptor pairs. DeepMatching exports source-best atomic hypotheses; common descriptor "
            "paths export mutual proposals. Native acceptance and added geometric rejection are separate "
            "columns. An empty score means it was not exported; in particular the current common-B "
            "benchmark path does not archive individual cosine values. No synthetic confidence is filled in."
        ),
        (
            "- Reported field residuals, photometric differences and visual alignment do not establish "
            "physical displacement accuracy on Adrien. Its table links each method's separate synthetic "
            "60% verdict rather than assigning a ground-truth verdict to the experimental image."
        ),
        (
            "- All neural inference uses optional external weights. ONNX outputs return through CPU arrays "
            "before the independent wgpu-py projection context. Common description and matching are CPU "
            "reference stages. The whole benchmark is not resident on GPU."
        ),
        "",
        "## Artifact verification",
        "",
        (
            "The independent [metric and animation audit](validation.json) recomputes analytic errors, "
            "checks every expected method/case/basis row, verifies raw-pair counts, and decodes both frames "
            "of every animation. Non-converged fields and repeated-call output changes are listed explicitly."
        ),
        "",
        (
            "The separate [raw-projection ablation](../raw_projection_ablation_20261107/index.md) "
            "uses the exact archived observations without the additional common local consensus; "
            "it tests whether robust fitting alone can remove false matches."
        ),
        "",
        "## Provenance",
        "",
        (
            "[Run manifest, source fingerprints, models, images and exact settings](manifest.json). "
            "[Complete measured summary](summary.json). [Hardware and software environment](environment.json). "
            "Each gallery links all observed pairs, including "
            "rejections, individual seed parameters and the full compressed field/provenance record."
        ),
        "",
    ]
    (root / "index.md").write_text("\n".join(lines))
    for name, records in by_case.items():
        content = [
            f"# {case_title(name)}: all methods and both projection requests",
            "",
            "[Return to the complete qualification report](index.md).",
            "",
            (
                "The GIF alternates between the unchanged reference and the target sampled through the "
                "estimated forward displacement. No analytic warp is used to create the corrected image. "
                "The exact same viewport and shared palette are used for both frames; out-of-image samples "
                "remain black. In tensile corrections, side bands can be physically outside the target "
                "field of view even for a correct estimate; black alone is not a failure verdict. "
                "The numeric population includes only analytically visible requested centres. "
                "Unlabelled PNG previews are retained for external image viewers."
            ),
            "",
        ]
        image = next(v for v in manifest["protocol"]["images"] if v["case"] == name)
        width, height = image["size"]
        content += [
            f"Native reference and target: {width} × {height} pixels. Case identifier: `{name}`.",
            "",
        ]
        if image["analytic_truth"] is not None:
            truth = image["analytic_truth"]
            content += [
                (
                    f"Prescribed stretch factors: x = {truth['stretch_x']!r}, y = {truth['stretch_y']!r}. "
                    f"Axial engineering strain is {100 * (truth['stretch_x'] - 1):.6f} percent. "
                    f"The analytical map uses centre ({width / 2:g}, {height / 2:g}) and "
                    "target = centre + diag(stretch_x, stretch_y) × (source − centre)."
                ),
                "",
            ]
        if name.startswith("adrien"):
            content += [
                (
                    "No analytic truth is available. Match residuals and visible alignment are diagnostics; "
                    "the 60% status below refers only to the separate synthetic stress test."
                ),
                "",
            ]
        content += [
            (
                "Error metrics use original-image pixels: RMSE, median error, P95 and maximum error "
                "are conditional on estimated centres. Correct coverage retains every visible request "
                "in its denominator. IRLS convergence is reported independently."
            ),
            "",
        ]
        content += table(records, root, name)
        for method in methods:
            subset = [r for r in records if r["method"] == method]
            content += ["", f"## {LABELS[method]}", ""]
            tests = []
            for basis in ("affine", "gaussian_rbf"):
                row60 = stress.get((method, basis))
                tests.append(basis + ": " + (verdict(row60) if row60 else "not measured"))
            content += ["60% synthetic status — " + "; ".join(tests) + ".", ""]
            for row in subset:
                basis = row["basis_requested"]
                folder = Path(name) / method / basis
                content += [
                    f"### Requested {basis}; effective {row.get('basis_effective') or 'none'}",
                    "",
                    (
                        f"**{verdict(row)}.** {row['assessment']['estimated']} of "
                        f"{row['assessment']['requested']} requested centres have an estimated field."
                    ),
                    "",
                ]
                if row.get("fit_error") or row.get("error"):
                    content += [
                        "Failure reason: " + str(row.get("fit_error") or row.get("error")),
                        "",
                    ]
                if row.get("visualizations", {}).get("diagnostic_only"):
                    content += [
                        (
                            "**Diagnostic only:** accepted pairs did not produce a field. The GIF shows "
                            "a fit to rejected proposals to make the failure visible. This fit contributes "
                            "no usable estimates and cannot pass."
                        ),
                        "",
                    ]
                a = row["assessment"]
                errors = a.get("errors_px", {})
                content += [
                    (
                        f"Correct coverage (<5 px / all requested): {number(a.get('correct_coverage'), 3)}. "
                        f"Conditional precision (<5 px / estimated): {number(a.get('conditional_precision'), 3)}. "
                        f"False estimates at ≥5 px: {a.get('false_estimates_ge5', '—')}. "
                        f"Median analytic error: {number(errors.get('median'), 3)} px."
                    ),
                    "",
                ]
                trials = row.get("trials", [])
                if trials:
                    content += [
                        "| Timed component | Median / min / max (s) |",
                        "|---|---:|",
                    ]
                    components = [
                        ("Complete affine call", [t["full_affine_pipeline_s"] for t in trials])
                    ]
                    stage_names = sorted({k for t in trials for k in t.get("extraction_s", {})})
                    for stage in stage_names:
                        components.append(
                            (
                                stage,
                                [
                                    t["extraction_s"][stage]
                                    for t in trials
                                    if stage in t["extraction_s"]
                                ],
                            )
                        )
                    for stage in ("field_fit_s", "patch_projection_s"):
                        values = [
                            t["projection_s"][stage]
                            for t in trials
                            if stage in t.get("projection_s", {})
                        ]
                        if values:
                            components.append((stage, values))
                    for stage, values in components:
                        content.append(
                            f"| {stage} | {number(float(np.median(values)), 4)} / {number(min(values), 4)} / {number(max(values), 4)} |"
                        )
                    content += [
                        "",
                        (
                            "Components are nested: matching_pipeline_s includes extraction sub-stages and "
                            "geometric verification. Do not sum this table. These repeated complete-call "
                            "times measure the affine path; the automatic-RBF request reuses its last pair set."
                        ),
                        "",
                    ]
                content += [
                    (
                        "Execution/input details are preserved verbatim in the downloadable per-seed result "
                        "under extraction: detector dimensions and point counts, descriptor backend, selected "
                        "ONNX graph, preprocessing and stage times. The shared run manifest records native "
                        "image dimensions and model hashes."
                    ),
                    "",
                ]
                rbf_times = row.get("rbf_fit_single_run_stages_s")
                if rbf_times:
                    content += [
                        "Automatic-basis single fit/patch times (seconds): " + str(rbf_times) + ".",
                        "",
                    ]
                diag = row.get("diagnostics") or {}
                content += [
                    (
                        f"Selection reason: {diag.get('selection_reason', 'unavailable')}. "
                        f"IRLS converged: {diag.get('irls_converged', 'unavailable')}. "
                        f"Final weight change: {number(diag.get('irls_final_max_weight_delta'), 6)}."
                    ),
                    "",
                    "Projection capacity selection: "
                    + str(diag.get("projection_capacity_selection", "unavailable"))
                    + ".",
                    "",
                    (
                        f"![{name}, {LABELS[method]}, {basis}: reference and corrected target]"
                        f"({link(folder, 'comparison.gif')})"
                    ),
                    "",
                ]
                visuals = row.get("visualizations", {})
                if visuals.get("zoom_gif"):
                    content += [
                        "**Native-pixel inspection:** a fixed central crop, not a region selected by error.",
                        "",
                        (
                            f"![Native-resolution reference/corrected crop]"
                            f"({link(folder, visuals['zoom_gif'])})"
                        ),
                        "",
                    ]
                fraction = visuals.get("valid_warp_fraction_of_reference_roi")
                photo = visuals.get("preview_absolute_intensity_difference", {})
                content += [
                    (
                        f"Valid target-sampling fraction of the reference ROI: {number(fraction, 3)}. "
                        f"Preview absolute intensity difference median/P95: {number(photo.get('median'))}"
                        f"/{number(photo.get('p95'))}. These are photometric diagnostics, not displacement error."
                    ),
                    "",
                    (
                        f"[Reference PNG]({link(folder, visuals.get('reference_png', 'image1_ref.png'))}) · "
                        f"[Target PNG]({link(folder, visuals.get('corrected_png', 'image1_def_uncorrected.png'))})"
                    ),
                    "",
                ]
                parent = Path(name) / method
                for file, label in (
                    (parent / "correspondences.csv.gz", "Every observed pair and rejection"),
                    (folder / "result.json", "Per-seed errors, estimates and timings"),
                    (folder / "model.json.gz", "Full fitted field and provenance"),
                    (
                        folder / "rejected_diagnostic_model.json.gz",
                        "Rejected-proposal diagnostic field",
                    ),
                    (parent / "execution_error.json", "Execution error details"),
                ):
                    if (root / file).exists():
                        content.append(f"- [{label}]({file.as_posix()})")
                content += [""]
        (root / f"case_{name}.md").write_text("\n".join(content))
        # Compact visual index for systematic offline inspection, not an accuracy metric.
        tiles = []
        for row in records:
            folder = root / name / row["method"] / row["basis_requested"]
            target_name = row.get("visualizations", {}).get(
                "corrected_png", "image1_def_uncorrected.png"
            )
            path = folder / target_name
            if not path.exists():
                continue
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((320, 220))
            tile = Image.new("RGB", (320, 270), "#172331")
            tile.paste(thumb, (0, 50))
            draw = ImageDraw.Draw(tile)
            draw.text((6, 4), row["method"] + " / " + row["basis_requested"], fill="white")
            draw.text((6, 24), verdict(row)[:48], fill="white")
            tiles.append(tile)
        if tiles:
            columns = 4
            sheet = Image.new(
                "RGB", (columns * 320, ((len(tiles) + columns - 1) // columns) * 270), "#172331"
            )
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i % columns) * 320, (i // columns) * 270))
            qa = root / "visual_review"
            qa.mkdir(exist_ok=True)
            sheet.save(qa / f"{name}.png")
    return root / "index.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(build_report(args.run))


if __name__ == "__main__":
    main()
