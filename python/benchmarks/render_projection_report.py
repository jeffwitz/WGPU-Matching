"""Build the Sphinx-rendered projection report from saved benchmark summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "projection"
REPORT = RESULTS / "matcher_projection_experiments_20261107.md"
LABELS = {
    "raco_aliked_lightglue": "RaCo-ALIKED-LightGlue+",
    "efficientloftr": "EfficientLoFTR",
}
CASES = (
    "tensile_10pct",
    "tensile_20pct",
    "tensile_30pct",
    "tensile_40pct",
    "tensile_50pct",
    "tensile_60pct",
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name / "summary.json").read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _strain(case: str) -> str:
    return case.split("_")[1].replace("pct", "%")


def _find(rows: list[dict[str, Any]], case: str, model: str, requested: str) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["dataset"] == case and row["model"] == model and row["basis_requested"] == requested
    )


def build_report() -> Path:
    traction = _load("dataset_run_traction_visual_20261107")
    adrien = _load("dataset_run_adrien_visual_20261107")
    traction_rows = traction["results"]
    adrien_rows = adrien["results"]
    lines = [
        "# Affine and RBF projection of learned matches",
        "",
        "**Experiment date:** 2026-11-07. This page reports the projection experiment on analytic tensile images and the Adrien experimental pair. GIFs are qualitative; tables retain the measured quality and timing quantities. Machine-readable summaries and per-match CSV files are linked at the end.",
        "",
        "## Main conclusion for tensile initialization",
        "",
        "For the tested analytic traction series, **affine projection is the most reliable initialization choice**. RaCo affine stays below 5 px for every visible seed from 10% through 60% strain (136/136 overall; 16/16 at 60%, RMSE 2.03 px), and automatic selection chooses affine in all six RaCo cases. EfficientLoFTR affine stays below 5 px through 40%, reaches 18/20 at 50%, then fails at 60% (0/16). Automatic RBF selection recovers only one 60% seed below 5 px, with 78.05 px RMSE; this is still a failed initialization. RBF gives no consistent coverage gain at lower strains.",
        "",
        "This conclusion is specific to these traction images and seeds. The Adrien pair has no analytic displacement truth; its RBF residuals and appearance cannot establish which field is physically more accurate.",
        "",
        "## Flicker GIFs",
        "",
        "Each GIF alternates between the reference and the deformed image sampled through the projected displacement field (850 ms per frame). PNG source previews are capped at 1920 px wide; GIFs are 1280 px wide. Matching and field fitting used native image coordinates.",
        "",
        "### Adrien experimental pair: frames 000000 → 000075",
        "",
        "**Affine projection.**",
        "",
        "![Adrien reference and RaCo affine-corrected target](gifs/adrien_raco_affine.gif)",
        "",
        "![Adrien reference and EfficientLoFTR affine-corrected target](gifs/adrien_efficientloftr_affine.gif)",
        "",
        "**Automatic Gaussian-RBF request.**",
        "",
        "![Adrien reference and RaCo Gaussian-RBF-corrected target](gifs/adrien_raco_gaussian_rbf.gif)",
        "",
        "![Adrien reference and EfficientLoFTR Gaussian-RBF-corrected target](gifs/adrien_efficientloftr_gaussian_rbf.gif)",
        "",
        "On Adrien, EfficientLoFTR yields more mask-valid matches (2,675/3,272 versus 680/701 for RaCo). Its match-field residuals are also lower: affine median/P95 are 6.76/15.95 px versus 7.50/20.47 px; RBF values are 4.93/10.85 px versus 5.51/15.71 px. These are residuals to the fitted match field, not errors against ground truth. Both RBF IRLS fits are flagged as non-converged.",
        "",
        "### Analytic traction: successful and failed large-strain cases",
        "",
        "RaCo affine provides a successful 60% initialization; EfficientLoFTR with automatically selected Gaussian RBF diverges at the same strain.",
        "",
        "![60% traction reference and RaCo affine-corrected target, successful initialization](gifs/tensile_60pct_raco_affine_success.gif)",
        "",
        "![60% traction reference and EfficientLoFTR Gaussian-RBF-corrected target, failed initialization](gifs/tensile_60pct_efficientloftr_gaussian_rbf_failure.gif)",
        "",
        "At low strain, EfficientLoFTR affine is also successful (28/28 seeds below 5 px at 10%, RMSE 0.08 px).",
        "",
        "![10% traction reference and EfficientLoFTR affine-corrected target, successful initialization](gifs/tensile_10pct_efficientloftr_affine_success.gif)",
        "",
        "## Experimental protocol",
        "",
        "- **Analytic traction:** reference and targets are 3840 × 2160 px; the matching networks receive 640 × 360 px content. Cases are 10%, 20%, 30%, 40%, 50%, and 60% strain. Seeds are kept only when their analytically transformed points remain inside the target. Projection error is measured against the exact analytic image displacement at native reference coordinates.",
        "- **Adrien:** frames `img_000000_0.09422.tiff` → `img_000075_1608.88508.tiff`, each 7920 × 6004 px; network content is 640 × 485 px. `Mask.tiff` restricts correspondences and seed windows. No analytic displacement field is available.",
        "- **Models:** RaCo-ALIKED-LightGlue+ and EfficientLoFTR, using the ONNX weights and SHA-256 digests recorded in the summary JSON. The configured provider is Microsoft `WebGpuExecutionProvider`.",
        "- **Projection:** affine or spatial-CV selection between affine and Gaussian-RBF; WGSL via `wgpu-py` on Intel Graphics LNL, Mesa 25.2.8, Vulkan. ONNX match outputs pass through host arrays before projection, so this is not GPU-resident end to end.",
        "- **Timing:** one measured inference after warmup per model/dataset, one fit per requested basis. CPU selection/scale pilot, GPU final fit, GPU patch projection, and total fit wall time are reported separately. These are single-run measurements, not repeated-run medians.",
        "- **Runtime:** Python 3.12.3, NumPy 2.5.4, `wgpu-py` 0.32.0.",
        "- All 28 summary records completed with `status = ok`; the separate IRLS convergence flag and final weight delta are shown in the result tables.",
        "",
        "## WGPU port coverage: native kernels versus hybrid inference",
        "",
        "There is no defensible single percentage for WGPU coverage: an operator count ignores operator cost, synchronization, and tensor copies. The stage-level status is:",
        "",
        "| Component | WGPU coverage | Work still on host/CPU |",
        "|---|---|---|",
        "| DeepMatching | Descriptor/pyramid kernels, correlation, hierarchy aggregation and descent execute as WGSL through `wgpu-py` (and the browser frontend). | Image upload, root/hierarchy metadata construction, command orchestration, and final result readback. This is a native WGSL matcher, not zero-copy end to end. |",
        "| HORG-DIC | Gaussian/DoG pyramid, detection, descriptors, exact k=2 search, Lowe ratio, seed candidates, local RANSAC and leave-one-out diagnostics are WGSL kernels. | Host orchestration and selected counters/feature data readback remain; they introduce synchronization points. |",
        "| CPM-HOG | Experimental pyramid and PatchMatch search passes execute as WGSL. | Host builds settings/dispatch sequence and reads back final diagnostics. This method remains experimentally qualified only on limited cases. |",
        "| Affine/RBF projection | Final coefficient solve, field evaluation and per-patch projection execute in WGSL. | Automatic spatial cross-validation and robust-scale pilot use CPU float64; ONNX correspondences enter via host arrays. |",
        "| EfficientLoFTR / RaCo-ALIKED-LightGlue+ | Inference is delegated to ONNX Runtime's native `WebGpuExecutionProvider`; these neural networks are **not ports to this repository's WGSL shaders**. | The measured graph includes CPU fallback operators and substantial host transfers. |",
        "| RANSAC/ECC affine preconditioning | Dedicated RANSAC, affine warp and ECC accumulation/solve kernels exist in WGSL. | Python drives the ECC iterations and updates parameters from small readbacks; this is a separate affine-synthetic preconditioning experiment, not a learned-matcher 4K pipeline. |",
        "",
        "### Measured ONNX placement on Intel Graphics LNL",
        "",
        "The following are ONNX Runtime diagnostic trace event counts and cumulative event durations, not hardware GPU timestamps. Copy event time can include queue waits/synchronization, so it is not a clean percentage of compute:",
        "",
        "| Graph/profile | WebGPU nodes / event time | CPU nodes / event time | Host-transfer evidence |",
        "|---|---:|---:|---|",
        "| EfficientLoFTR baseline | 1,296 / 1,389 ms | 1,256 / 110 ms | 12 `MemcpyToHost` events; 33.5 MB estimated summed tensors, 1,102 ms event duration. CPU `ScatterND` runs 8 times; CPU `Expand`/`Concat` also appear. |",
        "| RaCo baseline | 2,314 / 2,956 ms | 544 / 285 ms | About 1.37 GB estimated summed tensors to host and 1.23 GB back to device; largest single host-bound tensor 251.7 MB. CPU `Selu` and `GridSample` appear. |",
        "| RaCo with SELU decomposed to WebGPU-supported primitives | 2,648 / 1,263 ms | 459 / 86 ms | Transfer estimates fall to 0.262 GB to host and 0.118 GB to device, but all 39 `GridSample` nodes still execute on CPU. |",
        "",
        "The SELU graph transform preserved accepted-seed counts and changed accepted seed positions by less than 0.00075 px on four test cases. Unprofiled pair time fell from 2.35–2.44 s to 1.42–1.62 s (about 31–40%), still above the 0.8 s target. The EfficientLoFTR `ScatterND` rewrite was rejected: its CPU nodes remained and the trace had more host copies. IOBinding and the ORT buffer-cache option showed no measured speed gain; no shared allocation between ONNX Runtime's Dawn device and `wgpu-py`'s wgpu-native device has been demonstrated.",
        "",
        "### Native 4K status",
        "",
        "The images are 4K (or larger in Adrien's case), but learned matcher inference is reduced to 640 × 360 content (640 × 485 for Adrien). There is **no implemented native-4K tiled/local refinement stage yet**. At 1/8 feature stride, a global EfficientLoFTR all-pairs FP32 matrix for a 3840 × 2160 pair would be 67.2 GB, so simply increasing the model input is not viable. The next missing stage is a bounded coarse-to-local pipeline, after eliminating or containing the costly ONNX CPU/copy paths.",
        "",
        "## Direct matcher results on analytic traction",
        "",
        "Raw errors compare retained direct matches with analytic displacement. Counts are relative to all retained matches.",
        "",
        "| Strain | Frame | Matcher | Raw matches | Mask-valid | Median error (px) | RMSE (px) | ≤2 px | <5 px | Inference (ms) |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for case in CASES:
        for model, model_label in LABELS.items():
            row = _find(traction_rows, case, model, "affine")
            lines.append(
                f"| {_strain(case)} | {row['frame']} | {model_label} | {row['raw_match_count']} "
                f"| {row['mask_roi_match_count']} | {_fmt(row['raw_direct_error_median_px'])} "
                f"| {_fmt(row['raw_direct_error_rmse_px'])} "
                f"| {row['direct_le2_count']}/{row['raw_match_count']} "
                f"| {row['direct_lt5_count']}/{row['raw_match_count']} "
                f"| {_fmt(row['inference_wall_ms'], 1)} |"
            )

    lines.extend(
        [
            "",
            "## Projected seed accuracy against analytic truth",
            "",
            "`Affine` rows force the affine basis. `Auto` rows request automatic selection and report the basis actually used after selection/fallback. Seed error is Euclidean displacement error at the requested DIC center. Field residual is the residual of observed matcher points to the fitted field; patch support RMS is a support residual. These are diagnostics, not analytic accuracy metrics.",
            "",
            "| Strain | Matcher | Request → effective | Seeds projected | Seed median (px) | Seed RMSE (px) | Seed P95 (px) | ≤2 px | <5 px | Field residual median/P95 (px) | Patch support RMS median/P95 (px) | IRLS converged | IRLS max Δw |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for case in CASES:
        for model, model_label in LABELS.items():
            for requested in ("affine", "gaussian_rbf"):
                row = _find(traction_rows, case, model, requested)
                request_label = "Affine" if requested == "affine" else "Auto"
                lines.append(
                    f"| {_strain(case)} | {model_label} "
                    f"| {request_label} → {row['basis_effective']} "
                    f"| {row['projected_seed_count']}/{row['seed_count']} "
                    f"| {_fmt(row['seed_error_median_px'])} | {_fmt(row['seed_error_rmse_px'])} "
                    f"| {_fmt(row['seed_error_p95_px'])} "
                    f"| {row['seed_error_le2_count']}/{row['seed_error_total_count']} "
                    f"| {row['seed_error_lt5_count']}/{row['seed_error_total_count']} "
                    f"| {_fmt(row['field_residual_median_px'])}/{_fmt(row['field_residual_p95_px'])} "
                    f"| {_fmt(row['seed_support_rms_median_px'], 4)}/{_fmt(row['seed_support_rms_p95_px'], 4)} "
                    f"| {row['irls_converged']} | {_fmt(row['irls_final_max_weight_delta'], 6)} |"
                )

    lines.extend(
        [
            "",
            "## Traction model selection and timing",
            "",
            "Automatic selection uses spatial cross-validation with a one-standard-error preference for affine. Candidate folds, scale and ridge sweeps are retained in the linked JSON summaries.",
            "",
            "| Strain | Matcher | Request → effective | Selection reason | RBF centers | ℓ normalized | Ridge | CPU select/scale (ms) | GPU fit (ms) | GPU patch projection (ms) | Patch API (ms) | Total fit wall (ms) |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in CASES:
        for model, model_label in LABELS.items():
            for requested in ("affine", "gaussian_rbf"):
                row = _find(traction_rows, case, model, requested)
                request_label = "Affine" if requested == "affine" else "Auto"
                lines.append(
                    f"| {_strain(case)} | {model_label} "
                    f"| {request_label} → {row['basis_effective']} "
                    f"| `{row['selection_reason']}` | {row['selected_center_count']} "
                    f"| {_fmt(row['selected_ell_normalized'], 3)} | {_fmt(row['selected_ridge'], 6)} "
                    f"| {_fmt(row['cpu_selection_and_scale_pilot_ms'], 1)} "
                    f"| {_fmt(row['gpu_final_fit_wall_ms'], 1)} "
                    f"| {_fmt(row['gpu_patch_projection_wall_ms'], 1)} "
                    f"| {_fmt(row['patch_api_wall_ms'], 1)} "
                    f"| {_fmt(row['fit_wall_ms'], 1)} |"
                )

    lines.extend(
        [
            "",
            "## Adrien experimental-image statistics without ground truth",
            "",
            "All 112 requested seeds were projected. `Input pairs` is raw matcher output; `mask-valid` is retained after requiring both endpoints to satisfy the mask ROI. Residuals do not give an analytic displacement error.",
            "",
            "| Matcher | Request → effective | Status | Input pairs | Mask-valid | Seeds | Field residual median/P95 (px) | Patch support RMS median/P95 (px) | IRLS converged | IRLS max Δw | Inference (ms) | CPU select (ms) | GPU fit (ms) | GPU patch projection (ms) | Patch API (ms) | Total fit (ms) |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in adrien_rows:
        request_label = "Affine" if row["basis_requested"] == "affine" else "Auto"
        lines.append(
            f"| {LABELS[row['model']]} | {request_label} → {row['basis_effective']} | {row['status']} "
            f"| {row['input_pairs']} | {row['retained_pairs']} "
            f"| {row['projected_seed_count']}/{row['seed_count']} "
            f"| {_fmt(row['field_residual_median_px'])}/{_fmt(row['field_residual_p95_px'])} "
            f"| {_fmt(row['seed_support_rms_median_px'], 4)}/{_fmt(row['seed_support_rms_p95_px'], 4)} "
            f"| {row['irls_converged']} | {_fmt(row['irls_final_max_weight_delta'], 6)} "
            f"| {_fmt(row['inference_wall_ms'], 1)} "
            f"| {_fmt(row['cpu_selection_and_scale_pilot_ms'], 1)} "
            f"| {_fmt(row['gpu_final_fit_wall_ms'], 1)} "
            f"| {_fmt(row['gpu_patch_projection_wall_ms'], 1)} "
            f"| {_fmt(row['patch_api_wall_ms'], 1)} "
            f"| {_fmt(row['fit_wall_ms'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "- **Affine is the recommended projection for these tensile initializations.** It gives the most stable coverage over the tested 10–60% series. RBF does not provide a dependable gain in seed accuracy or coverage; EfficientLoFTR RBF at 60% visibly diverges and its IRLS fit did not converge.",
            "- **EfficientLoFTR looks better on Adrien**, with more retained mask-valid matches and slightly smaller fitted-field residuals. The affine-corrected views are very close. RBF views look plausible, but both RBF fits have `irls_converged = false`; without analytic truth this is qualitative, not validation of physical accuracy.",
            "- **Do not transfer the synthetic 60% result to Adrien.** The traction dataset has exact analytic truth and a homogeneous imposed map; the experimental pair has no independent displacement truth.",
            "- **Not GPU-resident end to end.** Projection fits use WGSL on WebGPU, but ONNX outputs cross host arrays, and automatic spatial-CV/scale selection is CPU work. Timings are specific to the recorded setup.",
            "",
            "## Full results and reproducibility",
            "",
            "- [Traction summary JSON — includes all fields and candidate scores](dataset_run_traction_visual_20261107/summary.json)",
            "- [Traction raw correspondences](dataset_run_traction_visual_20261107/raw_correspondences.csv) · [per-seed projection predictions](dataset_run_traction_visual_20261107/patch_predictions.csv)",
            "- [Adrien summary JSON — includes all fields and candidate scores](dataset_run_adrien_visual_20261107/summary.json)",
            "- [Adrien raw correspondences](dataset_run_adrien_visual_20261107/raw_correspondences.csv) · [per-seed projection predictions](dataset_run_adrien_visual_20261107/patch_predictions.csv)",
            "- [ONNX operator-placement and transfer audit](../features/gpu_performance_audit_20261106.md) · [RaCo SELU-to-WebGPU optimization study](../features/raco_selu_webgpu_optimization_20261106.md)",
            "- [Combined EOG PNG sequence](eog_sequence_traction_adrien_20261107/README.md)",
            "- GIF generator: [render_projection_gifs.py](../../python/benchmarks/render_projection_gifs.py). Re-run with `python python/benchmarks/render_projection_gifs.py`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return REPORT


if __name__ == "__main__":
    print(build_report())
