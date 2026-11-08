"""Reproducible analytic benchmark for affine/RBF WGPU projection."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import wgpu
from wgpu_matching import CorrespondenceSet, ProjectionOptions, WGPUMatching

ROOT = Path(__file__).resolve().parents[2]


def displacement(points: np.ndarray) -> np.ndarray:
    radial = points - np.asarray([250.0, 210.0])
    bump = np.exp(-np.sum(radial * radial, axis=1) / (2 * 90.0**2))
    return np.column_stack(
        (6.0 + 0.03 * points[:, 0] + 14.0 * bump, -3.0 + 0.01 * points[:, 1] + 6.0 * bump)
    )


def make_correspondences() -> CorrespondenceSet:
    x, y = np.meshgrid(np.linspace(40, 440, 24), np.linspace(40, 340, 18))
    source = np.column_stack((x.ravel(), y.ravel()))
    rng = np.random.default_rng(20261106)
    target = source + displacement(source) + rng.normal(0.0, 0.08, source.shape)
    outliers = rng.choice(len(source), 14, replace=False)
    target[outliers] += rng.normal(0.0, 50.0, (len(outliers), 2))
    return CorrespondenceSet(
        source_xy=source,
        target_xy=target,
        image_size=(512, 420),
        matcher_id="synthetic-affine-plus-local-Gaussian",
        match_ids=tuple(f"pair-{index}" for index in range(len(source))),
        raw_scores=np.ones(len(source), dtype=np.float64),
        coordinate_mapping={"frame": "native synthetic pixels"},
    )


def truth_patch(center: np.ndarray, width: int, height: int) -> np.ndarray:
    xs = np.linspace(-(width - 1) / 2, (width - 1) / 2, 5)
    ys = np.linspace(-(height - 1) / 2, (height - 1) / 2, 5)
    offsets = np.asarray([(x, y) for y in ys for x in xs])
    values = displacement(center[None, :] + offsets)
    center_value = displacement(center[None, :])[0]
    gradient = np.linalg.lstsq(offsets, values - center_value, rcond=None)[0]
    return np.asarray(
        [
            center_value[0],
            center_value[1],
            gradient[0, 0],
            gradient[1, 0],
            gradient[0, 1],
            gradient[1, 1],
        ]
    )


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def benchmark(repeats: int) -> dict[str, Any]:
    pairs = make_correspondences()
    queries = np.asarray([[150.0, 130.0], [250.0, 210.0], [350.0, 285.0], [210.0, 290.0]])
    width, height = 128, 96
    result: dict[str, Any] = {
        "protocol": {
            "fixture": "synthetic affine field plus localized Gaussian displacement",
            "image_size_px": [512, 420],
            "direct_pair_count": len(pairs.source_xy),
            "outlier_count": 14,
            "noise_sigma_px": 0.08,
            "requested_patches": len(queries),
            "patch_size_px": [width, height],
            "warmups": 1,
            "repeats": repeats,
            "time_scope": "projection only; excludes feature extraction and image decoding",
            "host_roundtrip": "input arrays are uploaded; final pair diagnostics and patch outputs are read back",
            "irls_damping": 0.0,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "wgpu_py": wgpu.__version__,
        },
        "methods": {},
    }
    options_by_basis = {
        "affine": ProjectionOptions(basis="affine"),
        "gaussian_rbf_auto": ProjectionOptions(basis="gaussian_rbf"),
    }
    with WGPUMatching(device_type="gpu") as matcher:
        result["adapter"] = matcher.adapter_info
        for name, options in options_by_basis.items():
            samples = []
            last_model = None
            last_patches = None
            for repeat in range(repeats + 1):
                start = time.perf_counter()
                model = matcher.fit_initialization_field(pairs, (512, 420), options=options)
                patches = model.initialize_patches(queries, (width, height))
                total_ms = (time.perf_counter() - start) * 1000.0
                if repeat:
                    samples.append(
                        {
                            "total_ms": total_ms,
                            "cpu_selection_and_scale_pilot_ms": model.diagnostics[
                                "cpu_selection_and_scale_pilot_ms"
                            ],
                            "gpu_final_fit_wall_ms": model.diagnostics["gpu_final_fit_wall_ms"],
                            "gpu_patch_projection_wall_ms": model.diagnostics[
                                "last_field_evaluation_wall_ms"
                            ],
                        }
                    )
                last_model, last_patches = model, patches
            assert last_model is not None and last_patches is not None
            truth = np.asarray([truth_patch(center, width, height) for center in queries])
            actual = np.asarray([patch.parameters for patch in last_patches], dtype=np.float64)
            center_errors = np.linalg.norm(actual[:, :2] - truth[:, :2], axis=1)
            gradient_errors = np.linalg.norm(actual[:, 2:] - truth[:, 2:], axis=1)
            residuals = last_model.residuals_px
            point_predictions = last_model.evaluate(pairs.source_xy)
            point_truth = displacement(pairs.source_xy)
            point_errors = np.linalg.norm(point_predictions - point_truth, axis=1)
            result["methods"][name] = {
                "effective_basis": last_model.effective_basis,
                "selection_reason": last_model.diagnostics["selection_reason"],
                "selected_center_count": last_model.diagnostics["selected_center_count"],
                "selected_ell_normalized": last_model.diagnostics["selected_ell_normalized"],
                "selected_ridge": last_model.diagnostics["selected_ridge"],
                "gpu_solver_failures": last_model.diagnostics["gpu_solver_failures"],
                "candidate_scores": last_model.diagnostics["candidate_scores"],
                "irls_converged": last_model.diagnostics["irls_converged"],
                "irls_final_max_weight_delta": last_model.diagnostics[
                    "irls_final_max_weight_delta"
                ],
                "host_synchronizations_per_fit": last_model.diagnostics["gpu_fit_host_sync_count"],
                "estimated_design_buffer_bytes": last_model.diagnostics["gpu_design_buffer_bytes"],
                "robust_weight_below_0_25_count": int(
                    np.count_nonzero(last_model.robust_weights < 0.25)
                ),
                "fit_residual_median_px": float(np.median(residuals)),
                "fit_residual_p95_px": float(np.percentile(residuals, 95)),
                "analytic_point_error_median_px": float(np.median(point_errors)),
                "analytic_point_error_rmse_px": float(np.sqrt(np.mean(point_errors**2))),
                "analytic_point_error_lt2px": int(np.count_nonzero(point_errors < 2.0)),
                "analytic_point_error_lt5px": int(np.count_nonzero(point_errors < 5.0)),
                "patch_center_error_median_px": float(np.median(center_errors)),
                "patch_center_error_max_px": float(np.max(center_errors)),
                "patch_gradient_error_frobenius_median": float(np.median(gradient_errors)),
                "patch_support_rms_px": [patch.support_rms_px for patch in last_patches],
                "timings": {
                    field: quantiles([sample[field] for sample in samples]) for field in samples[0]
                },
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/projection/gpu_projection_synthetic_20261107.json",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
