"""Evaluate status-only 64→32→16 image initialization on fixed archived pairs."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from diagnose_shear_support import map_centres_to_grid
from solver_image_cache import read_solver_images
from wgpu_matching.seed_recovery import retry_small_subsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-frozen", type=Path, help="optional pre-calibrated image-residual suspicion rule"
    )
    parser.add_argument(
        "--exact-truth-directory",
        type=Path,
        help="post-hoc truth for archives recorded before exact-center correction",
    )
    args = parser.parse_args()
    calibration = json.loads(args.residual_frozen.read_text()) if args.residual_frozen else None
    if calibration and (
        calibration.get("selected") not in ("global", "min_quadrant")
        or calibration.get("comparison") != "score < threshold"
        or calibration.get("invalid_score_policy") != "suspect"
    ):
        raise ValueError("invalid or unavailable frozen residual classifier")
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    source = args.archive / "affine_ncc"
    source_protocol = json.loads((source / f"combined_seed{args.seed_offset}.json").read_text())[
        "protocol"
    ]
    exact = source_protocol.get("synthetic_truth_convention") == "analytic_exact_center"
    exact_shear = source_protocol.get("shear_truth_convention") == "analytic_exact_center"
    if not (exact or exact_shear or args.exact_truth_directory):
        raise ValueError("source archive does not certify exact-center analytical truth")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.coarse.glob("*.npz"))
    if not paths:
        raise ValueError("no image-derived coarse inputs")
    rows = []
    for path in paths:
        archive = source / f"windows_seed{args.seed_offset}" / path.name
        reference, target = read_solver_images(
            source / f"solver_images_seed{args.seed_offset}", archive
        )
        with np.load(path) as coarse, np.load(archive) as saved:
            centers = coarse["centres"].copy()
            if not np.array_equal(centers, saved["centres_xy_native"]):
                raise ValueError("incompatible centers")
            valid = coarse["valid_roi"].astype(bool)
            shape = (len(np.unique(centers[:, 1])), len(np.unique(centers[:, 0])))
            suspect = None
            residual_stats = None
            if calibration:
                from wgpu_matching import WGPUMatching
                from wgpu_matching.affine_residual import AffineResidualWGPU

                residual_start = time.perf_counter()
                with WGPUMatching(device_type="gpu") as matcher:
                    residual = AffineResidualWGPU(matcher.device).evaluate(
                        reference, target, coarse["parameters"], coarse["accepted"]
                    )
                kind = calibration["selected"]
                score = (
                    residual["correlations"][:, 0]
                    if kind == "global"
                    else np.min(residual["correlations"][:, 1:], axis=1)
                ).astype(np.float64)
                threshold = calibration["candidates"][kind]["threshold"]
                suspect = coarse["accepted"].astype(bool) & (
                    ~residual["valid"] | (score < threshold)
                )
                residual_stats = {
                    "rule": kind,
                    "threshold": threshold,
                    "unavailable_accepted": int(
                        (coarse["accepted"].astype(bool) & ~residual["valid"]).sum()
                    ),
                    "suspect": int(suspect.sum()),
                    "wall_s_including_upload_and_readback": time.perf_counter() - residual_start,
                    "uploaded_image_bytes": residual["uploaded_image_bytes"],
                    "score_readback_bytes": residual["readback_bytes"],
                }
            stage_timings, solver = [], None

            def solve(
                size,
                indices,
                initial,
                *,
                reference=reference,
                centers=centers,
                target=target,
                stage_timings=stage_timings,
            ):
                nonlocal solver
                if solver is not None:
                    solver.close()
                stride = size // 2
                before_reference = time.perf_counter()
                solver = ICGNWGPU(
                    device_type="gpu",
                    prefer_subgroups=False,
                    subset_size=size,
                    subset_stride=stride,
                    interpolation="linear",
                    motion_model="affine",
                    outlier_policy="disabled",
                )
                solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=0.65)
                reference_s = time.perf_counter() - before_reference
                full_indices = map_centres_to_grid(
                    centers, size, stride, solver.grid_width, solver.grid_height
                )[indices]
                count = solver.grid_width * solver.grid_height
                state = np.zeros((count, 8), np.float32)
                state[full_indices, :6] = initial
                state[full_indices, 6] = 1
                active = np.zeros(count, np.uint32)
                active[full_indices] = 1
                before_solve = time.perf_counter()
                if hasattr(solver, "register_seeded"):
                    grid_shape = (solver.grid_height, solver.grid_width)
                    result = solver.register_seeded(
                        target, state[:, :6].reshape(*grid_shape, 6), active.reshape(grid_shape).astype(bool)
                    )
                    api = "public_v1"
                else:
                    result = solver._register_seeded_frame(target, state, active)
                    api = "private_legacy"
                solve_s = time.perf_counter() - before_solve
                stage_timings.append(
                    {
                        "subset": size,
                        "requested": len(indices),
                        "reference_setup_s": reference_s,
                        "solve_wall_s": solve_s,
                        "solver_grid_windows": count,
                        "seeded_api": api,
                    }
                )
                return (
                    np.asarray(result.parameters).reshape(-1, 6)[full_indices],
                    np.asarray(result.status).reshape(-1)[full_indices] > 0.5,
                )

            try:
                start = time.perf_counter()
                output = retry_small_subsets(
                    centers,
                    shape,
                    valid,
                    coarse["parameters"],
                    coarse["accepted"],
                    solve,
                    suspect=suspect,
                )
                retry_wall = time.perf_counter() - start
            finally:
                if solver is not None:
                    solver.close()
            # Ground truth is deliberately read only after the retry is complete.
            if args.exact_truth_directory:
                with np.load(args.exact_truth_directory / path.name) as corrected:
                    truth = corrected["truth_parameters"].copy()
                    visible = corrected["visible"].astype(bool)
            else:
                truth, visible = saved["truth_parameters"], saved["visible"].astype(bool)
            if residual_stats is not None:
                baseline_error = np.linalg.norm(
                    coarse["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1
                )
                true_bad = visible & coarse["accepted"].astype(bool) & (baseline_error >= 2)
                true_good = visible & coarse["accepted"].astype(bool) & (baseline_error < 2)
                residual_stats.update(
                    {
                        "true_bad": int(true_bad.sum()),
                        "flagged_bad": int((true_bad & suspect).sum()),
                        "true_good": int(true_good.sum()),
                        "flagged_good": int((true_good & suspect).sum()),
                    }
                )
            outputs = {
                "64": {
                    "parameters": coarse["parameters"],
                    "accepted": coarse["accepted"].astype(bool),
                },
                "32": output["snapshots"][32],
                "16": output["snapshots"][16],
            }
            metrics = {}
            for key, result in outputs.items():
                error = np.linalg.norm(result["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                accepted = result["accepted"].astype(bool)
                good = visible & accepted & (error < 2)
                bad = visible & accepted & (error >= 2)
                gradient_error = np.linalg.norm(
                    result["parameters"][:, [1, 2, 4, 5]] - truth[:, [1, 2, 4, 5]], axis=1
                )
                metrics[key] = {
                    "visible": int(visible.sum()),
                    "accepted_visible": int((visible & accepted).sum()),
                    "correct": int(good.sum()),
                    "false": int(bad.sum()),
                    "accepted_invisible": int((~visible & accepted).sum()),
                    "gradient_rmse_correct": float(np.sqrt(np.mean(gradient_error[good] ** 2)))
                    if good.any()
                    else None,
                }
            rows.append(
                {
                    "pair": path.stem,
                    "family": path.stem.split("_texture", 1)[0],
                    "metrics": metrics,
                    "attempts_32": int(np.count_nonzero(output["attempts"] >= 1)),
                    "attempts_16": int(np.count_nonzero(output["attempts"] >= 2)),
                    "accepted_32": int((output["stage"] == 32).sum()),
                    "accepted_16": int((output["stage"] == 16).sum()),
                    "residual": residual_stats,
                    "retry_wall_s": retry_wall,
                    "stage_timings": stage_timings,
                }
            )
            np.savez_compressed(
                args.output_dir / path.name,
                centres=centers,
                valid_roi=valid,
                visible=visible,
                truth=truth,
                coarse_parameters=outputs["64"]["parameters"],
                coarse_accepted=outputs["64"]["accepted"],
                stage32_parameters=outputs["32"]["parameters"],
                stage32_accepted=outputs["32"]["accepted"],
                parameters=output["parameters"],
                accepted=output["accepted"],
                stage=output["stage"],
                source=output["source"],
                attempts=output["attempts"],
                rejection_reason=output["rejection_reason"],
                suspect=output["suspect"],
                **(
                    {
                        "residual_correlations": residual["correlations"],
                        "residual_valid": residual["valid"],
                    }
                    if residual_stats is not None
                    else {}
                ),
            )
        print(f"Completed {path.stem}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
