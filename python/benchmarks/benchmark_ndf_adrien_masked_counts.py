"""Run an image-only HORG affine seed and IC-GN propagation on the masked Adrien pair."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qualify_ndf_pair_icgn import _frontier_propagation, centres, fit_global_affine_ransac
from wgpu_matching.api import HORGOptions, WGPUMatching


def _parameters_from_affine(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    mapped = np.column_stack((np.ones(len(points)), points)) @ coefficients
    gradient = coefficients[1:, :].T
    return np.column_stack((
        mapped[:, 0] - points[:, 0],
        np.full(len(points), gradient[0, 0] - 1),
        np.full(len(points), gradient[0, 1]),
        mapped[:, 1] - points[:, 1],
        np.full(len(points), gradient[1, 0]),
        np.full(len(points), gradient[1, 1] - 1),
    )).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adrien-dir", type=Path,
                        default=Path("/home/jeff/CNRS/DIC/AdrienVariational"))
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    reference_path = args.adrien_dir / "img_000000_0.09422.tiff"
    target_path = args.adrien_dir / "img_000075_1608.88508.tiff"
    mask_path = args.adrien_dir / "Mask.tiff"
    reference = np.asarray(Image.open(reference_path).convert("L"), dtype=np.uint8)
    target = np.asarray(Image.open(target_path).convert("L"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if reference.shape != target.shape or mask.shape != reference.shape:
        raise ValueError("reference, target, and ROI mask dimensions must agree")
    height, width = reference.shape
    requested = centres(width, height, subset=64, stride=64)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    matcher = WGPUMatching(device_type="gpu")
    start = time.perf_counter()
    matcher_error = None
    try:
        field = matcher.match_horg(
            reference, target,
            options=HORGOptions(
                patch_size=64, stride=64, reduction=2, max_features=4096, seed_points=None
            ),
            include_correspondences=True,
        )
    except ValueError as error:
        matcher_error = str(error)
        field = None
    matching_s = time.perf_counter() - start
    pairs = np.asarray(
        field.metadata.get("raw_correspondences_xyxy", []) if field else [], dtype=np.float64
    ).reshape(-1, 4)
    accepted_pairs = np.asarray(
        field.metadata.get("raw_correspondence_accepted", []) if field else [], dtype=bool
    ).reshape(-1)
    selected = pairs[accepted_pairs]
    coefficients, inliers, residual = fit_global_affine_ransac(
        selected[:, :2], selected[:, 2:], threshold=5.0, min_inliers=24
    ) if len(selected) >= 24 else (None, np.zeros(len(selected), dtype=bool), None)
    diagnostics = {
        "algorithm": "HORG affine fit, WebGPU local normalized cross-correlation, then IC-GN",
        "matcher_error": matcher_error,
        "truth_used": False,
        "roi_used_for_matching_or_affine_selection": False,
        "roi_used_for_dic_reference_validity": True,
        "raw_correspondences": len(pairs),
        "matcher_accepted_correspondences": int(accepted_pairs.sum()),
        "affine_inliers": int(inliers.sum()),
        "affine_fit_available": coefficients is not None,
        "affine_fit_median_residual_px": residual,
        "matching_and_affine_seconds": matching_s,
        "local_ncc_search": "WebGPU, 320x243 raster, radius 8 and patch radius 4; no ROI mask used in matching",
        "solver": "WebGPU inverse-compositional Gauss-Newton, affine, 64 px subset/stride, 30 iterations, 1e-3 tolerance, ZNCC minimum 0.65",
    }
    if coefficients is None:
        accepted = np.zeros(len(requested), dtype=bool)
        propagated = {
            "status": accepted,
            "seed_mask": accepted,
            "rings": [],
            "extra_accepted": 0,
            "wall_s": 0.0,
            "parameters": np.zeros((len(requested), 6), dtype=np.float32),
            "ring": np.zeros(len(requested), dtype=np.uint8),
            "attempts": np.zeros(len(requested), dtype=np.uint8),
        }
        _save_maps(out, requested, accepted, accepted, propagated["ring"])
        np.savez_compressed(
            out / "per_window_status.npz", centres_xy=requested,
            initial_status=accepted.astype(np.uint8),
            propagated_status=accepted.astype(np.uint8), ring=propagated["ring"],
            attempts=propagated["attempts"],
        )
    else:
        # This model raster keeps the parallel WebGPU local-correlation dispatch
        # below the device's per-dimension workgroup limit for the 17x17 search.
        model_width, model_height = 320, 243
        reference_model = np.asarray(
            Image.fromarray(reference).resize((model_width, model_height), Image.Resampling.BOX),
            dtype=np.uint8,
        )
        target_model = np.asarray(
            Image.fromarray(target).resize((model_width, model_height), Image.Resampling.BOX),
            dtype=np.uint8,
        )
        scale_x, scale_y = width / model_width, height / model_height
        feature_width, feature_height = (model_width + 3) // 4, (model_height + 3) // 4
        yy, xx = np.meshgrid(np.arange(feature_height), np.arange(feature_width), indexing="ij")
        source_model = np.stack(
            ((xx + 0.5) * model_width / feature_width - 0.5,
             (yy + 0.5) * model_height / feature_height - 0.5), axis=-1,
        )
        source_native = np.stack(
            ((source_model[..., 0] + 0.5) * scale_x - 0.5,
             (source_model[..., 1] + 0.5) * scale_y - 0.5), axis=-1,
        )
        target_native = np.concatenate(
            (np.ones((*source_native.shape[:2], 1)), source_native), axis=-1
        ) @ coefficients
        affine_displacement = np.stack(
            ((target_native[..., 0] - source_native[..., 0]) / scale_x,
             (target_native[..., 1] - source_native[..., 1]) / scale_y), axis=-1,
        ).astype(np.float32)
        ncc_start = time.perf_counter()
        local_ncc = matcher.refine_ndf_local_ncc(
            reference_model, target_model, affine_displacement,
            search_radius=8, patch_radius=4, geometry="translation",
        )
        ncc_s = time.perf_counter() - ncc_start
        refined_displacement = np.where(
            local_ncc.valid[..., None], local_ncc.displacement_xy, affine_displacement
        ).astype(np.float32)
        solver = ICGNWGPU(
            device_type="gpu", prefer_subgroups=False, subset_size=64,
            subset_stride=64, interpolation="linear", motion_model="affine",
            outlier_policy="disabled",
        )
        solver.set_reference(
            reference, iterations=30, tolerance=1e-3, zncc_min=0.65,
            roi_mask=mask,
        )
        state = np.zeros((len(requested), 8), dtype=np.float32)
        model_points = np.stack(
            ((requested[:, 0] + 0.5) / scale_x - 0.5,
             (requested[:, 1] + 0.5) / scale_y - 0.5), axis=-1,
        )
        norm_x = 2.0 * (model_points[:, 0] + 0.5) / model_width - 1.0
        norm_y = 2.0 * (model_points[:, 1] + 0.5) / model_height - 1.0
        sample_grid = torch.from_numpy(np.stack((norm_x, norm_y), axis=-1)).float()
        sample_grid = sample_grid.reshape(1, len(requested), 1, 2)
        field_tensor = torch.from_numpy(refined_displacement).permute(2, 0, 1)[None]
        sampled_displacement = torch.nn.functional.grid_sample(
            field_tensor, sample_grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0, :, :, 0].T.numpy()
        native_displacement = sampled_displacement * np.array([scale_x, scale_y], dtype=np.float32)
        state[:, :6] = _parameters_from_affine(requested, coefficients)
        state[:, 0] = native_displacement[:, 0]
        state[:, 3] = native_displacement[:, 1]
        state[:, 6] = 1
        active = np.ones(solver.roi.selected.shape, dtype=np.uint32)
        solve_start = time.perf_counter()
        result = solver._register_seeded_frame(target, state, active)
        solve_s = time.perf_counter() - solve_start
        initial_parameters = np.asarray(result.parameters).reshape(-1, 6)
        initial_status = np.asarray(result.status).reshape(-1) > 0.5
        propagated = _frontier_propagation(solver, target, requested, initial_parameters, initial_status)
        diagnostics.update({
            "one_pass_accepted": int(initial_status.sum()),
            "propagated_accepted": int(propagated["status"].sum()),
            "propagation_additional_accepted": int(propagated["extra_accepted"]),
            "propagation_rings": propagated["rings"],
            "solve_seconds": solve_s,
            "propagation_seconds": propagated["wall_s"],
            "local_ncc_seconds": ncc_s,
            "local_ncc_valid_cells": int(local_ncc.valid.sum()),
            "affine_coefficients_xy": coefficients.tolist(),
        })
        _save_maps(out, requested, initial_status, propagated["status"], propagated["ring"])
        np.savez_compressed(
            out / "per_window_status.npz",
            centres_xy=requested,
            roi_mask=mask.astype(np.uint8),
            initial_status=initial_status.astype(np.uint8),
            propagated_status=propagated["status"].astype(np.uint8),
            ring=propagated["ring"],
            attempts=propagated["attempts"],
            initial_parameters=initial_parameters,
            propagated_parameters=propagated["parameters"],
        )
        solver.close()
    matcher.close()
    (out / "summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Masked Adrien pair: accepted windows only",
        "",
        "Digital image correlation (DIC) windows were initialized from histogram of oriented and rotated gradients (HORG) correspondences and an image-derived affine estimate, refined by normalized cross-correlation (NCC), then processed by the WebGPU inverse-compositional Gauss-Newton (IC-GN) solver. Zero-mean normalized cross-correlation (ZNCC) is the solver's acceptance score. The supplied region-of-interest (ROI) mask was used only by the solver's reference validity; it did not select matching points or the affine fit. This experimental pair has no analytical displacement truth, so the report gives acceptance counts and maps only, not accuracy.",
        "",
        f"- HORG correspondences: {diagnostics['raw_correspondences']} raw, {diagnostics['matcher_accepted_correspondences']} accepted.",
        f"- Affine fit: {diagnostics['affine_inliers']} inliers; available={diagnostics['affine_fit_available']}.",
        f"- WebGPU local NCC valid cells: {diagnostics.get('local_ncc_valid_cells', 0)}; elapsed {diagnostics.get('local_ncc_seconds', 0):.3f} s.",
        f"- IC-GN accepted: {diagnostics.get('one_pass_accepted', 0)} after one pass; {diagnostics.get('propagated_accepted', 0)} after propagation.",
        f"- Propagation added: {diagnostics.get('propagation_additional_accepted', 0)}; elapsed {diagnostics.get('propagation_seconds', 0):.3f} s.",
        "",
        "Maps: `accepted_windows.png` (green after one pass, blue added by propagation, gray rejected) and `ring_map.png` (ring zero is the seed; successive propagation rings use distinct colors; gray is rejected). Exact ring indices remain in the per-window archive.",
        "", "Reproduce the count-and-map run with:", "", "```console",
        "PYTHONPATH=python/src:python/benchmarks .venv/bin/python python/benchmarks/benchmark_ndf_adrien_masked_counts.py \\",
        "  --live-dic-source /path/to/Live-DIC/python/src \\",
        "  --output-dir results/ndf_dic_20261109/adrien_masked_horg_propagation_20261109",
        "```",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "summary.json")


def _save_maps(
    output: Path, points: np.ndarray, initial: np.ndarray, propagated: np.ndarray,
    rings: np.ndarray,
) -> None:
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    grid_x = np.searchsorted(xs, points[:, 0])
    grid_y = np.searchsorted(ys, points[:, 1])
    colors = np.full((len(ys), len(xs), 3), 75, dtype=np.uint8)
    colors[grid_y[propagated], grid_x[propagated]] = (55, 170, 65)
    added = propagated & ~initial
    colors[grid_y[added], grid_x[added]] = (55, 100, 220)
    image = Image.fromarray(colors).resize((len(xs) * 5, len(ys) * 5), Image.Resampling.NEAREST)
    image.save(output / "accepted_windows.png")
    ring_grid = np.full((len(ys), len(xs)), 255, dtype=np.uint8)
    ring_grid[grid_y[propagated], grid_x[propagated]] = np.minimum(rings[propagated], 6)
    palette = np.asarray([
        (55, 170, 65), (55, 100, 220), (240, 180, 45), (195, 80, 195),
        (45, 185, 190), (230, 115, 45), (230, 220, 75), (75, 75, 75),
    ], dtype=np.uint8)
    ring_rgb = palette[ring_grid.clip(max=7)]
    ring_image = Image.fromarray(ring_rgb).resize(
        (len(xs) * 5, len(ys) * 5), Image.Resampling.NEAREST
    )
    ring_image.save(output / "ring_map.png")


if __name__ == "__main__":
    main()
