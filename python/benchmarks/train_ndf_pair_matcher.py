"""Train and validate a small pair-conditioned global matcher on speckles.

This deliberately bounded first prediction experiment is a gate for a larger
deformation-field model, not a production training recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from ndf_field import (
    feature_grid_displacement_field,
    fit_bspline_field,
    recommended_control_shape,
)
from ndf_pair_model import NDFPairMatcher, pair_matching_loss
from numpy.random import default_rng
from wgpu_matching.dic_synthetic import generate_speckle_pair


def make_dataset(
    count: int, offset: int, width: int, height: int, native_scale: int, speckle_size: float = 3.5
):
    examples = []
    modes = ("translation", "heterogeneous")
    for index in range(count):
        mode = modes[index % len(modes)]
        pair = generate_speckle_pair(
            51000 + offset + 2 * index,
            91000 + offset + 2 * index,
            width * native_scale,
            height * native_scale,
            motion_mode=mode,
            texture_options={
                "characteristic_size_px": speckle_size * native_scale,
                "coverage": 0.31,
                "blur_sigma_px": (0.45 * speckle_size / 3.5) * native_scale,
                # The edge softness is dimensionless. Keep it fixed as the
                # native raster scale changes; the historical scale-eight
                # training value is 0.18 + 0.12 * (0.45 * 8) = 0.612.
                "softness": 0.612,
                "noise_std": 0.025,
            },
        )
        reference_native = torch.from_numpy(pair.reference.copy())[None, None]
        target_native = torch.from_numpy(pair.target.copy())[None, None]
        flow_native = torch.from_numpy(pair.flow_xy.copy()).permute(2, 0, 1)[None]
        visible_native = torch.from_numpy(pair.visible.copy())[None, None].float()
        reference = torch.nn.functional.interpolate(reference_native, size=(height, width), mode="area")
        target = torch.nn.functional.interpolate(target_native, size=(height, width), mode="area")
        flow = torch.nn.functional.interpolate(
            flow_native, size=(height, width), mode="bilinear", align_corners=False
        ) / native_scale
        visible = torch.nn.functional.interpolate(visible_native, size=(height, width), mode="nearest")[:, 0] > 0.5
        examples.append(
            (
                reference,
                target,
                flow,
                visible,
                mode,
            )
        )
    return examples


def load_or_make_dataset(cache_path: Path, count: int, offset: int, width: int, height: int,
                         native_scale: int, speckle_size: float):
    """Reuse deterministic synthetic pairs across matched training ablations."""
    configuration = {
        "count": count,
        "seed_offset": offset,
        "width": width,
        "height": height,
        "native_scale": native_scale,
        "speckle_size": speckle_size,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("configuration") != configuration:
            raise ValueError(f"Dataset cache configuration mismatch: {cache_path}")
        print(json.dumps({"stage": "dataset_cache_hit", "path": str(cache_path)}), flush=True)
        return cached["examples"]
    print(json.dumps({"stage": "dataset_generation_started", "path": str(cache_path), **configuration}), flush=True)
    examples = make_dataset(count, offset, width, height, native_scale, speckle_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"configuration": configuration, "examples": examples}, temporary_path)
    temporary_path.replace(cache_path)
    print(json.dumps({"stage": "dataset_cache_written", "path": str(cache_path)}), flush=True)
    return examples


def stack_batch(examples):
    return tuple(torch.cat([sample[index] for sample in examples], dim=0) for index in range(4))


def local_affine_consensus(source_xy, target_xy, width, height, *, radius=20.0, threshold=2.5):
    """Fit a local affine map per coarse source cell using image-only matches."""
    rng = default_rng(417)
    source_xy = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    result = np.zeros((len(source_xy), 6), dtype=np.float64)
    usable = np.zeros(len(source_xy), dtype=bool)
    finite = np.isfinite(source_xy).all(axis=1) & np.isfinite(target_xy).all(axis=1)
    in_bounds = (
        (target_xy[:, 0] >= 0)
        & (target_xy[:, 0] < width)
        & (target_xy[:, 1] >= 0)
        & (target_xy[:, 1] < height)
    )
    eligible = finite & in_bounds
    for index, point in enumerate(source_xy):
        candidates = np.flatnonzero(
            eligible & (np.linalg.norm(source_xy - point, axis=1) <= radius)
        )
        if len(candidates) < 8:
            continue
        local_source = source_xy[candidates]
        local_target = target_xy[candidates]
        design = np.column_stack((np.ones(len(candidates)), local_source))
        best = np.zeros(len(candidates), dtype=bool)
        for _ in range(20):
            chosen = rng.choice(len(candidates), size=3, replace=False)
            if abs(np.linalg.det(design[chosen])) < 1e-7:
                continue
            coefficients = np.linalg.solve(design[chosen], local_target[chosen])
            residual = np.linalg.norm(design @ coefficients - local_target, axis=1)
            inliers = residual <= threshold
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() < 6:
            continue
        coefficients = np.linalg.lstsq(design[best], local_target[best], rcond=None)[0]
        affine = np.asarray(
            [coefficients[0, 0], coefficients[1, 0] - 1,
             coefficients[2, 0], coefficients[0, 1], coefficients[1, 1],
             coefficients[2, 1] - 1]
        )
        result[index] = affine
        usable[index] = True
    return result, usable


@torch.no_grad()
def evaluate(model, examples, native_scale: int, *, project_field: bool = False):
    model.eval()
    metrics = {"translation": [], "heterogeneous": []}
    hard_errors = {"translation": [], "heterogeneous": []}
    zero_flow_errors = {"translation": [], "heterogeneous": []}
    spline_errors = {"translation": [], "heterogeneous": []}
    spline_jacobian_errors = {"translation": [], "heterogeneous": []}
    single_spline_errors = {"translation": [], "heterogeneous": []}
    single_spline_jacobian_errors = {"translation": [], "heterogeneous": []}
    raw_grid_errors = {"translation": [], "heterogeneous": []}
    raw_grid_jacobian_errors = {"translation": [], "heterogeneous": []}
    consensus_errors = {"translation": [], "heterogeneous": []}
    consensus_gradient_errors = {"translation": [], "heterogeneous": []}
    consensus_coverage = {"translation": [], "heterogeneous": []}
    top1_values = {"translation": [], "heterogeneous": []}
    for example in examples:
        reference, target, flow, visible, mode = example
        prediction = model(reference, target)
        height, width = prediction.confidence.shape[-2:]
        truth = torch.nn.functional.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
        mask = torch.nn.functional.interpolate(visible[:, None].float(), size=(height, width), mode="nearest")[:, 0] > 0.5
        error = torch.linalg.vector_norm(prediction.displacement_xy - truth.permute(0, 2, 3, 1), dim=-1)
        metrics[mode].extend((error[mask] * native_scale).tolist())
        hard_error = torch.linalg.vector_norm(
            prediction.hard_displacement_xy - truth.permute(0, 2, 3, 1), dim=-1
        )
        hard_errors[mode].extend((hard_error[mask] * native_scale).tolist())
        zero_error = torch.linalg.vector_norm(truth.permute(0, 2, 3, 1), dim=-1)
        zero_flow_errors[mode].extend((zero_error[mask] * native_scale).tolist())
        image_size = (reference.shape[-1], reference.shape[-2])
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=flow.dtype),
            torch.arange(width, dtype=flow.dtype),
            indexing="ij",
        )
        source_points = torch.stack(
            (
                (xx + 0.5) * (image_size[0] / width) - 0.5,
                (yy + 0.5) * (image_size[1] / height) - 0.5,
            ),
            dim=-1,
        ).reshape(-1, 2)
        predicted_flow = prediction.displacement_xy[0].reshape(-1, 2)
        fit_points = source_points
        fit_values = predicted_flow
        spacing = (32.0, 16.0, 8.0)
        shapes = tuple(recommended_control_shape(image_size, step) for step in spacing)
        fitted = fit_bspline_field(fit_points, fit_values, image_size, shapes, bending=1e-3)
        fitted_u, fitted_f = fitted.evaluate(source_points)
        spline_error = torch.linalg.vector_norm(
            fitted_u - truth.permute(0, 2, 3, 1)[0].reshape(-1, 2), dim=-1
        ).reshape(height, width)
        spline_errors[mode].extend((spline_error[mask[0]] * native_scale).tolist())
        gradient_x = torch.gradient(truth[0, 0], spacing=(image_size[1] / height, image_size[0] / width))
        gradient_y = torch.gradient(truth[0, 1], spacing=(image_size[1] / height, image_size[0] / width))
        true_f = torch.stack(
            (
                torch.stack((1 + gradient_x[1], gradient_x[0]), dim=-1),
                torch.stack((gradient_y[1], 1 + gradient_y[0]), dim=-1),
            ),
            dim=-2,
        ).reshape(-1, 2, 2)
        jac_error = torch.linalg.matrix_norm(fitted_f - true_f, ord="fro", dim=(1, 2))
        spline_jacobian_errors[mode].extend(jac_error[mask[0].reshape(-1)].tolist())
        single_shape = recommended_control_shape(image_size, 8.0)
        single = fit_bspline_field(
            fit_points, fit_values, image_size, (single_shape,), bending=1e-3
        )
        single_u, single_f = single.evaluate(source_points)
        single_error = torch.linalg.vector_norm(
            single_u - truth.permute(0, 2, 3, 1)[0].reshape(-1, 2), dim=-1
        ).reshape(height, width)
        single_f_error = torch.linalg.matrix_norm(single_f - true_f, ord="fro", dim=(1, 2))
        single_spline_errors[mode].extend((single_error[mask[0]] * native_scale).tolist())
        single_spline_jacobian_errors[mode].extend(single_f_error[mask[0].reshape(-1)].tolist())
        raw_grid = feature_grid_displacement_field(
            prediction.displacement_xy[0].to(torch.float64), image_size
        )
        raw_u, raw_f = raw_grid.evaluate(source_points)
        raw_error = torch.linalg.vector_norm(
            raw_u - truth.permute(0, 2, 3, 1)[0].reshape(-1, 2), dim=-1
        ).reshape(height, width)
        raw_f_error = torch.linalg.matrix_norm(raw_f - true_f, ord="fro", dim=(1, 2))
        raw_grid_errors[mode].extend((raw_error[mask[0]] * native_scale).tolist())
        raw_grid_jacobian_errors[mode].extend(raw_f_error[mask[0].reshape(-1)].tolist())
        if not project_field:
            continue
        local_parameters, local_usable = local_affine_consensus(
            source_points.numpy(),
            prediction.hard_target_xy[0].reshape(-1, 2).numpy(),
            image_size[0],
            image_size[1],
        )
        truth_np = truth[0].permute(1, 2, 0).reshape(-1, 2).numpy()
        local_flow = np.column_stack(
            (
                local_parameters[:, 0] + local_parameters[:, 1] * source_points[:, 0].numpy()
                + local_parameters[:, 2] * source_points[:, 1].numpy(),
                local_parameters[:, 3] + local_parameters[:, 4] * source_points[:, 0].numpy()
                + local_parameters[:, 5] * source_points[:, 1].numpy(),
            )
        )
        local_true = true_f.numpy()
        local_f = local_parameters[:, [1, 2, 4, 5]].reshape(-1, 2, 2) + np.eye(2)
        local_flow_error = np.linalg.norm(local_flow - truth_np, axis=1)
        local_gradient_error = np.linalg.norm(local_f - local_true, axis=(1, 2))
        evaluation_mask = mask[0].reshape(-1).numpy() & local_usable
        consensus_coverage[mode].append(float(evaluation_mask.sum() / max(int(mask.sum()), 1)))
        consensus_errors[mode].extend((local_flow_error[evaluation_mask] * native_scale).tolist())
        consensus_gradient_errors[mode].extend(local_gradient_error[evaluation_mask].tolist())
        # The top-1 score asks whether each source cell selects the target cell
        # containing its analytic correspondence, among the full coarse grid.
        feat_h, feat_w = height, width
        yy, xx = torch.meshgrid(torch.arange(feat_h), torch.arange(feat_w), indexing="ij")
        sx = (xx + 0.5) * (reference.shape[-1] / feat_w) - 0.5
        sy = (yy + 0.5) * (reference.shape[-2] / feat_h) - 0.5
        target_x = sx[None] + truth[:, 0]
        target_y = sy[None] + truth[:, 1]
        tx = torch.round((target_x + 0.5) / (reference.shape[-1] / feat_w) - 0.5).long()
        ty = torch.round((target_y + 0.5) / (reference.shape[-2] / feat_h) - 0.5).long()
        valid = mask & (tx >= 0) & (tx < feat_w) & (ty >= 0) & (ty < feat_h)
        target_index = ty.clamp(0, feat_h - 1) * feat_w + tx.clamp(0, feat_w - 1)
        predicted_y = ((prediction.hard_target_xy[..., 1] + 0.5) / (reference.shape[-2] / feat_h) - 0.5)
        predicted_x = ((prediction.hard_target_xy[..., 0] + 0.5) / (reference.shape[-1] / feat_w) - 0.5)
        predicted_index = predicted_y.round().long().clamp(0, feat_h - 1) * feat_w
        predicted_index += predicted_x.round().long().clamp(0, feat_w - 1)
        top1_values[mode].extend((predicted_index[valid] == target_index[valid]).float().tolist())
    return {
        mode: {
            "visible_coarse_samples": len(metrics[mode]),
            "median_flow_error_px": float(np.median(metrics[mode])) if metrics[mode] else None,
            "hard_top1_median_flow_error_px": float(np.median(hard_errors[mode])) if hard_errors[mode] else None,
            "hard_top1_fraction_within_5px": float(np.mean(np.asarray(hard_errors[mode]) <= 5.0)) if hard_errors[mode] else None,
            "p90_flow_error_px": float(np.quantile(metrics[mode], 0.9)) if metrics[mode] else None,
            "zero_flow_median_error_px": float(np.median(zero_flow_errors[mode])) if zero_flow_errors[mode] else None,
            "fraction_within_2px": float(np.mean(np.asarray(metrics[mode]) <= 2.0)) if metrics[mode] else None,
            "fraction_within_5px": float(np.mean(np.asarray(metrics[mode]) <= 5.0)) if metrics[mode] else None,
            "spline_projected_flow_rmse_px": float(np.sqrt(np.mean(np.square(spline_errors[mode])))) if spline_errors[mode] else None,
            "spline_jacobian_rmse": float(np.sqrt(np.mean(np.square(spline_jacobian_errors[mode])))) if spline_jacobian_errors[mode] else None,
            "single_spline_flow_rmse_px": float(np.sqrt(np.mean(np.square(single_spline_errors[mode])))) if single_spline_errors[mode] else None,
            "single_spline_jacobian_rmse": float(np.sqrt(np.mean(np.square(single_spline_jacobian_errors[mode])))) if single_spline_jacobian_errors[mode] else None,
            "raw_bilinear_grid_flow_rmse_px": float(np.sqrt(np.mean(np.square(raw_grid_errors[mode])))) if raw_grid_errors[mode] else None,
            "raw_bilinear_grid_jacobian_rmse": float(np.sqrt(np.mean(np.square(raw_grid_jacobian_errors[mode])))) if raw_grid_jacobian_errors[mode] else None,
            "local_consensus_coverage_fraction": float(np.mean(consensus_coverage[mode])) if consensus_coverage[mode] else None,
            "local_consensus_displacement_rmse_px": float(np.sqrt(np.mean(np.square(consensus_errors[mode])))) if consensus_errors[mode] else None,
            "local_consensus_jacobian_rmse": float(np.sqrt(np.mean(np.square(consensus_gradient_errors[mode])))) if consensus_gradient_errors[mode] else None,
            "top1_cell_accuracy": float(np.mean(top1_values[mode])) if top1_values[mode] else None,
        }
        for mode in metrics
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--native-scale", type=int, default=1,
                        help="generate at a larger pixel scale, then area-downsample both images")
    parser.add_argument("--speckle-size", type=float, default=3.5,
                        help="characteristic speckle size in model-input pixels")
    parser.add_argument("--train-pairs", type=int, default=48)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physical-weight", type=float, default=0.0)
    parser.add_argument(
        "--physical-pyramid-levels",
        type=int,
        default=1,
        help="number of successively downsampled image scales for advection losses",
    )
    parser.add_argument("--dataset-cache-dir", type=Path,
                        help="cache deterministic low-resolution training and validation pairs for ablations")
    parser.add_argument("--contextual-features", action="store_true",
                        help="add residual dilated convolutions to expand the feature support")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--weights", type=Path, help="evaluate a saved checkpoint without training")
    args = parser.parse_args()
    if args.physical_pyramid_levels < 1:
        parser.error("--physical-pyramid-levels must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    dataset_factory = load_or_make_dataset if args.dataset_cache_dir else make_dataset
    if args.weights:
        train = []
    else:
        train = dataset_factory(
            args.dataset_cache_dir / "train.pt" if args.dataset_cache_dir else args.output_dir / "train.pt",
            args.train_pairs, 0, args.width, args.height, args.native_scale, args.speckle_size,
        ) if args.dataset_cache_dir else dataset_factory(
            args.train_pairs, 0, args.width, args.height, args.native_scale, args.speckle_size
        )
    validation = (
        dataset_factory(
            args.dataset_cache_dir / "validation.pt", args.validation_pairs, 100000,
            args.width, args.height, args.native_scale, args.speckle_size,
        )
        if args.dataset_cache_dir
        else make_dataset(
            args.validation_pairs, 100000, args.width, args.height,
            args.native_scale, args.speckle_size,
        )
    )
    model = NDFPairMatcher(contextual_features=args.contextual_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    best_score = float("inf")
    best_state = None
    started = time.perf_counter()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    for epoch in range(0 if args.weights else args.epochs):
        model.train()
        order = torch.randperm(len(train)).tolist()
        epoch_losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [train[i] for i in order[start : start + args.batch_size]]
            reference, target, flow, visible = stack_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = pair_matching_loss(
                model,
                reference,
                target,
                flow,
                visible,
                physical_weight=args.physical_weight,
                physical_pyramid_levels=args.physical_pyramid_levels,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        validation_metrics = evaluate(model, validation, args.native_scale)
        score = sum(validation_metrics[m]["median_flow_error_px"] for m in validation_metrics) / 2
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "physical_loss": float(batch_metrics["physical_loss"]),
            "validation": validation_metrics,
            "temperature": float(model.log_temperature.detach().exp()),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, validation, args.native_scale, project_field=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "ndf_pair_matcher.pt"
    torch.save(model.state_dict(), weights_path)
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    report = {
        "protocol": {
            "seed": args.seed,
            "train_texture_seeds": "51000..",
            "validation_texture_seeds": "151000.. (independent from training)",
            "image_size_px": [args.width, args.height],
            "native_scale": args.native_scale,
            "speckle_size_px_at_model_input": args.speckle_size,
            "synthetic_native_size_px": [args.width * args.native_scale, args.height * args.native_scale],
            "feature_stride": 4,
            "candidate_count_per_query": (args.width // 4) * (args.height // 4),
            "training_pairs": args.train_pairs,
            "validation_pairs": args.validation_pairs,
            "epochs": args.epochs,
            "physical_loss_weight": args.physical_weight,
            "physical_pyramid_levels": args.physical_pyramid_levels,
            "checkpoint_evaluation_only": bool(args.weights),
            "elapsed_s": time.perf_counter() - started,
            "device": "CPU",
            "method": "shared CNN features, exhaustive global coarse correlation, soft correspondence expectation",
            "contextual_features": args.contextual_features,
            "feature_receptive_field_px": 41 if args.contextual_features else 17,
            "oracle_used_for_training_only": True,
        },
        "best_validation": final_metrics,
        "history": history,
        "weights": weights_path.name,
        "weights_sha256": weights_sha256,
    }
    (args.output_dir / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation": final_metrics, "weights": str(weights_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
