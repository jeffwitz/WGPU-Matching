"""Train and hold out a tiny speckle-specific score detector.

Example: ``python python/benchmarks/train_dic_detector.py --epochs 10``.
This is an optional research script requiring the ``research`` extra. It writes
small NumPy weights and a provenance report; no weights are downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from wgpu_matching.dic_model import build_dic_score_model
from wgpu_matching.dic_synthetic import generate_speckle_pair

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/features/dic_detector_training_20261107"
GRID_STRIDE = 4
GRID_OFFSET = GRID_STRIDE // 2


def _score_targets(torch: Any, functional: Any, image: Any) -> Any:
    """Return a balanced top-quartile structure-tensor teacher on the 4 px grid."""
    sobel_x = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device
        ).view(1, 1, 3, 3)
        / 8.0
    )
    sobel_y = sobel_x.transpose(2, 3)
    gx = functional.conv2d(image, sobel_x, padding=1)
    gy = functional.conv2d(image, sobel_y, padding=1)
    hxx = functional.avg_pool2d(gx * gx, 9, stride=1, padding=4)
    hxy = functional.avg_pool2d(gx * gy, 9, stride=1, padding=4)
    hyy = functional.avg_pool2d(gy * gy, 9, stride=1, padding=4)
    trace = hxx + hyy
    discriminant = torch.sqrt(torch.clamp((hxx - hyy) ** 2 + 4.0 * hxy**2, min=0.0))
    minimum = 0.5 * torch.clamp(trace - discriminant, min=0.0)
    grid = minimum[:, :, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
    flat = grid.flatten(1)
    threshold = torch.quantile(flat.detach(), 0.72, dim=1).view(-1, 1, 1, 1)
    return (grid >= threshold).to(image.dtype)


def _pair_batch(torch: Any, samples: list[Any], device: Any) -> tuple[Any, Any, Any, Any]:
    ref = torch.from_numpy(np.stack([sample.reference for sample in samples]))[:, None].to(device)
    target = torch.from_numpy(np.stack([sample.target for sample in samples]))[:, None].to(device)
    flow = torch.from_numpy(np.stack([sample.flow_xy for sample in samples])).to(device)
    visible = torch.from_numpy(np.stack([sample.visible for sample in samples])).to(device)
    return ref, target, flow, visible


def _train_epoch(
    torch: Any,
    functional: Any,
    model: Any,
    optimizer: Any,
    samples: list[Any],
    batch_size: int,
    device: Any,
) -> dict[str, float]:
    order = np.arange(len(samples))
    # Deterministic shuffle follows the already seeded torch/NumPy generators.
    np.random.shuffle(order)
    losses: list[float] = []
    detector_losses: list[float] = []
    equivariance_losses: list[float] = []
    model.train()
    for start in range(0, len(order), batch_size):
        batch = [samples[index] for index in order[start : start + batch_size]]
        reference, target, flow, visible = _pair_batch(torch, batch, device)
        logits_reference = model(reference)
        logits_target = model(target)
        scores_reference = logits_reference[
            :, :, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE
        ]
        scores_target = logits_target[:, :, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
        labels_reference = _score_targets(torch, functional, reference)
        labels_target = _score_targets(torch, functional, target)
        detect_loss = 0.5 * (
            functional.binary_cross_entropy_with_logits(scores_reference, labels_reference)
            + functional.binary_cross_entropy_with_logits(scores_target, labels_target)
        )

        height, width = reference.shape[-2:]
        ys = torch.arange(GRID_OFFSET, height, GRID_STRIDE, device=device, dtype=reference.dtype)
        xs = torch.arange(GRID_OFFSET, width, GRID_STRIDE, device=device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        flow_grid = flow[:, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
        target_x = xx[None] + flow_grid[..., 0]
        target_y = yy[None] + flow_grid[..., 1]
        normalized_grid = torch.stack(
            (2.0 * (target_x + 0.5) / width - 1.0, 2.0 * (target_y + 0.5) / height - 1.0),
            dim=-1,
        )
        projected_target = functional.grid_sample(
            torch.sigmoid(logits_target),
            normalized_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        valid_grid = visible[:, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
        if valid_grid.any():
            equiv_loss = functional.smooth_l1_loss(
                torch.sigmoid(scores_reference)[valid_grid[:, None]],
                projected_target[valid_grid[:, None]],
            )
        else:
            equiv_loss = detect_loss.new_zeros(())
        loss = detect_loss + 0.5 * equiv_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        detector_losses.append(float(detect_loss.detach().cpu()))
        equivariance_losses.append(float(equiv_loss.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "saliency_bce": float(np.mean(detector_losses)),
        "equivariance": float(np.mean(equivariance_losses)),
    }


def _detect_points(
    model: Any, torch: Any, image: np.ndarray, budget: int, device: Any
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        value = torch.from_numpy(image)[None, None].to(device)
        scores = torch.sigmoid(
            model(value)[0, 0, GRID_OFFSET::GRID_STRIDE, GRID_OFFSET::GRID_STRIDE]
        )
        scores = scores.cpu().numpy()
    height, width = scores.shape
    keep = np.ones(scores.shape, dtype=np.bool_)
    linear_indices = np.arange(scores.size, dtype=np.int64).reshape(scores.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.full_like(scores, -np.inf)
            shifted_indices = np.full(scores.shape, np.iinfo(np.int64).max, dtype=np.int64)
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            shifted[target_y, target_x] = scores[source_y, source_x]
            shifted_indices[target_y, target_x] = linear_indices[source_y, source_x]
            keep &= (scores > shifted) | ((scores == shifted) & (linear_indices < shifted_indices))
    maxima = np.where(keep, scores, -np.inf)
    ys, xs = np.nonzero(np.isfinite(maxima))
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    values = maxima[ys, xs]
    order = np.argsort(-values, kind="stable")
    tile_columns, tile_rows = 16, 9
    per_tile = max(1, math.ceil(budget / (tile_columns * tile_rows)))
    counts = np.zeros((tile_rows, tile_columns), dtype=np.int32)
    selected: list[int] = []
    for rank in order:
        tx = min(tile_columns - 1, int(xs[rank] / max(width, 1) * tile_columns))
        ty = min(tile_rows - 1, int(ys[rank] / max(height, 1) * tile_rows))
        if counts[ty, tx] >= per_tile:
            continue
        counts[ty, tx] += 1
        selected.append(int(rank))
        if len(selected) >= budget:
            break
    if len(selected) < budget:
        chosen = set(selected)
        selected.extend(
            int(rank) for rank in order if int(rank) not in chosen and len(selected) < budget
        )
    chosen = np.asarray(selected, dtype=np.int32)
    return np.column_stack(
        ((xs[chosen] * GRID_STRIDE + GRID_OFFSET), (ys[chosen] * GRID_STRIDE + GRID_OFFSET))
    ).astype(np.float32)


def _repeatability(
    reference_xy: np.ndarray, target_xy: np.ndarray, pair: Any
) -> dict[str, float | int]:
    height, width = pair.reference.shape
    source_x = np.clip(np.rint(reference_xy[:, 0]).astype(int), 0, width - 1)
    source_y = np.clip(np.rint(reference_xy[:, 1]).astype(int), 0, height - 1)
    expected = reference_xy + pair.flow_xy[source_y, source_x]
    visible_a = pair.visible[source_y, source_x]
    if pair.target_visible is not None:
        target_x = np.clip(np.rint(target_xy[:, 0]).astype(int), 0, width - 1)
        target_y = np.clip(np.rint(target_xy[:, 1]).astype(int), 0, height - 1)
        visible_b = pair.target_visible[target_y, target_x]
    else:
        matrix = np.asarray(pair.metadata["affine_matrix"], dtype=np.float64)
        inverse = np.linalg.inv(matrix)
        center = np.array([width / 2.0, height / 2.0])
        translation = np.asarray(pair.metadata["translation_px"], dtype=np.float64)
        amplitude = np.asarray(pair.metadata["heterogeneity_amplitude_px"], dtype=np.float64)
        source_from_target = center + (target_xy - center - translation) @ inverse.T
        for _ in range(12):
            phase_x = 2.0 * np.pi * source_from_target[:, 0] / width
            phase_y = 2.0 * np.pi * source_from_target[:, 1] / height
            heterogeneity = np.column_stack(
                (
                    amplitude[0] * np.sin(phase_y) * np.cos(phase_x),
                    amplitude[1] * np.sin(phase_x) * np.cos(phase_y),
                )
            )
            source_from_target = (
                center + (target_xy - center - translation - heterogeneity) @ inverse.T
            )
        visible_b = (
            (source_from_target[:, 0] >= 0)
            & (source_from_target[:, 0] < width)
            & (source_from_target[:, 1] >= 0)
            & (source_from_target[:, 1] < height)
        )
    source = reference_xy[visible_a]
    expected_visible = expected[visible_a]
    target = target_xy[visible_b]
    denominator = min(len(source), len(target))
    if not denominator:
        return {
            "mutual": 0,
            "denominator": 0,
            "at_1px": 0,
            "at_2px": 0,
            "at_5px": 0,
            "repeatability_1px": 0.0,
            "repeatability_2px": 0.0,
            "repeatability_5px": 0.0,
        }
    squared = np.sum((expected_visible[:, None, :] - target[None, :, :]) ** 2, axis=2)
    nearest_target = squared.argmin(axis=1)
    nearest_source = squared.argmin(axis=0)
    mutual = nearest_source[nearest_target] == np.arange(len(source))
    errors = np.sqrt(squared[np.arange(len(source)), nearest_target])
    correct_1px = mutual & (errors <= 1.0)
    correct_2px = mutual & (errors <= 2.0)
    correct_5px = mutual & (errors <= 5.0)
    return {
        "mutual": int(mutual.sum()),
        "denominator": int(denominator),
        "at_1px": int(correct_1px.sum()),
        "at_2px": int(correct_2px.sum()),
        "at_5px": int(correct_5px.sum()),
        "repeatability_1px": float(correct_1px.sum() / denominator),
        "repeatability_2px": float(correct_2px.sum() / denominator),
        "repeatability_5px": float(correct_5px.sum() / denominator),
    }


def _export_weights(model: Any, output: Path) -> str:
    state = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in model.state_dict().items()
    }
    weights_path = output / "dic_detector_weights.npz"
    np.savez_compressed(weights_path, **state)
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    return digest


def _evaluate_split(
    model: Any,
    torch: Any,
    image_size: int,
    pairs: int,
    budget: int,
    texture_seed_start: int,
    warp_seed_start: int,
    device: Any,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index in range(pairs):
        pair = generate_speckle_pair(
            texture_seed=texture_seed_start + index,
            warp_seed=warp_seed_start + index,
            width=image_size,
            height=image_size,
        )
        points_reference = _detect_points(model, torch, pair.reference, budget, device)
        points_target = _detect_points(model, torch, pair.target, budget, device)
        rows.append(
            {
                "texture_seed": pair.metadata["texture_seed"],
                "warp_seed": pair.metadata["warp_seed"],
                "motion_mode": pair.metadata["motion_mode"],
                "reference_points": len(points_reference),
                "target_points": len(points_target),
                "repeatability_2px": _repeatability(points_reference, points_target, pair),
                "motion": {
                    "angle_rad": pair.metadata["angle_rad"],
                    "stretch_x": pair.metadata["stretch_x"],
                    "stretch_y": pair.metadata["stretch_y"],
                    "translation_px": pair.metadata["translation_px"],
                    "heterogeneity_amplitude_px": pair.metadata["heterogeneity_amplitude_px"],
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=24)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=73191)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (
        min(
            args.epochs, args.samples_per_epoch, args.batch_size, args.validation_pairs, args.budget
        )
        < 1
    ):
        parser.error("epochs, samples, batch size, validation pairs, and budget must be positive")
    if args.image_size < 64 or args.image_size % GRID_STRIDE:
        parser.error("image-size must be at least 64 and divisible by 4")

    try:
        import torch
        from torch.nn import functional
    except ImportError as error:
        raise SystemExit("Install the optional research extra to train DIC-Detector") from error
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model = build_dic_score_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    started = time.perf_counter()
    history: list[dict[str, float]] = []
    training_motion_counts: Counter[str] = Counter()
    for epoch in range(args.epochs):
        samples = [
            generate_speckle_pair(
                texture_seed=10_000 + epoch * args.samples_per_epoch + index,
                warp_seed=500_000 + epoch * args.samples_per_epoch + index,
                width=args.image_size,
                height=args.image_size,
            )
            for index in range(args.samples_per_epoch)
        ]
        training_motion_counts.update(sample.metadata["motion_mode"] for sample in samples)
        metrics = _train_epoch(
            torch, functional, model, optimizer, samples, args.batch_size, device
        )
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

    validation_rows = _evaluate_split(
        model,
        torch,
        args.image_size,
        args.validation_pairs,
        args.budget,
        2_000_000,
        3_000_000,
        device,
    )
    test_rows = _evaluate_split(
        model,
        torch,
        args.image_size,
        args.validation_pairs,
        args.budget,
        4_000_000,
        5_000_000,
        device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = _export_weights(model, args.output_dir)
    report = {
        "model": "DICScoreNet-v0",
        "architecture": "Conv2d 1->8 3x3 + SiLU; depthwise Conv2d 8->8 3x3 + SiLU; Conv2d 8->1 1x1",
        "training_objective": "balanced BCE for top-28% local structure-tensor scores plus 0.5x analytic-flow score equivariance; no IC-GN success supervision",
        "seed": args.seed,
        "split": {
            "train_texture_seed_range": [10_000, 10_000 + args.epochs * args.samples_per_epoch - 1],
            "validation_texture_seeds": [2_000_000 + i for i in range(args.validation_pairs)],
            "validation_warp_seeds": [3_000_000 + i for i in range(args.validation_pairs)],
            "training_and_validation_textures_disjoint": True,
            "test_texture_seeds": [4_000_000 + i for i in range(args.validation_pairs)],
            "test_warp_seeds": [5_000_000 + i for i in range(args.validation_pairs)],
            "test_textures_disjoint_from_training_and_validation": True,
        },
        "training": {
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "batch_size": args.batch_size,
            "image_size_px": args.image_size,
            "device": str(device),
            "torch_version": torch.__version__,
            "motion_mode_counts": dict(sorted(training_motion_counts.items())),
            "elapsed_s": time.perf_counter() - started,
            "history": history,
        },
        "validation": {
            "split_role": "diagnostic only; no checkpoint or hyperparameter selection",
            "budget": args.budget,
            "threshold_px": 2.0,
            "cases": validation_rows,
            "mean_repeatability": float(
                np.mean([row["repeatability_2px"]["repeatability_2px"] for row in validation_rows])
            ),
        },
        "test": {
            "split_role": "held-out final texture and deformation seeds",
            "budget": args.budget,
            "threshold_px": 2.0,
            "cases": test_rows,
            "mean_repeatability": float(
                np.mean([row["repeatability_2px"]["repeatability_2px"] for row in test_rows])
            ),
        },
        "weight_file": "dic_detector_weights.npz",
        "weights_sha256": digest,
        "inference_status": "PyTorch was used for training and oracle checks; WGSL learned-weight inference is implemented and score parity is validated separately",
        "scientific_limit": "The score target is a structure-tensor proxy and the objective does not supervise descriptor uniqueness or IC-GN convergence.",
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'training_report.json'}", flush=True)
    print(f"weights_sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
