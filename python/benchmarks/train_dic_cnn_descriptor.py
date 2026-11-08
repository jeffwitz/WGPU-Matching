"""Train the compact speckle descriptor from DIC-Detector keypoints.

The research pipeline uses analytic correspondences for positive patches and
same-image DIC-Detector points as a large hard-negative pool. Patch tensors are
stored as float16 to keep generated datasets bounded in size. The exported
weights are consumed by the native WGSL inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from wgpu_matching import DICDetectorOptions, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair

ROOT = Path(__file__).resolve().parents[2]
MOTIONS = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")
PATCH_SIZE = 64
SUPPORT_RADIUS = PATCH_SIZE / 2
EXCLUSION_RADIUS_PX = 48.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bilinear(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample a scalar or vector image at floating pixel-centre coordinates."""
    height, width = array.shape[:2]
    xy = np.asarray(points, dtype=np.float32)
    x0 = np.floor(xy[:, 0]).astype(np.int32)
    y0 = np.floor(xy[:, 1]).astype(np.int32)
    wx = (xy[:, 0] - x0).astype(np.float32)
    wy = (xy[:, 1] - y0).astype(np.float32)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tail = (1,) * (array.ndim - 2)
    wx, wy = wx.reshape((-1, *tail)), wy.reshape((-1, *tail))
    return (
        array[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + array[y0, x1] * wx * (1.0 - wy)
        + array[y1, x0] * (1.0 - wx) * wy
        + array[y1, x1] * wx * wy
    )


def sample_patches(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinearly sample 64x64 patches and retain subpixel keypoint centres."""
    xy = np.asarray(points, dtype=np.float32)
    offsets = np.arange(PATCH_SIZE, dtype=np.float32) - (PATCH_SIZE - 1) / 2
    ox, oy = np.meshgrid(offsets, offsets)
    locations = xy[:, None, None, :] + np.stack((ox, oy), axis=-1)[None]
    flat = locations.reshape(-1, 2)
    values = bilinear(np.asarray(image, dtype=np.float32), flat)
    return values.reshape(len(xy), 1, PATCH_SIZE, PATCH_SIZE).astype(np.float32)


def support_is_valid(pair, points: np.ndarray, *, support: int = PATCH_SIZE) -> np.ndarray:
    """Require a whole descriptor support to stay visible in both images.

    A 17x17 sample lattice detects invalid strips and discontinuities across
    the 64-pixel patch. It deliberately rejects partially visible supports;
    the DIC solver and descriptor must not learn from different material sides.
    """
    xy = np.asarray(points, dtype=np.float32)
    height, width = pair.reference.shape
    half = support / 2
    valid = (
        (xy[:, 0] >= half)
        & (xy[:, 1] >= half)
        & (xy[:, 0] < width - half)
        & (xy[:, 1] < height - half)
    )
    if not np.any(valid):
        return valid
    grid = np.linspace(-half + 0.5, half - 0.5, 17, dtype=np.float32)
    ox, oy = np.meshgrid(grid, grid)
    samples = xy[valid, None, :] + np.stack((ox, oy), axis=-1).reshape(1, -1, 2)
    flat = samples.reshape(-1, 2)
    h, w = pair.visible.shape
    sx = np.clip(np.rint(flat[:, 0]).astype(int), 0, w - 1)
    sy = np.clip(np.rint(flat[:, 1]).astype(int), 0, h - 1)
    material_valid = pair.visible[sy, sx].copy()
    flow = bilinear(pair.flow_xy, flat).astype(np.float32)
    mapped = flat + flow
    th, tw = pair.target.shape
    tx = np.rint(mapped[:, 0]).astype(int)
    ty = np.rint(mapped[:, 1]).astype(int)
    in_bounds = (tx >= 0) & (tx < tw) & (ty >= 0) & (ty < th)
    target_valid = np.zeros_like(in_bounds)
    if pair.target_visible is not None:
        target_valid[in_bounds] = pair.target_visible[ty[in_bounds], tx[in_bounds]]
    else:
        target_valid = in_bounds
    lattice_valid = (material_valid & target_valid).reshape(np.count_nonzero(valid), -1).all(1)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= lattice_valid
    return valid


def target_support_is_valid(pair, points: np.ndarray, *, support: int = PATCH_SIZE) -> np.ndarray:
    """Check a target-image patch against its physical-validity mask."""
    return mask_support_is_valid(
        pair.target_visible,
        pair.target.shape,
        points,
        support=support,
    )


def mask_support_is_valid(
    mask: np.ndarray | None,
    image_shape: tuple[int, int],
    points: np.ndarray,
    *,
    support: int = PATCH_SIZE,
) -> np.ndarray:
    """Require an axis-aligned patch to fit and remain within a validity mask."""
    xy = np.asarray(points, dtype=np.float32)
    height, width = image_shape
    half = support / 2
    valid = (
        (xy[:, 0] >= half)
        & (xy[:, 1] >= half)
        & (xy[:, 0] < width - half)
        & (xy[:, 1] < height - half)
    )
    if mask is None or not np.any(valid):
        return valid
    grid = np.linspace(-half + 0.5, half - 0.5, 17, dtype=np.float32)
    ox, oy = np.meshgrid(grid, grid)
    samples = xy[valid, None, :] + np.stack((ox, oy), axis=-1).reshape(1, -1, 2)
    flat = samples.reshape(-1, 2)
    xx = np.clip(np.rint(flat[:, 0]).astype(int), 0, width - 1)
    yy = np.clip(np.rint(flat[:, 1]).astype(int), 0, height - 1)
    lattice_valid = mask[yy, xx].reshape(np.count_nonzero(valid), -1).all(1)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= lattice_valid
    return valid


def build_dataset(
    count: int,
    point_count: int,
    negative_count: int,
    seed: int,
    matcher: WGPUMatching,
    image_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    source_rows, target_rows, source_neg_rows, target_neg_rows = [], [], [], []
    source_xy_rows, target_xy_rows, source_neg_xy_rows, target_neg_xy_rows = [], [], [], []
    texture_seeds, warp_seeds, modes, support_stats = [], [], [], []
    for index in range(count):
        texture_seed = seed + index * 1009
        warp_seed = seed + 7_000_001 + index * 7919
        mode = MOTIONS[index % len(MOTIONS)]
        pair = generate_speckle_pair(
            texture_seed, warp_seed, image_size, image_size, motion_mode=mode
        )
        reference = np.rint(pair.reference * 255).astype(np.uint8)
        target = np.rint(pair.target * 255).astype(np.uint8)
        detector_options = DICDetectorOptions(
            max_keypoints=min(8192, max(256, int(np.ceil(negative_count * 2.0)))),
            stride_px=4,
        )
        source_features = matcher.detect_dic(reference, options=detector_options).features
        target_features = matcher.detect_dic(target, options=detector_options).features
        source_points = source_features.xy
        target_points = target_features.xy
        source_mapped = source_points + bilinear(pair.flow_xy, source_points).astype(np.float32)
        source_support_valid = (
            source_features.valid
            & support_is_valid(pair, source_points)
            & target_support_is_valid(pair, source_mapped)
        )
        source_negative_valid = source_features.valid & mask_support_is_valid(
            pair.visible, pair.reference.shape, source_points
        )
        target_support_valid = target_features.valid & target_support_is_valid(
            pair, target_points
        )
        source_candidates = source_points[source_support_valid]
        source_negative_candidates = source_points[source_negative_valid]
        target_candidates = target_points[target_support_valid]
        if len(source_candidates) < point_count or min(
            len(source_negative_candidates), len(target_candidates)
        ) < point_count:
            raise RuntimeError(
                f"{mode} pair {texture_seed}/{warp_seed} has only "
                f"{len(source_candidates)} mapped query supports, "
                f"{len(source_negative_candidates)} source negatives and "
                f"{len(target_candidates)} target negatives; need at least {point_count}"
            )
        pair_negative_count = min(
            negative_count, len(source_negative_candidates), len(target_candidates)
        )

        query_ids = rng.choice(len(source_candidates), point_count, replace=False)
        source_xy = source_candidates[query_ids]
        target_xy = source_xy + bilinear(pair.flow_xy, source_xy).astype(np.float32)
        # The source material support and the fixed target-image descriptor
        # support are both checked, including discontinuity and image borders.
        target_neg_ids = rng.choice(len(target_candidates), pair_negative_count, replace=False)
        source_neg_ids = rng.choice(
            len(source_negative_candidates), pair_negative_count, replace=False
        )
        source_image = reference.astype(np.float32) / 255.0
        target_image = target.astype(np.float32) / 255.0
        source_rows.append(sample_patches(source_image, source_xy))
        target_rows.append(sample_patches(target_image, target_xy))
        source_neg_rows.append(
            sample_patches(source_image, source_negative_candidates[source_neg_ids])
        )
        target_neg_rows.append(sample_patches(target_image, target_candidates[target_neg_ids]))
        source_xy_rows.append(source_xy)
        target_xy_rows.append(target_xy)
        source_neg_xy_rows.append(source_negative_candidates[source_neg_ids])
        target_neg_xy_rows.append(target_candidates[target_neg_ids])
        texture_seeds.append(texture_seed)
        warp_seeds.append(warp_seed)
        modes.append(mode)
        support_stats.append(
            {
                "source_detected": len(source_points),
                "source_full_support": int(source_support_valid.sum()),
                "target_detected": len(target_points),
                "target_full_support": int(target_support_valid.sum()),
            }
        )
        if (index + 1) % 4 == 0 or index + 1 == count:
            print(f"generated {index + 1}/{count} detector-sampled image pairs", flush=True)
    shared_negative_count = min(len(row) for row in source_neg_rows)
    return {
        # Float16 halves the dataset footprint; intensities are already 8-bit.
        "source": np.asarray(source_rows, dtype=np.float16),
        "target_positive": np.asarray(target_rows, dtype=np.float16),
        "source_negative": np.asarray(
            [row[:shared_negative_count] for row in source_neg_rows], dtype=np.float16
        ),
        "target_negative": np.asarray(
            [row[:shared_negative_count] for row in target_neg_rows], dtype=np.float16
        ),
        "source_xy": np.asarray(source_xy_rows, dtype=np.float32),
        "target_xy": np.asarray(target_xy_rows, dtype=np.float32),
        "source_negative_xy": np.asarray(
            [row[:shared_negative_count] for row in source_neg_xy_rows], dtype=np.float32
        ),
        "target_negative_xy": np.asarray(
            [row[:shared_negative_count] for row in target_neg_xy_rows], dtype=np.float32
        ),
        "actual_negative_candidates_per_pair": np.asarray(
            [len(row) for row in source_neg_rows], dtype=np.int32
        ),
        "texture_seeds": np.asarray(texture_seeds, dtype=np.int64),
        "warp_seeds": np.asarray(warp_seeds, dtype=np.int64),
        "motion_modes": np.asarray(modes, dtype="U16"),
        "support_stats_json": np.asarray(
            json.dumps(support_stats, sort_keys=True), dtype="U2048"
        ),
    }


def create_model(torch: Any):
    class DICPatchCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=4, padding=1)
            self.depthwise = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.projection = torch.nn.Linear(8 * 4 * 4, 64)

        def forward(self, patches):
            value = torch.relu(self.conv1(patches))
            value = torch.relu(self.depthwise(value))
            value = torch.nn.functional.adaptive_avg_pool2d(value, (4, 4))
            value = self.projection(value.flatten(1))
            return torch.nn.functional.normalize(value, dim=-1, eps=1e-8)

    return DICPatchCNN()


def _hard_negative_loss(torch, query, positive, negative, positive_xy, negative_xy):
    temperature = 0.08
    logits = query @ negative.T / temperature
    distances = torch.cdist(positive_xy, negative_xy)
    logits = logits.masked_fill(distances < EXCLUSION_RADIUS_PX, -1e4)
    hard_count = min(128, logits.shape[1])
    hard = torch.topk(logits, hard_count, dim=1).values
    positive_logits = torch.sum(query * positive, dim=-1) / temperature
    return torch.nn.functional.softplus(torch.logsumexp(hard, dim=1) - positive_logits).mean()


def affine_augment_patches(torch, patches, rng: np.random.Generator):
    """Apply independent affine nuisance transforms without changing labels."""
    count = len(patches)
    angle = torch.from_numpy(rng.uniform(-np.pi, np.pi, count).astype(np.float32))
    scale_x = torch.from_numpy(np.exp(rng.uniform(np.log(0.7), np.log(1.45), count)).astype(np.float32))
    scale_y = torch.from_numpy(np.exp(rng.uniform(np.log(0.7), np.log(1.45), count)).astype(np.float32))
    shear = torch.from_numpy(rng.uniform(-0.25, 0.25, count).astype(np.float32))
    cosine, sine = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros((count, 2, 3), dtype=torch.float32)
    theta[:, 0, 0] = cosine * scale_x
    theta[:, 0, 1] = -sine * scale_y + shear
    theta[:, 1, 0] = sine * scale_x
    theta[:, 1, 1] = cosine * scale_y
    theta[:, 0, 2] = torch.from_numpy(rng.uniform(-0.08, 0.08, count).astype(np.float32))
    theta[:, 1, 2] = torch.from_numpy(rng.uniform(-0.08, 0.08, count).astype(np.float32))
    grid = torch.nn.functional.affine_grid(theta, patches.shape, align_corners=False)
    return torch.nn.functional.grid_sample(
        patches, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def _pair_metrics(torch, model, source, target, source_neg, target_neg, source_xy, target_xy, source_neg_xy, target_neg_xy):
    query_source = model(source)
    query_target = model(target)
    negatives_source = model(source_neg)
    negatives_target = model(target_neg)
    target_candidates = torch.cat((query_target, negatives_target), dim=0)
    source_candidates = torch.cat((query_source, negatives_source), dim=0)
    target_candidate_xy = torch.cat((target_xy, target_neg_xy), dim=0)
    source_candidate_xy = torch.cat((source_xy, source_neg_xy), dim=0)
    target_logits = query_source @ target_candidates.T
    source_logits = query_target @ source_candidates.T
    target_allowed = torch.cdist(target_xy, target_candidate_xy) >= EXCLUSION_RADIUS_PX
    source_allowed = torch.cdist(source_xy, source_candidate_xy) >= EXCLUSION_RADIUS_PX
    diagonal = torch.arange(len(source), device=source.device)
    target_allowed[diagonal, diagonal] = True
    source_allowed[diagonal, diagonal] = True
    target_logits = target_logits.masked_fill(~target_allowed, -2.0)
    source_logits = source_logits.masked_fill(~source_allowed, -2.0)
    target_rank = torch.argsort(torch.argsort(target_logits, dim=1, descending=True), dim=1)
    source_rank = torch.argsort(torch.argsort(source_logits, dim=1, descending=True), dim=1)
    target_correct = target_rank[diagonal, diagonal] == 0
    source_correct = source_rank[diagonal, diagonal] == 0
    return {
        "top1": float((target_correct.float().mean() + source_correct.float().mean()) / 2),
        "top5": float(
            ((target_rank[diagonal, diagonal] < 5).float().mean()
             + (source_rank[diagonal, diagonal] < 5).float().mean())
            / 2
        ),
        "source_descriptors": query_source,
        "target_descriptors": query_target,
    }


def train(
    torch,
    train_data,
    validation,
    *,
    epochs: int,
    seed: int,
    use_affine_augmentation: bool,
):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    model = create_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None

    def tensor(array):
        return torch.from_numpy(np.asarray(array).astype(np.float32, copy=False)).to(device)

    train_arrays = {
        name: tensor(train_data[name])
        for name in ("source", "target_positive", "source_negative", "target_negative")
    }
    train_xy = {name: torch.from_numpy(train_data[name].astype(np.float32)).to(device) for name in (
        "source_xy", "target_xy", "source_negative_xy", "target_negative_xy"
    )}
    validation_rows = []
    for index in range(len(validation["source"])):
        item = {
            name: tensor(validation[name][index])
            for name in ("source", "target_positive", "source_negative", "target_negative")
        }
        item.update({
            name: torch.from_numpy(validation[name][index].astype(np.float32)).to(device)
            for name in ("source_xy", "target_xy", "source_negative_xy", "target_negative_xy")
        })
        validation_rows.append(item)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_data["source"]))
        losses = []
        for pair_index in order:
            values = train_arrays
            q = values["source"].shape[1]
            # All query positives are trained together; the candidate pool is
            # global to each image pair and top-128 online hard negatives are
            # selected independently for every query.
            patches = (
                values["source"][pair_index],
                values["target_positive"][pair_index],
                values["source_negative"][pair_index],
                values["target_negative"][pair_index],
            )
            if use_affine_augmentation:
                patches = tuple(affine_augment_patches(torch, value, rng) for value in patches)
            source, target, source_negative, target_negative = (
                model(value) for value in patches
            )
            xy = train_xy
            loss = 0.5 * (
                _hard_negative_loss(
                    torch,
                    source,
                    target,
                    target_negative,
                    xy["target_xy"][pair_index],
                    xy["target_negative_xy"][pair_index],
                )
                + _hard_negative_loss(
                    torch,
                    target,
                    source,
                    source_negative,
                    xy["source_xy"][pair_index],
                    xy["source_negative_xy"][pair_index],
                )
            )
            # Include cross-query hard negatives from the aligned positives.
            # Nearby material points are masked to avoid false-negative labels.
            for query, candidate, query_xy, candidate_xy in (
                (source, target, xy["target_xy"][pair_index], xy["target_xy"][pair_index]),
                (target, source, xy["source_xy"][pair_index], xy["source_xy"][pair_index]),
            ):
                cross = query @ candidate.T / 0.08
                allowed = torch.cdist(query_xy, candidate_xy) >= EXCLUSION_RADIUS_PX
                allowed.fill_diagonal_(False)
                cross = cross.masked_fill(~allowed, -1e4)
                hard = torch.topk(cross, min(64, q), dim=1).values
                positive_logits = torch.sum(query * candidate, dim=-1) / 0.08
                loss = loss + 0.25 * torch.nn.functional.softplus(
                    torch.logsumexp(hard, dim=1) - positive_logits
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        family = {mode: {"top1": [], "top5": []} for mode in MOTIONS}
        with torch.inference_mode():
            for index, item in enumerate(validation_rows):
                metrics = _pair_metrics(
                    torch,
                    model,
                    item["source"], item["target_positive"],
                    item["source_negative"], item["target_negative"],
                    item["source_xy"], item["target_xy"],
                    item["source_negative_xy"], item["target_negative_xy"],
                )
                mode = str(validation["motion_modes"][index])
                family[mode]["top1"].append(metrics["top1"])
                family[mode]["top5"].append(metrics["top5"])
        family_metrics = {
            mode: {
                key: float(np.mean(values[key])) if values[key] else None
                for key in ("top1", "top5")
            }
            for mode, values in family.items()
        }
        valid_family_top1 = [entry["top1"] for entry in family_metrics.values() if entry["top1"] is not None]
        macro_top1 = float(np.mean(valid_family_top1)) if valid_family_top1 else 0.0
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_macro_family_top1": macro_top1,
            "validation_by_motion": family_metrics,
        }
        history.append(row)
        if macro_top1 > best_score:
            best_score = macro_top1
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 2 == 0 or epoch + 1 == epochs:
            print(row, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def export(path: Path, model, train_meta: dict[str, Any]) -> str:
    metadata = {
        **train_meta,
        "architecture": "dic_patch_cnn_v1",
        "support_px": PATCH_SIZE,
        "feature_channels": 8,
        "output_dimension": 64,
        "normalization": "L2",
        "operations": "3x3 stride-4 convolution, depthwise 3x3, 4x4 spatial average pooling, linear projection",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        conv1_weight=model.conv1.weight.detach().numpy().astype(np.float32),
        conv1_bias=model.conv1.bias.detach().numpy().astype(np.float32),
        depthwise_weight=model.depthwise.weight.detach().numpy().astype(np.float32),
        depthwise_bias=model.depthwise.bias.detach().numpy().astype(np.float32),
        projection_weight=model.projection.weight.detach().numpy().astype(np.float32),
        projection_bias=model.projection.bias.detach().numpy().astype(np.float32),
    )
    digest = sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps({**metadata, "weights_file": path.name, "weights_sha256": digest}, indent=2)
        + "\n"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=24)
    parser.add_argument("--validation-pairs", type=int, default=10)
    parser.add_argument("--points-per-pair", type=int, default=128)
    parser.add_argument("--negative-points-per-image", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--device-type", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--no-affine-augmentation",
        action="store_true",
        help="Disable independent random rotation, scale, shear and subpixel patch jitter",
    )
    args = parser.parse_args()
    if min(args.train_pairs, args.validation_pairs, args.points_per_pair, args.epochs) < 1:
        parser.error("pair counts, point budgets and epochs must be positive")
    if args.negative_points_per_image < args.points_per_pair:
        parser.error("negative point budget must be at least the positive query budget")
    try:
        import torch
    except ImportError as error:
        parser.error(f"Install the optional research extra to train the model: {error}")
    started = time.perf_counter()
    with WGPUMatching(device_type=args.device_type) as matcher:
        print(f"DIC keypoints will be produced by {matcher.device_name}", flush=True)
        train_data = build_dataset(
            args.train_pairs,
            args.points_per_pair,
            args.negative_points_per_image,
            args.seed,
            matcher,
            args.image_size,
        )
        validation = build_dataset(
            args.validation_pairs,
            args.points_per_pair,
            args.negative_points_per_image,
            args.seed + 700_000_003,
            matcher,
            args.image_size,
        )
    model, history, selected_epoch = train(
        torch,
        train_data,
        validation,
        epochs=args.epochs,
        seed=args.seed,
        use_affine_augmentation=not args.no_affine_augmentation,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train_data), ("validation", validation)):
        np.savez_compressed(
            output / f"{name}_split.npz",
            texture_seeds=data["texture_seeds"],
            warp_seeds=data["warp_seeds"],
            motion_modes=data["motion_modes"],
            support_stats_json=data["support_stats_json"],
        )
    model_meta = {
        "seed": args.seed,
        "train_pairs": args.train_pairs,
        "validation_pairs": args.validation_pairs,
        "query_points_per_pair": args.points_per_pair,
        "detector_negative_candidates_per_image": args.negative_points_per_image,
        "actual_train_negative_candidates_per_pair": train_data[
            "actual_negative_candidates_per_pair"
        ].tolist(),
        "actual_validation_negative_candidates_per_pair": validation[
            "actual_negative_candidates_per_pair"
        ].tolist(),
        "motion_families": list(MOTIONS),
        "point_source": "WGSL DIC-Detector spatially distributed response points",
        "positive_sampling": "analytic floating-point correspondence; bilinear image sampling",
        "support_validity": "17x17 lattice over full 64x64 support; source and mapped target must be visible",
        "negative_mining": "online top-128 negatives from the full same-image DIC-Detector candidate pool, symmetrically in both directions; 48 px false-negative exclusion",
        "validation_metric": "macro mean of per-motion-family top-1 retrieval; independent texture identities",
        "patch_augmentation": (
            "independent random rotation [-180,180] degrees, log-uniform anisotropic scale [0.7,1.45], shear [-0.25,0.25], and translation [-0.08,0.08] normalized patch units; bilinear resampling with border padding"
            if not args.no_affine_augmentation
            else "disabled"
        ),
        "torch_version": torch.__version__,
    }
    weights_path = output / "dic_patch_cnn_descriptor.npz"
    digest = export(weights_path, model, model_meta)
    report = {
        "metadata": model_meta,
        "training_texture_seed_sha256": hashlib.sha256(train_data["texture_seeds"].tobytes()).hexdigest(),
        "validation_texture_seed_sha256": hashlib.sha256(validation["texture_seeds"].tobytes()).hexdigest(),
        "weights_sha256": digest,
        "selected_epoch": selected_epoch,
        "selected_validation_by_motion": history[selected_epoch - 1]["validation_by_motion"],
        "total_wall_s": time.perf_counter() - started,
        "history": history,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {weights_path}; SHA-256 {digest}", flush=True)


if __name__ == "__main__":
    main()
