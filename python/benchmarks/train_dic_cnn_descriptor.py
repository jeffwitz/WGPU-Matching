"""Train a compact 64-pixel patch CNN descriptor on analytic speckle pairs.

PyTorch is an optional research dependency. The exported weights are consumed
by the native WGSL inference path in ``wgpu_matching``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from wgpu_matching.dic_synthetic import generate_speckle_pair

ROOT = Path(__file__).resolve().parents[2]
MOTIONS = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")
PATCH_SIZE = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_patches(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = image.shape
    offsets = np.arange(PATCH_SIZE, dtype=np.float32) - 31.0
    xx, yy = np.meshgrid(offsets, offsets)
    patches = np.empty((len(points), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for index, (cx, cy) in enumerate(points):
        x = cx + xx
        y = cy + yy
        x0 = np.clip(np.floor(x).astype(np.int32), 0, width - 1)
        y0 = np.clip(np.floor(y).astype(np.int32), 0, height - 1)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        wx = x - np.floor(x)
        wy = y - np.floor(y)
        patches[index] = (
            image[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + image[y0, x1] * wx * (1.0 - wy)
            + image[y1, x0] * (1.0 - wx) * wy
            + image[y1, x1] * wx * wy
        )
    return patches[:, None]


def sample_points(pair, rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = pair.reference.shape
    points = []
    for _ in range(count * 40):
        x = int(rng.integers(88, width - 88))
        y = int(rng.integers(88, height - 88))
        if not pair.visible[y, x]:
            continue
        tx, ty = x + pair.flow_xy[y, x]
        if not (40 <= tx < width - 40 and 40 <= ty < height - 40):
            continue
        if pair.target_visible is not None and not pair.target_visible[round(ty), round(tx)]:
            continue
        points.append((x, y, float(tx), float(ty)))
        if len(points) == count:
            break
    if len(points) < max(16, count // 2):
        raise RuntimeError(f"Only {len(points)} of {count} requested points are visible")
    array = np.asarray(points, dtype=np.float32)
    return array[:, :2], array[:, 2:]


def build_dataset(count: int, point_count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    source_rows, target_rows, coordinate_rows = [], [], []
    texture_seeds, warp_seeds, modes = [], [], []
    for index in range(count):
        texture_seed = seed + index * 1009
        warp_seed = seed + 7_000_001 + index * 7919
        mode = MOTIONS[index % len(MOTIONS)]
        pair = generate_speckle_pair(texture_seed, warp_seed, 512, 512, motion_mode=mode)
        source_xy, target_xy = sample_points(pair, rng, point_count)
        source_image = pair.reference.astype(np.float32)
        target_image = pair.target.astype(np.float32)
        source_rows.append(sample_patches(source_image, source_xy))
        target_rows.append(sample_patches(target_image, target_xy))
        coordinate_rows.append(source_xy)
        texture_seeds.append(texture_seed)
        warp_seeds.append(warp_seed)
        modes.append(mode)
        if (index + 1) % 8 == 0 or index + 1 == count:
            print(f"generated {index + 1}/{count} analytic image pairs", flush=True)
    shared_count = min(len(row) for row in source_rows)
    if shared_count < 16:
        raise RuntimeError("Fewer than 16 shared visible points across the split")
    return {
        "source": np.asarray([row[:shared_count] for row in source_rows], dtype=np.float32),
        "target": np.asarray([row[:shared_count] for row in target_rows], dtype=np.float32),
        "source_xy": np.asarray([row[:shared_count] for row in coordinate_rows], dtype=np.float32),
        "texture_seeds": np.asarray(texture_seeds, dtype=np.int64),
        "warp_seeds": np.asarray(warp_seeds, dtype=np.int64),
        "motion_modes": np.asarray(modes, dtype="U16"),
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


def train(torch, train, validation, *, epochs: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    model = create_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    src = torch.from_numpy(train["source"]).to(device)
    tgt = torch.from_numpy(train["target"]).to(device)
    coords = torch.from_numpy(train["source_xy"]).to(device)
    val_src = torch.from_numpy(validation["source"]).to(device)
    val_tgt = torch.from_numpy(validation["target"]).to(device)
    val_coords = torch.from_numpy(validation["source_xy"]).to(device)
    rng = np.random.default_rng(seed)
    history = []
    best_validation_score = -1.0
    best_epoch = 0
    best_state = None

    def metrics(a, b, positions):
        batch, points = positions.shape[:2]
        fa = model(a.flatten(0, 1)).reshape(batch, points, 64)
        fb = model(b.flatten(0, 1)).reshape(batch, points, 64)
        sim = torch.bmm(fa, fb.transpose(1, 2))
        allowed = torch.cdist(positions, positions) >= 48.0
        diagonal = torch.eye(points, dtype=torch.bool)[None]
        allowed |= diagonal
        correct = sim.masked_fill(~allowed, -2).argmax(-1) == torch.arange(points)[None]
        return fa, fb, float(correct.float().mean())

    for epoch in range(epochs):
        order = rng.permutation(len(src))
        losses = []
        model.train()
        for begin in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[begin : begin + batch_size], dtype=torch.long)
            batch, points = ids.numel(), src.shape[1]
            a = model(src[ids].flatten(0, 1)).reshape(batch, points, 64)
            b = model(tgt[ids].flatten(0, 1)).reshape(batch, points, 64)
            logits = torch.bmm(a, b.transpose(1, 2)) / 0.08
            spatial = torch.cdist(coords[ids], coords[ids]) >= 48.0
            diagonal = torch.eye(points, dtype=torch.bool)[None]
            allowed = spatial | diagonal
            logits = logits.masked_fill(~allowed, -1e4)
            labels = torch.arange(points).repeat(batch)
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels)
                + torch.nn.functional.cross_entropy(logits.transpose(1, 2).flatten(0, 1), labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            _, _, top1 = metrics(val_src, val_tgt, val_coords)
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_top1": top1}
        )
        if top1 > best_validation_score:
            best_validation_score = top1
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(history[-1], flush=True)
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
    parser.add_argument("--train-pairs", type=int, default=32)
    parser.add_argument("--validation-pairs", type=int, default=8)
    parser.add_argument("--points-per-pair", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    if min(args.train_pairs, args.validation_pairs, args.points_per_pair, args.epochs) < 1:
        parser.error("pair counts, point budget and epochs must be positive")
    try:
        import torch
    except ImportError as error:
        parser.error(f"Install the optional research extra to train the model: {error}")
    started = time.perf_counter()
    train_data = build_dataset(args.train_pairs, args.points_per_pair, args.seed)
    validation = build_dataset(args.validation_pairs, args.points_per_pair, args.seed + 700_000_003)
    model, history, selected_epoch = train(
        torch,
        train_data,
        validation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "train_split.npz",
        texture_seeds=train_data["texture_seeds"],
        warp_seeds=train_data["warp_seeds"],
        motion_modes=train_data["motion_modes"],
    )
    np.savez_compressed(
        output / "validation_split.npz",
        texture_seeds=validation["texture_seeds"],
        warp_seeds=validation["warp_seeds"],
        motion_modes=validation["motion_modes"],
    )
    weights_path = output / "dic_patch_cnn_descriptor.npz"
    model_meta = {
        "seed": args.seed,
        "train_pairs": args.train_pairs,
        "validation_pairs": args.validation_pairs,
        "points_per_pair": args.points_per_pair,
        "motion_families": list(MOTIONS),
        "negative_mining": "same-texture locations at least 48 source pixels apart; hardest current similarities dominate InfoNCE",
        "torch_version": torch.__version__,
        "checkpoint_selection": "maximum validation top-1 retrieval; first epoch wins ties",
    }
    digest = export(weights_path, model, model_meta)
    report = {
        "metadata": model_meta,
        "training_texture_seed_sha256": hashlib.sha256(
            train_data["texture_seeds"].tobytes()
        ).hexdigest(),
        "validation_texture_seed_sha256": hashlib.sha256(
            validation["texture_seeds"].tobytes()
        ).hexdigest(),
        "weights_sha256": digest,
        "selected_epoch": selected_epoch,
        "selected_validation_top1": history[selected_epoch - 1]["validation_top1"],
        "total_wall_s": time.perf_counter() - started,
        "history": history,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {weights_path}; SHA-256 {digest}", flush=True)


if __name__ == "__main__":
    main()
