"""Generate analytic HORG descriptor pairs and train compact DIC projections.

Requires the optional PyTorch research dependency. It writes NumPy weights that
are consumed by the WGSL inference path; PyTorch is not needed at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from wgpu_matching import FeatureSet, WGPUMatching
from wgpu_matching.dic_synthetic import generate_speckle_pair

ROOT = Path(__file__).resolve().parents[2]
MODES = ("translation", "rigid", "affine", "heterogeneous", "discontinuous")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_points(pair, rng: np.random.Generator, count: int, margin: int = 88) -> tuple[np.ndarray, np.ndarray]:
    height, width = pair.reference.shape
    candidates = []
    for _ in range(count * 30):
        x = int(rng.integers(margin, width - margin))
        y = int(rng.integers(margin, height - margin))
        if not pair.visible[y, x]:
            continue
        tx, ty = x + pair.flow_xy[y, x]
        if not (margin <= tx < width - margin and margin <= ty < height - margin):
            continue
        if pair.target_visible is not None and not pair.target_visible[round(ty), round(tx)]:
            continue
        candidates.append((x, y, float(tx), float(ty)))
        if len(candidates) == count:
            break
    if len(candidates) < max(16, count // 2):
        raise RuntimeError(f"Only {len(candidates)} of {count} requested training points are visible")
    points = np.asarray(candidates, dtype=np.float32)
    return points[:, :2], points[:, 2:]


def build_dataset(
    matcher: WGPUMatching,
    *,
    count: int,
    seed_start: int,
    points_per_pair: int,
    reductions: tuple[int, ...],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed_start)
    ref_rows, target_rows, coords_rows, labels = [], [], [], []
    for index in range(count):
        texture_seed = seed_start + 1009 * index
        warp_seed = seed_start + 9_000_001 + 7919 * index
        mode = MODES[index % len(MODES)]
        pair = generate_speckle_pair(texture_seed, warp_seed, 512, 512, motion_mode=mode)
        source_xy, target_xy = _sample_points(pair, rng, points_per_pair)
        source = FeatureSet(source_xy, np.ones(len(source_xy), dtype=np.float32))
        target = FeatureSet(target_xy, np.ones(len(target_xy), dtype=np.float32))
        reference_image = np.rint(pair.reference * 255).astype(np.uint8)
        target_image = np.rint(pair.target * 255).astype(np.uint8)
        reference_parts = [
            matcher.describe_horg_keypoints(reference_image, source, reduction=f).descriptors
            for f in reductions
        ]
        target_parts = [
            matcher.describe_horg_keypoints(target_image, target, reduction=f).descriptors
            for f in reductions
        ]
        ref_rows.append(np.concatenate(reference_parts, axis=1))
        target_rows.append(np.concatenate(target_parts, axis=1))
        coords_rows.append(source_xy)
        labels.append(mode)
        if (index + 1) % 8 == 0 or index + 1 == count:
            print(f"generated {index + 1}/{count} pairs ({mode})", flush=True)
    common_count = min(len(row) for row in ref_rows)
    if common_count < 16:
        raise RuntimeError("At least one motion family produced fewer than 16 usable points")
    print(f"using {common_count} shared valid points per pair after visibility filtering", flush=True)
    return {
        "reference": np.asarray([row[:common_count] for row in ref_rows], dtype=np.float32),
        "target": np.asarray([row[:common_count] for row in target_rows], dtype=np.float32),
        "source_xy": np.asarray([row[:common_count] for row in coords_rows], dtype=np.float32),
        "motion_mode": np.asarray(labels, dtype="U16"),
        "texture_seed": np.asarray(
            [seed_start + 1009 * i for i in range(count)], dtype=np.int64
        ),
        "warp_seed": np.asarray(
            [seed_start + 9_000_001 + 7919 * i for i in range(count)], dtype=np.int64
        ),
    }


def _torch_model(torch: Any, architecture: str, input_dim: int):
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.architecture = architecture
            if architecture == "linear":
                self.first = torch.nn.Linear(input_dim, 64)
                self.second = None
            elif architecture in {"residual_mlp", "multiscale_mlp"}:
                self.first = torch.nn.Linear(input_dim, 96)
                self.second = torch.nn.Linear(96, 64)
            else:
                raise ValueError(architecture)

        def forward(self, value):
            if self.architecture == "linear":
                output = self.first(value)
            else:
                output = self.second(torch.relu(self.first(value)))
                if self.architecture == "residual_mlp":
                    output = value + 0.25 * output
            return torch.nn.functional.normalize(output, dim=-1, eps=1e-8)

    return Projection()


def _loss_and_diagnostics(torch, model, reference, target, xy, temperature, exclusion_px):
    described_ref = model(reference)
    described_target = model(target)
    similarities = torch.bmm(described_ref, described_target.transpose(1, 2)) / temperature
    displacement = torch.cdist(xy, xy)
    allowed = displacement >= exclusion_px
    identity = torch.eye(xy.shape[1], dtype=torch.bool, device=xy.device)[None]
    allowed = allowed | identity
    logits = similarities.masked_fill(~allowed, -1e4)
    labels = torch.arange(xy.shape[1], device=xy.device).repeat(xy.shape[0])
    loss = 0.5 * (
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels)
        + torch.nn.functional.cross_entropy(logits.transpose(1, 2).flatten(0, 1), labels)
    )
    with torch.no_grad():
        positive = similarities.diagonal(dim1=1, dim2=2)
        nonself = allowed & ~identity
        negatives = similarities.masked_fill(~nonself, -1e4)
        hard_negative = negatives.max(dim=2).values
        winner = logits.argmax(dim=2)
        correct = winner == torch.arange(xy.shape[1], device=xy.device)[None]
        return loss, {
            "positive_similarity": float(positive.mean()),
            "hard_negative_similarity": float(hard_negative.mean()),
            "positive_minus_hard_negative": float((positive - hard_negative).mean()),
            "retrieval_top1": float(correct.float().mean()),
        }


def _export(path: Path, model, architecture: str, reductions, input_dim, train_meta):
    first_weight = model.first.weight.detach().cpu().numpy().astype(np.float32)
    first_bias = model.first.bias.detach().cpu().numpy().astype(np.float32)
    if model.second is None:
        second_weight = np.empty((0, 0), dtype=np.float32)
        second_bias = np.empty((0,), dtype=np.float32)
        hidden_dim = 0
    else:
        second_weight = model.second.weight.detach().cpu().numpy().astype(np.float32)
        second_bias = model.second.bias.detach().cpu().numpy().astype(np.float32)
        hidden_dim = int(first_weight.shape[0])
    metadata = {
        **train_meta,
        "architecture": architecture,
        "image_reductions": list(reductions),
        "input_dimension": int(input_dim),
        "hidden_dimension": hidden_dim,
        "output_dimension": 64,
        "normalization": "L2",
        "inference": "WGSL linear projection or one-hidden-layer ReLU MLP",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        first_weight=first_weight,
        first_bias=first_bias,
        second_weight=second_weight,
        second_bias=second_bias,
    )
    report = {**metadata, "weights_file": path.name, "weights_sha256": _sha256(path)}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


def train_model(torch, architecture, train, validation, *, epochs, batch_size, seed, device):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if architecture in {"linear", "residual_mlp"}:
        train_ref_values = train["reference"][..., 64:128]
        train_target_values = train["target"][..., 64:128]
        validation_ref_values = validation["reference"][..., 64:128]
        validation_target_values = validation["target"][..., 64:128]
    else:
        train_ref_values, train_target_values = train["reference"], train["target"]
        validation_ref_values, validation_target_values = validation["reference"], validation["target"]
    input_dim = int(train_ref_values.shape[-1])
    model = _torch_model(torch, architecture, input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_ref = torch.from_numpy(train_ref_values).to(device)
    train_tar = torch.from_numpy(train_target_values).to(device)
    train_xy = torch.from_numpy(train["source_xy"]).to(device)
    val_ref = torch.from_numpy(validation_ref_values).to(device)
    val_tar = torch.from_numpy(validation_target_values).to(device)
    val_xy = torch.from_numpy(validation["source_xy"]).to(device)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_ref))
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start : start + batch_size], device=device)
            loss, _ = _loss_and_diagnostics(
                torch,
                model,
                train_ref[indices],
                train_tar[indices],
                train_xy[indices],
                temperature=0.08,
                exclusion_px=48.0,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            _, metrics = _loss_and_diagnostics(
                torch, model, val_ref, val_tar, val_xy, 0.08, 48.0
            )
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(architecture, history[-1], flush=True)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=64)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--points-per-pair", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as error:
        parser.error(f"Install the optional CPU PyTorch package to train: {error}")
    if min(args.train_pairs, args.validation_pairs, args.points_per_pair, args.epochs) < 1:
        parser.error("pair counts, point budget and epochs must be positive")
    reductions = (2, 4, 8)
    start = time.perf_counter()
    with WGPUMatching(device_type="gpu") as matcher:
        train = build_dataset(
            matcher,
            count=args.train_pairs,
            seed_start=args.seed,
            points_per_pair=args.points_per_pair,
            reductions=reductions,
        )
        validation = build_dataset(
            matcher,
            count=args.validation_pairs,
            seed_start=args.seed + 700_000_003,
            points_per_pair=args.points_per_pair,
            reductions=reductions,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    train_path, validation_path = args.output / "train_pairs.npz", args.output / "validation_pairs.npz"
    np.savez_compressed(train_path, **train)
    np.savez_compressed(validation_path, **validation)
    train_meta = {
        "seed": args.seed,
        "train_texture_seeds_sha256": hashlib.sha256(train["texture_seed"].tobytes()).hexdigest(),
        "validation_texture_seeds_sha256": hashlib.sha256(validation["texture_seed"].tobytes()).hexdigest(),
        "train_pairs": args.train_pairs,
        "validation_pairs": args.validation_pairs,
        "points_per_pair": args.points_per_pair,
        "motion_families": list(MODES),
        "hard_negatives": "highest cosine distractors among same-texture points at least 48 source pixels apart; InfoNCE emphasizes high-scoring negatives",
        "exclusion_radius_px": 48,
        "dataset_files": {
            train_path.name: _sha256(train_path),
            validation_path.name: _sha256(validation_path),
        },
        "feature_teacher": "HORG-64 WGSL at exact analytic corresponding coordinates",
        "wall_s_before_training": time.perf_counter() - start,
    }
    results = {}
    for architecture in ("linear", "residual_mlp", "multiscale_mlp"):
        model, history = train_model(
            torch,
            architecture,
            train,
            validation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        output_path = args.output / f"dic_descriptor_{architecture}.npz"
        model_reductions = (4,) if architecture in {"linear", "residual_mlp"} else reductions
        _export(output_path, model, architecture, model_reductions, model.first.in_features, train_meta)
        results[architecture] = history
    report = {
        "training": train_meta,
        "device": args.device,
        "torch_version": torch.__version__,
        "total_wall_s": time.perf_counter() - start,
        "history": results,
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {args.output / 'training_report.json'}", flush=True)


if __name__ == "__main__":
    main()
