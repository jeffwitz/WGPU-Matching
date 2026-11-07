"""Compare the learned WGSL detector scores with the PyTorch oracle."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from wgpu_matching import DICDetectorOptions, WGPUMatching
from wgpu_matching.dic_model import build_dic_score_model
from wgpu_matching.dic_synthetic import generate_speckle

GRID_STRIDE = 4
GRID_OFFSET = 2


def _torch_sample_scores(torch: Any, model: Any, image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    from torch.nn import functional

    height, width = image.shape
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
    # The WGSL shader reports the subpixel NMS location but its score belongs to
    # the original integer grid candidate. Reconstruct that candidate centre.
    candidate_xy = (np.rint((xy - GRID_OFFSET) / GRID_STRIDE) * GRID_STRIDE + GRID_OFFSET).astype(
        np.float32
    )
    coordinates = torch.from_numpy(candidate_xy)[None, None]
    normalized = torch.stack(
        (
            2.0 * (coordinates[..., 0] + 0.5) / width - 1.0,
            2.0 * (coordinates[..., 1] + 0.5) / height - 1.0,
        ),
        dim=-1,
    )
    with torch.inference_mode():
        score_map = torch.sigmoid(model(image_tensor))
        sampled = functional.grid_sample(
            score_map,
            normalized,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
    return sampled[0, 0, 0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--keypoints", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device-type", choices=("auto", "gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if min(args.width, args.height, args.keypoints, args.repeats) < 1 or not args.weights.is_file():
        parser.error("dimensions, budget, and repeats must be positive and weights must exist")

    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "Install the optional research extra for PyTorch parity validation"
        ) from error

    with np.load(args.weights, allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    torch_model = build_dic_score_model().eval()
    torch_model.load_state_dict(state)
    generated, texture_metadata = generate_speckle(
        seed=88312,
        width=args.width,
        height=args.height,
        characteristic_size_px=4.0,
        coverage=0.32,
        blur_sigma_px=0.5,
        noise_std=0.0,
        illumination_variation=0.0,
    )
    image = np.rint(generated * 255.0).astype(np.uint8)
    settings = DICDetectorOptions(
        stride_px=4,
        tensor_radius_px=6,
        max_keypoints=args.keypoints,
        minimum_score=0.0,
    )
    with WGPUMatching(device_type=args.device_type) as matcher:
        matcher.detect_dic(image, options=settings, learned_weights_path=args.weights)
        wall_times = []
        result = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = matcher.detect_dic(image, options=settings, learned_weights_path=args.weights)
            wall_times.append((time.perf_counter() - started) * 1000.0)
        adapter = matcher.device_name
    assert result is not None
    torch_scores = _torch_sample_scores(torch, torch_model, image, result.features.xy)
    error = np.abs(torch_scores - result.features.scores)
    report = {
        "adapter": adapter,
        "weights": str(args.weights.resolve()),
        "weights_sha256": result.metadata["weights_sha256"],
        "image_size_px": [args.width, args.height],
        "texture": texture_metadata,
        "selected_keypoints": len(result.features.xy),
        "steady_state_wall_ms": {
            "median": float(statistics.median(wall_times)),
            "p10": float(np.percentile(wall_times, 10)),
            "p90": float(np.percentile(wall_times, 90)),
            "repeats": wall_times,
        },
        "gpu_stage_wall_ms": result.metadata["full_detection_wall_ms"],
        "dispatch_and_readback_ms": result.metadata["dispatch_and_readback_ms"],
        "readback_bytes": result.metadata["readback_bytes"],
        "parity": {
            "maximum_absolute_score_error": float(error.max()) if len(error) else 0.0,
            "median_absolute_score_error": float(np.median(error)) if len(error) else 0.0,
            "mean_absolute_score_error": float(np.mean(error)) if len(error) else 0.0,
            "within_1e-4": int(np.count_nonzero(error <= 1e-4)),
            "within_1e-3": int(np.count_nonzero(error <= 1e-3)),
        },
        "scores": {
            "wgsl": result.features.scores.tolist(),
            "pytorch": torch_scores.tolist(),
            "absolute_error": error.tolist(),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in report.items() if key != "scores"}, indent=2))


if __name__ == "__main__":
    main()
