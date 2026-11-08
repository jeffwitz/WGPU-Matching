"""Export the official DeDoDe-B dense descriptor plus keypoint sampling graph."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


class KeypointDescriptor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        grid = self.model.forward({"image": image})["description_grid"].float()
        sampled = torch.nn.functional.grid_sample(
            grid, keypoints[:, None], mode="bilinear", align_corners=False
        )[:, :, 0].transpose(1, 2)
        return torch.nn.functional.normalize(sampled, p=2, dim=-1, eps=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--keypoints", type=int, default=8192)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if args.height < 64 or args.width < 64 or args.keypoints < 1:
        parser.error("spatial dimensions must be >=64 and keypoints positive")
    from wgpu_matching import DeDoDeBDescriptor

    descriptor = DeDoDeBDescriptor(
        max_side=max(args.height, args.width), device="cpu", weights_path=args.weights
    )
    model = KeypointDescriptor(descriptor.model).eval()
    image = torch.zeros((1, 3, args.height, args.width), dtype=torch.float32)
    points = torch.rand((1, args.keypoints, 2), dtype=torch.float32) * 1.8 - 0.9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (image, points),
        args.output,
        input_names=["image", "keypoints"],
        output_names=["descriptors"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    import onnx

    graph = onnx.load(str(args.output))
    properties = {
        "wgpu_matching.model": "DeDoDe-B official descriptor + align_corners=False sampling",
        "wgpu_matching.source_revision": descriptor.metadata["source_revision"],
        "wgpu_matching.weights_sha256": str(descriptor.metadata["weights_sha256"]),
        "wgpu_matching.graph_sha256": digest,
        "wgpu_matching.input_shape": f"1x3x{args.height}x{args.width};1x{args.keypoints}x2",
        "wgpu_matching.output_shape": f"1x{args.keypoints}x256",
    }
    for key, value in properties.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(graph, str(args.output))
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes); sha256={digest}")


if __name__ == "__main__":
    main()
