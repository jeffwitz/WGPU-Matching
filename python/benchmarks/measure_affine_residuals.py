"""Compute WebGPU affine photometric residual indicators before truth scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images
from wgpu_matching import WGPUMatching
from wgpu_matching.affine_residual import AffineResidualWGPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exact-truth-directory",
        type=Path,
        help="post-hoc exact-center truth for an older rounded-truth archive",
    )
    args = parser.parse_args()
    source = args.archive / "affine_ncc"
    protocol = json.loads((source / f"combined_seed{args.seed_offset}.json").read_text())[
        "protocol"
    ]
    if not args.exact_truth_directory and not (
        protocol.get("synthetic_truth_convention") == "analytic_exact_center"
        or protocol.get("shear_truth_convention") == "analytic_exact_center"
    ):
        raise ValueError("source archive must contain exact-center analytical truth")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with WGPUMatching(device_type="gpu") as matcher:
        evaluator = AffineResidualWGPU(matcher.device)
        for f in sorted(args.coarse.glob("*.npz")):
            archive = source / f"windows_seed{args.seed_offset}" / f.name
            reference, target = read_solver_images(
                source / f"solver_images_seed{args.seed_offset}", archive
            )
            with np.load(f) as coarse:
                accepted = coarse["accepted"].astype(bool)
                result = evaluator.evaluate(reference, target, coarse["parameters"], accepted)
                # No truth or visibility was read before scoring the images.
                score_truth = (
                    args.exact_truth_directory / f.name if args.exact_truth_directory else archive
                )
                with np.load(score_truth) as source_data:
                    truth = source_data["truth_parameters"]
                    visible = source_data["visible"].astype(bool)
                    error = np.linalg.norm(
                        coarse["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1
                    )
                    good = accepted & visible & (error < 2)
                    bad = accepted & visible & (error >= 2)
                    np.savez_compressed(
                        args.output_dir / f.name,
                        correlations=result["correlations"],
                        valid_score=result["valid"],
                        accepted=accepted,
                        visible=visible,
                        correct=good,
                        false_accepted=bad,
                        error_px=error,
                    )
                    rows.append(
                        {
                            "pair": f.stem,
                            "accepted_visible": int((accepted & visible).sum()),
                            "false_accepted": int(bad.sum()),
                            "valid_score_accepted_visible": int(
                                (result["valid"] & accepted & visible).sum()
                            ),
                            "uploaded_image_bytes": result["uploaded_image_bytes"],
                            "score_readback_bytes": result["readback_bytes"],
                        }
                    )
            print(f"Completed {f.stem}", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"pairs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
