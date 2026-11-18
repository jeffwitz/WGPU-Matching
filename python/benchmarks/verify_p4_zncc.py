"""Compare 4K final-solve correlation against the independent residual shader."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images
from wgpu_matching import WGPUMatching
from wgpu_matching.affine_residual import AffineResidualWGPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    source = args.archive / "affine_ncc"
    archive = source / f"windows_seed{args.seed_offset}" / args.coarse.name
    reference, target = read_solver_images(
        source / f"solver_images_seed{args.seed_offset}", archive
    )
    with np.load(args.coarse) as saved:
        params = saved["parameters"].copy()
        accepted = saved["accepted"].astype(bool)
    with ICGNWGPU(
        device_type="gpu",
        subset_size=64,
        subset_stride=64,
        interpolation="linear",
        outlier_policy="disabled",
    ) as engine:
        engine.set_reference(reference, zncc_min=0.65)
        shape = (engine.grid_height, engine.grid_width)
        result = engine.register_seeded(
            target, params.reshape(*shape, 6), accepted.reshape(shape), include_zncc=True
        )
        with WGPUMatching(device_type="gpu") as matcher:
            other = AffineResidualWGPU(matcher.device).evaluate(
                reference, target, result.parameters.reshape(-1, 6), result.status.ravel() == 1
            )
        scores = result.zncc.ravel()
        common = np.isfinite(scores) & other["valid"]
        differences = np.abs(scores[common] - other["correlations"][common, 0])
        record = {
            "pair": args.coarse.stem,
            "valid": int(common.sum()),
            "mask_equal": bool(np.array_equal(np.isfinite(scores), other["valid"])),
            "max_difference": float(differences.max()) if differences.size else None,
            "rmse": float(np.sqrt(np.mean(differences**2))) if differences.size else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if not record["mask_equal"] or not record["valid"] or record["max_difference"] > 2e-5:
        raise SystemExit("Final correlation does not match independent residual shader")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
