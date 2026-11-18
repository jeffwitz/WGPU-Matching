"""Compare the public GPU affine-frontier pass with a NumPy snapshot oracle."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images
from wgpu_matching.seed_propagation import neighbours, transport


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

    rows = []
    for path in sorted(args.coarse.glob("*.npz")):
        source = args.archive / "affine_ncc"
        archive = source / f"windows_seed{args.seed_offset}" / path.name
        reference, _ = read_solver_images(source / f"solver_images_seed{args.seed_offset}", archive)
        with np.load(path) as saved:
            centres = saved["centres"].copy()
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
            valid = np.asarray(engine.reference_valid, bool).ravel()
            mask = (accepted & valid).reshape(shape)
            predicted, active = engine.affine_frontier_seeds(params.reshape(*shape, 6), mask)
        expected = np.zeros_like(params, dtype=np.float64)
        expected_active = np.zeros(len(params), bool)
        for j in np.flatnonzero(valid & ~mask.ravel()):
            nearby = [i for i in neighbours(shape, j) if mask.ravel()[i]]
            if nearby:
                expected[j] = np.median(
                    [transport(params[i], centres[j] - centres[i]) for i in nearby], axis=0
                )
                expected_active[j] = True
        diff = np.abs(expected[expected_active] - predicted.reshape(-1, 6)[expected_active])
        rows.append(
            {
                "pair": path.stem,
                "active_equal": bool(np.array_equal(active.ravel(), expected_active)),
                "active_count": int(expected_active.sum()),
                "max_abs_parameter_diff": float(diff.max()) if diff.size else 0.0,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    if any(not row["active_equal"] or row["max_abs_parameter_diff"] > 2e-4 for row in rows):
        raise SystemExit("GPU and NumPy frontier outputs differ")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
