"""Qualify image-derived 64-to-32 subset transfer on archived synthetic pairs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from diagnose_shear_support import map_centres_to_grid
from solver_image_cache import read_solver_images
from wgpu_matching.seed_propagation import transfer_to_fine_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    original = args.archive / "affine_ncc"
    report = json.loads((original / f"combined_seed{args.seed_offset}.json").read_text())
    if report["protocol"].get("synthetic_truth_convention") != "analytic_exact_center":
        raise ValueError("requires source archives with exact-center analytical truth")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.coarse.glob("*.npz"))
    if not paths:
        raise ValueError("no coarse results")
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=32, subset_stride=16,
                      interpolation="linear", motion_model="affine", outlier_policy="disabled")
    rows = []
    try:
        for path in paths:
            family = path.stem.split("_texture", 1)[0]
            archive = original / f"windows_seed{args.seed_offset}" / path.name
            reference, target = read_solver_images(original / f"solver_images_seed{args.seed_offset}", archive)
            with np.load(path) as coarse, np.load(archive) as source:
                centers = coarse["centres"].copy()
                if not np.array_equal(centers, source["centres_xy_native"]):
                    raise ValueError("coarse and source window coordinates differ")
                shape = (len(np.unique(centers[:, 1])), len(np.unique(centers[:, 0])))
                # This selection sees image-verified states only, never truth or visibility.
                initial = transfer_to_fine_grid(centers, shape, coarse["parameters"], coarse["accepted"], centers)
                available = initial["available"].astype(bool)
                solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
                indices = map_centres_to_grid(centers, 32, 16, solver.grid_width, solver.grid_height)
                count = solver.grid_width * solver.grid_height
                state = np.zeros((count, 8), np.float32)
                state[indices[available], :6] = initial["parameters"][available]
                state[indices[available], 6] = 1
                active = np.zeros(count, np.uint32)
                active[indices[available]] = 1
                result = solver._register_seeded_frame(target, state, active)
                parameters = np.asarray(result.parameters).reshape(-1, 6)[indices]
                accepted = available & (np.asarray(result.status).reshape(-1)[indices] > .5)
                # Truth and material-boundary labels are used solely after inference.
                truth, visible = source["truth_parameters"], source["visible"].astype(bool)
                if not np.array_equal(coarse["visible"].astype(bool), visible):
                    raise ValueError("coarse and source visibility metadata disagree")
                error = np.linalg.norm(parameters[:, [0, 3]] - truth[:, [0, 3]], axis=1)
                good = visible & accepted & (error < 2)
                bad = visible & accepted & (error >= 2)
                prior_accepted = coarse["accepted"].astype(bool)
                prior_error = np.linalg.norm(coarse["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                prior_good = visible & prior_accepted & (prior_error < 2)
                prior_bad = visible & prior_accepted & (prior_error >= 2)
                crossing = np.zeros(len(centers), bool)
                source_crossing = np.zeros(len(centers), bool)
                opposite = np.zeros(len(centers), bool)
                if family == "discontinuous":
                    boundary = 3840 * report["protocol"]["discontinuity_boundary_fraction"]
                    x = centers[:, 0]
                    crossing = (x - 15.5 < boundary) & (x + 15.5 >= boundary)
                    eligible = np.flatnonzero(available)
                    source_x = x[initial["source"][eligible]]
                    source_crossing[eligible] = (source_x - 31.5 < boundary) & (source_x + 31.5 >= boundary)
                    opposite[eligible] = (x[eligible] < boundary) != (source_x < boundary)
                rows.append({"pair": path.stem, "family": family, "visible": int(visible.sum()),
                             "coarse_correct": int(prior_good.sum()), "coarse_false": int(prior_bad.sum()),
                             "coarse_accepted_visible": int((visible & prior_accepted).sum()),
                             "available_visible": int((visible & available).sum()),
                             "fine_correct": int(good.sum()), "fine_false": int(bad.sum()),
                             "fine_accepted_visible": int((visible & accepted).sum()),
                             "accepted_invisible": int((~visible & accepted).sum()),
                             "coarse_bad_recovered": int((prior_bad & good).sum()),
                             "coarse_good_lost": int((prior_good & ~good).sum()),
                             "new_correct": int((good & ~prior_good).sum()),
                             "fine_accepted_crossing": int((accepted & crossing).sum()),
                             "visible_opposite_source": int((visible & available & opposite).sum()),
                             "visible_source_crossing": int((visible & available & source_crossing).sum())})
                np.savez_compressed(args.output_dir / path.name, centres=centers, truth=truth, visible=visible,
                                    available=available, source=initial["source"], initial=initial["parameters"],
                                    parameters=parameters, accepted=accepted, error_px=error, correct=good,
                                    prior_accepted=prior_accepted, prior_error_px=prior_error,
                                    prior_correct=prior_good, support_crossing=crossing,
                                    source_crossing=source_crossing, opposite_source=opposite)
            print(f"Completed {path.stem}", flush=True)
    finally:
        solver.close()
    (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
