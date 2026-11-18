"""Re-solve fine supports at coarse centers from image-verified coarse states."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from diagnose_shear_support import map_centres_to_grid
from solver_image_cache import read_solver_images
from wgpu_matching.seed_propagation import transfer_to_fine_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=59000000)
    parser.add_argument("--gradient", type=float, default=2.0)
    parser.add_argument("--coarse-directory", type=Path,
                        help="optional image-derived coarse results; defaults to the base evaluation")
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.base / f"g{args.gradient}/inputs/affine_ncc"
    original = args.coarse_directory or (args.base / f"g{args.gradient}/evaluation")
    exact = args.base / "exact_center_analysis"
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=32, subset_stride=16,
                      interpolation="linear", motion_model="affine", outlier_policy="disabled")
    rows = []
    try:
        for path in sorted(original.glob("*.npz")):
            archive = source / f"windows_seed{args.seed_offset}" / path.name
            reference, target = read_solver_images(source / f"solver_images_seed{args.seed_offset}", archive)
            truth_path = exact / f"g{args.gradient}" / path.name
            if not truth_path.exists() and args.gradient == 2.0:
                truth_path = exact / path.name  # Historical gradient-2.0 export.
            with np.load(path) as coarse, np.load(truth_path) as known:
                centers = coarse["centres"]
                shape = (len(np.unique(centers[:, 1])), len(np.unique(centers[:, 0])))
                # No analytical truth, visibility or band information enters this selection.
                initial = transfer_to_fine_grid(centers, shape, coarse["parameters"], coarse["accepted"], centers)
                available = initial["available"].astype(bool)
                solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
                index = map_centres_to_grid(centers, 32, 16, solver.grid_width, solver.grid_height)
                count = solver.grid_width * solver.grid_height
                state = np.zeros((count, 8), np.float32)
                state[index[available], :6] = initial["parameters"][available]
                state[index[available], 6] = 1
                active = np.zeros(count, np.uint32)
                active[index[available]] = 1
                started = time.perf_counter()
                output = solver._register_seeded_frame(target, state, active)
                solve_s = time.perf_counter() - started
                parameters = np.asarray(output.parameters).reshape(-1, 6)[index]
                accepted = (np.asarray(output.status).reshape(-1)[index] > .5) & available
                # Only now may analytical truth and visibility be used.
                truth, visible = known["truth_parameters"], known["visible"].astype(bool)
                error = np.linalg.norm(parameters[:, [0, 3]] - truth[:, [0, 3]], axis=1)
                good = visible & accepted & (error < 2)
                bad = visible & accepted & (error >= 2)
                coarse_error = np.linalg.norm(coarse["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                coarse_good = visible & coarse["accepted"].astype(bool) & (coarse_error < 2)
                coarse_bad = visible & coarse["accepted"].astype(bool) & (coarse_error >= 2)
                rows.append({"pair": path.stem, "visible": int(visible.sum()),
                             "available_visible": int((visible & available).sum()),
                             "accepted_visible": int((visible & accepted).sum()), "correct": int(good.sum()),
                             "false": int(bad.sum()), "accepted_invisible": int((~visible & accepted).sum()),
                             "original_correct": int(coarse_good.sum()), "original_false": int(coarse_bad.sum()),
                             "original_bad_recovered": int((coarse_bad & good).sum()),
                             "original_bad_rejected": int((coarse_bad & ~accepted).sum()),
                             "original_bad_still_false": int((coarse_bad & bad).sum()),
                             "original_good_lost": int((coarse_good & ~good).sum()),
                             "newly_correct": int((good & ~coarse_good).sum()),
                             "solve_wall_s_unwarmed": solve_s})
                np.savez_compressed(args.output_dir / path.name, centres=centers, visible=visible, available=available,
                                    source=initial["source"], initial=initial["parameters"], parameters=parameters,
                                    accepted=accepted, correct=good, error_px=error, original_bad=coarse_bad,
                                    original_good=coarse_good)
                print(f"{path.stem}: {int(good.sum())}/{int(visible.sum())}, false {int(bad.sum())}", flush=True)
    finally:
        solver.close()
    (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
