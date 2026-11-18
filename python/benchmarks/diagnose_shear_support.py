"""Compare oracle-initialized affine IC-GN support/interpolation at identical centers."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images


def map_centres_to_grid(centres, subset, stride, grid_width, grid_height):
    """Return exact row-major grid indices; fail if a center is unavailable."""
    coordinates = (np.asarray(centres, dtype=np.float64) - (subset - 1) / 2) / stride
    rounded = np.rint(coordinates)
    if (not np.allclose(coordinates, rounded, rtol=0, atol=1e-8)
            or np.any(rounded[:, 0] < 0) or np.any(rounded[:, 0] >= grid_width)
            or np.any(rounded[:, 1] < 0) or np.any(rounded[:, 1] >= grid_height)):
        raise ValueError("the requested centers do not coincide with the destination solver grid")
    return (rounded[:, 1].astype(np.int64) * grid_width + rounded[:, 0].astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.base / "g2.0/inputs/affine_ncc"
    original = args.base / "g2.0/evaluation"
    exact = args.base / "exact_center_analysis"
    rows = []
    configs = ((64, 64, "linear"), (64, 64, "cubic"), (32, 16, "linear"), (32, 16, "cubic"))
    for subset, stride, interpolation in configs:
        solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=subset, subset_stride=stride,
                          interpolation=interpolation, motion_model="affine", outlier_policy="disabled")
        try:
            for path in sorted(original.glob("*.npz")):
                archive = source / "windows_seed59000000" / path.name
                reference, target = read_solver_images(source / "solver_images_seed59000000", archive)
                truth_path = exact / "g2.0" / path.name
                if not truth_path.exists():
                    truth_path = exact / path.name  # Historical gradient-2.0 export.
                with np.load(path) as initial, np.load(truth_path) as known:
                    centers = initial["centres"]
                    truth, visible = known["truth_parameters"], known["visible"].astype(bool)
                    init_error = np.linalg.norm(initial["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                    original_bad = visible & initial["accepted"].astype(bool) & (init_error >= 2)
                    solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
                    index = map_centres_to_grid(centers, subset, stride, solver.grid_width, solver.grid_height)
                    count = solver.grid_width * solver.grid_height
                    state = np.zeros((count, 8), np.float32)
                    state[index, :6] = truth
                    state[index, 6] = 1
                    active = np.zeros(count, np.uint32)
                    active[index] = 1
                    start = time.perf_counter()
                    output = solver._register_seeded_frame(target, state, active)
                    solve_s = time.perf_counter() - start
                    params = np.asarray(output.parameters).reshape(-1, 6)[index]
                    status = np.asarray(output.status).reshape(-1)[index] > .5
                    error = np.linalg.norm(params[:, [0, 3]] - truth[:, [0, 3]], axis=1)
                    good = visible & status & (error < 2)
                    bad = visible & status & (error >= 2)
                    previous_good = known["correct"].astype(bool)
                    oracle_previous_good = (known["oracle_status"].astype(bool) & visible
                                            & (known["oracle_error_px"] < 2))
                    label = f"s{subset}_{interpolation}"
                    if (subset == 64 and interpolation == "linear"
                            and (not np.array_equal(status, known["oracle_status"].astype(bool))
                                 or not np.allclose(error, known["oracle_error_px"], rtol=0, atol=1e-4))):
                        raise ValueError("64-pixel linear oracle differs from archived exact-center control")
                    np.savez_compressed(args.output_dir / f"{path.stem}_{label}.npz", centres=centers,
                                        truth=truth, visible=visible, status=status, parameters=params,
                                        error_px=error, original_bad=original_bad, original_good=previous_good,
                                        original_oracle_good=oracle_previous_good)
                    rows.append({"pair": path.stem, "subset": subset, "stride": stride,
                                 "interpolation": interpolation, "visible": int(visible.sum()),
                                 "accepted_visible": int((visible & status).sum()), "correct": int(good.sum()),
                                 "false": int(bad.sum()), "accepted_invisible": int((~visible & status).sum()),
                                 "original_bad": int(original_bad.sum()), "original_bad_recovered": int((original_bad & good).sum()),
                                 "original_bad_rejected": int((original_bad & ~status).sum()),
                                 "original_bad_still_false": int((original_bad & bad).sum()),
                                 "original_good_lost": int((previous_good & ~good).sum()),
                                 "original_oracle_good_lost": int((oracle_previous_good & ~good).sum()),
                                 "original_oracle_good_gained": int((good & ~oracle_previous_good).sum()),
                                 "solve_wall_s_unwarmed": solve_s})
                    print(f"{path.stem} {label}: {int(good.sum())}/{int(visible.sum())}, false {int(bad.sum())}", flush=True)
        finally:
            solver.close()
    (args.output_dir / "summary.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
