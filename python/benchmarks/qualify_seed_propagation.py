"""Evaluate oracle-seeded frontier propagation with the live WebGPU solver."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from wgpu_matching.dic_synthetic import generate_speckle_pair
from wgpu_matching.seed_propagation import propagate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dic-source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.archive.glob("affine_ncc/windows_seed*/*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("no archived windows found")
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=64,
                      subset_stride=64, interpolation="linear", motion_model="affine",
                      outlier_policy="disabled")
    rows = []
    for path in paths:
        name = path.stem
        texture = int(name.split("texture")[1].split("_")[0])
        warp = int(name.split("warp")[1])
        with np.load(path) as data:
            archive = {key: data[key].copy() for key in data.files}
        pair = generate_speckle_pair(texture, warp, 3840, 2160, motion_mode="heterogeneous",
                                     texture_options={"characteristic_size_px": 7.0,
                                                      "coverage": .31, "blur_sigma_px": .9,
                                                      "softness": .612, "noise_std": .025})
        reference = np.rint(pair.reference * 255).astype(np.uint8)
        target = np.rint(pair.target * 255).astype(np.uint8)
        solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
        centres = archive["centres_xy_native"]
        shape = (len(np.unique(centres[:, 1])), len(np.unique(centres[:, 0])))
        count = len(centres)
        truth = archive["truth_parameters"]
        visible = archive["visible"].astype(bool)
        eligible = np.flatnonzero(archive["oracle_icgn_correct_lt2px"] & visible)
        valid = np.ones(count, dtype=bool)

        def run_solve(indices, candidates, count=count, target=target):
            state = np.zeros((count, 8), np.float32)
            state[indices, :6] = candidates
            state[indices, 6] = 1
            active = np.zeros(count, np.uint32)
            active[indices] = 1
            result = solver._register_seeded_frame(target, state, active)
            params = np.asarray(result.parameters).reshape(-1, 6)
            status = np.asarray(result.status).reshape(-1) > .5
            return params[indices], status[indices]

        rng = np.random.default_rng(args.seed + texture)
        for k in (1, 4, 16):
            for draw in range(args.draws):
                chosen = rng.choice(eligible, size=min(k, len(eligible)), replace=False)
                initial = np.zeros((count, 6), np.float32)
                initial[chosen], statuses = run_solve(chosen, truth[chosen])
                seeds = np.zeros(count, bool)
                seeds[chosen] = statuses
                for affine in (False, True):
                    for consistency in (False, True):
                        schedule = []
                        def solve(indices, candidates, schedule=schedule):
                            schedule.append(indices.copy())
                            return run_solve(indices, candidates)
                        started = time.perf_counter()
                        result = propagate(centres, shape, valid, initial, seeds, solve,
                                           affine=affine, consistency=consistency)
                        elapsed = time.perf_counter() - started
                        control_state = archive["initial_parameters"].copy()
                        control_status = np.zeros(count, bool)
                        for indices in schedule:
                            params, status = run_solve(indices, control_state[indices])
                            control_state[indices] = params
                            control_status[indices] = status
                        error = np.linalg.norm(result["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
                        control_error = np.linalg.norm(control_state[:, [0, 3]] - truth[:, [0, 3]], axis=1)
                        accepted = result["accepted"]
                        correct = accepted & visible & (error < 2)
                        bad = accepted & visible & (error >= 2)
                        tag = f"{name}_k{k}_draw{draw}_aff{int(affine)}_cons{int(consistency)}"
                        np.savez_compressed(out / f"{tag}.npz", centres=centres, visible=visible,
                                            parameters=result["parameters"], accepted=accepted,
                                            correct=correct, error=error, parent=result["parent"],
                                            origin=result["origin"], generation=result["generation"],
                                            attempts=result["attempts"], reason=result["rejection_reason"],
                                            control_parameters=control_state, control_status=control_status)
                        rows.append({"pair": name, "k": k, "draw": draw, "affine": affine,
                                     "consistency": consistency, "seeds": int(seeds.sum()),
                                     "visible": int(visible.sum()), "oracle": len(eligible),
                                     "correct": int(correct.sum()), "false_accepted": int(bad.sum()),
                                     "accepted": int(accepted.sum()), "passes": result["passes"],
                                     "solver_calls": result["solver_calls"],
                                     "active_windows": result["active_windows"], "wall_s": elapsed,
                                     "control_correct": int((control_status & visible & (control_error < 2)).sum()),
                                     "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "window_export": tag + ".npz"})
        (out / f"{name}.json").write_text(json.dumps([r for r in rows if r["pair"] == name], indent=2) + "\n")
        print(f"Completed {name}", flush=True)
    (out / "summary.json").write_text(json.dumps({"seed": args.seed, "draws": args.draws,
                                                   "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
