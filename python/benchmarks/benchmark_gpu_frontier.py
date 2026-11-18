"""Compare GPU-resident and NumPy affine frontiers using the actual IC-GN solver."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from solver_image_cache import read_solver_images
from wgpu_matching.gpu_frontier import GPUFrontier
from wgpu_matching.seed_propagation import propagate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--live-dic-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_dic_source.resolve()))
    from live_dic import ICGNWGPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sorted((args.inputs / "affine_ncc").glob("windows_seed*/*.npz")):
        seed_file = args.seeds / source.name
        if not seed_file.is_file():
            raise ValueError(f"missing seed result for {source.name}")
        offset = source.parent.name.removeprefix("windows_seed")
        reference, target = read_solver_images(args.inputs / "affine_ncc" / f"solver_images_seed{offset}", source)
        with np.load(source) as archive, np.load(seed_file) as saved:
            centres = saved["centres"].copy()
            valid = saved["valid_roi"].astype(bool)
            seed_mask = saved["confirmed_seeds"].astype(bool)
            state = saved["parameters"].copy()
            visible = archive["visible"].astype(bool)
            truth = archive["truth_parameters"].copy()
        solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=64, subset_stride=64,
                          interpolation="linear", motion_model="affine", outlier_policy="disabled")
        try:
            solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65)
            shape = (solver.grid_height, solver.grid_width)
            count = len(centres)

            def solve(indices, candidates, count=count, solver=solver, target=target):
                initial = np.zeros((count, 8), dtype=np.float32)
                initial[indices, :6] = candidates
                initial[indices, 6] = 1
                active = np.zeros(count, dtype=np.uint32)
                active[indices] = 1
                result = solver._register_seeded_frame(target, initial, active)
                return (np.asarray(result.parameters).reshape(-1, 6)[indices],
                        np.asarray(result.status).reshape(-1)[indices] > .5)

            start = time.perf_counter()
            cpu = propagate(centres, shape, valid, state, seed_mask, solve)
            cpu_wall = time.perf_counter() - start
            start = time.perf_counter()
            frontier = GPUFrontier(solver)
            pipeline_wall = time.perf_counter() - start
            start = time.perf_counter()
            gpu = frontier.propagate(target, valid, state, seed_mask)
            gpu_wall = time.perf_counter() - start
            error_cpu = np.linalg.norm(cpu["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
            error_gpu = np.linalg.norm(gpu["parameters"][:, [0, 3]] - truth[:, [0, 3]], axis=1)
            common = cpu["accepted"] & gpu["accepted"]
            equal = {key: bool(np.array_equal(cpu[key], gpu[key]))
                     for key in ("accepted", "parent", "origin", "generation", "attempts", "rejection_reason")}
            maximum = float(np.max(np.abs(cpu["parameters"][common] - gpu["parameters"][common]))) if common.any() else 0.
            rows.append({"pair": source.stem, "visible": int(visible.sum()),
                         "cpu_correct": int((visible & cpu["accepted"] & (error_cpu < 2)).sum()),
                         "gpu_correct": int((visible & gpu["accepted"] & (error_gpu < 2)).sum()),
                         "cpu_false": int((visible & cpu["accepted"] & (error_cpu >= 2)).sum()),
                         "gpu_false": int((visible & gpu["accepted"] & (error_gpu >= 2)).sum()),
                         "accepted_changed": int(np.count_nonzero(cpu["accepted"] != gpu["accepted"])),
                         "max_common_parameter_abs": maximum, "exact_metadata": equal,
                         "cpu_passes": cpu["passes"], "gpu_passes": gpu["passes"],
                         "cpu_active": cpu["active_windows"], "gpu_active": gpu["active_windows"],
                         "cpu_wall_s": cpu_wall, "gpu_wall_s": gpu_wall, "pipeline_create_s": pipeline_wall,
                         "counter_readbacks": gpu["counter_readbacks"], "counter_bytes": gpu["counter_readback_bytes"],
                         "final_bytes": gpu["final_readback_bytes"], "converged": gpu["converged"]})
            np.savez_compressed(args.output_dir / source.name, cpu_parameters=cpu["parameters"],
                                gpu_parameters=gpu["parameters"], cpu_accepted=cpu["accepted"],
                                gpu_accepted=gpu["accepted"], cpu_error=error_cpu, gpu_error=error_gpu,
                                cpu_parent=cpu["parent"], gpu_parent=gpu["parent"])
            print(f"Completed {source.stem}", flush=True)
        finally:
            solver.close()
    if not rows:
        parser.error("no prepared cases found")
    (args.output_dir / "summary.json").write_text(json.dumps({"pairs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
