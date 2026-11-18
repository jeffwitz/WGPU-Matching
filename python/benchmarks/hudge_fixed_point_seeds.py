"""Find candidate affine seeds using the displacement increment of the native solver.

Jacobian columns are differences of one-iteration solves on the original images.
Each accepted search step must decrease the measured increment norm. These are
proposals. The independent multi-iteration verifier and material check decide acceptance.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask
from PIL import Image


def refine(folder, original, live_src, out, images=None):
    sys.path.insert(0, str(live_src))
    from live_dic import ICGNWGPU

    info = json.loads(
        Path("results/hudge_hole_moments_20261117/proposals/summary.json").read_text()
    )
    reference, target = [np.asarray(Image.open(p)) for p in (images or info["input_sha256"])]
    old = np.load(original)
    proposal = np.load(folder / "seeds.npz")
    parameters = proposal["parameters"].copy()
    active = np.flatnonzero(proposal["accepted"] & ~old["accepted"])
    half = 5.5
    scale = np.array([1, half, half, 1, half, half])
    state = parameters[active] * scale
    initial = state.copy()
    solver = ICGNWGPU(
        device_type="gpu",
        prefer_subgroups=False,
        subset_size=12,
        subset_stride=4,
        interpolation="bspline",
        motion_model="affine",
        outlier_policy="disabled",
    )
    try:
        solver.set_reference(
            reference,
            iterations=1,
            tolerance=0.001,
            zncc_min=0.65,
            roi_mask=material_mask(reference),
            roi_policy="partial",
            roi_truncation="visibility",
        )
        enabled = np.zeros(len(parameters), np.uint32)
        enabled[active] = 1
        packed = np.zeros((len(parameters), 8), np.float32)

        def increment(s):
            packed[active, :6] = s / scale
            r = solver._register_seeded_frame(
                target, packed, enabled.reshape(solver.roi.selected.shape)
            )
            step = r.parameters.reshape(-1, 6)[active] * scale - s
            status = r.status.ravel()[active]
            return step, np.where(
                (status >= 0) & np.isfinite(step).all(1), np.linalg.norm(step, axis=1), np.inf
            )

        delta, norm = increment(state)
        trace = []
        for iteration in range(15):
            columns = []
            for axis in np.eye(6) * 0.04:
                diff, dn = increment(state + axis)
                columns.append(np.where(np.isfinite(dn[:, None]), (diff - delta) / 0.04, 0))
            jac = np.stack(columns, axis=-1)
            jac = np.where(np.isfinite(jac), jac, 0)
            # Damped least squares keeps nearly unobservable directions bounded.
            jt = jac.swapaxes(1, 2)
            direction = np.linalg.solve(
                jt @ jac + 0.0001 * np.eye(6), -jt @ np.nan_to_num(delta)[:, :, None]
            )[:, :, 0]
            direction /= np.maximum(1, np.linalg.norm(direction, axis=1) / 2)[:, None]
            nxt, best_step, best = state.copy(), delta.copy(), norm.copy()
            for length in (1.0, 0.5, 0.25):
                trial = state + length * direction
                step, n = increment(trial)
                improve = (n < best) & (np.linalg.norm(trial - initial, axis=1) < 12)
                nxt[improve], best_step[improve], best[improve] = (
                    trial[improve],
                    step[improve],
                    n[improve],
                )
            state, delta, norm = nxt, best_step, best
            trace.append(int((norm < 0.001).sum()))
            print(iteration, trace[-1], flush=True)
    finally:
        solver.close()
    parameters[active] = state / scale
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "seeds.npz",
        parameters=parameters,
        centres=proposal["centres"],
        accepted=proposal["accepted"],
    )
    (out / "iterations.json").write_text(json.dumps(trace))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("results/hudge_hole_rbf_20261117/curve_transport_191/initialisation.npz"),
    )
    parser.add_argument(
        "--live-src",
        type=Path,
        default=Path("/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", nargs=2, type=Path)
    args = parser.parse_args()
    refine(args.folder, args.original, args.live_src, args.output, args.images)
