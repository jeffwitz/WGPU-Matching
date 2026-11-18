"""Frozen paired affine/RBF/derivative-RBF restart and gradient-gate ablation."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hudge_rbf_restart import fit_seed_field
from hudge_strains_data import material_mask
from PIL import Image
from summarize_hudge_strains import photometric_check
from wgpu_matching.seed_initialization import initialize
from wgpu_matching.seed_propagation import propagate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[50, 100, 150, 191])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "stage1": "archived rectified_native affine restart, identical for every variant",
        "methods": ["affine", "rbf_position", "rbf_gradient"], "tau_g": [.25, .5, 1.],
        "tau_u": 8., "subset": 32, "stride": 32, "iterations": 30, "tolerance": .001,
        "zncc_min": .65, "rounds_max": 5, "sparse_stride": 1,
        "length_px": 64., "gradient_length_px": 16., "ridge": 1e-4,
        "notes": ["No truth used; no temporal sequence; archived accepted states immutable",
                  "Every variant proposes at all eligible centres, unlike former sparse stride 2",
                  "RBF coefficients fitted on CPU; all IC-GN calls on WebGPU",
                  "Only positive determinant and condition <=100 proposals enter sparse trials",
                  "No claims of correctness without associated displacement truth"],
    }
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    sys.path.insert(0, "/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    from live_dic import ICGNWGPU
    data = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    baseline = Path("results/hudge_affine_restart_20261115/all_levels/rectified_native")
    reference = np.asarray(Image.open(data/"Img_F.tif"))
    solver = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=32,
                      subset_stride=32, interpolation="linear", motion_model="affine",
                      outlier_policy="disabled")
    solver.set_reference(reference, iterations=30, tolerance=1e-3, zncc_min=.65,
                         roi_mask=material_mask(reference))
    shape = solver.roi.selected.shape
    rows = []
    for level in args.levels:
        target = np.asarray(Image.open(data/f"Img_wa_g{level:03d}.tif"))
        with np.load(baseline/f"level_{level:03d}.npz") as z:
            centres, valid, initial, original = [z[k].copy() for k in
                                               ["centres", "valid_roi", "parameters", "accepted"]]
        assert np.array_equal(valid, solver.roi.selected.ravel())
        count = len(centres)
        for method in protocol["methods"]:
            for tau_g in protocol["tau_g"]:
                started = time.perf_counter()
                folder = args.output / f"{method}_g{tau_g:.2f}"
                folder.mkdir(exist_ok=True)
                parameters, accepted = initial.copy(), original.copy()
                rounds, calls = [], []
                history = []

                def solve(indices, values, target=target, count=count, calls=calls, history=history):
                    packed = np.zeros((count, 8), np.float32)
                    packed[indices, :6], packed[indices, 6] = values, 1
                    active = np.zeros(count, np.uint32)
                    active[indices] = 1
                    result = solver._register_seeded_frame(target, packed, active.reshape(shape))
                    out = np.asarray(result.parameters).reshape(-1, 6)[indices].copy()
                    raw = np.asarray(result.status).ravel()[indices]
                    calls.append(len(indices))
                    history.append((indices.copy(), values.copy(), out, raw.copy()))
                    return out, (raw == 1) & np.isfinite(out).all(axis=1)

                for iteration in range(5):
                    before = accepted.copy()
                    field, fit = fit_seed_field(centres, parameters, accepted, method=method)
                    proposal = field.evaluate(centres)
                    gradients = proposal[:, [1, 2, 4, 5]].reshape(-1, 2, 2)+np.eye(2)
                    eligible = valid & (np.linalg.det(gradients) > 0) & (np.linalg.cond(gradients) <= 100)
                    restart = initialize(centres, shape, valid, [(method, proposal, eligible)], solve,
                                         sparse_stride=1, tau_u=8., tau_g=tau_g,
                                         minimum_anchors=3, batch_confirmation=True)
                    new = restart["accepted"] & ~accepted
                    parameters[new] = restart["parameters"][new]
                    accepted |= new
                    merged = propagate(centres, shape, valid, parameters, accepted, solve,
                                       tau_u=8., tau_g=tau_g)
                    parameters, accepted = merged["parameters"], merged["accepted"]
                    assert np.array_equal(parameters[original], initial[original])
                    rounds.append({"round": iteration+1, "before": int(before.sum()),
                                   "after": int(accepted.sum()), "restart_added": int(new.sum()),
                                   "invalid_proposals": int(np.sum(valid & ~eligible)),
                                   "sparse_accepted": restart["hypothesis_acceptances"][0]["accepted"], **fit})
                    np.savez_compressed(folder/f"level_{level:03d}_round_{iteration+1}.npz",
                                        proposal=proposal, before=before, parameters=parameters,
                                        accepted=accepted, restart_reason=restart["rejection_reason"],
                                        merged_reason=merged["rejection_reason"], matrix=field.matrix,
                                        translation=field.translation, sites=field.sites,
                                        coefficients=field.coefficients, length=field.length)
                    if accepted.sum() == before.sum():
                        break
                elapsed = time.perf_counter()-started
                corr, bounds, _ = photometric_check(reference/255., target/255., centres, parameters, accepted, 32)
                grad = parameters[:, [1, 2, 4, 5]].reshape(-1, 2, 2)+np.eye(2)
                added = accepted & ~original
                row = {"level": level, "method": method, "tau_g": tau_g,
                       "stage1": int(original.sum()), "final": int(accepted.sum()), "valid": int(valid.sum()),
                       "new": int(added.sum()), "rounds": rounds, "wall_s": elapsed,
                       "solver_calls": len(calls), "solver_windows": sum(calls),
                       "below_065": int(np.sum(accepted & (corr < .65))),
                       "out_of_bounds": int(np.sum(accepted & ~bounds)),
                       "nonpositive_jacobians": int(np.sum(accepted & (np.linalg.det(grad) <= 0))),
                       "min_new_correlation": float(np.nanmin(corr[added])) if added.any() else None}
                np.savez_compressed(folder/f"level_{level:03d}.npz", centres=centres, valid_roi=valid,
                                    parameters=parameters, accepted=accepted, baseline_accepted=original,
                                    correlation=corr, in_bounds=bounds)
                # Preserve every actual candidate solve, including rejected states.
                if history:
                    np.savez_compressed(folder/f"level_{level:03d}_calls.npz",
                                        call_sizes=np.array(calls), indices=np.concatenate([h[0] for h in history]),
                                        inputs=np.concatenate([h[1] for h in history]),
                                        outputs=np.concatenate([h[2] for h in history]),
                                        raw_status=np.concatenate([h[3] for h in history]))
                rows.append(row)
                (args.output/"summary.json").write_text(json.dumps(rows, indent=2))
                print(level, method, tau_g, row["stage1"], "->", row["final"], flush=True)


if __name__ == "__main__":
    main()
