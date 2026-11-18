"""Compare quadratic 32px supports with dense affine 16px retries on WebGPU."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hudge_rbf_restart import fit_seed_field
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import map_coordinates


def sample_patch(image, centre, p, size):
    y, x = np.mgrid[:size, :size].astype(float)-(size-1)/2
    dx = p[0]+p[1]*x+p[2]*y
    dy = p[3]+p[4]*x+p[5]*y
    if len(p) == 12:
        dx += .5*p[6]*x*x+p[7]*x*y+.5*p[8]*y*y
        dy += .5*p[9]*x*x+p[10]*x*y+.5*p[11]*y*y
    qx, qy = centre[0]+x+dx, centre[1]+y+dy
    bounds = bool(np.all((qx >= 0) & (qy >= 0) & (qx <= image.shape[1]-1) & (qy <= image.shape[0]-1)))
    patch = map_coordinates(image.astype(float), [qy, qx], order=1, mode="constant", cval=0., prefilter=False)
    fxx, fxy, fyx, fyy = 1+p[1]+0*x, p[2]+0*x, p[4]+0*x, 1+p[5]+0*x
    if len(p) == 12:
        fxx += p[6]*x+p[7]*y
        fxy += p[7]*x+p[8]*y
        fyx += p[9]*x+p[10]*y
        fyy += p[10]*x+p[11]*y
    return patch, bounds, float(np.min(fxx*fyy-fxy*fyx))


def check_patch(ref, target, centre, p, size):
    a = sample_patch(ref, centre, np.zeros(6), size)[0]
    b, inside, determinant = sample_patch(target, centre, p, size)
    a, b = a-a.mean(), b-b.mean()
    corr = float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
    return corr, inside, determinant


def quadratic_states(field, centres, curvature=True):
    p = np.zeros((len(centres), 12))
    p[:, :6] = field.evaluate(centres)
    if curvature:
        h = field.hessian(centres)
        p[:, 6:] = h[:, :, [0, 0, 1], [0, 1, 1]].reshape(-1, 6)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[50, 100, 150, 191])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--continuations", type=int, default=1,
                        help="Diagnostic: warm-start unconverged outputs for additional solver calls")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"baseline": "hudge_rbf_restart_20261115/all_levels/rbf_gradient_g0.50",
                "models": ["affine32", "quadratic32_zero", "quadratic32_rbf", "affine16_dense"],
                "subset_stride": {"affine32": [32, 32], "quadratic32_zero": [32, 32],
                                  "quadratic32_rbf": [32, 32], "affine16_dense": [16, 8]},
                "rounds": 3, "iterations": args.iterations, "continuations": args.continuations,
                "tolerance": .001, "zncc_min": .65,
                "gradient_veto": False, "acceptance": "solver status 1 AND final correlation>=0.65 AND in bounds AND positive determinant over support",
                "truth": "unavailable; accepted candidates, not proven correct seeds",
                "area": "same union of originally failed 32px supports; fine supports fully contained",
                "initialization": "same Gaussian field fitted on archived accepted states; update with accepted retries"}
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    sys.path.insert(0, "/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    from live_dic import ICGNWGPU
    data = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    reference = np.asarray(Image.open(data/"Img_F.tif"))
    solvers = {}
    for key, size, stride, motion in [("affine32", 32, 32, "affine"),
                                       ("quadratic32", 32, 32, "quadratic"),
                                       ("affine16_dense", 16, 8, "affine")]:
        engine = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=size,
                          subset_stride=stride, interpolation="linear", motion_model=motion,
                          outlier_policy="disabled")
        # The external quadratic backend rejects ROI configuration. Use its
        # selective active mask with the exact full-support affine ROI instead.
        kwargs = {"roi_mask": material_mask(reference)} if motion == "affine" else {}
        engine.set_reference(reference, iterations=args.iterations, tolerance=1e-3, zncc_min=.65, **kwargs)
        solvers[key] = engine
    rows = []
    for level in args.levels:
        source = np.load(Path("results")/protocol["baseline"]/f"level_{level:03d}.npz")
        old_centres, old_params, old_accepted = source["centres"], source["parameters"], source["accepted"]
        failed_indices = np.flatnonzero(source["valid_roi"] & ~old_accepted)
        failed_centres = old_centres[failed_indices]
        target = np.asarray(Image.open(data/f"Img_wa_g{level:03d}.tif"))
        for model in protocol["models"]:
            start = time.perf_counter()
            engine = solvers["quadratic32" if model.startswith("quadratic") else model]
            size, stride = protocol["subset_stride"][model]
            valid_grid = (solvers["affine32"].roi.selected if model.startswith("quadratic")
                          else engine.roi.selected)
            assert valid_grid.shape == (engine.grid_height, engine.grid_width)
            yy, xx = np.indices(valid_grid.shape)
            centres = np.column_stack((xx.ravel()*stride+(size-1)/2, yy.ravel()*stride+(size-1)/2))
            distance = np.max(np.abs(centres[:, None, :]-failed_centres[None, :, :]), axis=2)
            region = valid_grid.ravel() & np.any(distance <= (32-size)/2+1e-8, axis=1)
            indices = np.flatnonzero(region)
            sites = centres[indices]
            components = 12 if model.startswith("quadratic") else 6
            parameters = np.zeros((len(sites), components))
            accepted = np.zeros(len(sites), bool)
            corr = np.zeros(len(sites))
            raw_status = np.zeros(len(sites))
            attempts = []
            folder = args.output/model
            folder.mkdir(exist_ok=True)
            area = np.zeros(reference.shape, bool)
            for iteration in range(3):
                fit_centres = np.vstack((old_centres[old_accepted], sites[accepted]))
                fit_params = np.vstack((old_params[old_accepted], parameters[accepted, :6]))
                field, _ = fit_seed_field(fit_centres, fit_params, np.ones(len(fit_centres), bool))
                proposal = quadratic_states(field, sites, model == "quadratic32_rbf") if components == 12 else field.evaluate(sites)
                active_local = np.flatnonzero(~accepted)
                if not len(active_local):
                    break
                packed = np.zeros((len(centres), 16 if components == 12 else 8), np.float32)
                packed[indices[active_local], :components] = proposal[active_local]
                active = np.zeros(len(centres), np.uint32)
                active[indices[active_local]] = 1
                result = engine._register_seeded_frame(target, packed, active.reshape(valid_grid.shape))
                out = np.asarray(result.parameters).reshape(-1, components)[indices[active_local]].copy()
                statuses = np.asarray(result.status).ravel()[indices[active_local]].copy()
                iterations = np.asarray(result.iterations).ravel()[indices[active_local]].copy()
                for block in range(1, args.continuations):
                    pending = np.flatnonzero((statuses == 0) & np.isfinite(out).all(axis=1))
                    if not len(pending):
                        break
                    packed[indices[active_local[pending]], :components] = out[pending]
                    active[:] = 0
                    active[indices[active_local[pending]]] = 1
                    result = engine._register_seeded_frame(target, packed, active.reshape(valid_grid.shape))
                    out[pending] = np.asarray(result.parameters).reshape(-1, components)[indices[active_local[pending]]]
                    statuses[pending] = np.asarray(result.status).ravel()[indices[active_local[pending]]]
                    iterations[pending] += np.asarray(result.iterations).ravel()[indices[active_local[pending]]]
                    np.savez_compressed(folder/f"level_{level:03d}_round_{iteration+1}_block_{block+1}.npz",
                                        local_indices=active_local[pending], output=out[pending],
                                        status=statuses[pending], accumulated_iterations=iterations[pending])
                checks = np.array([check_patch(reference, target, sites[i], p, size)
                                   for i, p in zip(active_local, out, strict=True)])
                keep = (statuses == 1) & np.isfinite(out).all(axis=1) & (checks[:, 0] >= .65) & (checks[:, 1] > 0) & (checks[:, 2] > 0)
                parameters[active_local], raw_status[active_local], corr[active_local] = out, statuses, checks[:, 0]
                accepted[active_local[keep]] = True
                np.savez_compressed(folder/f"level_{level:03d}_round_{iteration+1}.npz", centres=sites[active_local],
                                    proposal=proposal[active_local], output=out, status=statuses,
                                    iterations=iterations, checks=checks, accepted=keep)
                attempts.append({"round": iteration+1, "attempted": len(active_local),
                                 "solver_accepted": int((statuses == 1).sum()), "retained": int(keep.sum())})
                if not keep.any():
                    break
            for c in sites[accepted]:
                x, y = np.rint(c-(size-1)/2).astype(int)
                area[y:y+size, x:x+size] = True
            covered = []
            at_centre = []
            for c in failed_centres:
                x, y = np.rint(c-15.5).astype(int)
                covered.append(float(np.mean(area[y:y+32, x:x+32])))
                at_centre.append(bool(np.any(accepted & np.all(np.isclose(sites, c), axis=1))))
            row = {"level": level, "model": model, "baseline": int(old_accepted.sum()),
                   "failed_regions": len(failed_centres), "new_candidates": len(sites),
                   "accepted_candidates": int(accepted.sum()), "old_centres_recovered": sum(at_centre),
                   "regions_fully_covered": sum(f == 1 for f in covered),
                   "mean_area_coverage": float(np.mean(covered)) if covered else 1.,
                   "covered_pixels": int(area.sum()), "original_failed_pixels": len(failed_centres)*1024,
                   "wall_s": time.perf_counter()-start, "rounds": attempts,
                   "min_accepted_correlation": float(corr[accepted].min()) if accepted.any() else None}
            np.savez_compressed(folder/f"level_{level:03d}.npz", centres=sites, parameters=parameters,
                                accepted=accepted, correlation=corr, status=raw_status, size=size,
                                failed_centres=failed_centres, failed_indices=failed_indices,
                                old_centres_recovered=np.asarray(at_centre, bool), region_coverage=covered)
            rows.append(row)
            (args.output/"summary.json").write_text(json.dumps(rows, indent=2))
            print(level, model, row["accepted_candidates"], "/", len(sites), "centres", sum(at_centre),
                  "area", row["mean_area_coverage"], flush=True)


if __name__ == "__main__":
    main()
