"""Paired iterative affine restarts from archived image-only plate solutions."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hudge_affine_restart import affine_states, compose, fit_global, rectify, uncompose
from hudge_strains_data import material_mask
from PIL import Image
from summarize_hudge_strains import photometric_check
from wgpu_matching.seed_initialization import initialize
from wgpu_matching.seed_propagation import propagate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[25,50,100,150,191])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/hudge_affine_restart_20261115"))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,"/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    from live_dic import ICGNWGPU
    data = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    baseline = Path("results/hudge_strains_20261114/silhouette_32/FEA")
    reference = np.asarray(Image.open(data/"Img_F.tif"))
    solver = ICGNWGPU(device_type="gpu",prefer_subgroups=False,subset_size=32,subset_stride=32,
                      interpolation="linear",motion_model="affine",outlier_policy="disabled")
    solver.set_reference(reference,iterations=30,tolerance=1e-3,zncc_min=.65,roi_mask=material_mask(reference))
    shape = solver.roi.selected.shape
    rows = []
    for level in args.levels:
        target = np.asarray(Image.open(data/f"Img_wa_g{level:03d}.tif"))
        with np.load(baseline/f"level_{level:03d}.npz") as z:
            centres, valid, initial, original_accepted = [z[k].copy() for k in ["centres","valid_roi","parameters","accepted"]]
        assert np.array_equal(valid,solver.roi.selected.ravel())
        count = len(centres)
        calls = []

        def solve_image(image, indices, values, count=count, calls=calls):
            packed = np.zeros((count,8),np.float32)
            packed[indices,:6] = values
            packed[indices,6] = 1
            active = np.zeros(count,np.uint32)
            active[indices] = 1
            result = solver._register_seeded_frame(image,packed,active.reshape(shape))
            calls.append(len(indices))
            return np.asarray(result.parameters).reshape(-1,6)[indices].copy(), np.asarray(result.status).ravel()[indices]>.5

        def native_solve(indices, values, target=target, solve_image=solve_image):
            return solve_image(target,indices,values)

        for method in ["native_retry","native_two_pass","rectified_native"]:
            started = time.perf_counter()
            parameters, accepted = initial.copy(), original_accepted.copy()
            rounds = []
            calls.clear()
            stop = "round limit"
            for iteration in range(args.rounds):
                before = int(accepted.sum())
                try:
                    matrix, translation, fit = fit_global(centres,parameters,accepted)
                except ValueError as error:
                    stop = str(error)
                    break
                proposal = affine_states(centres,matrix,translation)
                rectified = rectify(target,matrix,translation) if method=="rectified_native" else target

                def restart_solve(indices, values, method=method, matrix=matrix, translation=translation,
                                  rectified=rectified, centres=centres, solve_image=solve_image, native_solve=native_solve):
                    if method=="native_retry":
                        return native_solve(indices,values)
                    residual = uncompose(centres[indices],values,matrix,translation) if method=="rectified_native" else values
                    first, ok = solve_image(rectified,indices,residual)
                    first = compose(centres[indices],first,matrix,translation) if method=="rectified_native" else first
                    finite = np.isfinite(first).all(1)
                    result, status = values.copy(), np.zeros(len(indices),bool)
                    selected = np.flatnonzero(ok & finite)
                    if len(selected):
                        result[selected], status[selected] = native_solve(indices[selected],first[selected])
                    return result,status

                restart = initialize(centres,shape,valid,[("fitted_affine",proposal)],restart_solve,
                                     sparse_stride=2,tau_u=8.,tau_g=.25,minimum_anchors=3,batch_confirmation=True)
                new = restart["accepted"] & ~accepted
                parameters[new] = restart["parameters"][new]
                accepted |= new
                merged = propagate(centres,shape,valid,parameters,accepted,native_solve,tau_u=8.,tau_g=.25)
                parameters, accepted = merged["parameters"], merged["accepted"]
                assert np.array_equal(parameters[original_accepted],initial[original_accepted])
                rounds.append({"iteration":iteration+1,"before":before,"restart_added":int(new.sum()),
                               "after":int(accepted.sum()),"matrix":matrix.tolist(),"translation":translation.tolist(),
                               "restart_hypotheses":restart["hypothesis_acceptances"],**fit})
                if int(accepted.sum())==before:
                    stop = "no added window"
                    break
            elapsed = time.perf_counter()-started
            corr, bounds, _ = photometric_check(reference/255.,target/255.,centres,parameters,accepted,32)
            row = {"level":level,"method":method,"baseline":int(original_accepted.sum()),"final":int(accepted.sum()),
                   "valid":int(valid.sum()),"rounds":rounds,"stop":stop,"restart_wall_s":elapsed,
                   "solver_calls":len(calls),"attempted_windows":sum(calls),
                   "final_zncc_min":float(np.nanmin(corr[accepted])) if accepted.any() else None,
                   "accepted_below_065":int((accepted & (corr<.65)).sum()),
                   "accepted_out_of_bounds":int((accepted & ~bounds).sum()),"truth_accuracy":"unavailable"}
            folder=args.output/method;folder.mkdir(exist_ok=True)
            (folder/f"level_{level:03d}.json").write_text(json.dumps(row,indent=2)+"\n")
            np.savez_compressed(folder/f"level_{level:03d}.npz",centres=centres,valid_roi=valid,
                                parameters=parameters,accepted=accepted,baseline_accepted=original_accepted,correlation=corr)
            rows.append(row)
            (args.output/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
            print(level,method,row["baseline"],"->",row["final"],"/",row["valid"],flush=True)


if __name__=="__main__":
    main()
