"""Solve dense material supports, export a usable non-affine field and raw pullback."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from hudge_dense_field import LocalField
from hudge_rbf_restart import fit_seed_field
from hudge_strains_data import material_mask
from PIL import Image
from qualify_hudge_adaptive_support import check_patch
from scipy.ndimage import map_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[191])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interpolation", choices=["linear", "bspline"], default="linear")
    parser.add_argument("--models", nargs="+", choices=["affine8", "affine16", "quadratic32", "quadratic16"],
                        default=["affine16", "quadratic32", "quadratic16"])
    parser.add_argument("--dense-seed-root", type=Path,
                        help="Previous dense run: level_###/affine16/seeds.npz; use accepted states only")
    parser.add_argument("--initial-contour", type=Path, help="Saved additive contour field; single level only")
    args = parser.parse_args()
    if args.initial_contour and (not args.dense_seed_root or len(args.levels) != 1):
        parser.error("--initial-contour requires --dense-seed-root and exactly one level")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, "/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    from live_dic import ICGNWGPU
    from live_dic.roi import prepare_roi
    data = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    reference = np.asarray(Image.open(data/"Img_F.tif"))
    mask = material_mask(reference)
    sizes = {"affine8": [8, 4], "affine16": [16, 8], "quadratic32": [32, 8], "quadratic16": [16, 8]}
    protocol = {"size_stride": {key: sizes[key] for key in args.models},
                "source": "complete coarse accepted states from earlier RBF/continuation experiments",
                "iterations_per_call": 30, "continuations": 4, "rounds": 3,
                "solver_interpolation": args.interpolation,
                "dense_seed_root": str(args.dense_seed_root) if args.dense_seed_root else None,
                "initial_contour": str(args.initial_contour) if args.initial_contour else None,
                "tolerance": .001, "correlation_min": .65, "gradient_veto": False,
                "field": "24-nearest normalized Gaussian blend of solved local maps; length=0.75*subset",
                "raw_remapping": "original target sampled bilinearly, no silhouette reshaping or mask painting",
                "qualification": "solver status + final correlation + bounds + positive support Jacobian; no displacement truth"}
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    for model, (size, stride) in protocol["size_stride"].items():
        quadratic = model.startswith("quadratic")
        engine = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=size, subset_stride=stride,
                          interpolation=args.interpolation, motion_model="quadratic" if quadratic else "affine", outlier_policy="disabled")
        kwargs = {} if quadratic else {"roi_mask": mask}
        engine.set_reference(reference, iterations=30, tolerance=.001, zncc_min=.65, **kwargs)
        roi = prepare_roi(mask, size, stride, policy="full")
        yy, xx = np.indices(roi.selected.shape)
        all_centres = np.column_stack((xx.ravel()*stride+(size-1)/2, yy.ravel()*stride+(size-1)/2))
        indices = np.flatnonzero(roi.selected.ravel())
        centres = all_centres[indices]
        count = 12 if quadratic else 6
        for level in args.levels:
            folder = args.output/f"level_{level:03d}"/model
            folder.mkdir(parents=True)
            target = np.asarray(Image.open(data/f"Img_wa_g{level:03d}.tif"))
            old = np.load(f"results/hudge_rbf_restart_20261115/all_levels/rbf_gradient_g0.50/level_{level:03d}.npz")
            new = np.load(f"results/hudge_adaptive_support_20261115/all_levels_continuation4/affine32/level_{level:03d}.npz")
            old_params, old_ok = old["parameters"].copy(), old["accepted"].copy()
            old_params[new["failed_indices"][new["accepted"]]] = new["parameters"][new["accepted"]]
            old_ok[new["failed_indices"][new["accepted"]]] = True
            field, _ = fit_seed_field(old["centres"], old_params, old_ok)
            if args.dense_seed_root:
                dense = np.load(args.dense_seed_root/f"level_{level:03d}"/"affine16"/"seeds.npz")
                field = LocalField(dense["centres"][dense["accepted"]],
                                   dense["parameters"][dense["accepted"]], length=12.)
                if args.initial_contour:
                    from hudge_contour_field import ContourField
                    correction = np.load(args.initial_contour)
                    field = ContourField(field, mask, material_mask(target),
                                         dense["centres"][dense["accepted"]], length=float(correction["length"]))
                    field.sites = correction["sites"]
                    field.coefficients = correction["coefficients"]
            parameters = np.zeros((len(centres), count))
            accepted = np.zeros(len(centres), bool)
            correlation = np.zeros(len(centres))
            statuses = np.zeros(len(centres))
            for round_index in range(3):
                pending = np.flatnonzero(~accepted)
                if not len(pending):
                    break
                seeds = np.zeros((len(pending), count))
                seeds[:, :6] = field.evaluate(centres[pending])
                proposal = seeds.copy()
                status = np.zeros(len(pending))
                for block in range(4):
                    active_local = np.flatnonzero((status == 0) & np.isfinite(seeds).all(axis=1))
                    if not len(active_local):
                        break
                    packed = np.zeros((len(all_centres), 16 if quadratic else 8), np.float32)
                    packed[indices[pending[active_local]], :count] = seeds[active_local]
                    active = np.zeros(len(all_centres), np.uint32)
                    active[indices[pending[active_local]]] = 1
                    result = engine._register_seeded_frame(target, packed, active.reshape(roi.selected.shape))
                    seeds[active_local] = np.asarray(result.parameters).reshape(-1, count)[indices[pending[active_local]]]
                    status[active_local] = np.asarray(result.status).ravel()[indices[pending[active_local]]]
                checks = np.array([check_patch(reference, target, c, p, size)
                                   for c, p in zip(centres[pending], seeds, strict=True)])
                keep = (status == 1) & np.isfinite(seeds).all(axis=1) & (checks[:, 0] >= .65) & (checks[:, 1] > 0) & (checks[:, 2] > 0)
                parameters[pending], statuses[pending], correlation[pending] = seeds, status, checks[:, 0]
                accepted[pending[keep]] = True
                np.savez_compressed(folder/f"round_{round_index+1}.npz", centres=centres[pending], proposal=proposal,
                                    output=seeds, status=status, checks=checks, accepted=keep)
                print(level, model, round_index+1, int(accepted.sum()), "/", len(centres), flush=True)
                if not accepted.any():
                    raise RuntimeError("no accepted sites")
                field = LocalField(centres[accepted], parameters[accepted], length=.75*size)
                if not keep.any():
                    break
            np.savez_compressed(folder/"seeds.npz", centres=centres, parameters=parameters, accepted=accepted,
                                status=statuses, correlation=correlation, size=size, stride=stride)
            y, x = np.indices(reference.shape)
            points = np.column_stack((x.ravel(), y.ravel()))
            states = field.evaluate(points)
            q = points + states[:, [0, 3]]
            remapped = map_coordinates(target.astype(float), [q[:, 1], q[:, 0]], order=1,
                                      mode="constant", cval=float(target[-1, -1]), prefilter=False).reshape(reference.shape)
            inside = ((q[:, 0] >= 0) & (q[:, 0] <= target.shape[1]-1) & (q[:, 1] >= 0) & (q[:, 1] <= target.shape[0]-1)).reshape(reference.shape)
            jacobian = states[:, [1, 2, 4, 5]].reshape(-1, 2, 2)+np.eye(2)
            determinant = np.linalg.det(jacobian).reshape(reference.shape)
            coverage = np.zeros(reference.shape, bool)
            for c in centres[accepted]:
                x0, y0 = np.rint(c-(size-1)/2).astype(int)
                coverage[y0:y0+size, x0:x0+size] = True
            np.savez_compressed(folder/"field.npz", displacement=states[:, [0, 3]].reshape(*reference.shape, 2).astype(np.float32),
                                gradient=(jacobian-np.eye(2)).reshape(*reference.shape, 2, 2).astype(np.float32),
                                in_bounds=inside, support_coverage=coverage, determinant=determinant.astype(np.float32))
            Image.fromarray(np.uint8(np.clip(np.rint(remapped), 0, 255))).save(folder/"def_cor.png")
            Image.fromarray(reference).save(folder/"ref.png")
            target_material = material_mask(target)
            mapped_material = map_coordinates(target_material.astype(float), [q[:, 1], q[:, 0]], order=0,
                                              mode="constant", cval=0).reshape(reference.shape) > .5
            np.savez_compressed(folder/"diagnostics.npz", reference_material=mask, mapped_material=mapped_material)
            image_corr = np.corrcoef(reference[mask & inside], remapped[mask & inside])[0, 1]
            row = {"level": level, "model": model, "accepted": int(accepted.sum()), "requested": len(centres),
                   "material_correlation": float(image_corr), "material_iou": float(np.sum(mask & mapped_material)/np.sum(mask | mapped_material)),
                   "nonpositive_material_jacobian": int(np.sum(mask & (determinant <= 0))),
                   "material_outside_target": int(np.sum(mask & ~inside)),
                   "material_support_coverage": float(np.mean(coverage[mask]))}
            rows.append(row)
            (args.output/"summary.json").write_text(json.dumps(rows, indent=2))
            print(row, flush=True)
        engine.close()


if __name__ == "__main__":
    main()
