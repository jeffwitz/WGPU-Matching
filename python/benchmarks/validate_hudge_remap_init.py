"""Reset-and-solve validation at 24-pixel windows, offset from 16-pixel anchors."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hudge_contour_field import ContourField
from hudge_dense_field import LocalField
from hudge_rbf_restart import fit_seed_field
from hudge_strains_data import material_mask
from PIL import Image
from qualify_hudge_adaptive_support import check_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=191)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, "/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src")
    from live_dic import ICGNWGPU
    from live_dic.roi import prepare_roi

    data = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    reference = np.asarray(Image.open(data/"Img_F.tif"))
    target = np.asarray(Image.open(data/f"Img_wa_g{args.level:03d}.tif"))
    mask = material_mask(reference)
    saved = np.load(args.field)
    dense = LocalField(saved["anchor_centres"], saved["anchor_parameters"], length=float(saved["base_length"]))
    corrected = ContourField(dense, mask, material_mask(target), saved["anchor_centres"], length=float(saved["length"]))
    corrected.sites, corrected.coefficients = saved["sites"], saved["coefficients"]
    old = np.load(f"results/hudge_rbf_restart_20261115/all_levels/rbf_gradient_g0.50/level_{args.level:03d}.npz")
    new = np.load(f"results/hudge_adaptive_support_20261115/all_levels_continuation4/affine32/level_{args.level:03d}.npz")
    p, accepted = old["parameters"].copy(), old["accepted"].copy()
    indices = new["failed_indices"][new["accepted"]]
    p[indices] = new["parameters"][new["accepted"]]
    accepted[indices] = True
    coarse, _ = fit_seed_field(old["centres"], p, accepted)
    size, stride = 24, 8
    roi = prepare_roi(mask, size, stride, policy="full")
    y, x = np.indices(roi.selected.shape)
    all_centres = np.column_stack((stride*x.ravel()+(size-1)/2, stride*y.ravel()+(size-1)/2))
    selected = np.flatnonzero(roi.selected.ravel())
    centres = all_centres[selected]
    rows = []
    for name, field in (("coarse36", coarse), ("dense16", dense), ("dense16_contours", corrected)):
        engine = ICGNWGPU(device_type="gpu", prefer_subgroups=False, subset_size=size, subset_stride=stride,
                          interpolation="bspline", motion_model="affine", outlier_policy="disabled")
        engine.set_reference(reference, iterations=30, tolerance=.001, zncc_min=.65, roi_mask=mask)
        initial = field.evaluate(centres)
        seeds, status = initial.copy(), np.zeros(len(centres))
        counts = []
        start = time.perf_counter()
        for block in range(4):
            active_local = np.flatnonzero((status == 0) & np.isfinite(seeds).all(axis=1))
            if not len(active_local):
                break
            packed = np.zeros((len(all_centres), 8), np.float32)
            packed[selected[active_local], :6] = seeds[active_local]
            active = np.zeros(len(all_centres), np.uint32)
            active[selected[active_local]] = 1
            result = engine._register_seeded_frame(target, packed, active.reshape(roi.selected.shape))
            seeds[active_local] = np.asarray(result.parameters).reshape(-1, 6)[selected[active_local]]
            status[active_local] = np.asarray(result.status).ravel()[selected[active_local]]
            counts.append(int(np.sum(status == 1)))
        seconds = time.perf_counter()-start
        checks = np.array([check_patch(reference, target, c, p, size) for c, p in zip(centres, seeds, strict=True)])
        keep = (status == 1) & np.isfinite(seeds).all(axis=1) & (checks[:, 0] >= .65) & (checks[:, 1] > 0) & (checks[:, 2] > 0)
        np.savez_compressed(args.output/f"{name}.npz", centres=centres, initial=initial, parameters=seeds,
                            status=status, checks=checks, accepted=keep)
        row = {"method": name, "requested": len(centres), "accepted": int(keep.sum()),
               "status_per_block": counts, "solver_seconds": seconds,
               "median_final_correlation_accepted": float(np.median(checks[keep, 0]))}
        rows.append(row)
        print(row, flush=True)
        engine.close()
    (args.output/"summary.json").write_text(json.dumps({"level": args.level, "size": size, "stride": stride,
        "centre_offset_from_anchor_grid_px": [4, 4], "iterations": 30, "continuations": 4,
        "tolerance": .001, "zncc_min": .65, "interpolation": "bspline", "results": rows,
        "limitation": "Offset centres overlap anchor supports; not an independent image test. No displacement truth."}, indent=2))


if __name__ == "__main__":
    main()
