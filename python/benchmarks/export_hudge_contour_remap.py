"""Reproducible observed-contour correction and raw-image remapping.

Run after qualify_hudge_dense_remap.py. This is a CPU reconstruction experiment
on top of WebGPU local solves, not an entirely GPU-resident production method.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from hudge_contour_field import ContourField
from hudge_dense_field import LocalField
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import label, map_coordinates


def hole_statistics(mask):
    labels, number = label(~mask)
    outside = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    rows = []
    for index in range(1, number+1):
        if index in outside:
            continue
        y, x = np.nonzero(labels == index)
        if len(x) < 10:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(np.column_stack((x, y)), rowvar=False, bias=True))
        rows.append({"label": index, "centre": [float(x.mean()), float(y.mean())],
                     "area": len(x), "axis_ratio": float(np.sqrt(eigenvalues[-1]/eigenvalues[0]))})
    return labels, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=float, default=16.)
    parser.add_argument("--max-sites", type=int, default=196)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA")
    ref_path, target_path = data/"Img_F.tif", data/f"Img_wa_g{args.level:03d}.tif"
    reference, target = np.asarray(Image.open(ref_path)), np.asarray(Image.open(target_path))
    mask, target_mask = material_mask(reference), material_mask(target)
    archive = np.load(args.seeds)
    ok = archive["accepted"]
    centres, parameters = archive["centres"][ok], archive["parameters"][ok]
    base = LocalField(centres, parameters, length=12.)
    field = ContourField(base, mask, target_mask, centres, length=args.length, max_sites=args.max_sites)
    y, x = np.indices(reference.shape)
    points = np.column_stack((x.ravel(), y.ravel()))
    material_points = points[mask.ravel()]
    history = []
    start = time.perf_counter()
    for iteration in range(6):
        update = field.step(contour_weight=2., ridge=.05)
        saved = field.coefficients.copy()
        used_step = 0.
        for factor in (1., .5, .25, .125, .0625):
            field.coefficients = saved+factor*update
            states = field.evaluate(material_points)
            determinant = np.linalg.det(np.eye(2)+states[:, [1, 2, 4, 5]].reshape(-1, 2, 2))
            if np.all(determinant > 0):
                used_step = factor
                break
        if not used_step:
            field.coefficients = saved
        history.append({"iteration": iteration+1, "step": used_step,
                        "boundary_distance_rms_target_px": float(np.sqrt(np.mean(field.residual()[0]**2)))})
        print(args.level, history[-1], flush=True)
    fitting_seconds = time.perf_counter()-start
    np.savez_compressed(args.output/"contour_field.npz", sites=field.sites, coefficients=field.coefficients,
                        length=field.length, anchor_centres=centres, anchor_parameters=parameters, base_length=12.)
    scores, images = [], []
    for name, candidate in (("dense", base), ("corrected", field)):
        states = candidate.evaluate(points)
        q = points+states[:, [0, 3]]
        remapped = map_coordinates(target.astype(float), q[:, ::-1].T, order=1, mode="constant",
                                  cval=float(target[-1, -1]), prefilter=False).reshape(reference.shape)
        mapped_mask = map_coordinates(target_mask.astype(float), q[:, ::-1].T, order=0,
                                      mode="constant", cval=0).reshape(reference.shape) > .5
        gradient = states[:, [1, 2, 4, 5]].reshape(*reference.shape, 2, 2)
        determinant = np.linalg.det(np.eye(2)+gradient)
        inside = ((q[:, 0] >= 0) & (q[:, 0] <= target.shape[1]-1) &
                  (q[:, 1] >= 0) & (q[:, 1] <= target.shape[0]-1)).reshape(reference.shape)
        row = {"stage": name, "material_correlation": float(np.corrcoef(reference[mask], remapped[mask])[0, 1]),
               "material_iou": float(np.sum(mask & mapped_mask)/np.sum(mask | mapped_mask)),
               "nonpositive_material_jacobian": int(np.sum(mask & (determinant <= 0))),
               "minimum_material_jacobian": float(np.min(determinant[mask])),
               "outside_target_material": int(np.sum(mask & ~inside))}
        ref_labels, ref_holes = hole_statistics(mask)
        mapped_labels, mapped_holes = hole_statistics(mapped_mask)
        hole_rows = []
        for hole in ref_holes:
            region = ref_labels == hole["label"]
            overlaps = [np.sum(region & (mapped_labels == h["label"])) for h in mapped_holes]
            if not overlaps or max(overlaps) == 0:
                hole_rows.append({"reference": hole, "matched": False})
                continue
            other = mapped_holes[int(np.argmax(overlaps))]
            other_region = mapped_labels == other["label"]
            hole_rows.append({"reference": hole, "matched": True, "corrected": other,
                              "iou": float(np.sum(region & other_region)/np.sum(region | other_region)),
                              "centre_error_px": float(np.linalg.norm(np.array(hole["centre"])-other["centre"]))})
        row["holes"] = hole_rows
        scores.append(row)
        image = Image.fromarray(np.uint8(np.clip(np.rint(remapped), 0, 255)))
        image.save(args.output/("image2_def_cor.png" if name == "corrected" else "dense_def_cor.png"))
        images.append(image)
        if name == "corrected":
            np.savez_compressed(args.output/"field.npz", displacement=states[:, [0, 3]].reshape(*reference.shape, 2).astype(np.float32),
                                gradient=gradient.astype(np.float32), determinant=determinant.astype(np.float32),
                                in_bounds=inside, reference_material=mask, remapped_material=mapped_mask)
    all_centres = archive["centres"]
    np.savez_compressed(args.output/"initialization.npz", centres=all_centres,
                        parameters=field.evaluate(all_centres).astype(np.float32), observed_anchor=ok,
                        measured_parameters=archive["parameters"], measured_status=archive["status"])
    Image.fromarray(reference).save(args.output/"image1_ref.png")
    Image.fromarray(target).save(args.output/"image0_def.png")
    frames = [Image.fromarray(reference), images[-1]]
    frames[0].save(args.output/"reference_corrected.gif", save_all=True, append_images=frames[1:], duration=900, loop=0)
    yy, xx = np.nonzero(mask)
    crop = (max(0, xx.min()-4), max(0, yy.min()-4), min(reference.shape[1], xx.max()+13), min(reference.shape[0], yy.max()+13))
    cropped = [im.crop(crop).resize((2*(crop[2]-crop[0]), 2*(crop[3]-crop[1])), Image.Resampling.NEAREST) for im in frames]
    cropped[0].save(args.output/"reference_corrected_zoom.gif", save_all=True, append_images=cropped[1:], duration=900, loop=0)
    for i, frame in enumerate(cropped, 1):
        frame.save(args.output/f"zoom_{i}.png")
    report = {"level": args.level, "anchors": int(ok.sum()), "requested": len(ok), "history": history,
              "fitting_cpu_seconds": fitting_seconds, "scores": scores, "crop_xyxy": list(map(int, crop)),
              "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ref_path, target_path, args.seeds)},
              "parameter_order": ["u", "du_dx", "du_dy", "v", "dv_dx", "dv_dy"],
              "coordinates": "native pixel centres; x right, y down; reference to original target",
              "method": f"accepted 16x16 local maps + additive Gaussian contour fit; {args.max_sites} sites; length {args.length}; weight 2; ridge .05; six steps",
              "limitation": "CPU field fit/remap; GPU local solves; no displacement truth; observed contours are fitting data, not independent validation"}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2))
    print(args.level, [{k: v for k, v in r.items() if k != "holes"} for r in scores], flush=True)


if __name__ == "__main__":
    main()
