"""Read-only, post-inference image diagnostics for HudgeStrains qualification.

No photometric diagnostic is labelled displacement accuracy. Image-derived extent
ratios describe visible envelopes, never nominal strain or supplied ground truth.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def photometric_check(reference, target, centres, parameters, accepted, subset):
    """Recompute final-state correlation with bilinear native-pixel sampling."""
    offset = np.arange(subset) - (subset-1)/2
    dx, dy = np.meshgrid(offset, offset)
    correlations = np.full(len(centres), np.nan)
    in_bounds = np.zeros(len(centres), bool)
    pullback = np.full(reference.shape, np.nan, np.float32)
    for i in np.flatnonzero(accepted):
        x, y = centres[i]
        u, ux, uy, v, vx, vy = parameters[i]
        qx = x + u + (1+ux)*dx + uy*dy
        qy = y + v + vx*dx + (1+vy)*dy
        if not (np.isfinite(qx).all() and np.isfinite(qy).all()
                and qx.min() >= 0 and qy.min() >= 0
                and qx.max() <= target.shape[1]-1 and qy.max() <= target.shape[0]-1):
            continue
        in_bounds[i] = True
        a = map_coordinates(reference, [y+dy, x+dx], order=1, mode="constant", cval=np.nan)
        b = map_coordinates(target, [qy, qx], order=1, mode="constant", cval=np.nan)
        ac, bc = a-a.mean(), b-b.mean()
        norm = np.linalg.norm(ac)*np.linalg.norm(bc)
        if norm > 1e-12:
            correlations[i] = np.sum(ac*bc)/norm
        left, top = round(x-offset[-1]), round(y-offset[-1])
        pullback[top:top+subset, left:left+subset] = b
    return correlations, in_bounds, pullback


def records(root):
    for family in ("FEA", "ClosedFormExpression"):
        for marker in sorted((root/family).glob("*/completed.json")):
            record = json.loads(marker.read_text())
            archive = next((marker.parent/"qualification/evaluation").glob("*.npz"))
            yield "learned", family, record["level"], 64, archive, record
        for subset in (64, 32):
            for marker in sorted((root/f"silhouette_{subset}"/family).glob("level_*.json")):
                record = json.loads(marker.read_text())
                yield f"silhouette_{subset}", family, record["level"], subset, marker.with_suffix(".npz"), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/hudge_strains_20261114"))
    args = parser.parse_args()
    data = Path("/home/jeff/CNRS/DIC/HudgeStrains")
    diagnostics = args.root/"diagnostics"
    diagnostics.mkdir(exist_ok=True)
    references = {family: np.asarray(Image.open(data/family/"Img_F.tif"), dtype=np.float32)/255
                  for family in ("FEA", "ClosedFormExpression")}
    rows = []
    for method, family, level, subset, archive, info in records(args.root):
        output = diagnostics/f"{method}_{family}_{level:03d}.npz"
        with np.load(archive) as a:
            centres, parameters, accepted, valid = (a[k] for k in ["centres", "parameters", "accepted", "valid_roi"])
            accepted = accepted.astype(bool)
            if output.exists():
                with np.load(output) as check:
                    corr, bounds = check["correlation"], check["in_bounds"]
            else:
                name = f"Img_wa_g{level:03d}.tif" if family == "FEA" else f"Img_G{level:03d}.tif"
                if level == 0:
                    name = "Img_F.tif"
                target = np.asarray(Image.open(data/family/name), dtype=np.float32)/255
                corr, bounds, _ = photometric_check(references[family], target, centres, parameters, accepted, subset)
                np.savez_compressed(output, correlation=corr, in_bounds=bounds)
            finite = accepted & np.isfinite(corr)
            row = {"method": method, "family": family, "level": level, "subset": subset,
                   "valid_roi": int(valid.sum()), "accepted": int(accepted.sum()),
                   "accepted_fraction": float(accepted.sum()/valid.sum()) if valid.any() else None,
                   "confirmed_seeds": int(a["confirmed_seeds"].sum()),
                   "accepted_nonfinite_parameters": int((accepted & ~np.isfinite(parameters).all(1)).sum()),
                   "accepted_outside_target": int((accepted & ~bounds).sum()),
                   "final_zncc_median": float(np.median(corr[finite])) if finite.any() else None,
                   "final_zncc_min": float(np.min(corr[finite])) if finite.any() else None,
                   "final_zncc_below_065": int((finite & (corr < .65)).sum()),
                   "max_displacement_norm_px": float(np.linalg.norm(parameters[accepted][:, [0, 3]], axis=1).max()) if accepted.any() else None,
                   "truth_accuracy": "unavailable: displacement association unresolved",
                   "identity_displacement_rmse_px": float(np.sqrt(np.mean(parameters[accepted][:, [0, 3]]**2))) if level == 0 and accepted.any() else None,
                   "archive": str(archive)}
            if method == "learned":
                row["wall_s"] = sum(info["stage_wall_seconds"])
            else:
                row["wall_s"] = info["target_wall_s"]
            rows.append(row)
    (args.root/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    if rows:
        with (args.root/"summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Post-inference diagnostics:", len(rows), "pairs/methods")


if __name__ == "__main__":
    main()
