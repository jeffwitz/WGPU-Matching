"""Save a fixed-scale plate comparison for every hole, in reference coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from experiment_hudge_hole_rbf import corr
from qualify_hudge_rectified_windows import material_mask, samples


def compare(root):
    info = json.loads(
        Path("results/hudge_hole_moments_20261117/proposals/summary.json").read_text()
    )
    ref, target = [np.asarray(Image.open(p), float) for p in info["input_sha256"]]
    paths = [
        root / "prior",
        root / "target_limited",
        root / "pass2/target_limited",
        root / "pass3/target_limited",
        root / "boundary_normal",
        root / "boundary_narrow",
    ]
    q0 = np.load(paths[0] / "map.npz")["coordinates"]
    h, w = q0.shape[:2]
    y, x = np.indices((h, w))
    mask = material_mask(ref)[:h, :w]
    tlab = label(~material_mask(target))[0]
    slab = label(~material_mask(ref))[0][:h, :w]
    rows = []
    for path in paths:
        q = np.load(path / "map.npz")["coordinates"]
        warp = samples(target, q)
        holes = []
        for i, j, *_ in info["matches"]:
            hole = info["source_holes"][i]
            centre = hole["centre"]
            rho = np.hypot(x - centre[0], y - centre[1]) / np.sqrt(hole["area"] / np.pi)
            sel = mask & (rho >= 1) & (rho < 1.25)
            src = slab == hole["component"]
            dst = (
                samples((tlab == info["target_candidates"][j]["component"]).astype(float), q) > 0.5
            )
            holes.append(
                dict(
                    x=centre[0],
                    y=centre[1],
                    pixels=int(sel.sum()),
                    correlation=corr(ref[:h, :w][sel], warp[sel]),
                    iou=float((src & dst).sum() / (src | dst).sum()),
                )
            )
        rows.append(dict(method=str(path.relative_to(root)), holes=holes))
    (root / "comparison/holes.json").write_text(json.dumps(rows, indent=2))
    frames = []
    for title, path in [
        ("Reference", root / "reference.png"),
        ("Initialisation precedente", paths[0] / "corrected.png"),
        ("Correction du mouchetis et de la normale au bord", paths[-1] / "corrected.png"),
    ]:
        im = Image.open(path)
        out = Image.new("RGB", (660, 490), "white")
        draw = ImageDraw.Draw(out)
        draw.text((10, 6), title, fill="black")
        for k, (i, j, *_) in enumerate(info["matches"]):
            v = info["source_holes"][i]
            cx, cy = np.rint(v["centre"]).astype(int)
            # Same field of view for each hole through all frames.
            r = int(np.ceil(np.sqrt(v["area"] / np.pi))) + 13
            crop = im.crop((cx - r, cy - r, cx + r, cy + r)).resize(
                (200, 200), Image.Resampling.NEAREST
            )
            xx, yy = 5 + (k % 3) * 220, 30 + (k // 3) * 230
            out.paste(crop, (xx, yy))
            draw.text((xx, yy + 203), f"Centre ({cx}, {cy}) px", fill="black")
        frames.append(out)
    frames[0].save(
        root / "comparison/holes.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1150,
        loop=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    compare(parser.parse_args().folder)
