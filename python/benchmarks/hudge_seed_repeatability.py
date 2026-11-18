"""Report independent-start agreement, solver sensitivity and unmasked remaps."""

import argparse
import json
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import corr
from PIL import Image, ImageDraw
from qualify_hudge_curve_transport import contours
from qualify_hudge_rectified_windows import material_mask, samples


def report(folder):
    baseline = np.load("results/hudge_hole_rbf_20261117/curve_transport_191/initialisation.npz")
    old = baseline["accepted"]
    centres = baseline["centres"]
    pick = np.load(folder / "prior/seeds.npz")["accepted"]
    info = json.loads(
        Path("results/hudge_hole_moments_20261117/proposals/summary.json").read_text()
    )
    ref, target = [np.asarray(Image.open(p), float) for p in info["input_sha256"]]
    mat = material_mask(ref)
    curves = contours(mat, [info["source_holes"][i] for i, j, *_ in info["matches"]])
    dist = np.stack([c.nearest(centres)[0] for c in curves])
    band = dist.min(0) <= 8
    regions = {
        "all": pick,
        "hole_8px": pick & band & (dist.argmin(0) >= 4),
        "external_8px": pick & band & (dist.argmin(0) < 4),
    }

    def read(name):
        return np.load(folder / name / "native_restart_verification.npz")

    main = read("fixed_point")

    def agreement(r):
        different = (main["parameters"] - r["parameters"])[:, [0, 3]]
        return main["accepted"] & r["accepted"] & (np.linalg.norm(different, axis=1) < 1)

    reproducible = agreement(read("fixed_prior")) & agreement(read("fixed_perturb"))
    stable = agreement(read("perturb_1")) & agreement(read("perturb_-1"))
    choices = {
        "eligible": pick,
        "previous": old,
        "dense_only": read("target_integrated")["accepted"],
        "fixed_point": main["accepted"],
        "reproducible": old | reproducible,
        "native_both_perturbations": old | stable,
    }
    metrics = {
        name: {region: int((mask & use).sum()) for region, use in regions.items()}
        for name, mask in choices.items()
    }
    out = folder / "comparison"
    out.mkdir(exist_ok=True)
    merged = baseline["parameters"].copy()
    new = reproducible & ~old
    merged[new] = main["parameters"][new]
    np.savez_compressed(
        out / "initialisation.npz",
        centres=centres,
        parameters=merged,
        accepted=old | new,
        previously_accepted=old,
        agrees_from_three_starts=new,
    )
    q = np.load(folder / "mesh/map.npz")["coordinates"]
    h, w = q.shape[:2]
    y, x = np.indices((h, w))
    g = np.stack((x, y), -1)
    m = mat[:h, :w]
    radius = np.min(
        [
            np.hypot(x - v["centre"][0], y - v["centre"][1]) / np.sqrt(v["area"] / np.pi)
            for v in info["source_holes"]
        ],
        axis=0,
    )
    crown = m & (radius >= 1) & (radius < 1.25)
    dy, dx = np.mgrid[:12, :12] - 5.5
    off = np.stack((dx, dy), -1)
    weights = np.exp(-(dx * dx + dy * dy) / 18)
    warped = {}
    for name, active in [("previous", old), ("reproducible", old | new)]:
        sums = np.zeros_like(q)
        mass = np.zeros((h, w))
        for i in np.flatnonzero(active):
            ix, iy = (centres[i] - 5.5).astype(int)
            sl = np.s_[iy : iy + 12, ix : ix + 12]
            p = merged[i]
            coords = g[sl] + p[[0, 3]] + off @ p[[1, 2, 4, 5]].reshape(2, 2).T
            sums[sl] += weights[..., None] * coords
            mass[sl] += weights
        # Add a continuous background proposal including voids; never replace
        # pixels by reference intensities or fill detected holes with background.
        coords = (sums + 0.3 * q) / (mass[..., None] + 0.3)
        after = samples(target, coords)
        det = np.linalg.det(
            np.stack((np.gradient(coords, axis=1), np.gradient(coords, axis=0)), -1)
        )
        metrics[name].update(
            correlation_material=corr(ref[:h, :w][m], after[m]),
            correlation_crown=corr(ref[:h, :w][crown], after[crown]),
            sampled_folds=int((m & (det <= 0)).sum()),
        )
        warped[name] = Image.fromarray(np.uint8(np.clip(after[:225, :440], 0, 255)))
    before = Image.fromarray(ref[:225, :440].astype(np.uint8))
    before.save(out / "image1_ref.png")
    warped["reproducible"].save(out / "image1_def_cor.png")
    before.save(
        out / "reference_corrected.gif",
        save_all=True,
        append_images=[warped["reproducible"]],
        duration=1000,
        loop=0,
    )
    im = before.convert("RGB").resize((880, 450))
    drawer = ImageDraw.Draw(im)
    for i in np.flatnonzero(pick):
        col = "cyan" if old[i] else "lime" if new[i] else "red"
        xx, yy = 2 * centres[i]
        drawer.ellipse((xx - 2, yy - 2, xx + 2, yy + 2), fill=col)
    im.save(out / "centres.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    report(parser.parse_args().folder)
