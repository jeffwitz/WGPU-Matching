"""Endpoint-only monotone alignment of the material strips along plate 191 sidewalls.

Fit only alternate normal-depth columns and inspect the intervening pixels.
The straightening below measures depth horizontally from each observed side.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask, profiles
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter1d, map_coordinates


def sample(image, x, y):
    return map_coordinates(image, [y, x], order=1, mode="nearest", prefilter=False)


def normalize(a):
    a = a - a.mean(axis=-1, keepdims=True)
    return np.asarray(a / np.maximum(1e-6, np.linalg.norm(a, axis=-1, keepdims=True)), dtype="f")


def corr(a, b):
    return float(normalize(a) @ normalize(b))


def fit_side(images, masks, prior, side):
    a, b = images
    sign = [1, -1][side]
    edges = [profiles(binary_fill_holes(m))[side][1] for m in masks]
    t0, t1 = [np.flatnonzero(np.isfinite(e)) for e in edges]
    edge0, edge1 = [e[t] for e, t in zip(edges, [t0, t1])]
    # Every displacement index on the deformed side is considered.
    nodes = np.arange(t0[0] + 5, t0[-1] - 4, 2)
    dst = t1
    scale = (t1[-1] - t1[0]) / (t0[-1] - t0[0])
    dt, depth = np.meshgrid(np.arange(-5, 6, 2), np.arange(0.5, 12, 2), indexing="ij")
    dt, depth = dt.ravel(), depth.ravel()
    sy = nodes[:, None] + dt
    sx = np.interp(sy, t0, edge0) + sign * depth
    desc = normalize(sample(a, sx, sy))
    score = np.full((len(nodes), len(dst)), -np.inf)
    idx = np.zeros(score.shape, int)
    options = []
    for sy_scale in scale * np.array([0.45, 0.7, 1, 1.4, 1.9]):
        for depth_scale in [0.25, 0.5, 0.8, 1.2, 1.65]:
            for shear in [-3, -1.5, 0, 1.5, 3]:
                dy = dst[:, None] + sy_scale * dt + shear * depth
                dx = np.interp(dy, t1, edge1) + sign * depth_scale * depth
                target = normalize(sample(b, dx, dy))
                sim = desc @ target.T
                sim[:, (dy.min(1) < t1[0]) | (dy.max(1) > t1[-1])] = -np.inf
                better = sim > score
                idx[better] = len(options)
                score = np.maximum(score, sim)
                options.append((depth_scale, shear))
    # Max-sum path, preserving tangential order and permitting nonuniform strain.
    guess = np.interp(nodes, [t0[0], t0[-1]], [t1[0], t1[-1]])
    value = score[0] - 0.5 * ((dst - guess[0]) / 35) ** 2
    parent = np.zeros(score.shape, int)
    steps = np.arange(1, int(np.ceil(5 * scale)) + 1)
    for i in range(1, len(nodes)):
        cost = np.full((len(steps), len(dst)), -np.inf)
        for k, j in enumerate(steps):
            cost[k, j:] = value[:-j] - 0.035 * ((j / (2 * scale)) - 1) ** 2
        best = cost.argmax(axis=0)
        parent[i] = np.arange(len(dst)) - steps[best]
        value = score[i] + cost.max(0)
    path = [int(np.argmax(value - 0.5 * ((dst - guess[-1]) / 35) ** 2))]
    for i in range(len(nodes) - 1, 0, -1):
        path.append(parent[i, path[-1]])
    path = np.array(path[::-1])
    par = np.array(options)[idx[np.arange(len(nodes)), path]]
    grid_y, grid_x = np.indices(prior.shape[:2])
    edge = np.interp(grid_y, t0, edge0)
    d = (grid_x - edge) * sign
    inside = masks[0][: prior.shape[0], : prior.shape[1]]
    band = (d >= 0) & (d < 8) & inside & (grid_y >= nodes[0]) & (grid_y <= nodes[-1])
    # Interpolate the estimated material coordinate and across-strip deformation.
    mat_t = np.interp(grid_y, nodes, dst[path])
    thick, shear = [np.interp(grid_y, nodes, gaussian_filter1d(column, 1.4)) for column in par.T]
    ty = mat_t + shear * d
    tx = np.interp(ty, t1, edge1) + sign * thick * d
    proposed = np.stack((tx, ty), -1)
    # Separate reference depths omitted in all descriptors.
    held = band & (np.floor(d).astype(int) % 2 == 1)
    qc = dict()
    for key, p in [("prior", prior), ("ribbon", proposed)]:
        warped = sample(b, p[..., 0], p[..., 1])
        qc[key] = corr(a[: d.shape[0], : d.shape[1]][held], warped[held])
    qc["held_pixels"] = int(held.sum())
    return qc, proposed, band, held


def main(root, folder, out):
    a, b = [
        np.asarray(Image.open(folder / name), dtype="f")
        for name in ["Img_F.tif", "Img_wa_g191.tif"]
    ]
    prior = np.load(root / "boundary_narrow/map.npz")["coordinates"]
    masks = [material_mask(im) for im in (a, b)]
    rows = {}
    tiles = []
    seed = np.load(root / "comparison/narrow_seeds.npz")
    xy = seed["centres"]
    previous = seed["accepted"]
    parameters = seed["parameters"].copy()
    tested = np.zeros(len(xy), bool)
    eligible = np.load(root / "prior/seeds.npz")["accepted"]
    dy, dx = np.mgrid[:12, :12] - 5.5
    yy, xx = np.indices(prior.shape[:2])
    origin = np.stack((xx, yy), axis=-1)
    for side, name in enumerate(["left", "right"]):
        row, proposed, band, held = fit_side((a, b), masks, prior, side)
        rows[name] = row
        for i in np.flatnonzero(eligible):
            x, y = xy[i]
            iy, ix = int(y), int(x)
            if not band[iy, ix]:
                continue
            sl = np.s_[int(y - 5.5) : int(y - 5.5) + 12, int(x - 5.5) : int(x - 5.5) + 12]
            use = masks[0][sl]
            design = np.c_[np.ones(use.sum()), dx[use], dy[use]]
            parameters[i] = np.linalg.lstsq(design, (proposed[sl] - origin[sl])[use], rcond=None)[
                0
            ].T.ravel()
            tested[i] = True
        ys, xs = np.nonzero(band)
        sl = np.s_[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        tiles.append(
            [
                im[sl]
                for im in [
                    a,
                    sample(b, prior[..., 0], prior[..., 1]),
                    sample(b, proposed[..., 0], proposed[..., 1]),
                ]
            ]
        )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "proposals.npz",
        parameters=parameters,
        accepted=tested | previous,
        centres=xy,
        tested=tested,
    )
    (out / "ribbon.json").write_text(json.dumps(rows, indent=2))
    print(rows)
    from PIL import ImageDraw

    im = Image.new("RGB", (480, 488), "white")
    draw = ImageDraw.Draw(im)
    for k, (name, col) in enumerate(zip(["Gauche", "Droite"], tiles)):
        draw.text((15 + k * 240, 7), name, fill="black")
        for j, (label, v) in enumerate(zip(["Ref.", "Champ", "Ruban"], col)):
            tile = Image.fromarray(np.uint8(np.clip(v, 0, 255))).resize(
                (48, 448), Image.Resampling.NEAREST
            )
            im.paste(tile, (15 + k * 240 + j * 73, 34))
            draw.text((15 + k * 240 + j * 73, 20), label, fill="black")
    im.save(out / "flancs.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/hudge_dense_flow_20261118"))
    p.add_argument("--folder", type=Path, default=Path("/home/jeff/CNRS/DIC/HudgeStrains/FEA"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.folder, args.output)
