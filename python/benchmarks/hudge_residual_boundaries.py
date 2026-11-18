"""Correct the remaining boundary-normal error after residual texture matching.

Nearest target curve points constrain the normal. The tangential registration
already estimated from the surrounding texture is retained as an initial state.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, label
from scipy.spatial import cKDTree

from experiment_hudge_hole_rbf import corr
from hudge_dense_residual_flow import limit_folds, write_seeds
from qualify_hudge_curve_transport import contours
from qualify_hudge_rectified_windows import material_mask, samples


def project(root, width, out):
    info = json.loads(
        Path("results/hudge_hole_moments_20261117/proposals/summary.json").read_text()
    )
    ref, target = [np.asarray(Image.open(p), float) for p in info["input_sha256"]]
    m0, m1 = material_mask(ref), material_mask(target)
    q = np.load(root / "pass3/target_limited/map.npz")["coordinates"]
    cs = contours(m0, [info["source_holes"][i] for i, j, *_ in info["matches"]])
    ct = contours(m1, [info["target_candidates"][j] for i, j, *_ in info["matches"]])
    h, w = q.shape[:2]
    y, x = np.indices((h, w))
    grid = np.stack((x, y), -1)
    out = out or root / "boundary_normal"
    out.mkdir(exist_ok=True)
    for _ in range(2):
        best = np.full((h, w), np.inf)
        offset = np.zeros_like(q)
        for a, b in zip(cs, ct, strict=True):
            line = a.at(a.grid)
            predicted = np.stack([samples(q[..., k], line) for k in range(2)], axis=-1)
            _, s = b.nearest(predicted)
            normal = b.at(s) - predicted
            normal = gaussian_filter1d(normal, 4, axis=0, mode="wrap" if a.closed else "nearest")
            dist, idx = cKDTree(line).query(grid)
            use = dist < best
            offset[use] = normal[idx[use]]
            best[use] = dist[use]
        q = limit_folds(q, q + offset * np.exp(-0.5 * (best / width) ** 2)[..., None], m0[:h, :w])
    np.savez_compressed(out / "map.npz", coordinates=q)
    write_seeds(
        out,
        q,
        grid,
        m0[:h, :w],
        np.load("results/hudge_hole_rbf_20261117/curve_transport_191/initialisation.npz"),
        np.load(root / "prior/seeds.npz")["accepted"],
    )
    warp = samples(target, q)
    after = Image.fromarray(np.uint8(np.clip(warp[:225, :440], 0, 255)))
    after.save(out / "corrected.png")
    before = Image.fromarray(ref[:225, :440].astype(np.uint8))
    before.save(
        out / "reference_corrected.gif", save_all=True, append_images=[after], duration=950, loop=0
    )
    r = np.min(
        [
            np.hypot(x - v["centre"][0], y - v["centre"][1]) / np.sqrt(v["area"] / np.pi)
            for v in info["source_holes"]
        ],
        axis=0,
    )
    m = m0[:h, :w]
    crown = m & (r >= 1) & (r < 1.25)
    void_src = np.isin(
        label(~m0)[0][:h, :w],
        [info["source_holes"][i]["component"] for i, j, *_ in info["matches"]],
    )
    void_dst = np.isin(
        label(~m1)[0], [info["target_candidates"][j]["component"] for i, j, *_ in info["matches"]]
    )
    pred = samples(void_dst.astype(float), q) > 0.5
    metrics = dict(
        material=corr(ref[:h, :w][m], warp[m]),
        crown=corr(ref[:h, :w][crown], warp[crown]),
        overlap=float((pred & void_src).sum() / (pred | void_src).sum()),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--width", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("width must be positive")
    project(args.folder, args.width, args.output)
