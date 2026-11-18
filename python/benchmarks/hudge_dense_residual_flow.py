"""Compare dense residual matching in reference and target coordinates on plate 191.

The OpenCV worker runs separately so qualification can use its normal Python
environment. The global initializer is the archived four-step RBF field.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def worker(folder):
    import cv2

    data = np.load(folder / "pair.npz")
    cv2.setNumThreads(4)
    opt = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    opt.setFinestScale(0)
    a, b = (np.ascontiguousarray(data[k], np.uint8) for k in ("a", "b"))
    np.savez_compressed(
        folder / "flow.npz", forward=opt.calc(a, b, None), reverse=opt.calc(b, a, None)
    )


def limit_folds(base, proposed, material):
    """Backtrack residual amplitude locally until both mesh triangles stay oriented."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def orientations(q):
        a, b, c, d = q[:-1, :-1], q[:-1, 1:], q[1:, :-1], q[1:, 1:]
        # All four corners of the piecewise bilinear map in every cell.
        return np.stack(
            (cross(b - a, c - a), cross(b - a, d - b), cross(d - c, c - a), cross(d - c, d - b))
        )

    relevant = material[:-1, :-1] | material[:-1, 1:] | material[1:, :-1] | material[1:, 1:]
    if np.any(orientations(base)[:, relevant] <= 0):
        raise ValueError("The starting deformation already folds on the material mesh")
    alpha = np.ones(base.shape[:2])
    for _ in range(160):
        trial = base + (proposed - base) * alpha[..., None]
        bad = relevant & (orientations(trial).min(0) < 0.005)
        central_det = np.linalg.det(
            np.stack((np.gradient(trial, axis=1), np.gradient(trial, axis=0)), -1)
        )
        bad_pixel = material & (central_det < 0.005)
        if not (bad.any() or bad_pixel.any()):
            return trial
        bad_pixel[:-1, :-1] |= bad
        bad_pixel[1:, 1:] |= bad
        envelope = gaussian_filter(maximum_filter(bad_pixel.astype(float), size=7), 2)
        alpha *= 1 - 0.55 * envelope
    raise RuntimeError("Residual backtracking did not converge")


def write_seeds(path, coords, grid, m, template, eligible):
    centres = template["centres"]
    ids = np.flatnonzero(eligible)
    # Local least-squares affine on the material portion of each original
    # window, without differentiating noisy individual flow pixels.
    states = template["parameters"].copy()
    dy, dx = np.mgrid[:12, :12] - 5.5
    for i in ids:
        c = centres[i]
        x, y = (c - 5.5).astype(int)
        support = m[y : y + 12, x : x + 12]
        A = np.c_[np.ones(support.sum()), dx[support], dy[support]]
        local = (coords - grid)[y : y + 12, x : x + 12][support]
        coef = np.linalg.lstsq(A, local, rcond=None)[0]
        if not template["accepted"][i]:
            states[i] = coef.T.ravel()
    np.savez_compressed(path / "seeds.npz", centres=centres, accepted=eligible, parameters=states)


def experiment(folder, start_map=None):
    from experiment_hudge_hole_rbf import corr
    from PIL import Image
    from qualify_hudge_rectified_windows import ROOT, material_mask, samples
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    info = json.loads(
        Path("results/hudge_hole_moments_20261117/proposals/summary.json").read_text()
    )
    ref, target = [np.asarray(Image.open(p), float) for p in info["input_sha256"]]
    prior = np.load(ROOT / "composition_v2/iteration_04.npz")
    p = prior["points"]
    h, w = p[:, 1].max() + 1, p[:, 0].max() + 1
    grid = p.reshape(h, w, 2).astype(float)
    q = (p + prior["parameters"][:, [0, 3]]).reshape(h, w, 2)
    J = np.eye(2) + prior["parameters"][:, [1, 2, 4, 5]].reshape(h, w, 2, 2)
    if start_map is not None:
        q = np.load(start_map)["coordinates"]
        if q.shape != grid.shape or not np.isfinite(q).all():
            raise ValueError("Incompatible starting map")
        J = np.stack((np.gradient(q, axis=1), np.gradient(q, axis=0)), -1)
    reference = ref[:h, :w]
    # Keep all the observed material, holes and external background in the warp.
    m = material_mask(ref)[:h, :w]
    radius = np.min(
        [
            np.linalg.norm(grid - v["centre"], axis=-1) / np.sqrt(v["area"] / np.pi)
            for v in info["source_holes"]
        ],
        axis=0,
    )
    crown = m & (radius >= 1) & (radius < 1.25)
    labels0 = material_mask(target)
    from scipy.ndimage import label

    lab, _ = label(~labels0)
    void = np.isin(lab, [info["target_candidates"][j]["component"] for i, j, *_ in info["matches"]])
    lab, _ = label(~material_mask(ref))
    holes = np.isin(lab[:h, :w], [v["component"] for v in info["source_holes"]])

    def vec(image, coords):
        return np.stack(
            [samples(image[..., k].astype(float), coords) for k in range(image.shape[-1])], -1
        )

    def dis(name, a, b):
        path = folder / name
        path.mkdir(exist_ok=True)
        np.savez_compressed(
            path / "pair.npz", a=np.uint8(np.clip(a, 0, 255)), b=np.uint8(np.clip(b, 0, 255))
        )
        subprocess.run(["/usr/bin/python3", __file__, str(path), "--worker"], check=True)
        z = np.load(path / "flow.npz")
        return z["forward"], z["reverse"]

    # Invert the initializer in target coordinates by nearest sampling followed
    # by Newton steps. Pixels with no valid preimage receive the observed background.
    ty, tx = np.mgrid[: target.shape[0], : min(480, target.shape[1])]
    tg = np.stack((tx, ty), -1)
    nearest = cKDTree(q.reshape(-1, 2)).query(tg)[1]
    inv = grid.reshape(-1, 2)[nearest].copy()
    for _ in range(6):
        err = vec(q, inv) - tg
        jac = vec(J.reshape(h, w, 4), inv).reshape(*inv.shape[:2], 2, 2)
        valid = np.linalg.det(jac) > 0.02
        delta = np.zeros_like(inv)
        delta[valid] = np.linalg.solve(jac[valid], err[valid, :, None])[:, :, 0]
        inv = np.clip(inv - np.clip(delta, -3, 3), [0, 0], [w - 1, h - 1])
    valid = np.linalg.norm(vec(q, inv) - tg, axis=-1) < 0.25
    rendered = samples(reference, inv)
    rendered[~valid] = ref[-1, -1]
    tf, tr = dis("target_frame", rendered, target[:, : tg.shape[1]])
    rf, rr = dis("reference_frame", reference, samples(target, q))
    proposals = {
        "prior": q,
        "reference_flow": vec(q, np.clip(grid + rf, [0, 0], [w - 1, h - 1])),
        "target_flow": q + vec(tf, q),
    }
    # Integrate a smoothed stationary residual velocity on the target pixel grid.
    # Small steps limit spatial distortion; inspect the sampled final determinant.
    velocity = gaussian_filter(tf.astype(float), (2, 2, 0))
    locations = q.copy()
    for _ in range(32):
        locations += vec(velocity, locations) / 32
    proposals["target_integrated"] = locations
    proposals["target_limited"] = limit_folds(q, proposals["target_flow"], m)
    results = {}
    template = np.load(ROOT / "curve_transport_191/initialisation.npz")
    eligible = np.load("results/hudge_boundary_frontier_20261115/partial12/seeds.npz")["selected"]
    for name, coords in proposals.items():
        path = folder / name
        path.mkdir(exist_ok=True)
        warp = samples(target, coords)
        det = np.linalg.det(
            np.stack((np.gradient(coords, axis=1), np.gradient(coords, axis=0)), -1)
        )
        pred = samples(void.astype(float), coords) > 0.5
        row = dict(
            material=corr(reference[m], warp[m]),
            crown=corr(reference[crown], warp[crown]),
            folds=int((m & (det <= 0)).sum()),
            overlap=float((pred & holes).sum() / (pred | holes).sum()),
            rms_intensity=float(np.sqrt(np.mean((reference[m] - warp[m]) ** 2))),
        )
        results[name] = row
        im = Image.fromarray(np.uint8(np.clip(warp[:225, :440], 0, 255)))
        im.save(path / "corrected.png")
        Image.fromarray(np.uint8(reference[:225, :440])).save(
            path / "reference_corrected.gif",
            save_all=True,
            append_images=[im],
            duration=1000,
            loop=0,
        )
        write_seeds(path, coords, grid, m, template, eligible)
        np.savez_compressed(path / "map.npz", coordinates=coords)
        print(name, row, flush=True)
    (folder / "metrics.json").write_text(json.dumps(results, indent=2))
    Image.fromarray(np.uint8(reference[:225, :440])).save(folder / "reference.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--start-map", type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.folder)
    else:
        experiment(args.folder, args.start_map)
