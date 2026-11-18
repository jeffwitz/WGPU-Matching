"""Image-pair registration on a nested bilinear displacement mesh, plate 191.

Alternate material pixels are withheld throughout the fit. The mesh describes
a correction to the archived displacement. Update derivatives use the target
gradient in original pixel units. Accept only steps that improve the objective
and preserve a positive sampled Jacobian on material pixels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import corr
from hudge_dense_residual_flow import write_seeds
from PIL import Image
from qualify_hudge_rectified_windows import material_mask
from scipy import sparse
from scipy.ndimage import binary_dilation, map_coordinates, spline_filter
from scipy.sparse.linalg import spsolve


def basis(h, w, spacing):
    y, x = np.indices((h, w))
    ny, nx = (h - 1) // spacing + 2, (w - 1) // spacing + 2
    idx = y // spacing * nx + x // spacing
    v, u = (y % spacing) / spacing, (x % spacing) / spacing
    cols = np.stack((idx, idx + 1, idx + nx, idx + nx + 1), -1).reshape(-1)
    vals = np.stack(((1 - v) * (1 - u), (1 - v) * u, v * (1 - u), v * u), -1).ravel()
    out = sparse.csr_matrix((vals, (np.repeat(np.arange(h * w), 4), cols)), shape=(h * w, nx * ny))
    dx = sparse.diags([-1.0, 1.0], [0, 1], shape=(nx - 1, nx))
    dy = sparse.diags([-1.0, 1.0], [0, 1], shape=(ny - 1, ny))
    edges = sparse.vstack((sparse.kron(sparse.eye(ny), dx), sparse.kron(dy, sparse.eye(nx))))
    return out, (edges.T @ edges).tocsr()


def fit(root):
    info = json.loads(
        Path("results/hudge_hole_moments_20261117/proposals/summary.json").read_text()
    )
    a, b = [np.asarray(Image.open(p), float) for p in info["input_sha256"]]
    coeff = spline_filter(b)

    def sample(xy):
        return map_coordinates(
            coeff, [xy[..., 1], xy[..., 0]], order=3, prefilter=False, mode="nearest"
        )

    prior = np.load(root / "prior/map.npz")["coordinates"]
    h, w = prior.shape[:2]
    material = material_mask(a)[:h, :w]
    y, x = np.indices((h, w))
    radius = np.min(
        [
            np.hypot(x - v["centre"][0], y - v["centre"][1]) / np.sqrt(v["area"] / np.pi)
            for v in info["source_holes"]
        ],
        axis=0,
    )
    crown = material & (radius >= 1) & (radius < 1.25)
    train = material & ((x + y) % 2 == 0)
    test = material & ~train
    observed = a[:h, :w]
    ref = observed[train]

    def diagnostics(coords):
        det = np.linalg.det(
            np.stack((np.gradient(coords, axis=1), np.gradient(coords, axis=0)), -1)
        )
        warped = sample(coords)
        return dict(
            train=corr(observed[train], warped[train]),
            test=corr(observed[test], warped[test]),
            crown_test=corr(observed[test & crown], warped[test & crown]),
            folds=int((det[material] <= 0).sum()),
        )

    log = []
    coords = prior.copy()
    log.append(diagnostics(coords))
    for spacing in (16, 8, 4):
        B, L = basis(h, w, spacing)
        A = B[train.ravel()]
        base = coords.copy()
        controls = np.zeros((B.shape[1], 2))
        # Quadratic roughness on corrections within this resolution level.
        reg = 1400 * L + 12 * sparse.eye(L.shape[0])
        penalty = sparse.block_diag((reg, reg), format="csr")
        for _ in range(16):
            z = coords[train]
            residual = sample(z) - ref
            g = [
                (sample(z + offset) - sample(z - offset)) / 0.5
                for offset in (np.array([0.25, 0]), np.array([0, 0.25]))
            ]
            design = sparse.hstack([A.multiply(col[:, None]) for col in g], format="csr")
            weights = 1 / np.sqrt(1 + (residual / 20) ** 2)
            cur = controls.T.ravel()
            grad = design.T @ (weights * residual) + penalty @ cur
            step = spsolve(design.T @ design.multiply(weights[:, None]) + penalty, -grad)

            def objective(r, c):
                return float(
                    np.sum(400 * (np.sqrt(1 + (r / 20) ** 2) - 1)) + 0.5 * c @ (penalty @ c)
                )

            old = objective(residual, cur)
            # Restrict nodes influencing at-risk material locally, so a single
            # nearly singular element cannot stop the update everywhere else.
            direction = step.reshape(2, -1).T.copy()
            for _guard in range(14):
                local = coords + (B @ direction).reshape(h, w, 2)
                jac = np.stack((np.gradient(local, axis=1), np.gradient(local, axis=0)), -1)
                risk = material & (np.linalg.det(jac) <= 0.005)
                if not risk.any():
                    break
                near = np.unique(B[binary_dilation(risk).ravel()].indices)
                direction[near] *= 0.4
            step = direction.T.ravel()
            for alpha in 2.0 ** -np.arange(10):
                c = cur + alpha * step
                trial = base + (B @ c.reshape(2, -1).T).reshape(h, w, 2)
                det = np.linalg.det(
                    np.stack((np.gradient(trial, axis=1), np.gradient(trial, axis=0)), -1)
                )
                if det[material].min() > 0 and objective(sample(trial[train]) - ref, c) < old:
                    coords = trial
                    controls = c.reshape(2, -1).T
                    break
            else:
                break
        row = diagnostics(coords)
        row["spacing"] = spacing
        log.append(row)
        print(row, flush=True)
    out = root / "mesh"
    out.mkdir(exist_ok=True)
    template = np.load("results/hudge_hole_rbf_20261117/curve_transport_191/initialisation.npz")
    eligible = np.load(root / "prior/seeds.npz")["accepted"]
    write_seeds(out, coords, np.stack((x, y), -1), material, template, eligible)
    np.savez_compressed(out / "map.npz", coordinates=coords)
    (out / "fit.json").write_text(json.dumps(log, indent=2))
    after = Image.fromarray(np.uint8(np.clip(sample(coords)[:225, :440], 0, 255)))
    after.save(out / "corrected.png")
    Image.fromarray(observed[:225, :440].astype(np.uint8)).save(
        out / "reference_corrected.gif", save_all=True, append_images=[after], duration=900, loop=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    fit(parser.parse_args().folder)
