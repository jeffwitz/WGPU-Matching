"""Coordinate-safe global affine restart helpers; no displacement truth inputs."""

import numpy as np
from scipy.ndimage import map_coordinates


def fit_global(centres, parameters, accepted):
    """Robust position-only affine fit to finite accepted centre correspondences."""
    use = accepted & np.isfinite(parameters).all(axis=1)
    x = centres[use]
    y = x + parameters[use][:, [0, 3]]
    if len(x) < 3:
        raise ValueError("fewer than three accepted correspondences")
    origin = x.mean(0)
    scale = np.maximum(x.std(0), 1)
    design = np.column_stack(((x-origin)/scale, np.ones(len(x))))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("accepted centres are collinear")
    weights = np.ones(len(x))
    for _ in range(10):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design*root[:, None], y*root[:, None], rcond=None)[0]
        residual = np.linalg.norm(design @ coefficients-y, axis=1)
        cutoff = max(1., 2.5*float(np.median(residual)))
        weights = np.minimum(1., cutoff/np.maximum(residual, 1e-12))
    matrix = coefficients[:2].T/scale[None, :]
    translation = coefficients[2]-matrix @ origin
    if np.linalg.det(matrix) <= 0 or np.linalg.cond(matrix) > 100:
        raise ValueError("nonpositive or ill-conditioned affine map")
    return matrix, translation, {"fit_count":len(x),"median_fit_residual_px":float(np.median(residual))}


def affine_states(centres, matrix, translation):
    values = np.zeros((len(centres), 6))
    values[:, [0, 3]] = centres @ (matrix-np.eye(2)).T + translation
    values[:, [1, 2, 4, 5]] = (matrix-np.eye(2)).ravel()
    return values


def compose(centres, residual, matrix, translation):
    """phi = A psi + t, with psi(x)=x+r(x): F_phi=A F_psi."""
    values = residual.copy()
    values[:, [0, 3]] = (centres+residual[:, [0, 3]]) @ matrix.T+translation-centres
    gradients = residual[:, [1, 2, 4, 5]].reshape(-1,2,2)+np.eye(2)
    values[:, [1, 2, 4, 5]] = (matrix @ gradients-np.eye(2)).reshape(-1,4)
    return values


def uncompose(centres, native, matrix, translation):
    inverse = np.linalg.inv(matrix)
    values = native.copy()
    values[:, [0, 3]] = (centres+native[:, [0, 3]]-translation) @ inverse.T-centres
    gradients = native[:, [1, 2, 4, 5]].reshape(-1,2,2)+np.eye(2)
    values[:, [1, 2, 4, 5]] = (inverse @ gradients-np.eye(2)).reshape(-1,4)
    return values


def rectify(target, matrix, translation):
    """Sample original target once, J(z)=I1(Az+t), never a previous rectification."""
    y, x = np.indices(target.shape, dtype=float)
    qx = matrix[0,0]*x+matrix[0,1]*y+translation[0]
    qy = matrix[1,0]*x+matrix[1,1]*y+translation[1]
    image = map_coordinates(target.astype(float), [qy,qx], order=1,
                            mode="constant", cval=float(target[-1,-1]), prefilter=False)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)
