"""Bounded-memory blending of measured local affine/quadratic displacement maps."""

import numpy as np
from scipy.spatial import cKDTree


class ReducedRBF:
    """Gaussian radial basis field, fitted to positions and soft derivatives.

    The basis sites are a deterministic farthest-point subset of observations.
    Derivative magnitude is never used as a rejection criterion.
    """

    def __init__(self, centres, parameters, length=32., gradient_length=4.,
                 max_sites=196, ridge=1e-3):
        from hudge_affine_restart import fit_global
        from hudge_rbf_restart import SeedField

        centres, parameters = np.asarray(centres, float), np.asarray(parameters, float)
        if not len(centres) or length <= 0 or gradient_length < 0 or ridge <= 0:
            raise ValueError("invalid observations or fit parameters")
        selected = [0]
        distances = np.full(len(centres), np.inf)
        for _ in range(min(max_sites, len(centres))-1):
            distances = np.minimum(distances, np.sum((centres-centres[selected[-1]])**2, axis=1))
            selected.append(int(np.argmax(distances)))
        sites = centres[selected]
        matrix, translation, _ = fit_global(centres, parameters[:, :6], np.ones(len(centres), bool))
        self.field = SeedField(matrix, translation, np.empty((0, 2)), np.empty((0, 2)), length)
        residual = parameters[:, :6]-self.field.evaluate(centres)
        delta = centres[:, None, :]-sites[None, :, :]
        kernel = np.exp(-np.sum(delta**2, axis=2)/(2*length**2))
        blocks, targets = [kernel], [residual[:, [0, 3]]]
        if gradient_length:
            for axis, indices in ((0, [1, 4]), (1, [2, 5])):
                blocks.append(-gradient_length*delta[:, :, axis]*kernel/length**2)
                targets.append(gradient_length*residual[:, indices])
        blocks.append(np.sqrt(ridge)*np.eye(len(sites)))
        targets.append(np.zeros((len(sites), 2)))
        self.field.sites = sites
        self.field.coefficients = np.linalg.lstsq(np.vstack(blocks), np.vstack(targets), rcond=None)[0]

    def evaluate(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        if not len(points):
            return np.empty((0, 6))
        return np.concatenate([self.field.evaluate(points[i:i+2048]) for i in range(0, len(points), 2048)])


class LocalField:
    """Normalized Gaussian partition of unity of local Taylor maps.

    Gradients include derivatives of the weights. Nearest-neighbour truncation
    makes the model piecewise smooth; it is not a globally fitted RBF model.
    """

    def __init__(self, centres, parameters, length=12., neighbours=24):
        self.centres = np.asarray(centres, float)
        self.parameters = np.asarray(parameters, float)
        self.length = float(length)
        self.neighbours = min(neighbours, len(centres))
        if self.neighbours < 1 or self.length <= 0:
            raise ValueError("nonempty observations and positive scale required")
        self.tree = cKDTree(self.centres)

    def evaluate(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        result = np.empty((len(points), 6))
        for start in range(0, len(points), 4096):
            q = points[start:start+4096]
            distances, indices = self.tree.query(q, k=self.neighbours)
            distances = distances.reshape(len(q), self.neighbours)
            indices = indices.reshape(len(q), self.neighbours)
            delta = q[:, None, :]-self.centres[indices]
            p = self.parameters[indices]
            g = p[:, :, [1, 2, 4, 5]].reshape(len(q), self.neighbours, 2, 2).copy()
            u = p[:, :, [0, 3]] + np.einsum("nkij,nkj->nki", g, delta)
            if p.shape[-1] == 12:
                h = np.empty((len(q), self.neighbours, 2, 2, 2))
                h[:, :, :, 0, 0] = p[:, :, [6, 9]]
                h[:, :, :, 0, 1] = h[:, :, :, 1, 0] = p[:, :, [7, 10]]
                h[:, :, :, 1, 1] = p[:, :, [8, 11]]
                u += .5*np.einsum("nkoij,nki,nkj->nko", h, delta, delta)
                g += np.einsum("nkoij,nkj->nkoi", h, delta)
            w = np.exp(-(distances**2-distances[:, :1]**2)/(2*self.length**2))
            w /= w.sum(axis=1, keepdims=True)
            value = np.einsum("nk,nko->no", w, u)
            gradient = np.einsum("nk,nkoi->noi", w, g)
            gradient += np.einsum("nk,nko,nki->noi", w, u-value[:, None, :], -delta/self.length**2)
            result[start:start+len(q), [0, 3]] = value
            result[start:start+len(q), [1, 2, 4, 5]] = gradient.reshape(-1, 4)
        return result
