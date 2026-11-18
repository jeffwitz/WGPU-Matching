"""Experimental Gaussian displacement fit with measured derivative constraints.

CPU fitting of at most 36 sites; native IC-GN remains on WebGPU. This research
helper does not alter the public projection API or its archived selection rules.
"""

from dataclasses import dataclass

import numpy as np
from hudge_affine_restart import affine_states, fit_global


@dataclass
class SeedField:
    matrix: np.ndarray
    translation: np.ndarray
    sites: np.ndarray
    coefficients: np.ndarray
    length: float

    def hessian(self, points):
        """Native-pixel second derivatives, axes (point, output, x/y, x/y)."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(self.sites):
            return np.zeros((len(points), 2, 2, 2))
        delta = points[:, None, :] - self.sites[None, :, :]
        kernel = np.exp(-np.sum(delta**2, axis=2)/(2*self.length**2))
        basis = (delta[:, :, :, None]*delta[:, :, None, :]/self.length**4
                 - np.eye(2)/self.length**2)*kernel[:, :, None, None]
        return np.einsum("nkij,ko->noij", basis, self.coefficients)

    def evaluate(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        states = affine_states(points, self.matrix, self.translation)
        if not len(self.sites):
            return states
        delta = points[:, None, :] - self.sites[None, :, :]
        kernel = np.exp(-np.sum(delta**2, axis=2) / (2*self.length**2))
        derivative = -delta * kernel[:, :, None] / self.length**2
        states[:, [0, 3]] += kernel @ self.coefficients
        gradients = np.einsum("nkj,ko->noj", derivative, self.coefficients)
        states[:, [1, 2, 4, 5]] += gradients.reshape(-1, 4)
        return states


def fit_seed_field(centres, parameters, accepted, *, method="rbf_gradient",
                   length=64., gradient_length=16., ridge=1e-4):
    """Soft least-squares displacement/gradient fit, using accepted states only.

    Minimize sum |u-u_i|² + gradient_length² sum |G-G_i|² + ridge |c|².
    Derivatives are native-pixel derivatives, not separately interpolated fields.
    The global robust affine is fixed; Gaussian residuals decay away from sites.
    """
    if method not in {"affine", "rbf_position", "rbf_gradient"}:
        raise ValueError("unknown field method")
    if length <= 0 or gradient_length < 0 or ridge <= 0:
        raise ValueError("invalid fit scales")
    use = np.asarray(accepted, bool) & np.isfinite(parameters).all(axis=1)
    matrix, translation, diagnostics = fit_global(centres, parameters, use)
    sites = np.asarray(centres)[use]
    field = SeedField(matrix, translation, np.empty((0, 2)), np.empty((0, 2)), length)
    if method != "affine":
        residual = parameters[use] - field.evaluate(sites)
        delta = sites[:, None, :] - sites[None, :, :]
        kernel = np.exp(-np.sum(delta**2, axis=2) / (2*length**2))
        blocks, targets = [kernel], [residual[:, [0, 3]]]
        if method == "rbf_gradient":
            for axis, indices in ((0, [1, 4]), (1, [2, 5])):
                blocks.append(-gradient_length * delta[:, :, axis] * kernel / length**2)
                targets.append(gradient_length * residual[:, indices])
        blocks.append(np.sqrt(ridge)*np.eye(len(sites)))
        targets.append(np.zeros((len(sites), 2)))
        design = np.vstack(blocks)
        coefficients = np.linalg.lstsq(design, np.vstack(targets), rcond=None)[0]
        field.sites, field.coefficients = sites, coefficients
        diagnostics["design_condition"] = float(np.linalg.cond(design))
    error = field.evaluate(sites)-parameters[use]
    diagnostics.update(position_fit_rms=float(np.sqrt(np.mean(np.sum(error[:, [0, 3]]**2, axis=1)))),
                       gradient_fit_rms=float(np.sqrt(np.mean(np.sum(error[:, [1, 2, 4, 5]]**2, axis=1)))),
                       length_px=length, gradient_length_px=gradient_length, ridge=ridge)
    return field, diagnostics
