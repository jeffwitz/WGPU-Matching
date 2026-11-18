"""Image-contour constraints on a displacement field, with interior anchors.

This CPU research fit uses observed masks, never circular templates or analytical
displacements. Contours constrain only motion normal to the target boundary.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def boundary_points(mask):
    """Pixel-edge midpoints separating the observed material and background."""
    y, x = np.nonzero(mask[:, 1:] != mask[:, :-1])
    horizontal = np.column_stack((x+.5, y))
    y, x = np.nonzero(mask[1:] != mask[:-1])
    return np.vstack((horizontal, np.column_stack((x, y+.5))))


class ContourField:
    def __init__(self, base, reference_mask, target_mask, anchors, length=16., max_sites=196):
        self.base = base
        self.length = float(length)
        self.anchors = np.asarray(anchors, float)
        self.boundary = boundary_points(reference_mask)[::3]
        pool = np.vstack((self.boundary, self.anchors))
        chosen, distances = [0], np.full(len(pool), np.inf)
        for _ in range(min(max_sites, len(pool))-1):
            distances = np.minimum(distances, np.sum((pool-pool[chosen[-1]])**2, axis=1))
            chosen.append(int(np.argmax(distances)))
        self.sites = pool[chosen]
        self.coefficients = np.zeros((len(self.sites), 2))
        self.distance = distance_transform_edt(target_mask)-distance_transform_edt(~target_mask)
        gy, gx = np.gradient(self.distance)
        self.distance_gradient = (gx, gy)

    def basis(self, points):
        delta = np.asarray(points)[:, None, :]-self.sites[None, :, :]
        kernel = np.exp(-np.sum(delta**2, axis=2)/(2*self.length**2))
        return kernel, -delta*kernel[:, :, None]/self.length**2

    def evaluate(self, points):
        points = np.asarray(points, float).reshape(-1, 2)
        result = self.base.evaluate(points)
        for start in range(0, len(points), 2048):
            k, dk = self.basis(points[start:start+2048])
            result[start:start+len(k), [0, 3]] += k @ self.coefficients
            result[start:start+len(k), [1, 2, 4, 5]] += np.einsum("nki,ko->noi", dk, self.coefficients).reshape(-1, 4)
        return result

    def residual(self):
        q = self.boundary+self.evaluate(self.boundary)[:, [0, 3]]
        distance = map_coordinates(self.distance, q[:, ::-1].T, order=1, mode="nearest")
        gradient = np.column_stack([map_coordinates(g, q[:, ::-1].T, order=1, mode="nearest")
                                    for g in self.distance_gradient])
        return distance, gradient

    def step(self, contour_weight=2., ridge=.05):
        """Linearized signed-distance fit; anchor displacements stay near the base.

        Ridge penalizes total correction coefficients. No threshold is placed on
        the magnitude of the physical displacement gradient.
        """
        distance, gradient = self.residual()
        kernel, _ = self.basis(self.boundary)
        anchors, _ = self.basis(self.anchors)
        n = len(self.sites)
        normal = np.concatenate((kernel*gradient[:, :1], kernel*gradient[:, 1:]), axis=1)
        anchor_design = np.block([[anchors, np.zeros_like(anchors)], [np.zeros_like(anchors), anchors]])
        current = self.coefficients.T.ravel()
        design = np.vstack((contour_weight*normal, anchor_design, np.sqrt(ridge)*np.eye(2*n)))
        target = np.concatenate((-contour_weight*distance, -anchor_design@current, -np.sqrt(ridge)*current))
        update = np.linalg.lstsq(design, target, rcond=None)[0].reshape(2, n).T
        return update
