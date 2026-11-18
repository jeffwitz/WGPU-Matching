"""Image-derived hole hypotheses; moments propose geometry, texture ranks it.

Research CPU reference. No displacement truth is accepted by this module.
Coordinates are floating pixel centres, ordered (x, y). Covariances describe
filled holes, not their boundary pixels. A source pixel has covariance I/12.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, label, map_coordinates
from scipy.optimize import least_squares, linear_sum_assignment


@dataclass
class Hole:
    component: int
    centre: np.ndarray
    covariance: np.ndarray
    area: int
    ellipse_residual: float

    @property
    def radius(self):
        return np.sqrt(self.area / np.pi)


def spd_power(matrix, power):
    values, vectors = np.linalg.eigh(matrix)
    if values.min() <= 0 or not np.isfinite(values).all():
        raise ValueError("hole covariance must be positive definite")
    return (vectors * values**power) @ vectors.T


def holes_from_mask(material, minimum_area=128):
    """Exclude exterior/cropped components; keep internal candidates for matching."""
    components, count = label(~np.asarray(material, bool))
    outside = np.unique(np.r_[components[0], components[-1],
                              components[:, 0], components[:, -1]])
    holes = []
    for component in range(1, count + 1):
        region = components == component
        if component in outside or region.sum() < minimum_area:
            continue
        points = np.argwhere(region)[:, ::-1].astype(float)
        centre = points.mean(0)
        offsets = points - centre
        covariance = offsets.T @ offsets / len(points) + np.eye(2) / 12
        border = np.argwhere(region & ~binary_erosion(region))[:, ::-1]
        # Pixel-centre border, hence an O(1/radius) discretisation residual.
        radius = np.linalg.norm((border - centre) @ spd_power(covariance, -.5), axis=1)
        residual = np.sqrt(np.mean((radius / 2 - 1)**2))
        holes.append(Hole(component, centre, covariance, len(points), float(residual)))
    return holes


def moment_affine(source, target, angle_degrees):
    theta = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    matrix = spd_power(target.covariance, .5) @ rotation @ spd_power(source.covariance, -.5)
    return matrix, target.centre - matrix @ source.centre


def match_holes(sources, targets, source_mask, target_mask):
    """Hungarian assignment from observable bounding-box geometry and area.

    Assumes complete, similarly oriented silhouettes, as the Hudge plate.
    It is NOT an arbitrary-rotation or arbitrary-occlusion hole matcher.
    Ambiguous row/column alternatives (<.15 cost margin) are rejected.
    """
    if not sources or not targets:
        return [], np.empty((len(sources), len(targets)))
    bounds = []
    for mask in (source_mask, target_mask):
        xy = np.argwhere(mask)[:, ::-1]
        bounds.append((xy.min(0) - .5, xy.max(0) + .5))
    (lo0, hi0), (lo1, hi1) = bounds
    scale = (hi1 - lo1) / (hi0 - lo0)
    costs = np.empty((len(sources), len(targets)))
    for i, a in enumerate(sources):
        predicted = lo1 + scale * (a.centre - lo0)
        for j, b in enumerate(targets):
            spatial = np.linalg.norm((b.centre - predicted) / (scale * a.radius))
            log_area = abs(np.log(b.area / (a.area * np.prod(scale))))
            costs[i, j] = spatial + .25 * log_area
    rows, cols = linear_sum_assignment(costs)
    matches = []
    for i, j in zip(rows, cols, strict=True):
        alternatives = np.r_[np.delete(costs[i], j), np.delete(costs[:, j], i)]
        margin = float(alternatives.min() - costs[i, j]) if len(alternatives) else np.inf
        if costs[i, j] <= 1.5 and margin >= .15:
            matches.append((int(i), int(j), float(costs[i, j]), margin))
    return matches, costs


def sector_points(mask, hole, radial_range, angle_degrees, aperture=40):
    yy, xx = np.indices(mask.shape)
    delta = np.stack((xx - hole.centre[0], yy - hole.centre[1]), -1)
    radius = np.linalg.norm(delta, axis=-1) / hole.radius
    angle = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0]))
    angular = (angle - angle_degrees + 180) % 360 - 180
    selected = (mask & (radius >= radial_range[0]) & (radius < radial_range[1])
                & (np.abs(angular) <= aperture / 2))
    return np.column_stack((xx[selected], yy[selected])).astype(float)


def sample(image, points, order=1):
    return map_coordinates(np.asarray(image, float), points[:, ::-1].T,
                           order=order, mode="constant", cval=0, prefilter=False)


def score_affine(reference_values, points, target_image, target_material, matrix, translation):
    """Same source samples for EVERY angle; no hypothesis-specific intersection.

    Invalid target samples penalise correlation through their actual intensities;
    never delete them. Reject if <90% have full bilinear material support, or any
    sample leaves the image. Return support fraction even for rejected proposals.
    """
    q = points @ matrix.T + translation
    return score_coordinates(reference_values, q, target_image, target_material)


def score_coordinates(reference_values, q, target_image, target_material):
    """Photometry for a prescribed warp, keeping every reference sample."""
    inside = ((q >= 0) & (q <= np.array(target_image.shape[::-1]) - 1)).all()
    fraction = float(np.mean(sample(target_material, q) >= 1 - 1e-8))
    if not inside or fraction < .9 or len(q) < 32:
        return -1., fraction
    a = reference_values - np.mean(reference_values)
    b = sample(target_image, q)
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return (float(a @ b / denominator) if denominator > 1e-12 else -1.), fraction


def rank_sector(reference, target, target_mask, points, source, destination,
                angles=None):
    if angles is None:
        angles = np.arange(0, 360, 5)
    values = sample(reference, points)
    rows = []
    for angle in angles:
        matrix, translation = moment_affine(source, destination, angle)
        score, fraction = score_affine(values, points, target, target_mask, matrix, translation)
        rows.append((float(angle), score, fraction))
    return np.asarray(rows)


def distinct_peaks(rows, count=3, separation=15):
    chosen = []
    for row in rows[np.argsort(-rows[:, 1], kind="stable")]:
        if row[1] <= -1:
            continue
        if all(abs((row[0] - previous[0] + 180) % 360 - 180) >= separation
               for previous in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return np.asarray(chosen).reshape(-1, 3)


def affine_parameters(matrix, translation, centres):
    displacement = np.asarray(centres) @ (matrix - np.eye(2)).T + translation
    gradient = matrix - np.eye(2)
    return np.column_stack((displacement[:, 0], np.full(len(centres), gradient[0, 0]),
                            np.full(len(centres), gradient[0, 1]), displacement[:, 1],
                            np.full(len(centres), gradient[1, 0]),
                            np.full(len(centres), gradient[1, 1])))


def refine_sector(reference, target, target_mask, points, matrix, translation, max_nfev=60):
    """CPU photometric affine proposal on a fixed material sector, not a DIC solve.

    Six free parameters. Translation is expressed at the sector centroid;
    gradient increments are scaled by support radius for numerical conditioning.
    No contour penalty, analytical field, or regularisation toward a circle.
    """
    centre = points.mean(0)
    offsets = points - centre
    scale = max(np.sqrt(np.mean(np.sum(offsets**2, axis=1))), 1.)
    q0 = centre @ matrix.T + translation
    values = sample(reference, points)
    a = values - values.mean()
    a /= max(np.linalg.norm(a), 1e-12)

    def unpack(p):
        f = matrix + p[2:].reshape(2, 2) / scale
        return f, q0 + p[:2] - f @ centre

    def residual(p):
        f, t = unpack(p)
        b = sample(target, points @ f.T + t)
        b -= b.mean()
        return b / max(np.linalg.norm(b), 1e-12) - a

    result = least_squares(residual, np.zeros(6), max_nfev=max_nfev,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7, diff_step=1e-3)
    f, t = unpack(result.x)
    score, fraction = score_affine(values, points, target, target_mask, f, t)
    if np.linalg.det(f) <= 0:
        score = -1.
    return f, t, {'score': score, 'material_fraction': fraction,
                  'optimizer_success': bool(result.success), 'evaluations': result.nfev,
                  'determinant': float(np.linalg.det(f))}


def quadratic_mapping(points, origin, matrix, translation, hessian):
    """q=F*x+t+1/2 H:(x-origin)^2; return q and analytical Jacobians."""
    delta = np.asarray(points) - origin
    h = np.asarray(hessian).reshape(2, 3)
    tensors = np.stack((h[:, [0, 1]], h[:, [1, 2]]), axis=-2)
    q = points @ matrix.T + translation + .5*np.einsum('ni,oij,nj->no', delta, tensors, delta)
    jac = matrix + np.einsum('oij,nj->noi', tensors, delta)
    return q, jac


def refine_quadratic_sector(reference, target, target_mask, points, matrix, translation):
    """Diagnostic twelve-parameter CPU fit; no change to the external solver."""
    origin = points.mean(0)
    scale = max(np.sqrt(np.mean(np.sum((points-origin)**2, axis=1))), 1.)
    values = sample(reference, points)
    a = values - values.mean()
    a /= max(np.linalg.norm(a), 1e-12)

    def unpack(p):
        f = matrix + p[2:6].reshape(2, 2)/scale
        t = translation + p[:2] - (f-matrix)@origin
        h = p[6:].reshape(2, 3)/scale**2
        return f, t, h

    def residual(p):
        f, t, h = unpack(p)
        q, _ = quadratic_mapping(points, origin, f, t, h)
        b = sample(target, q)
        b -= b.mean()
        return b/max(np.linalg.norm(b), 1e-12)-a

    result = least_squares(residual, np.zeros(12), max_nfev=60,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7, diff_step=1e-3)
    f, t, h = unpack(result.x)
    q, jac = quadratic_mapping(points, origin, f, t, h)
    score, fraction = score_coordinates(values, q, target, target_mask)
    minimum_det = float(np.linalg.det(jac).min())
    if minimum_det <= 0:
        score = -1.
    return f, t, h, origin, {'score': score, 'material_fraction': fraction,
                             'optimizer_success': bool(result.success), 'evaluations': result.nfev,
                             'determinant': minimum_det}
