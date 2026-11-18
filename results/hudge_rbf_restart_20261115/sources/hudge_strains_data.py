"""Image-only support geometry and explicitly separate displacement-file inspection."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import label


def material_mask(image, minimum_background_component=128):
    """Exclude large flat-background components; retain quantized speckle islands.

    Specific to the supplied constant-background renderings, not a general segmenter.
    The corner intensity is observable; no analytical material geometry is consulted.
    """
    image = np.asarray(image)
    components, _ = label(image == image[-1, -1])
    sizes = np.bincount(components.ravel())
    excluded = sizes >= minimum_background_component
    excluded[0] = False
    return ~excluded[components]


def silhouette_proposal(reference, target, centres):
    """Axis-aligned extent hypothesis from two images, not displacement truth.

    Requires identifiable complete silhouettes. It is intentionally a simple
    comparator and is not valid for arbitrary crops, rotations or occlusions.
    The returned six-parameter field remains only an initialization for the solver.
    """
    bounds = []
    for image in (reference, target):
        yy, xx = np.nonzero(material_mask(image))
        if not len(xx):
            raise ValueError("image has no non-background material")
        bounds.append(np.array([xx.min() - .5, yy.min() - .5,
                                xx.max() + .5, yy.max() + .5]))
    b0, b1 = bounds
    scales = (b1[2:] - b1[:2]) / (b0[2:] - b0[:2])
    translation = b1[:2] - scales * b0[:2]
    centres = np.asarray(centres)
    values = np.zeros((len(centres), 6))
    values[:, [0, 3]] = centres * (scales - 1) + translation
    values[:, 1], values[:, 5] = scales - 1
    return values, {"reference_bounds": b0.tolist(), "target_bounds": b1.tolist(),
                    "extent_ratio_xy": scales.tolist(), "translation_xy": translation.tolist()}


def read_ascii_vtu(path):
    """Read named displacement data without guessing units, image map or frame index."""
    tree = ET.parse(path)
    points = tree.find(".//Points/DataArray")
    displacement = tree.find(".//PointData/DataArray[@Name='DEPL']")
    if points is None or displacement is None:
        raise ValueError("expected points and named DEPL point data")
    if any(element.get("format") != "ascii" for element in (points, displacement)):
        raise ValueError("only explicit ASCII arrays supported")
    pts = np.fromstring(points.text, sep=" ").reshape(-1, int(points.get("NumberOfComponents")))
    values = np.fromstring(displacement.text, sep=" ").reshape(-1, int(displacement.get("NumberOfComponents")))
    if len(pts) != len(values) or not np.isfinite(pts).all() or not np.isfinite(values).all():
        raise ValueError("invalid displacement data")
    return pts, values


def evaluate_vtu_truth(path, image_centres, physical_to_pixel, pixel_origin):
    """Evaluate a supplied triangular field at exact centres using an explicit map.

    This is scoring-only. Callers must establish which mesh state belongs to which
    image and provide the map; neither is fitted from the estimated displacements.
    Original triangles are used, so voids are not bridged by a new triangulation.
    """
    points, values = read_ascii_vtu(path)
    tree = ET.parse(path)
    cells = tree.find(".//Cells/DataArray[@Name='connectivity']")
    types = tree.find(".//Cells/DataArray[@Name='types']")
    offsets = tree.find(".//Cells/DataArray[@Name='offsets']")
    if any(x is None or x.get("format") != "ascii" for x in (cells, types, offsets)):
        raise ValueError("expected explicit ASCII triangle connectivity")
    triangle = np.fromstring(cells.text, sep=" ", dtype=int).reshape(-1, 3)
    if not np.all(np.fromstring(types.text, sep=" ", dtype=int) == 5):
        raise ValueError("only linear triangles supported")
    if not np.array_equal(np.fromstring(offsets.text, sep=" ", dtype=int), 3*np.arange(1,len(triangle)+1)):
        raise ValueError("incompatible triangle offsets")
    matrix = np.asarray(physical_to_pixel, dtype=float).reshape(2,2)
    origin = np.asarray(pixel_origin, dtype=float).reshape(2)
    inverse = np.linalg.inv(matrix)
    physical = (np.asarray(image_centres)-origin) @ inverse.T
    vertices = points[triangle, :2]
    edges = np.stack((vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=2)
    inverse_edges = np.linalg.inv(edges)
    field_edges = np.stack((values[triangle[:,1]]-values[triangle[:,0]],
                            values[triangle[:,2]]-values[triangle[:,0]]),axis=2)
    physical_gradients = field_edges @ inverse_edges
    lower, upper = vertices.min(1),vertices.max(1)
    parameters = np.full((len(physical),6),np.nan)
    elements = np.full(len(physical),-1,int)
    for i, point in enumerate(physical):
        candidates = np.flatnonzero(((point >= lower-1e-10) & (point <= upper+1e-10)).all(1))
        for element in candidates:
            bary = inverse_edges[element] @ (point-vertices[element,0])
            if bary.min() >= -1e-10 and bary.sum() <= 1+1e-10:
                u = values[triangle[element,0]] + physical_gradients[element] @ (point-vertices[element,0])
                pixel_u = matrix @ u
                gradient = matrix @ physical_gradients[element] @ inverse
                parameters[i] = [pixel_u[0], *gradient[0], pixel_u[1], *gradient[1]]
                elements[i] = element
                break
    return parameters, elements >= 0
