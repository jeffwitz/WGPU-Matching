"""Render the reference plate with explicit truth, kept separate from inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def forward(points, nonlinear=False):
    f = np.array([[1., .08], [.15, 3.9]])
    q = points @ f.T + [2., 10.]
    jac = np.broadcast_to(f, (*points.shape[:-1], 2, 2)).copy()
    if nonlinear:
        k = 2*np.pi/140
        q[..., 0] += 4*np.sin(k*points[..., 1])
        q[..., 1] += 8*np.sin(k*points[..., 0])
        jac[..., 0, 1] += 4*k*np.cos(k*points[..., 1])
        jac[..., 1, 0] += 8*k*np.cos(k*points[..., 0])
    return q, jac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nonlinear', action='store_true')
    parser.add_argument('--reference', type=Path, default=Path('/home/jeff/CNRS/DIC/HudgeStrains/FEA/Img_F.tif'))
    parser.add_argument('--grid', type=Path, default=Path('results/hudge_boundary_frontier_20261115/partial12/seeds.npz'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ref = np.asarray(Image.open(args.reference))
    y, x = np.indices(ref.shape)
    target_points = np.stack((x, y), -1).astype(float)
    f = np.array([[1., .08], [.15, 3.9]])
    source_points = (target_points - [2., 10.]) @ np.linalg.inv(f).T
    for _ in range(12):
        q, jac = forward(source_points, args.nonlinear)
        error = q - target_points
        det = jac[..., 0, 0]*jac[..., 1, 1] - jac[..., 0, 1]*jac[..., 1, 0]
        step = np.stack((jac[..., 1, 1]*error[..., 0] - jac[..., 0, 1]*error[..., 1],
                         -jac[..., 1, 0]*error[..., 0] + jac[..., 0, 0]*error[..., 1]), -1)/det[..., None]
        source_points -= step
    q, jac = forward(source_points, args.nonlinear)
    assert np.max(np.linalg.norm(q-target_points, axis=-1)) < 1e-7
    assert det.min() > 0
    target = map_coordinates(ref.astype(float), source_points[..., ::-1].reshape(-1, 2).T,
                             order=3, mode='constant', cval=float(ref[-1, -1])).reshape(ref.shape)
    Image.fromarray(ref).save(args.output/'Img_F.tif')
    Image.fromarray(np.uint8(np.clip(np.rint(target), 0, 255))).save(args.output/'Img_wa_g191.tif')
    grid = dict(np.load(args.grid))
    grid['accepted'][:] = False
    grid['parameters'][:] = 0
    grid['scores'][:] = 0
    np.savez_compressed(args.output/'empty.npz', **grid)
    q, jac = forward(grid['centres'], args.nonlinear)
    gradient = jac - np.eye(2)
    parameters = np.column_stack((q[:, 0]-grid['centres'][:, 0], gradient[:, 0],
                                  q[:, 1]-grid['centres'][:, 1], gradient[:, 1]))
    np.savez_compressed(args.output/'truth_scoring_only.npz', parameters=parameters, centres=grid['centres'],
                        matrix=f, translation=np.array([2., 10.]))
    (args.output/'render.json').write_text(json.dumps({
        'nonlinear': args.nonlinear, 'reference': str(args.reference),
        'formula': 'F=[[1,.08],[.15,3.9]], t=[2,10]; nonlinear adds [4 sin(2pi*y/140),8 sin(2pi*x/140)]',
        'sampling': 'inverse Newton 12 steps; cubic scipy interpolation; round/clamp uint8',
        'limitations': 'Same texture as historical plate; diagnostic twin, not independent generalization. '
        'Truth arrays are scoring-only. Interpolation and the local affine solver have discretization errors.'}, indent=2))


if __name__ == '__main__':
    main()
