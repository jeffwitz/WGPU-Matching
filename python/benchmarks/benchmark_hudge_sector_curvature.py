"""Paired affine/quadratic sector fits from identical image-only initial states."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from hudge_hole_moments import (
    Hole,
    quadratic_mapping,
    refine_quadratic_sector,
    refine_sector,
    sample,
    score_affine,
    score_coordinates,
    sector_points,
)
from hudge_strains_data import material_mask
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = json.loads((args.input/'summary.json').read_text())
    paths = list(source['input_sha256'])
    reference, target = [np.asarray(Image.open(p), float) for p in paths]
    masks = [material_mask(x) for x in (reference, target)]
    target_mask = masks[1].astype(float)
    baseline = np.load(source['baseline'])
    centres = baseline['centres']
    eligible = baseline['selected'] & ~baseline['accepted']
    proposals = [[] for _ in centres]
    supports, records = [], []
    for record in source['sectors']:
        hole = source['source_holes'][record['source_hole']]
        hole = Hole(hole['component'], np.array(hole['centre']), np.array(hole['covariance']),
                    hole['area'], hole['ellipse_residual'])
        points = sector_points(masks[0], hole, record['radial_range'], record['sector_angle'], aperture=50)
        parity = np.sum(points.astype(int), axis=1) % 2 == 0
        fit_points, validation = points[parity], points[~parity]
        supports.append(validation)
        delta = centres-hole.centre
        radius = np.linalg.norm(delta, axis=1)/hole.radius
        angle = (np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0]))-record['sector_angle']+180) % 360-180
        active = np.flatnonzero(eligible & (radius >= record['radial_range'][0])
                               & (radius < record['radial_range'][1]) & (np.abs(angle) <= 25))
        values = sample(reference, validation)
        for peak in record['peaks']:
            f, t = np.array(peak['matrix']), np.array(peak['translation'])
            af, at, a_fit = refine_sector(reference, target, target_mask, fit_points, f, t)
            a_score, _ = score_affine(values, validation, target, target_mask, af, at)
            qf, qt, h, origin, q_fit = refine_quadratic_sector(reference, target, target_mask, fit_points, f, t)
            q, jac = quadratic_mapping(validation, origin, qf, qt, h)
            q_score, _ = score_coordinates(values, q, target, target_mask)
            q_fit['heldout_min_det'] = float(np.linalg.det(jac).min())
            accepted = q_fit['score'] >= .65 and q_score >= .65 and q_fit['heldout_min_det'] > 0
            records.append({'sector': record['id'], 'angle': peak['angle'],
                            'affine_validation': a_score, 'quadratic_validation': q_score,
                            'affine': a_fit, 'quadratic': q_fit, 'qualified_proposal': accepted})
            if not accepted:
                continue
            q, jac = quadratic_mapping(centres[active], origin, qf, qt, h)
            g = jac-np.eye(2)
            p = np.column_stack((q[:, 0]-centres[active, 0], g[:, 0],
                                 q[:, 1]-centres[active, 1], g[:, 1]))
            for i, seed in zip(active, p, strict=True):
                proposals[i].append((q_score, record['id'], seed, h))
    rows = []
    for i, candidates in enumerate(proposals):
        for rank, (score, sector, p, h) in enumerate(sorted(candidates, key=lambda c: -c[0])[:6]):
            rows.append((i, rank, sector, p, h))
    np.savez_compressed(args.output/'proposals.npz', centres=centres[[r[0] for r in rows]],
                        rank=np.array([r[1] for r in rows]), sector=np.array([r[2] for r in rows]),
                        parameters=np.array([r[3] for r in rows]).reshape(-1, 6),
                        curvature=np.array([r[4] for r in rows]).reshape(-1, 6))
    np.savez_compressed(args.output/'validation_supports.npz', points=np.concatenate(supports),
                        offsets=np.r_[0, np.cumsum([len(p) for p in supports])])
    (args.output/'summary.json').write_text(json.dumps({'input': str(args.input), 'pairs': records,
        'proposals': len(rows), 'centres': len({r[0] for r in rows}),
        'seconds': time.perf_counter()-start,
        'limits': 'CPU photometric quadratic fit, same initial state and 60 evaluations for affine control. '
        'Checkerboard holdout is spatially correlated, not an independent statistical test. '
        'Only local affine tangents are sent to the external GPU solver; curvature stays fixed in sector validation.'}, indent=2))
    print(len(rows), 'hypotheses', len({r[0] for r in rows}), 'centres', flush=True)


if __name__ == '__main__':
    main()
