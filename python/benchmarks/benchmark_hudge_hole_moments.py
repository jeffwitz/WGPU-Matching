"""Generate auditable hole-moment/sector hypotheses for native GPU seed solves."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from hudge_hole_moments import (
    affine_parameters,
    distinct_peaks,
    holes_from_mask,
    match_holes,
    moment_affine,
    rank_sector,
    refine_sector,
    sample,
    score_affine,
    sector_points,
)
from hudge_strains_data import material_mask
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/home/jeff/CNRS/DIC/HudgeStrains/FEA'))
    parser.add_argument('--level', type=int, default=191)
    parser.add_argument('--baseline', type=Path,
                        default=Path('results/hudge_boundary_frontier_20261115/partial12/seeds.npz'))
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--heldout-sectors', action='store_true',
                        help='Fit on even pixels; require .65 correlation on held-out odd pixels')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [args.data/'Img_F.tif', args.data/f'Img_wa_g{args.level:03d}.tif']
    images = [np.asarray(Image.open(path), float) for path in paths]
    masks = [material_mask(image) for image in images]
    holes = [holes_from_mask(mask) for mask in masks]
    matches, costs = match_holes(*holes, *masks)
    baseline = np.load(args.baseline)
    centres = baseline['centres']
    # Frozen grid, only unresolved selected windows may receive hypotheses.
    eligible = baseline['selected'] & ~baseline['accepted']
    proposals = [[] for _ in centres]
    records, scans, sensitivity, validation_points = [], [], [], []
    target_mask = masks[1].astype(float)
    for source_index, target_index, cost, margin in matches:
        source, destination = holes[0][source_index], holes[1][target_index]
        for lower, upper in [(1.02, 1.5), (1.5, 2.), (2., 3.)]:
            for sector_angle in np.arange(0, 360, 45):
                points = sector_points(masks[0], source, (lower, upper), sector_angle, aperture=50)
                if len(points) < 32:
                    continue
                sector_id = len(scans)
                split = np.sum(points.astype(int), axis=1) % 2 == 0
                fit_points = points[split] if args.heldout_sectors else points
                check_points = points[~split] if args.heldout_sectors else points
                rows = rank_sector(*images, target_mask, fit_points, source, destination)
                scans.append(rows)
                validation_points.append(check_points)
                delta = centres - source.centre
                radius = np.linalg.norm(delta, axis=1) / source.radius
                angular = (np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0])) - sector_angle + 180) % 360 - 180
                active = np.flatnonzero(eligible & (radius >= lower) & (radius < upper)
                                       & (np.abs(angular) <= 25))
                record = {'id': sector_id, 'source_hole': source_index, 'target_hole': target_index,
                          'radial_range': [lower, upper], 'sector_angle': int(sector_angle),
                          'samples': len(points), 'unresolved_centres': len(active), 'peaks': []}
                for peak in distinct_peaks(rows):
                    matrix, translation = moment_affine(source, destination, peak[0])
                    refined_f, refined_t, fit = refine_sector(*images, target_mask, fit_points, matrix, translation)
                    heldout_score, _ = score_affine(sample(images[0], check_points), check_points,
                                                    images[1], target_mask, refined_f, refined_t)
                    fit['heldout_score'] = heldout_score
                    record['peaks'].append({'angle': float(peak[0]), 'raw_score': float(peak[1]),
                                            'raw_material_fraction': float(peak[2]), 'fit': fit,
                                            'matrix': refined_f.tolist(), 'translation': refined_t.tolist()})
                    for variant, f, t, score in [('moments', matrix, translation, float(peak[1])),
                                                ('sector_refined', refined_f, refined_t, fit['score'])]:
                        if score <= -1:
                            continue
                        if args.heldout_sectors:
                            validation_score, _ = score_affine(sample(images[0], check_points), check_points,
                                                               images[1], target_mask, f, t)
                            if score < .65 or validation_score < .65:
                                continue
                            score = validation_score
                        parameters = affine_parameters(f, t, centres[active])
                        for i, p in zip(active, parameters, strict=True):
                            proposals[i].append((score, sector_id, variant, float(peak[0]), p))
                if args.sensitivity and record['peaks']:
                    best = max(record['peaks'], key=lambda x: x['fit']['score'])
                    # Fixed perturbations; evaluated for ALL best sector hypotheses,
                    # including failed ones. No post-hoc successful-sector selection.
                    for da, dx, dy in [(0, 0, 0), (-5, 0, 0), (5, 0, 0),
                                       (0, -.5, 0), (0, .5, 0), (0, 0, -.5), (0, 0, .5)]:
                        f, t = moment_affine(source, destination, best['angle'] + da)
                        f, t, fit = refine_sector(*images, target_mask, fit_points, f, t + [dx, dy])
                        original_f = np.asarray(best['matrix'])
                        original_t = np.asarray(best['translation'])
                        change = points @ (f - original_f).T + t - original_t
                        sensitivity.append({'sector': sector_id, 'perturbation': [da, dx, dy],
                                            'rms_mapping_change': float(np.sqrt(np.mean(np.sum(change**2, axis=1)))),
                                            **fit})
                records.append(record)
        print('hole', source_index, '->', target_index, 'cost', cost, 'margin', margin, flush=True)
    indices, parameters, ranks, provenance = [], [], [], []
    for i, choices in enumerate(proposals):
        # At most six hypotheses per centre, ranked using image texture only.
        for rank, (score, sector, variant, angle, p) in enumerate(sorted(choices, key=lambda c: -c[0])[:6]):
            indices.append(i)
            parameters.append(p)
            ranks.append(rank)
            provenance.append({'centre_index': i, 'rank': rank, 'sector': sector,
                               'variant': variant, 'angle': angle, 'score': score})
    indices = np.asarray(indices, int)
    np.savez_compressed(args.output/'proposals.npz', centres=centres[indices],
                        parameters=np.asarray(parameters).reshape(-1, 6), rank=np.asarray(ranks, int),
                        indices=indices, sector=np.array([p['sector'] for p in provenance], int))
    np.savez_compressed(args.output/'validation_supports.npz', points=np.concatenate(validation_points),
                        offsets=np.r_[0, np.cumsum([len(p) for p in validation_points])])
    np.savez_compressed(args.output/'scans.npz', scans=np.asarray(scans), assignment_cost=costs)

    def serialize_hole(hole):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(hole).items()}

    report = {'level': args.level, 'baseline': str(args.baseline),
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'source_holes': [serialize_hole(h) for h in holes[0]],
              'target_candidates': [serialize_hole(h) for h in holes[1]],
              'matches': matches, 'sectors': records, 'provenance': provenance,
              'sensitivity': sensitivity, 'unique_proposed_centres': len(np.unique(indices)),
              'hypotheses': len(indices), 'seconds': time.perf_counter() - started,
              'heldout_sectors': args.heldout_sectors,
              'backend': 'CPU proposal and sector refinement; GPU solves are a separate command',
              'limitations': 'Image segmentation and silhouette-based assignment specific to these complete plate renders. '
              'Moment affine is only an initializer. No ground truth or contour fit in refinement. '
              'Sector refinement is CPU least squares, not the external IC-GN solver. '
              'A converged sector optimizer is not an accepted DIC measurement.'}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print('proposals', len(indices), 'centres', len(np.unique(indices)), 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
