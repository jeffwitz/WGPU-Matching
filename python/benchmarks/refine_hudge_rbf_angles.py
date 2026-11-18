"""Choose hole angles through the fitted nonlinear RBF warp, not an affine annulus."""

import argparse
import json
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import MixedField, corr, evaluate, fit
from hudge_contour_field import boundary_points
from hudge_hole_moments import Hole, moment_affine, sample
from hudge_rbf_restart import SeedField
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.ndimage import label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weight', type=float, default=1.)
    parser.add_argument('--name', default='joint_angles')
    parser.add_argument('--angle-radius', type=float, default=15.)
    args = parser.parse_args()
    root = args.input
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    report = json.loads((root/'summary.json').read_text())
    reference, target = [np.asarray(Image.open(p), float) for p in info['input_sha256']]
    mask = material_mask(reference)
    components, _ = label(~mask)
    tcomp, _ = label(~material_mask(target))
    source_holes = np.isin(components[:225, :440], [h['component'] for h in info['source_holes']])
    target_holes = np.isin(tcomp, [info['target_candidates'][j]['component'] for _, j, _, _ in info['matches']])
    old = np.load('results/hudge_boundary_frontier_20261115/partial12/seeds.npz')
    centres, states = old['centres'][old['accepted']], old['parameters'][old['accepted']]
    model = np.load(root/'seeds_only.npz')
    base = SeedField(model['matrix'], model['translation'], np.empty((0, 2)), np.empty((0, 2)), 16.)
    sites, lengths = model['sites'], model['length']
    holes, contours = [], []
    for i, j, _, _ in info['matches']:
        def make(h):
            return Hole(h['component'], np.asarray(h['centre']), np.asarray(h['covariance']), h['area'], h['ellipse_residual'])
        a, b = make(info['source_holes'][i]), make(info['target_candidates'][j])
        points = boundary_points(components == a.component)
        angle = np.arctan2(points[:, 1]-a.centre[1], points[:, 0]-a.centre[0])
        points = points[np.argsort(angle)]
        contours.append(points[np.linspace(0, len(points)-1, 48, dtype=int)])
        holes.append((a, b))
    hole_centres = np.array([a.centre for a, _ in holes])

    def solve(angles):
        targets, gradients = [], []
        for angle, points, (a, b) in zip(angles, contours, holes, strict=True):
            f, t = moment_affine(a, b, angle)
            targets.append(points@(f-np.eye(2)).T+t)
            gradients.append(f-np.eye(2))
        return fit(centres, states, sites, lengths, base, np.vstack(contours), np.vstack(targets),
                   hole_centres, np.asarray(gradients), args.weight, True).coefficients

    # The least-squares design is fixed and right-hand sides depend linearly on
    # cos(theta) and sin(theta). Thirteen solves give exact response coefficients
    # for all later angular queries; no refit or image-wide remap for each angle.
    zero = solve(np.zeros(6))
    cos_coeff, sin_coeff = [], []
    for i in range(6):
        angles = np.zeros(6)
        angles[i] = 180
        c = (zero-solve(angles))/2
        angles[i] = 90
        s = solve(angles)-zero+c
        cos_coeff.append(c)
        sin_coeff.append(s)
    constant = zero-np.sum(cos_coeff, axis=0)
    cos_coeff, sin_coeff = np.array(cos_coeff), np.array(sin_coeff)
    yy, xx = np.mgrid[:225, :440]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    radii = np.stack([np.linalg.norm(pixels-a.centre, axis=1)/a.radius for a, _ in holes])
    nearest = np.argmin(radii, axis=0)
    radius = radii[nearest, np.arange(len(pixels))]
    selection = mask[:225, :440].ravel() & (radius >= 1.) & (radius < 1.5)
    indices = np.flatnonzero(selection)
    points = pixels[indices]
    parity = points.sum(1) % 2 == 0
    values = reference[points[:, 1], points[:, 0]]
    constant_q = points+evaluate(MixedField(base, sites, constant, lengths), points)[:, [0, 3]]
    empty_base = SeedField(np.eye(2), np.zeros(2), np.empty((0, 2)), np.empty((0, 2)), 16.)
    cos_q = np.array([evaluate(MixedField(empty_base, sites, c, lengths), points)[:, [0, 3]] for c in cos_coeff])
    sin_q = np.array([evaluate(MixedField(empty_base, sites, c, lengths), points)[:, [0, 3]] for c in sin_coeff])
    initial = np.array([h['neighbour_angle'] for h in report['hole_affines']])
    angles = initial.copy()
    history = []

    def mapping(theta):
        return constant_q + np.einsum('h,hni->ni', np.cos(np.deg2rad(theta)), cos_q) + np.einsum('h,hni->ni', np.sin(np.deg2rad(theta)), sin_q)

    initial_q = mapping(angles)
    for sweep in range(3):
        for hole in range(6):
            selected = parity & (nearest[indices] == hole)
            scores, candidates = [], initial[hole]+np.arange(-args.angle_radius, args.angle_radius+.01, .5)
            for angle in candidates:
                trial = angles.copy()
                trial[hole] = angle
                q = mapping(trial)[selected]
                score = corr(values[selected], sample(target, q)) if len(q) >= 32 else -1.
                if not ((q >= 0) & (q <= np.array(target.shape[::-1])-1)).all():
                    score = -1.
                scores.append(score)
            angles[hole] = candidates[np.argmax(scores)]
            history.append({'sweep': sweep, 'hole': hole+1, 'angle': float(angles[hole]), 'score': max(scores)})
    final_q = mapping(angles)
    changes = []
    for i in range(6):
        selected = ~parity & (nearest[indices] == i)
        changes.append({'hole': i+1, 'initial_angle': initial[i], 'angle': angles[i], 'validation_pixels': int(selected.sum()),
                        'before_correlation': corr(values[selected], sample(target, initial_q[selected])),
                        'after_correlation': corr(values[selected], sample(target, final_q[selected]))})
    coeff = constant+np.einsum('h,hki->ki', np.cos(np.deg2rad(angles)), cos_coeff)+np.einsum('h,hki->ki', np.sin(np.deg2rad(angles)), sin_coeff)
    field = MixedField(base, sites, coeff, lengths)
    # Validate the linear response construction on the final parameters.
    exact = solve(angles)
    residual = float(np.max(np.abs(coeff-exact)))
    if residual > 1e-6:
        raise RuntimeError('affine angle response reconstruction failed')
    states_pixel = evaluate(field, pixels)
    q = pixels+states_pixel[:, [0, 3]]
    warp = sample(target, q).reshape(xx.shape)
    material = mask[:225, :440]
    pred_void = (sample(target_holes, q) >= .5).reshape(xx.shape)
    det = np.linalg.det(np.eye(2)+states_pixel[:, [1, 2, 4, 5]].reshape(-1, 2, 2)).reshape(xx.shape)
    e = evaluate(field, centres)-states
    row = {'variant': args.name, 'weight': args.weight, 'angle_radius': args.angle_radius,
           'seed_displacement_rms': float(np.sqrt(np.mean(np.sum(e[:, [0, 3]]**2, axis=1)))),
           'seed_gradient_rms': float(np.sqrt(np.mean(np.sum(e[:, [1, 2, 4, 5]]**2, axis=1)))),
           'material_correlation': corr(reference[:225, :440][material], warp[material]),
           'hole_iou': float((pred_void & source_holes).sum()/max((pred_void | source_holes).sum(), 1)),
           'nonpositive_material_jacobians': int((material & (det <= 0)).sum()),
           'response_max_coefficient_error': residual, 'per_hole_validation': changes, 'history': history,
           'note': 'Angles optimized on even reference material pixels, checked on odd ones; no independent displacement truth. '
           'Same moments and same seed observations. Search range recorded in angle_radius.'}
    (root/f'{args.name}.json').write_text(json.dumps(row, indent=2))
    np.savez_compressed(root/f'{args.name}.npz', matrix=field.matrix, translation=field.translation,
                        sites=field.sites, coefficients=field.coefficients, length=field.length,
                        points=pixels, parameters=states_pixel, angles=angles)
    Image.fromarray(np.uint8(np.clip(warp, 0, 255))).save(root/f'{args.name}.png')
    def frame(a, text):
        canvas = Image.new('RGB', (880, 484), 'white')
        canvas.paste(Image.fromarray(np.uint8(np.clip(a, 0, 255))).convert('RGB').resize((880, 450)), (0, 34))
        ImageDraw.Draw(canvas).text((10, 10), text, fill='black')
        return canvas
    frames = [frame(reference[:225, :440], 'Reference'), frame(warp, 'RBF with jointly refined hole angles - proposal')]
    frames[0].save(root/f'reference_{args.name}.gif', save_all=True, append_images=frames[1:], duration=1000, loop=0)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
