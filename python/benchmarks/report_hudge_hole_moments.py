"""Publish fixed-denominator diagnostics, truth scoring, and remapped images."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hudge_hole_moments import Hole
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, map_coordinates


def write_csv(path, rows):
    if rows:
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def score_truth(root, name):
    truth = np.load(root/'truth_scoring_only.npz')
    results = np.load(root/name/'seeds.npz')
    centres = results['centres']
    if 'parameters' in truth:
        exact = truth['parameters']
    else:
        f = truth['matrix']
        u = centres@(f-np.eye(2)).T+truth['translation']
        exact = np.column_stack((u[:, 0], np.tile((f-np.eye(2))[0], (len(u), 1)),
                                 u[:, 1], np.tile((f-np.eye(2))[1], (len(u), 1))))
    error = np.linalg.norm(results['parameters'][:, [0, 3]]-exact[:, [0, 3]], axis=1)
    grad_error = np.linalg.norm(results['parameters'][:, [1, 2, 4, 5]]-exact[:, [1, 2, 4, 5]], axis=1)
    accepted = results['accepted']
    anchors = np.zeros(len(centres), bool)
    for path in (root/name).glob('extra_*.npz'):
        extra = np.load(path)
        anchors[extra['indices'][extra['retained']]] = True
    rows = []
    for i in np.flatnonzero(accepted & (error >= 2)):
        rows.append({'index': int(i), 'x': centres[i, 0], 'y': centres[i, 1],
                     'translation_error': error[i], 'gradient_error': grad_error[i],
                     'correlation': results['scores'][i], 'ring': results['rings'][i],
                     'from_moment_candidate': bool(anchors[i])})
    write_csv(root/name/'false_accepted.csv', rows)
    return {'case': root.name, 'variant': name, 'selected': int(results['selected'].sum()),
            'accepted': int(accepted.sum()), 'correct_lt2': int((accepted & (error < 2)).sum()),
            'false_ge2': int((accepted & (error >= 2)).sum()),
            'anchors': int(anchors.sum()), 'false_anchors': int((anchors & (error >= 2)).sum()),
            'median_translation_error': float(np.median(error[accepted])) if accepted.any() else None,
            'median_gradient_error': float(np.median(grad_error[accepted])) if accepted.any() else None}


def frame(image, mask, covered, title):
    view = np.repeat(np.uint8(np.clip(image, 0, 255))[..., None], 3, -1)
    y, x = np.indices(mask.shape)
    hatch = 205+25*((x+y)//4 % 2)
    view[~mask] = np.repeat(hatch[..., None], 3, -1)[~mask]
    view[mask & ~covered] = [240, 0, 160]
    picture = Image.fromarray(view[:222, :434]).resize((868, 444))
    canvas = Image.new('RGB', (868, 480), 'white')
    canvas.paste(picture, (0, 36))
    ImageDraw.Draw(canvas).text((12, 10), title, fill=(20, 35, 50))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/hudge_hole_moments_20261117'))
    args = parser.parse_args()
    root = args.root
    root.mkdir(exist_ok=True)
    summary = json.loads((root/'proposals/summary.json').read_text())
    reference, target = [np.asarray(Image.open(p)) for p in summary['input_sha256']]
    mask = material_mask(reference)
    old = np.load(summary['baseline'])
    methods = [name for name in ['control', 'moments', 'validated', 'quadratic'] if (root/name/'field.npz').exists()]
    frames, metrics, bands = {}, [], []
    reference_frame = frame(reference, mask, mask, 'Reference - hatching: excluded material / void')
    reference_frame.save(root/'reference.png')
    reference_holes = [Hole(h['component'], np.array(h['centre']), np.array(h['covariance']),
                             h['area'], h['ellipse_residual']) for h in summary['source_holes']]
    yy, xx = np.indices(reference.shape)
    pixels = np.stack((xx, yy), -1)
    hole_radius = np.stack([np.linalg.norm(pixels-h.centre, axis=-1)/h.radius for h in reference_holes])
    nearest_hole = np.argmin(hole_radius, axis=0)
    for name in methods:
        state = np.load(root/name/'seeds.npz')
        np.testing.assert_array_equal(state['parameters'][old['accepted']], old['parameters'][old['accepted']])
        field = np.load(root/name/'field.npz')
        q = pixels+field['displacement']
        remap = map_coordinates(target.astype(float), q[..., ::-1].reshape(-1, 2).T,
                                order=1, mode='constant', cval=0).reshape(reference.shape)
        covered = field['covered']
        du = [np.gradient(field['displacement'][..., k].astype(float)) for k in range(2)]
        det = (1+du[0][1])*(1+du[1][0])-du[0][0]*du[1][1]
        measured = binary_erosion(covered & mask)
        report = json.loads((root/name/'summary.json').read_text())
        metrics.append({'variant': name, 'accepted': int(state['accepted'].sum()),
                        'new_vs_archive': int((state['accepted'] & ~old['accepted']).sum()),
                        'nonpositive_field_jacobians': int((measured & (det <= 0)).sum()),
                        'times': report['times']})
        frames[name] = frame(remap, mask, covered, name + ' - magenta: unresolved material; hatching: not evaluated')
        frames[name].save(root/f'{name}.png')
        reference_frame.save(root/f'reference_{name}.gif', save_all=True,
                             append_images=[frames[name]], duration=900, loop=0)
        # Plain native images for eog; mask exported independently, never fabricate a void warp.
        native = np.uint8(np.clip(remap, 0, 255))
        native[~(covered & mask)] = reference[-1, -1]
        Image.fromarray(reference).save(root/name/'image1_ref.png')
        Image.fromarray(native).save(root/name/'image1_def_cor.png')
        Image.fromarray(np.uint8(covered & mask)*255).save(root/name/'image1_valid.png')
        for hole_index, hole in enumerate(reference_holes):
            delta = pixels-hole.centre
            angles = (np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0]))+360) % 360
            for lo, hi in [(1., 1.25), (1.25, 1.5), (1.5, 2.), (2., 3.)]:
                for angle in range(0, 360, 45):
                    region = (mask & (nearest_hole == hole_index) & (hole_radius[hole_index] >= lo)
                              & (hole_radius[hole_index] < hi) & (angles >= angle) & (angles < angle+45))
                    valid = region & covered
                    corr = None
                    if valid.sum() >= 3 and np.std(reference[valid]) > 0 and np.std(remap[valid]) > 0:
                        corr = float(np.corrcoef(reference[valid], remap[valid])[0, 1])
                    bands.append({'variant': name, 'hole': hole_index, 'r_min': lo, 'r_max': hi,
                                  'angle_min': angle, 'angle_max': angle+45, 'pixels': int(region.sum()),
                                  'covered': int(valid.sum()), 'correlation_covered': corr})
    write_csv(root/'annular_sectors.csv', bands)
    frames['control'].save(root/'control_vs_unchecked.gif', save_all=True,
                           append_images=[frames['moments']], duration=1000, loop=0)
    if 'quadratic' in frames:
        frames['control'].save(root/'control_vs_quadratic.gif', save_all=True,
                               append_images=[frames['quadratic']], duration=1000, loop=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    for axis, image, key in zip(axes, [reference, target], ['source_holes', 'target_candidates'], strict=True):
        axis.imshow(image, cmap='gray', vmin=0, vmax=255)
        axis.set_xlim(0, 434)
        axis.set_ylim(222 if key == 'source_holes' else 1024, 0)
        for i, j, _, _ in summary['matches']:
            h = summary[key][i if key == 'source_holes' else j]
            centre = np.array(h['centre'])
            values, vectors = np.linalg.eigh(h['covariance'])
            theta = np.linspace(0, 2*np.pi, 200)
            ellipse = centre[:, None] + vectors @ (2*np.sqrt(values)[:, None]*np.array([np.cos(theta), np.sin(theta)]))
            axis.plot(*ellipse, color='orangered', lw=1)
            axis.text(*centre, str(i+1), color='cyan', ha='center', va='center')
        axis.set_xlabel('x [native pixels]')
        axis.set_ylabel('y [native pixels]')
        axis.set_title('Reference: six matched holes' if key == 'source_holes' else 'Target: matched by geometry, not component index')
    fig.savefig(root/'hole_assignment.png', dpi=170)
    plt.close(fig)

    scans = np.load(root/'proposals/scans.npz')['scans']
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True, sharex=True, sharey=True)
    for i, axis in enumerate(axes.ravel()):
        records = [r for r in summary['sectors'] if r['source_hole'] == i]
        for r in records:
            a = scans[r['id']]
            axis.plot(a[:, 0], np.maximum(a[:, 1], -.2), alpha=.25)
        axis.axhline(.65, color='black', ls='--', lw=.8)
        axis.set_title(f'Hole {i+1}: {len(records)} fixed sectors')
        axis.set_ylim(-.2, 1)
        axis.set_xlabel('Moment-family angle [degrees]')
        axis.set_ylabel('Normalized correlation')
    fig.savefig(root/'angular_scores.png', dpi=160)
    plt.close(fig)

    truth_rows = []
    for twin in ['twin_affine', 'twin_nonlinear']:
        if not (root/twin/'truth_scoring_only.npz').exists():
            continue
        for name in ['moments', 'validated', 'silhouette', 'oracle']:
            if (root/twin/name/'seeds.npz').exists():
                truth_rows.append(score_truth(root/twin, name))
                twin_target = np.asarray(Image.open(root/twin/'Img_wa_g191.tif'))
                field = np.load(root/twin/name/'field.npz')
                q = pixels + field['displacement']
                warped = map_coordinates(twin_target.astype(float), q[..., ::-1].reshape(-1, 2).T,
                                         order=1, mode='constant', cval=0).reshape(reference.shape)
                picture = frame(warped, mask, field['covered'], twin+' / '+name)
                picture.save(root/f'{twin}_{name}.png')
                reference_frame.save(root/f'{twin}_{name}.gif', save_all=True,
                                     append_images=[picture], duration=900, loop=0)
    write_csv(root/'truth_scores.csv', truth_rows)
    final = {'real_case': metrics, 'twins': truth_rows,
             'warning': 'Real-case accepted counts are not correctness counts. Averaged diagnostic fields may fold.'}
    (root/'comparison.json').write_text(json.dumps(final, indent=2))
    if (root/'quadratic_proposals/summary.json').exists():
        pairs = json.loads((root/'quadratic_proposals/summary.json').read_text())['pairs']
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        a = np.array([r['affine_validation'] for r in pairs])
        b = np.array([r['quadratic_validation'] for r in pairs])
        ax.scatter(a, b, s=12, alpha=.45)
        ax.plot([-1, 1], [-1, 1], color='black', lw=1)
        ax.axhline(.65, ls='--', color='gray')
        ax.axvline(.65, ls='--', color='gray')
        ax.set(xlabel='Affine: held-out correlation', ylabel='Quadratic: held-out correlation',
               title='Same initial state and sample split; -1 = invalid support')
        fig.savefig(root/'curvature_ablation.png', dpi=170)
        plt.close(fig)
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
