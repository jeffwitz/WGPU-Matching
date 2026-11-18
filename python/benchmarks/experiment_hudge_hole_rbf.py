"""Development experiment: inject hole moments into a Gaussian displacement fit.

All observations are image-derived. Hole affines are soft geometric hypotheses,
not measured material gradients. No solver run or documentation build is needed.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hudge_affine_restart import affine_states, fit_global
from hudge_contour_field import boundary_points
from hudge_hole_moments import Hole, moment_affine, sample
from hudge_rbf_restart import SeedField
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.linalg import lstsq
from scipy.ndimage import label, map_coordinates


def basis(points, sites, length):
    delta = points[:, None, :] - sites[None, :, :]
    lengths = np.broadcast_to(np.asarray(length), (len(sites),))
    k = np.exp(-np.sum(delta**2, axis=2)/(2*lengths[None, :]**2))
    return k, -delta*k[..., None]/lengths[None, :, None]**2


class MixedField:
    def __init__(self, base, sites, coefficients, lengths):
        self.matrix, self.translation = base.matrix, base.translation
        self.sites, self.coefficients, self.length = sites, coefficients, lengths

    def evaluate(self, points):
        values = affine_states(points, self.matrix, self.translation)
        k, dk = basis(points, self.sites, self.length)
        values[:, [0, 3]] += k @ self.coefficients
        values[:, [1, 2, 4, 5]] += np.einsum('nki,ko->noi', dk, self.coefficients).reshape(-1, 4)
        return values


def select_sites(points, count):
    chosen, distance = [0], np.full(len(points), np.inf)
    for _ in range(min(count, len(points))-1):
        distance = np.minimum(distance, np.sum((points-points[chosen[-1]])**2, axis=1))
        chosen.append(int(np.argmax(distance)))
    return points[chosen]


def fit(centres, states, sites, length, base, virtual_points, virtual_u,
        hole_centres, hole_gradients, weight, use_gradient):
    # Group-normalised objective. A hole group has weight w against the entire
    # measured seed group, regardless of how many virtual points are sampled.
    k, dk = basis(centres, sites, length)
    residual = states-base.evaluate(centres)
    root = 1/np.sqrt(len(centres))
    blocks = [root*k, root*4*dk[..., 0], root*4*dk[..., 1]]
    targets = [root*residual[:, [0, 3]], root*4*residual[:, [1, 4]], root*4*residual[:, [2, 5]]]
    if weight:
        kh, _ = basis(virtual_points, sites, length)
        s = np.sqrt(weight/len(virtual_points))
        blocks.append(s*kh)
        targets.append(s*(virtual_u-base.evaluate(virtual_points)[:, [0, 3]]))
        if use_gradient:
            _, dh = basis(hole_centres, sites, length)
            g0 = base.evaluate(hole_centres)[:, [1, 2, 4, 5]].reshape(-1, 2, 2)
            s = 4*np.sqrt(weight/len(hole_centres))
            for axis in range(2):
                blocks.append(s*dh[..., axis])
                targets.append(s*(hole_gradients-g0)[:, :, axis])
    blocks.append(np.sqrt(1e-7)*np.eye(len(sites)))
    targets.append(np.zeros((len(sites), 2)))
    coeff = lstsq(np.vstack(blocks), np.vstack(targets), lapack_driver='gelsy')[0]
    return MixedField(base, sites, coeff, length)


def evaluate(field, points):
    return np.concatenate([field.evaluate(points[i:i+1024]) for i in range(0, len(points), 1024)])


def corr(a, b):
    a, b = a.astype(float)-a.mean(), b.astype(float)-b.mean()
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--length', type=float, default=16)
    parser.add_argument('--local-holes', action='store_true',
                        help='Add short-range bases at observed hole contours; same basis for all variants')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    prior = Path('results/hudge_hole_moments_20261117/proposals')
    info = json.loads((prior/'summary.json').read_text())
    scans = np.load(prior/'scans.npz')['scans']
    reference, target = [np.asarray(Image.open(p), float) for p in info['input_sha256']]
    mask, target_mask = material_mask(reference), material_mask(target)
    components, _ = label(~mask)
    target_components, _ = label(~target_mask)
    target_holes = np.isin(target_components, [info['target_candidates'][j]['component']
                                              for _, j, _, _ in info['matches']])
    archive = np.load('results/hudge_boundary_frontier_20261115/partial12/seeds.npz')
    centres = archive['centres'][archive['accepted']]
    states = archive['parameters'][archive['accepted']]
    f0, t0, _ = fit_global(centres, states, np.ones(len(centres), bool))
    base = SeedField(f0, t0, np.empty((0, 2)), np.empty((0, 2)), args.length)
    holes, constraints, diagnostics = [], {'photo': [], 'neighbours': []}, []
    for source_id, target_id, _, _ in info['matches']:
        def make(data):
            return Hole(data['component'], np.asarray(data['centre']), np.asarray(data['covariance']),
                        data['area'], data['ellipse_residual'])
        source = make(info['source_holes'][source_id])
        destination = make(info['target_candidates'][target_id])
        contour = boundary_points(components == source.component)
        # Equal angular sampling of the OBSERVED source contour, not a perfect circle.
        angles = np.arctan2(contour[:, 1]-source.centre[1], contour[:, 0]-source.centre[0])
        contour = contour[np.argsort(angles)]
        contour = contour[np.linspace(0, len(contour)-1, 48, dtype=int)]
        radius = np.linalg.norm(centres-source.centre, axis=1)/source.radius
        near = (radius >= 1.02) & (radius <= 2)
        if near.sum() < 6:
            raise ValueError('insufficient neighbouring measured windows for a hole')
        observed_q = centres[near] + states[near][:, [0, 3]]
        trial_angles = np.arange(-180, 180, .5)
        errors = []
        for angle in trial_angles:
            f, t = moment_affine(source, destination, angle)
            errors.append(np.median(np.linalg.norm(centres[near]@f.T+t-observed_q, axis=1)))
        neighbour_angle = float(trial_angles[np.argmin(errors)])
        ids = [r['id'] for r in info['sectors'] if r['source_hole'] == source_id and r['radial_range'][0] == 1.02]
        pooled_scores = scans[ids, :, 1].mean(0)
        photo_angle = float(scans[0, np.argmax(pooled_scores), 0])
        item = {'hole': source_id+1, 'reference_centre': source.centre.tolist(),
                'target_centre': destination.centre.tolist(), 'neighbour_count': int(near.sum()),
                'neighbour_angle': neighbour_angle, 'neighbour_median_residual_px': min(errors),
                'photo_angle': photo_angle, 'photo_mean_sector_score': float(pooled_scores.max())}
        for name, angle in [('photo', photo_angle), ('neighbours', neighbour_angle)]:
            f, t = moment_affine(source, destination, angle)
            constraints[name].append((contour, contour@(f-np.eye(2)).T+t, f-np.eye(2)))
            item[name+'_matrix'] = f.tolist()
            item[name+'_translation'] = t.tolist()
        holes.append((source, destination))
        diagnostics.append(item)
    hole_centres = np.array([a.centre for a, _ in holes])
    # Same sites and regularisation in every comparison.
    pool = np.vstack((centres, hole_centres, np.vstack([v[0] for v in constraints['neighbours']])))
    sites = np.vstack((select_sites(pool, 320), hole_centres))
    lengths = np.full(len(sites), args.length)
    if args.local_holes:
        local_sites = np.vstack([v[0] for v in constraints['neighbours']])
        sites = np.vstack((sites, local_sites))
        lengths = np.r_[lengths, np.full(len(local_sites), 4.)]
    configs = [('seeds_only', 0., 'centres', False), ('hole_centres', 1., 'centres', False),
               ('photo_affines', 1., 'photo', True)]
    configs += [(f'neighbour_affines_w{w:g}', w, 'neighbours', True) for w in [.1, 1., 10.]]
    configs += [('neighbour_positions_only', 1., 'neighbours', False)]
    yy, xx = np.mgrid[:225, :440]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    ref_crop, material = reference[:225, :440], mask[:225, :440]
    hole_distances = np.stack([np.linalg.norm(points-a.centre, axis=1)/a.radius for a, _ in holes])
    nearest = np.argmin(hole_distances, axis=0)
    frames, rows = {}, []
    source_holes = (components[:225, :440] > 0) & np.isin(components[:225, :440], [a.component for a, _ in holes])
    roi = np.zeros_like(material)
    roi[4:215, 4:427] = True
    for name, weight, kind, with_gradient in configs:
        if kind == 'centres':
            virtual_points = hole_centres
            virtual_u = np.array([b.centre-a.centre for a, b in holes])
            gradients = np.zeros((len(holes), 2, 2))
        else:
            virtual_points = np.vstack([c[0] for c in constraints[kind]])
            virtual_u = np.vstack([c[1] for c in constraints[kind]])
            gradients = np.stack([c[2] for c in constraints[kind]])
        field = fit(centres, states, sites, lengths, base, virtual_points, virtual_u,
                    hole_centres, gradients, weight, with_gradient)
        values = evaluate(field, points)
        q = points+values[:, [0, 3]]
        warped = sample(target, q).reshape(ref_crop.shape)
        inside = ((q >= 0) & (q <= np.array(target.shape[::-1])-1)).all(1).reshape(ref_crop.shape)
        predicted_void = (sample(target_holes, q).reshape(ref_crop.shape) >= .5) & roi & inside
        intersection = (predicted_void & source_holes).sum()
        union = (predicted_void | source_holes).sum()
        det = np.linalg.det(np.eye(2)+values[:, [1, 2, 4, 5]].reshape(-1, 2, 2)).reshape(material.shape)
        error = evaluate(field, centres)-states
        row = {'variant': name, 'weight': weight, 'length': args.length,
               'seed_displacement_rms': float(np.sqrt(np.mean(np.sum(error[:, [0, 3]]**2, axis=1)))),
               'seed_gradient_rms': float(np.sqrt(np.mean(np.sum(error[:, [1, 2, 4, 5]]**2, axis=1)))),
               'material_correlation': corr(ref_crop[material], warped[material]),
               'hole_iou': float(intersection/max(union, 1)),
               'nonpositive_material_jacobians': int((material & (det <= 0)).sum()),
               'out_of_image_material': int((material & ~inside).sum()), 'bands': []}
        for lo, hi in [(1., 1.25), (1.25, 1.5), (1.5, 2.), (2., 3.)]:
            r = hole_distances[nearest, np.arange(len(points))].reshape(material.shape)
            band = material & (r >= lo) & (r < hi)
            row['bands'].append({'radius': [lo, hi], 'pixels': int(band.sum()),
                                 'correlation': corr(ref_crop[band], warped[band])})
        rows.append(row)
        np.savez_compressed(root/f'{name}.npz', matrix=field.matrix, translation=field.translation,
                            sites=field.sites, coefficients=field.coefficients, length=field.length,
                            points=points, parameters=values)
        Image.fromarray(np.uint8(np.clip(warped, 0, 255))).save(root/f'{name}.png')
        picture = Image.fromarray(np.uint8(np.clip(warped, 0, 255))).convert('RGB').resize((880, 450))
        canvas = Image.new('RGB', (880, 484), 'white')
        canvas.paste(picture, (0, 34))
        ImageDraw.Draw(canvas).text((10, 10), name+' - proposed RBF warp, not a new DIC measurement', fill='black')
        frames[name] = canvas
        print(json.dumps({k: v for k, v in row.items() if k != 'bands'}), flush=True)
    ref_picture = Image.fromarray(np.uint8(ref_crop)).convert('RGB').resize((880, 450))
    ref_frame = Image.new('RGB', (880, 484), 'white')
    ref_frame.paste(ref_picture, (0, 34))
    ImageDraw.Draw(ref_frame).text((10, 10), 'Reference - actual hole shapes, no overlays', fill='black')
    ref_picture.save(root/'reference.png')
    for name, picture in frames.items():
        ref_frame.save(root/f'reference_{name}.gif', save_all=True, append_images=[picture], duration=1000, loop=0)
        frames['seeds_only'].save(root/f'baseline_{name}.gif', save_all=True, append_images=[picture], duration=1000, loop=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for ax, item in zip(axes.ravel(), diagnostics, strict=True):
        source, destination = holes[item['hole']-1]
        ax.imshow(reference, cmap='gray', vmin=0, vmax=255)
        ax.set_xlim(source.centre[0]-2*source.radius, source.centre[0]+2*source.radius)
        ax.set_ylim(source.centre[1]+2*source.radius, source.centre[1]-2*source.radius)
        radius = np.linalg.norm(centres-source.centre, axis=1)/source.radius
        near = (radius >= 1.02) & (radius <= 2)
        ax.scatter(*centres[near].T, s=5, color='cyan')
        ax.set_title(f"Hole {item['hole']}: texture {item['photo_angle']:.0f} deg / neighbours {item['neighbour_angle']:.1f} deg\n"
                     f"{item['neighbour_count']} measured windows; shape-family residual {item['neighbour_median_residual_px']:.1f} px", fontsize=10)
    fig.savefig(root/'angle_supports.png', dpi=150)
    plt.close(fig)
    report = {'hole_affines': diagnostics, 'variants': rows, 'seconds': time.perf_counter()-started,
              'local_hole_bases': args.local_holes, 'basis_count': len(sites),
              'objective': 'mean seed position residual squared + 4^2 mean seed gradient residual squared '
              '+ w mean virtual hole position residual squared + optional w*4^2 mean hole-centre gradient residual squared '
              '+ 1e-7 sum coefficient squared',
              'limits': 'Hole gradients are virtual extensions into voids, not material measurements. '
              'Hole overlap is fitted information, not independent validation. No clipping of displacement-gradient magnitude. '
              'CPU prototype; old accepted seeds and previous results unchanged.'}
    (root/'summary.json').write_text(json.dumps(report, indent=2))
    table=''.join(f"<tr><td>{r['variant']}</td><td>{r['seed_displacement_rms']:.2f}</td><td>{r['material_correlation']:.3f}</td><td>{r['hole_iou']:.3f}</td><td>{r['nonpositive_material_jacobians']}</td></tr>" for r in rows)
    sections=''.join(f'<h2>{name}</h2><p>Référence puis cible remappée ; les trous affichés proviennent de la cible, sans masque circulaire ajouté.</p><img src="reference_{name}.gif"><p><a href="{name}.png">Image remappée seule</a> · <a href="baseline_{name}.gif">RBF de référence / variante</a></p>' for name in frames)
    page='''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Essai RBF avec affines des trous</title>
<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:15px;color:#233242}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}h2{margin-top:36px}</style>
<h1>Influence des affines des trous sur le champ RBF</h1>
<p>RBF : fonctions de base radiale gaussiennes. Essai de développement sur la plaque 191, mêmes 3 163 fenêtres déjà validées dans chaque ajustement. Aucun nouveau calcul de corrélation d'images numériques (DIC). Les formes contraignent le champ ; leur recouvrement n'est pas une preuve indépendante de justesse.</p>
<p>L'angle choisi uniquement par la texture est incertain. Une seconde estimation utilise les positions des fenêtres validées dans la couronne. Les deux hypothèses sont conservées pour comparaison. Le champ peut être faux entre observations, même si les trous paraissent bien corrigés.</p>
<table><tr><th>Variante</th><th>Écart aux graines (px)</th><th>Corrélation matière</th><th>Intersection/union des trous</th><th>Déterminants ≤0 en matière</th></tr>'''+table+'''</table>
<p>Les gradients issus des trous sont des extensions virtuelles dans le vide, pas des mesures dans la matière. Aucune borne n'est appliquée à leur amplitude. Toutes les images viennent des pixels natifs ; les sorties précédentes sont conservées.</p>
<p><a href="summary.json">Matrices affines de chaque trou et chiffres bruts</a> · <a href="reference.png">Référence</a></p><img src="angle_supports.png">'''+sections+'</html>'
    (root/'index.html').write_text(page)


if __name__ == '__main__':
    main()
