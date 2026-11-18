"""Residual GPU correlation and exact composition of Gaussian RBF maps.

Development experiment, not a new production solver. The original image is
sampled after composition; remapped images are used only to identify residuals.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import MixedField, corr, evaluate, fit, select_sites
from hudge_contour_field import boundary_points
from hudge_hole_moments import Hole, moment_affine, sample
from hudge_rbf_restart import SeedField
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.ndimage import label


class Composition:
    """Phi_new = Phi_old o (identity + residual); exact chain-rule Jacobian."""

    def __init__(self, initial, residuals=()):
        self.initial, self.residuals = initial, list(residuals)

    def transform(self, points):
        """Position-only path for patch scoring; same composed transformation."""
        q = np.asarray(points, float).copy()
        for field in [*reversed(self.residuals), self.initial]:
            delta = q[:, None, :]-field.sites[None, :, :]
            k = np.exp(-np.sum(delta**2, axis=2)/(2*np.asarray(field.length)[None, :]**2))
            q = q@field.matrix.T+field.translation+k@field.coefficients
        return q

    def evaluate(self, points):
        q = np.asarray(points, float).copy()
        jac = np.broadcast_to(np.eye(2), (len(q), 2, 2)).copy()
        for field in [*reversed(self.residuals), self.initial]:
            state = field.evaluate(q)
            jac = (np.eye(2) + state[:, [1, 2, 4, 5]].reshape(-1, 2, 2)) @ jac
            q += state[:, [0, 3]]
        result = np.zeros((len(q), 6))
        result[:, [0, 3]] = q-points
        result[:, [1, 2, 4, 5]] = (jac-np.eye(2)).reshape(-1, 4)
        return result


def load_field(path):
    z = np.load(path)
    base = SeedField(z['matrix'], z['translation'], np.empty((0, 2)), np.empty((0, 2)), 16.)
    return MixedField(base, z['sites'], z['coefficients'], z['length'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, default=Path('results/hudge_hole_rbf_20261117/local_holes/joint_angles_weak.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--hole-weight', type=float, default=.1)
    parser.add_argument('--resume', action='store_true', help='Continue from fully archived accepted compositions')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=args.resume)
    start = time.perf_counter()
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref, target = [np.asarray(Image.open(p), float) for p in info['input_sha256']]
    mask = material_mask(ref)
    ref_labels, _ = label(~mask)
    target_labels, _ = label(~material_mask(target))
    hole_labels = [info['target_candidates'][j]['component'] for _, j, _, _ in info['matches']]
    source_labels = [info['source_holes'][i]['component'] for i, _, _, _ in info['matches']]
    yy, xx = np.mgrid[:240, :448]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    material = mask[:240, :448]
    source_void = np.isin(ref_labels[:240, :448], source_labels)
    radius = np.stack([np.linalg.norm(pixels-np.array(h['centre']), axis=1)/np.sqrt(h['area']/np.pi)
                       for h in info['source_holes']]).min(0).reshape(material.shape)
    bands = [material & (radius >= lo) & (radius < hi) for lo, hi in [(1, 1.25), (1.25, 1.5), (1.5, 2), (2, 3)]]
    parity = (xx+yy) % 2 == 0
    initial = load_field(args.initial)
    field = Composition(initial)
    rows, steps = [], []
    frames = []
    if args.resume:
        previous = json.loads((root/'summary.json').read_text())
        if previous['initial'] != str(args.initial) or previous['hole_weight'] != args.hole_weight:
            raise ValueError('resume must preserve initial field and hole weight')
        rows, steps = previous['rows'], previous['steps']
        for step in steps:
            if step['chosen_alpha'] > 0:
                residual_field = load_field(root/f"residual_{step['iteration']:02d}.npz")
                residual_field.coefficients *= step['chosen_alpha']
                field.residuals.append(residual_field)
        for row in rows:
            image = Image.open(root/f"iteration_{row['iteration']:02d}.gif")
            image.seek(1)
            frames.append(image.convert('RGB').copy())

    def inspect(candidate):
        states = evaluate(candidate, pixels)
        q = pixels+states[:, [0, 3]]
        warp = sample(target, q).reshape(material.shape)
        void = np.isin(sample(target_labels, q, order=0).astype(int), hole_labels).reshape(material.shape)
        det = np.linalg.det(np.eye(2)+states[:, [1, 2, 4, 5]].reshape(-1, 2, 2)).reshape(material.shape)
        inside = ((q >= 0) & (q <= np.array(target.shape[::-1])-1)).all(1).reshape(material.shape)
        row = {'material_correlation': corr(ref[:240, :448][material], warp[material]),
               'band_correlation': [corr(ref[:240, :448][b], warp[b]) for b in bands],
               'hole_iou': float((void & source_void).sum()/max((void | source_void).sum(), 1)),
               'nonpositive_material_jacobians': int((material & (det <= 0)).sum()),
               'out_of_image_material': int((material & ~inside).sum()),
               'material_pixels': int(material.sum())}
        # Selection uses a fixed reference pixel population, never a changing
        # valid subset; withheld parity is descriptive, not independent truth.
        row['selection_score'] = .5*(corr(ref[:240, :448][material & parity], warp[material & parity])
                                    + corr(ref[:240, :448][bands[0] & parity], warp[bands[0] & parity]))
        row['odd_pixel_band_correlation'] = [corr(ref[:240, :448][b & ~parity], warp[b & ~parity]) for b in bands]
        return row, states, warp, det

    def save(iteration, row, states, warp, det):
        name = f'iteration_{iteration:02d}'
        Image.fromarray(np.uint8(np.clip(warp[:225, :440], 0, 255))).save(root/f'{name}.png')
        np.savez_compressed(root/f'{name}.npz', points=pixels, parameters=states, determinant=det)
        def frame(image, title):
            canvas = Image.new('RGB', (880, 480), 'white')
            canvas.paste(Image.fromarray(np.uint8(np.clip(image[:225, :440], 0, 255))).convert('RGB').resize((880, 450)), (0, 30))
            ImageDraw.Draw(canvas).text((10, 8), title, fill='black')
            return canvas
        reference = frame(ref, 'Reference')
        after = frame(warp, f'Composition RBF - iteration {iteration}')
        reference.save(root/f'{name}.gif', save_all=True, append_images=[after], duration=1100, loop=0)
        frames.append(after)
        row['iteration'] = iteration
        rows.append(row)
        (root/'summary.json').write_text(json.dumps({'initial': str(args.initial), 'rows': rows, 'steps': steps,
            'hole_weight': args.hole_weight, 'wall_s': time.perf_counter()-start,
            'limits': 'No displacement truth. GPU residual solves; CPU RBF fits and remaps. '
            'Hole shape is fitted information. Correlation is not a convergence or accuracy proof. '
            'Selection uses even material pixels; odd ones are correlated spatially. '
            'Original seed archives and previous fields are unchanged.'}, indent=2))
        print(json.dumps(row), flush=True)

    # Numerical check of composition order AND its analytic Jacobian away from
    # image interpolation. A nonzero smooth residual exercises the chain rule.
    identity = SeedField(np.eye(2), np.zeros(2), np.empty((0, 2)), np.empty((0, 2)), 8.)
    probe = MixedField(identity, np.array([[120., 100.]]), np.array([[.4, -.3]]), np.array([30.]))
    composed = Composition(initial, [probe])
    probes = np.array([[80.2, 80.3], [120.5, 105.2], [310.2, 150.8]])
    v = composed.evaluate(probes)
    expected_q = probes+probe.evaluate(probes)[:, [0, 3]]
    expected_q += initial.evaluate(expected_q)[:, [0, 3]]
    np.testing.assert_allclose(probes+v[:, [0, 3]], expected_q, atol=1e-10)
    np.testing.assert_allclose(composed.transform(probes), expected_q, atol=1e-5)
    for axis in range(2):
        d = np.eye(2)[axis]*1e-4
        fd = (composed.evaluate(probes+d)[:, [0, 3]]-composed.evaluate(probes-d)[:, [0, 3]])/2e-4
        np.testing.assert_allclose(fd, v[:, [1+axis, 4+axis]], atol=1e-6, rtol=1e-5)

    row, states, warp, det = inspect(field)
    if not args.resume:
        save(0, row, states, warp, det)
    sys.path.insert(0, '/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src')
    from live_dic import ICGNWGPU
    from live_dic.roi import unpack_support
    size, stride, half = 12, 4, 5.5
    engine = ICGNWGPU(device_type='gpu', prefer_subgroups=False, subset_size=size,
                     subset_stride=stride, interpolation='bspline', motion_model='affine', outlier_policy='disabled')
    engine.set_reference(ref[:240, :448], iterations=30, tolerance=.001, zncc_min=.65,
                         roi_mask=material, roi_policy='partial', roi_truncation='visibility')
    roi = engine.roi
    cy, cx = np.indices(roi.selected.shape)
    centres = np.column_stack((cx.ravel()*stride+half, cy.ravel()*stride+half))
    active = np.flatnonzero(roi.selected.ravel())
    supports = {int(i): unpack_support(w, size).reshape(size, size) for i, w in zip(roi.partial_indices, roi.support_words, strict=True)}
    dy, dx = np.mgrid[:size, :size].astype(float)-half
    offsets = np.column_stack((dx.ravel(), dy.ravel()))
    holes, contours = [], []
    for i, _, _, _ in info['matches']:
        h = info['source_holes'][i]
        a = Hole(h['component'], np.array(h['centre']), np.array(h['covariance']), h['area'], h['ellipse_residual'])
        contour = boundary_points(ref_labels == a.component)
        angle = np.arctan2(contour[:, 1]-a.centre[1], contour[:, 0]-a.centre[0])
        contour = contour[np.argsort(angle)]
        contours.append(contour[np.linspace(0, len(contour)-1, 48, dtype=int)])
        holes.append(a)
    sites = np.vstack((select_sites(centres[active], 400), np.vstack(contours)))
    lengths = np.r_[np.full(400, 8.), np.full(sum(map(len, contours)), 4.)]
    for iteration in range(max(r['iteration'] for r in rows)+1, args.iterations+1):
        begin = time.perf_counter()
        packed = np.zeros((len(centres), 8), np.float32)
        enabled = np.zeros(len(centres), np.uint32)
        enabled[active] = 1
        # This private GPU entry point uploads raw uint8 bytes; it does not
        # perform the public API's floating-point image conversion.
        solver_image = np.ascontiguousarray(np.clip(warp, 0, 255), dtype=np.uint8)
        result = engine._register_seeded_frame(solver_image, packed, enabled.reshape(roi.selected.shape))
        residual = np.asarray(result.parameters).reshape(-1, 6).copy()
        status = np.asarray(result.status).ravel().copy()
        keep = (status == 1) & roi.selected.ravel() & np.isfinite(residual).all(1)
        scores = np.full(len(centres), np.nan)
        # Score the proposed composition on ORIGINAL target pixels, rather than
        # accepting a match explained only by the interpolated remapped image.
        for i in np.flatnonzero(keep):
            support = supports.get(int(i), np.ones((size, size), bool)).ravel()
            p = centres[i]+offsets[support]
            r = residual[i]
            f = np.eye(2)+r[[1, 2, 4, 5]].reshape(2, 2)
            qp = centres[i]+r[[0, 3]]+offsets[support]@f.T
            q = field.transform(qp)
            scores[i] = corr(sample(ref, p), sample(target, q))
            keep[i] &= scores[i] >= .65 and np.linalg.det(f) > 0 and ((q >= 0) & (q <= np.array(target.shape[::-1])-1)).all()
        if keep.sum() < 16:
            steps.append({'iteration': iteration, 'stop': 'insufficient residual observations', 'accepted': int(keep.sum())})
            np.savez_compressed(root/f'failed_residual_{iteration:02d}.npz', centres=centres,
                                parameters=residual, status=status, accepted=keep, scores=scores)
            row = row.copy()
            row['stop'] = 'insufficient residual observations'
            save(iteration, row, states, warp, det)
            break
        q = pixels+states[:, [0, 3]]
        remapped_labels = sample(target_labels, q, order=0).astype(int).reshape(material.shape)
        virtual_points, virtual_u, hole_centres, gradients, hole_rows = [], [], [], [], []
        for a, contour, target_label in zip(holes, contours, hole_labels, strict=True):
            hy, hx = np.nonzero(remapped_labels == target_label)
            if len(hx) < 12:
                continue
            cloud = np.column_stack((hx, hy))
            c = cloud.mean(0)
            cov = (cloud-c).T@(cloud-c)/len(cloud)+np.eye(2)/12
            b = Hole(target_label, c, cov, len(cloud), 0.)
            rr = np.linalg.norm(centres-a.centre, axis=1)/a.radius
            near = keep & (rr >= 1.02) & (rr < 2.)
            if near.sum() < 6:
                continue
            angles = np.arange(-20, 20.01, .5)
            errors = []
            observed = centres[near]+residual[near][:, [0, 3]]
            for angle in angles:
                f, t = moment_affine(a, b, angle)
                errors.append(np.median(np.linalg.norm(centres[near]@f.T+t-observed, axis=1)))
            angle = angles[np.argmin(errors)]
            f, t = moment_affine(a, b, angle)
            virtual_points.append(contour)
            virtual_u.append(contour@(f-np.eye(2)).T+t)
            hole_centres.append(a.centre)
            gradients.append(f-np.eye(2))
            hole_rows.append({'hole': len(hole_rows)+1, 'angle': float(angle), 'F': f.tolist(), 't': t.tolist(), 'neighbour_residual_px': float(min(errors))})
        correction = fit(centres[keep], residual[keep], sites, lengths, identity,
                         np.vstack(virtual_points), np.vstack(virtual_u), np.array(hole_centres),
                         np.array(gradients), args.hole_weight, True)
        np.savez_compressed(root/f'residual_{iteration:02d}.npz', centres=centres, parameters=residual,
                            status=status, accepted=keep, original_target_correlation=scores,
                            matrix=correction.matrix, translation=correction.translation,
                            sites=sites, length=lengths, coefficients=correction.coefficients)
        trials = []
        best = None
        for alpha in [1., .5, .25, .125]:
            step = MixedField(identity, sites, alpha*correction.coefficients, lengths)
            trial_field = Composition(initial, [*field.residuals, step])
            trial_row, trial_states, trial_warp, trial_det = inspect(trial_field)
            trial_row['alpha'] = alpha
            trials.append(trial_row)
            safe = (trial_row['nonpositive_material_jacobians'] <= row['nonpositive_material_jacobians']
                    and trial_row['out_of_image_material'] <= row['out_of_image_material'])
            if safe and trial_row['selection_score'] > row['selection_score']+1e-5:
                if best is None or trial_row['selection_score'] > best[1]['selection_score']:
                    best = (trial_field, trial_row, trial_states, trial_warp, trial_det)
                # Backtracking: keep the first safe improving step. This avoids
                # four expensive full-image evaluations when the full step works.
                break
        steps.append({'iteration': iteration, 'solver_accepted': int(((status == 1) & roi.selected.ravel()).sum()),
                      'original_target_checked': int(keep.sum()), 'holes': hole_rows, 'trials': trials,
                      'seconds': time.perf_counter()-begin, 'chosen_alpha': best[1]['alpha'] if best else 0.})
        if best is None:
            row = row.copy()
            row['stop'] = 'no safe improving composition among tested steps'
            save(iteration, row, states, warp, det)
            break
        field, row, states, warp, det = best
        save(iteration, row, states, warp, det)
    engine.close()
    frames[0].save(root/'iterations.gif', save_all=True, append_images=frames[1:], duration=1100, loop=0)
    Image.fromarray(np.uint8(ref[:225, :440])).save(root/'image1_ref.png')
    Image.fromarray(np.uint8(np.clip(warp[:225, :440], 0, 255))).save(root/'image1_def_cor.png')
    table = ''.join(f"<tr><td>{r['iteration']}</td><td>{r['material_correlation']:.4f}</td><td>{r['band_correlation'][0]:.4f}</td><td>{100*r['hole_iou']:.2f}%</td><td>{r['nonpositive_material_jacobians']}</td><td>{r['out_of_image_material']}</td></tr>" for r in rows)
    figures = ''.join(f'<h2>Itération {r["iteration"]}</h2><img src="iteration_{r["iteration"]:02d}.gif"><p><a href="iteration_{r["iteration"]:02d}.png">Image remappée</a></p>' for r in rows)
    (root/'index.html').write_text('''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Composition des champs RBF</title>
<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:15px}img{max-width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}</style>
<h1>Remapping, nouvelle identification, composition</h1>
<p>RBF : fonctions de base radiale. À chaque itération : corrélation affine résiduelle sur GPU (processeur graphique), contrôle sur la cible originale, moments des trous remappés, ajustement RBF, puis composition Φ<sub>nouveau</sub>(x) = Φ<sub>ancien</sub>(x + δu(x)). Les jacobiennes se multiplient dans ce même ordre. Les images affichées sont toutes échantillonnées une seule fois dans la cible originale.</p>
<p>Supports de 12 pixels, pas de 4 pixels ; seuil de corrélation 0,65 inchangé. Le pas 1 est essayé, puis réduit à 1/2, 1/4 ou 1/8 si nécessaire, sur un ensemble fixe de pixels matériels ; on refuse une augmentation des plis ou des sorties d’image. Aucun déplacement analytique n’est utilisé. Prototype de développement : calcul local GPU, ajustement et composition CPU.</p>
<p>Ces corrélations ne constituent pas une mesure d’erreur de déplacement. Le recouvrement des trous participe à l’ajustement. Les anciennes graines restent intactes. La couronne est définie par 1 ≤ r/r₀ &lt; 1,25.</p>
<table><tr><th>Itération</th><th>Corrélation matière</th><th>Couronne</th><th>Recouvrement trous</th><th>Déterminants ≤0</th><th>Matière hors image</th></tr>'''+table+'''</table>
<p><a href="summary.json">Tous les essais et valeurs brutes</a> · <a href="image1_ref.png">image1_ref</a> · <a href="image1_def_cor.png">image1_def_cor</a></p>
<h2>Succession des images remappées</h2><img src="iterations.gif">'''+figures+'</html>')


if __name__ == '__main__':
    main()
