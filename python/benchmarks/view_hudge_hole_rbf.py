"""Small development image viewer; no Sphinx or CI build."""

import json
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import corr
from hudge_hole_moments import Hole, moment_affine, sample
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.ndimage import label


def main():
    root = Path('results/hudge_hole_rbf_20261117')
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    reference, target = [np.asarray(Image.open(p), float) for p in info['input_sha256']]
    material = material_mask(reference)[:225, :440]
    labels, _ = label(~material_mask(target))
    target_holes = np.isin(labels, [info['target_candidates'][j]['component'] for _, j, _, _ in info['matches']])
    labels0, _ = label(~material_mask(reference))
    source_holes = np.isin(labels0[:225, :440], [h['component'] for h in info['source_holes']])
    reports = {}
    for name in ['length16', 'local_holes']:
        reports[name] = json.loads((root/name/'summary.json').read_text())
        for row in reports[name]['variants']:
            archive = np.load(root/name/(row['variant']+'.npz'))
            q = archive['points']+archive['parameters'][:, [0, 3]]
            predicted = (sample(target_holes, q) >= .5).reshape(source_holes.shape)
            row['hole_iou'] = float((predicted & source_holes).sum()/max((predicted | source_holes).sum(), 1))
        (root/name/'summary.json').write_text(json.dumps(reports[name], indent=2))
        # Refresh the existing experiment table after defining overlap on matched
        # internal holes only (not the exterior or flat speckle components).
        page = (root/name/'index.html').read_text()
        table = '<table><tr><th>Variante</th><th>Écart aux graines (px)</th><th>Corrélation matière</th><th>Recouvrement des trous</th><th>Déterminants ≤0</th></tr>'
        for r in reports[name]['variants']:
            table += f"<tr><td>{r['variant']}</td><td>{r['seed_displacement_rms']:.2f}</td><td>{r['material_correlation']:.3f}</td><td>{r['hole_iou']:.3f}</td><td>{r['nonpositive_material_jacobians']}</td></tr>"
        begin, end = page.index('<table>'), page.index('</table>')+len('</table>')
        (root/name/'index.html').write_text(page[:begin]+table+'</table>'+page[end:])
    local = reports['local_holes']
    variants = local['variants'].copy()
    joint_name = 'joint_angles_weak' if (root/'local_holes/joint_angles_weak.json').exists() else 'joint_angles'
    joint_path = root/'local_holes'/f'{joint_name}.json'
    joint = json.loads(joint_path.read_text()) if joint_path.exists() else None
    if joint:
        archive = np.load(root/'local_holes'/f'{joint_name}.npz')
        radius = np.stack([np.linalg.norm(archive['points']-np.array(h['centre']), axis=1)/np.sqrt(h['area']/np.pi)
                           for h in info['source_holes']]).min(0).reshape(material.shape)
        q = archive['points']+archive['parameters'][:, [0, 3]]
        warped = sample(target, q).reshape(material.shape)
        joint['bands'] = []
        for lo, hi in [(1., 1.25), (1.25, 1.5), (1.5, 2.), (2., 3.)]:
            valid = material & (radius >= lo) & (radius < hi)
            joint['bands'].append({'radius': [lo, hi], 'correlation': corr(reference[:225, :440][valid], warped[valid])})
        variants.append(joint)
        joint_path.write_text(json.dumps(joint, indent=2))
    affines = []
    for k, (i, j, _, _) in enumerate(info['matches']):
        def make(h):
            return Hole(h['component'], np.array(h['centre']), np.array(h['covariance']), h['area'], h['ellipse_residual'])
        a, b = make(info['source_holes'][i]), make(info['target_candidates'][j])
        angle = joint['per_hole_validation'][k]['angle'] if joint else local['hole_affines'][k]['neighbour_angle']
        f, t = moment_affine(a, b, angle)
        affines.append({'hole': k+1, 'angle_degrees': angle, 'F': f.tolist(), 't': t.tolist(),
                        'source_centroid': a.centre.tolist(), 'target_centroid': b.centre.tolist(),
                        'angle_source': 'RBF material photometry' if joint else 'accepted neighbouring windows',
                        'interpretation': 'Estimated affine shape transport of the hole, not measured material gradient.'})
    (root/'identified_hole_affines.json').write_text(json.dumps(affines, indent=2))
    Image.fromarray(np.uint8(reference[:225, :440])).save(root/'image1_ref.png')
    chosen = joint_name if joint else 'neighbour_affines_w0.1'
    Image.open(root/'local_holes'/f'{chosen}.png').save(root/'image1_def_cor.png')
    # Enlarged per-hole crops without any circular template overlaid on the warp.
    def crops(image, title):
        canvas = Image.new('RGB', (900, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), title, fill='black')
        for i, h in enumerate(info['source_holes']):
            c = np.array(h['centre'])
            radius = np.sqrt(h['area']/np.pi)+12
            box = tuple(map(int, [c[0]-radius, c[1]-radius, c[0]+radius, c[1]+radius]))
            patch = image.crop(box).convert('RGB').resize((280, 280))
            x, y = (i % 3)*300+10, (i//3)*310+45
            canvas.paste(patch, (x, y))
            draw.text((x, y-18), f'Trou {i+1}', fill='black')
        return canvas
    before = crops(Image.fromarray(np.uint8(reference)), 'Référence - mêmes régions matérielles')
    for name in ['seeds_only', 'neighbour_affines_w0.1', 'neighbour_affines_w1', chosen]:
        after = crops(Image.open(root/'local_holes'/f'{name}.png'), 'Champ proposé : '+name)
        before.save(root/f'holes_{name}.gif', save_all=True, append_images=[after], duration=1000, loop=0)
    rows = ''.join(f"<tr><td>{r['variant']}</td><td>{100*r['hole_iou']:.1f} %</td><td>{r['material_correlation']:.3f}</td><td>{r['bands'][0]['correlation']:.3f}</td><td>{r['seed_displacement_rms']:.2f}</td><td>{r['nonpositive_material_jacobians']}</td></tr>" for r in variants)
    affine_rows = ''.join(f"<tr><td>{h['hole']}</td><td>{h['angle_degrees']:.1f}°</td><td><pre>{np.array2string(np.array(h['F']),precision=3)}</pre></td><td>{np.round(h['t'],2).tolist()}</td></tr>" for h in affines)
    page = '''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Trous et identification RBF — essai en cours</title>
<style>body{font:17px/1.5 system-ui;max-width:1150px;margin:30px auto;padding:15px;color:#243242}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}h2{margin-top:35px}pre{margin:0}</style>
<h1>Les affines des trous apportent-elles quelque chose au champ RBF ?</h1>
<p>Essai de développement, plaque 191. RBF signifie fonctions de base radiale. Les mêmes 3 163 fenêtres validées servent à tous les ajustements. Les affines des trous ajoutent des contraintes souples de position sur leur frontière et, selon la variante, de gradient au centre du vide. Aucun seuil ne borne l'amplitude du gradient.</p>
<p><strong>Les trous affichés proviennent réellement du remapping de la cible : aucun cercle ni masque de référence n'est plaqué sur ces images.</strong> Le champ dans le vide est une extension géométrique, pas une mesure de matière. Aucun nouveau calcul de corrélation d'images numériques n'a été effectué.</p>
<h2>Effet sur les six trous</h2><p>Référence puis image remappée. Examiner aussi la texture au voisinage immédiat des trous.</p><img src="holes_'''+chosen+'''.gif">
<h2>Comparaison sur la même base RBF localisée</h2><p>Gaussiennes de largeur 16 pixels dans la plaque, complétées par des gaussiennes de largeur 4 pixels près des trous. La corrélation de la couronne utilise 1 ≤ r/r₀ &lt; 1,25, avec r₀ rayon du disque de même aire. Le recouvrement est l'intersection divisée par l'union des six trous appariés ; il fait partie des données ajustées.</p>
<table><tr><th>Variante</th><th>Recouvrement trous</th><th>Corrélation matière</th><th>Corrélation couronne</th><th>Écart aux graines (px)</th><th>Déterminants ≤0 en matière</th></tr>'''+rows+'''</table>
<p>Un recouvrement élevé ne démontre pas un bon transport du mouchetis. L'écart aux graines mesure l'effet du réajustement sur leurs déplacements ; ce n'est pas une erreur à une vérité analytique. Les résultats précédents restent intacts.</p>
<h2>Champ complet</h2><img src="local_holes/reference_'''+chosen+'''.gif">
<p><a href="image1_ref.png">image1_ref</a> · <a href="image1_def_cor.png">image1_def_cor</a> — mêmes dimensions pour un défilement dans eog.</p>
<h2>Transformations affines estimées des trous</h2><p>Convention : y = F x + t. La forme fournit cinq contraintes ; l'angle restant est estimé à partir du voisinage, puis éventuellement affiné par la photométrie du champ RBF. Ces matrices ne sont pas des mesures du gradient dans la matière de la couronne.</p>
<table><tr><th>Trou</th><th>Angle normalisé</th><th>F</th><th>t [px]</th></tr>'''+affine_rows+'''</table>
<p><a href="identified_hole_affines.json">Matrices et centroïdes</a> · <a href="local_holes/summary.json">Chiffres des variantes fixes</a> · <a href="local_holes/'''+joint_name+'''.json">Affinement des angles et contrôle par trou</a></p>
<h2>Variantes à inspecter</h2><p><a href="local_holes/index.html">Toutes les variantes localisées, y compris les échecs</a> · <a href="length16/index.html">Gaussiennes de 16 pixels seules</a></p>
<p>Référence / RBF sans information des trous :</p><img src="holes_seeds_only.gif">
<p>Référence / faible pondération des affines des trous :</p><img src="holes_neighbour_affines_w0.1.gif">
<p>Référence / pondération plus forte :</p><img src="holes_neighbour_affines_w1.gif">
</html>'''
    (root/'index.html').write_text(page)


if __name__ == '__main__':
    main()
