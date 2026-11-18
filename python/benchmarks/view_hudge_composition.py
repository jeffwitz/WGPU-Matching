"""Inspect native remaps and individual hole crowns from a composition run."""

import argparse
import json
from pathlib import Path

import numpy as np
from experiment_hudge_hole_rbf import corr
from hudge_hole_moments import sample
from hudge_strains_data import material_mask
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    root = args.folder
    report = json.loads((root/'summary.json').read_text())
    info = json.loads(Path('results/hudge_hole_moments_20261117/proposals/summary.json').read_text())
    ref, target = [np.asarray(Image.open(p), float) for p in info['input_sha256']]
    material = material_mask(ref)[:240, :448]
    lab0, _ = label(~material_mask(ref))
    lab1, _ = label(~material_mask(target))
    target_material = material_mask(target)
    outside_ids = np.unique(np.r_[lab0[0], lab0[-1], lab0[:, 0], lab0[:, -1]])
    exterior = np.isin(lab0, outside_ids)
    outer_band = material & (distance_transform_edt(~exterior)[:240, :448] <= 8)
    def panel(image, title):
        canvas = Image.new('RGB', (900, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), title, fill='black')
        source = Image.fromarray(np.uint8(np.clip(image, 0, 255)))
        for i, h in enumerate(info['source_holes']):
            c = np.array(h['centre'])
            r = np.sqrt(h['area']/np.pi)+12
            patch = source.crop(tuple(map(int, [c[0]-r, c[1]-r, c[0]+r, c[1]+r]))).convert('RGB').resize((280, 280))
            x, y = (i % 3)*300+10, (i//3)*310+45
            canvas.paste(patch, (x, y))
            draw.text((x, y-18), f'Trou {i+1}', fill='black')
        return canvas
    reference_panel = panel(ref, 'Reference - pixels natifs, sans contour superpose')
    panels = []
    sections = []
    for row in report['rows']:
        iteration = row['iteration']
        data = np.load(root/f'iteration_{iteration:02d}.npz')
        points = data['points']
        q = points+data['parameters'][:, [0, 3]]
        warp = sample(target, q).reshape(material.shape)
        target_label = sample(lab1, q, order=0).astype(int).reshape(material.shape)
        mapped_material = (sample(target_material, q) >= .5).reshape(material.shape)
        row['material_to_background_pixels'] = int((material & ~mapped_material).sum())
        row['outer_8px_correlation'] = corr(ref[:240, :448][outer_band], warp[outer_band])
        row['outer_8px_pixels'] = int(outer_band.sum())
        row['outer_8px_to_background'] = int((outer_band & ~mapped_material).sum())
        row['intensity_rmse_8bit'] = float(np.sqrt(np.mean((ref[:240, :448][material]-warp[material])**2)))
        per_hole = []
        for i, j, _, _ in info['matches']:
            hole = info['source_holes'][i]
            radius = np.linalg.norm(points-np.array(hole['centre']), axis=1).reshape(material.shape)/np.sqrt(hole['area']/np.pi)
            crown = material & (radius >= 1) & (radius < 1.25)
            a = lab0[:240, :448] == hole['component']
            b = target_label == info['target_candidates'][j]['component']
            per_hole.append({'hole': i+1, 'crown_pixels': int(crown.sum()),
                             'crown_correlation': corr(ref[:240, :448][crown], warp[crown]),
                             'hole_iou': float((a & b).sum()/max((a | b).sum(), 1))})
        row['per_hole'] = per_hole
        after = panel(warp, f'RBF composee - iteration {iteration}')
        after.save(root/f'holes_{iteration:02d}.png')
        reference_panel.save(root/f'holes_{iteration:02d}.gif', save_all=True, append_images=[after], duration=1200, loop=0)
        panels.append(after)
        # Fixed difference scale, no contrast normalisation per iteration.
        difference = np.abs(ref[:240, :448]-warp)
        Image.fromarray(np.uint8(np.clip(difference[:225, :440]*2, 0, 255))).save(root/f'difference_{iteration:02d}.png')
        table = ''.join(f"<tr><td>{r['hole']}</td><td>{r['crown_correlation']:.3f}</td><td>{100*r['hole_iou']:.1f}%</td></tr>" for r in per_hole)
        sections.append(f'<h3>Itération {iteration}</h3><img src="holes_{iteration:02d}.gif"><table><tr><th>Trou</th><th>Corrélation couronne</th><th>Recouvrement</th></tr>{table}</table><p>Différence absolue ×2, même échelle à chaque itération :</p><img src="difference_{iteration:02d}.png">')
    panels[0].save(root/'holes_iterations.gif', save_all=True, append_images=panels[1:], duration=1200, loop=0)
    (root/'summary.json').write_text(json.dumps(report, indent=2))
    page = (root/'index.html').read_text().split('<!-- hole details -->')[0]
    page = page.replace('</html>', '')
    edges = ''.join(f"<tr><td>{r['iteration']}</td><td>{r['outer_8px_correlation']:.3f}</td><td>{r['outer_8px_to_background']}/{r['outer_8px_pixels']}</td><td>{r['intensity_rmse_8bit']:.2f}</td></tr>" for r in report['rows'])
    validation = []
    verified = []
    for path in sorted(root.glob('original_image_verification_*.json')):
        result = json.loads(path.read_text())
        verified.append(result)
        validation.append(f"<tr><td>{result['iteration']}</td><td>{result['composed_accepted']}/{result['eligible']}</td><td>{result['new_original_image_solves']}</td><td>{result['old_not_recovered']}</td><td>{result['union_preserving_old_seeds']}</td><td><a href='{path.name}'>Détail</a></td></tr>")
    validation_text = ''
    if validation:
        validation_text = '''<h2>Vérification sur les deux images originales</h2>
<p>Les six paramètres affines locaux sont extraits du champ composé et transmis au solveur de corrélation d’images numériques par Gauss–Newton à composition inverse (IC-GN). Même grille de 12 pixels et pas de 4 pixels ; au plus quatre blocs de 30 itérations, tolérance 0,001 et corrélation minimale 0,65. Les fenêtres sont également contrôlées sur la matière et les pixels originaux. Ce comptage est distinct de celui du solveur sur l’image remappée. Une acceptation photométrique ne démontre pas une erreur de déplacement faible en l’absence de vérité.</p>
<p>La référence archivée contient 3 163 fenêtres acceptées. Leur état est conservé dans l’archive fusionnée ; les anciens points non retrouvés par ce nouvel essai n’en sont pas supprimés.</p>
<table><tr><th>Champ initial, itération</th><th>Acceptées sur images originales</th><th>Nouvelles / archive</th><th>Anciennes non retrouvées</th><th>Union conservant l’archive</th><th>Données</th></tr>'''+''.join(validation)+'</table>'
    if len(verified) >= 2:
        before, after = report['rows'][0], report['rows'][-1]
        verdict = (f"<p><strong>Résultat sur la plaque trouée, image 191 :</strong> la composition améliore la corrélation de la couronne "
                   f"de {before['band_correlation'][0]:.3f} à {after['band_correlation'][0]:.3f}, "
                   f"mais les résolutions acceptées sur les images originales passent seulement de {verified[0]['composed_accepted']} "
                   f"à {verified[-1]['composed_accepted']} sur {verified[-1]['eligible']} fenêtres. "
                   "Le gain de remapping est réel ; ce test ne démontre pas un déblocage massif de l’initialisation. "
                   "Les derniers passages saturent et détériorent certains contours. Les bords extérieurs restent partiellement perdus.</p>")
        page = page.replace('</h1>', '</h1>'+verdict, 1)
    (root/'index.html').write_text(page+'<!-- hole details -->'+validation_text+'''<h2>Contrôle des bords extérieurs</h2>
<p>Une projection située dans l’image peut néanmoins tomber dans le fond : ces deux défauts sont comptés séparément. Bande matérielle de huit pixels près du bord extérieur ; erreur quadratique moyenne d’intensité sur toute la matière, en niveaux de gris 0–255.</p>
<table><tr><th>Itération</th><th>Corrélation bande extérieure</th><th>Pixels de cette bande projetés dans le fond</th><th>Erreur d’intensité</th></tr>'''+edges+'''</table>
<h2>Inspection des six couronnes</h2><p>Le trou seul ne suffit pas : chaque animation alterne les textures natives de référence et de la cible remappée. La corrélation est calculée dans la couronne matérielle 1 ≤ r/r₀ &lt; 1,25. Aucune vérité de déplacement n’est disponible.</p>'''+''.join(sections)+'</html>')
    parent = root.parent/'index.html'
    if parent.exists():
        page = parent.read_text()
        marker = '<!-- composition link -->'
        if marker not in page:
            page = page.replace('<h1>', marker+f'<p><strong>Nouveau : <a href="{root.name}/index.html">remapping, nouvelle identification et composition — images de chaque itération</a></strong></p><h1>', 1)
            parent.write_text(page)
    print(json.dumps([{'iteration': r['iteration'], 'per_hole': r['per_hole']} for r in report['rows']], indent=2))


if __name__ == '__main__':
    main()
