"""Publish boundary-frontier results with uncovered material explicitly marked."""
import json
import shutil
from pathlib import Path

import numpy as np
from hudge_strains_data import material_mask
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def main():
    root=Path('results/hudge_boundary_frontier_20261115')
    gallery=Path('results/hudge_dense_remap_20261115/near_hole_propagation')
    gallery.mkdir(exist_ok=True)
    ref=np.asarray(Image.open(root/'partial12/reference.png'))
    mask=material_mask(ref);lab,_=label(~mask)
    outside=np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]])
    dist=distance_transform_edt(~((lab>0)&~np.isin(lab,outside)))
    old=np.asarray(Image.open('results/hudge_dense_remap_20261115/refined/level_191/image2_def_cor.png'))
    summaries=[]
    for size in [16,12,8]:
        folder=root/f'partial{size}';summary=json.loads((folder/'summary.json').read_text())
        f=np.load(folder/'field.npz');covered=f['covered'];u=f['displacement']
        uy,ux=np.gradient(u,axis=(0,1));det=(1+ux[...,0])*(1+uy[...,1])-uy[...,0]*ux[...,1]
        valid=binary_erosion(covered)
        summary['assembled_field_nonpositive_determinants']=int(np.sum(valid&(det<=0)))
        image=np.asarray(Image.open(folder/'def_cor_supported.png'))
        for band in summary['bands']:
            lo,hi=band['band'];use=mask&(dist>lo)&(dist<=hi)&covered
            band['old_contour_correlation_same_pixels']=float(np.corrcoef(ref[use],old[use])[0,1])
        summaries.append(summary)
        # Holes/exterior are hatched: not evaluated, not evidence of restored circles.
        y,x=np.indices(mask.shape);background=np.where(((x+y)//4)%2,55,75).astype(np.uint8)
        frames=[]
        for raw in [ref,image]:
            rgb=np.repeat(raw[...,None],3,axis=2)
            rgb[~mask]=np.repeat(background[~mask,None],3,axis=1)
            rgb[mask&~covered]=[255,0,180]
            frames.append(Image.fromarray(rgb).crop((0,0,440,225)).resize((880,450),Image.Resampling.NEAREST))
        frames[0].save(gallery/f'partial{size}.gif',save_all=True,append_images=frames[1:],duration=900,loop=0)
        frames[1].save(gallery/f'partial{size}.png')
        shutil.copyfile(folder/'seeds.npz',gallery/f'seeds{size}.npz')
    (root/'comparison.json').write_text(json.dumps(summaries,indent=2))
    shutil.copyfile(root/'comparison.json',gallery/'comparison.json')
    rows=[]
    for summary in summaries:
        for b in summary['bands'][:3]:
            rows.append(f"<tr><td>{summary['size']}</td><td>{b['band'][0]}–{b['band'][1]}</td><td>{b['accepted_centres']}/{b['selected_centres']}</td><td>{b['covered']}/{b['pixels']}</td><td>{b['correlation_covered']:.3f}</td><td>{b['old_contour_correlation_same_pixels']:.3f}</td></tr>")
    html='''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Propagation jusqu'aux bords</title><style>body{font:17px/1.5 system-ui;max-width:1150px;margin:30px auto;padding:20px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}h2{margin-top:45px}</style>
<h1>Propagation sur supports tronqués — niveau 191</h1>
<p>Calcul sur les images originales, sans correction de contour. Les fenêtres sont limitées à la matière visible depuis leur centre. Les graines viennent de voisins résolus, avec transport des six paramètres et respect des liens du masque.</p>
<p><strong>Magenta : matière non couverte par une fenêtre retenue. Hachures : trous et extérieur non évalués.</strong> La forme des zones hachurées ne prouve aucun recalage des trous. Il faut regarder le mouchetis.</p>
<p>Les fenêtres de 12 pixels constituent le meilleur compromis de couverture dans ce pilote. La bande 0–4 pixels reste mal recalée (corrélation 0,640 sur 98,2 % des pixels). Le passage à 8 pixels réduit la couverture à 66,0 % dans cette bande : il ne résout pas le problème.</p>
<h2>Alternance référence / résultat, fenêtres de 12 pixels</h2><img src="partial12.gif" alt="Référence et remapping sur supports résolus">
<h2>Comparaison, mêmes pixels pour les deux corrélations</h2><table><tr><th>Support (px)</th><th>Distance au trou (px)</th><th>Centres retenus / admissibles</th><th>Pixels couverts / matière</th><th>Propagation</th><th>Ancienne correction de contour</th></tr>'''+''.join(rows)+'''</table>
<p>Chaque corrélation regroupe les intensités de la bande couverte, ce n'est pas un score de convergence. Les dénominateurs diffèrent entre supports.</p>
<h2>Limite de l'assemblage</h2><p>Le remapping combine les déplacements locaux seulement sur leurs supports validés. Cet assemblage reste imparfait : les différences finies détectent des déterminants non positifs dans le champ combiné (29 pixels en 16, 125 en 12), même si les affines locales retenues ont chacune un déterminant positif. Il n'est pas qualifié comme transformation matérielle globale. Les graines locales sont exportées séparément.</p>
<h2>Fenêtres de 16 pixels</h2><img src="partial16.gif" alt="Supports 16">
<h2>Fenêtres de 8 pixels — perte de couverture</h2><img src="partial8.gif" alt="Supports 8">
<p><a href="comparison.json">Résultats bruts</a> · <a href="seeds12.npz">Graines locales 12 pixels</a> · <a href="../index.html">Retour à la page précédente</a></p></html>'''
    (gallery/'index.html').write_text(html)
    page=Path('results/hudge_dense_remap_20261115/index.html')
    s=page.read_text(); marker='<section id="propagation-bords"><h2>Nouvel essai : propagation sur supports tronqués</h2><p>Sans correction de contour. Magenta : matière non résolue ; hachures : hors matière. Le bord immédiat reste insuffisamment recalé.</p><a href="near_hole_propagation/index.html">Ouvrir les résultats, les limites et les comparaisons</a><img src="near_hole_propagation/partial12.gif" alt="Propagation jusqu’aux bords"></section>'
    if 'id="propagation-bords"' not in s:s=s.replace('</h1>','</h1>'+marker,1)
    page.write_text(s)


if __name__=='__main__':main()
