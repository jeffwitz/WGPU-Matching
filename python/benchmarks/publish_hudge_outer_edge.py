"""Publish the exterior-boundary experiment next to the existing image gallery."""
import json
import shutil
from pathlib import Path


def main():
    source=Path('results/hudge_boundary_frontier_20261115/outer_completed')
    gallery=Path('results/hudge_dense_remap_20261115/near_hole_propagation/outer_edge')
    gallery.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/'summary.json').read_text())
    for name in ['before.png','after.png','reference.png','before_after.gif','reference_corrected.gif','summary.json','local_initializers.npz']:
        shutil.copyfile(source/name,gallery/name)
    rows=''.join(f"<tr><td>{r['band'][0]}–{r['band'][1]} px</td><td>{r['before']}/{r['pixels']}</td><td>{r['after']}/{r['pixels']}</td><td>{r['correlation_new_pixels']:.3f}</td></tr>" for r in report['bands'])
    page='''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Reprise des bords extérieurs</title>
<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:16px;color:#243242}img{max-width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}section{margin:35px 0}</style>
<h1>Reprise des bords extérieurs — niveau 191</h1>
<p>Les supports tronqués de 16, 24 puis 32 pixels récupèrent des observations là où les supports de 12 pixels échouent. La marge technique seule apporte peu. Un décalage de grille de deux pixels complète la reprise.</p>
<p><strong>Les déplacements déjà couverts et la couronne de 16 pixels autour des trous sont conservés à l'identique.</strong> Aucun contour n'est ajusté, aucune extrapolation ne remplit les zones perdues. Seuls les pixels couverts par des supports résolus peuvent être ajoutés.</p>
<section><h2>Avant / après, même recadrage</h2><p>Magenta : matière non couverte ou ajout rejeté. Hachures : hors matière non évaluée.</p><img src="before_after.gif" alt="Avant et après reprise des bords"></section>
<section><h2>Référence / image remappée</h2><img src="reference_corrected.gif" alt="Référence et image corrigée"></section>
<section><h2>Résultat après contrôle des nouveaux pixels</h2>
<p>Dans la bande extérieure de 12 pixels, les pixels manquants passent de <strong>5 141 à 2 236</strong>, soit une diminution de 56,5 %. Des segments restent non résolus.</p>
<table><tr><th>Distance au bord extérieur</th><th>Pixels couverts avant</th><th>Après</th><th>Corrélation sur les seuls ajouts</th></tr>'''+rows+'''</table>
<p>Ces corrélations regroupent les intensités des pixels ajoutés ; ce ne sont pas des erreurs de déplacement ni des taux de convergence.</p>
<p>Sur 4 732 pixels proposés dans la bande de 20 pixels, 3 122 sont conservés après un contrôle supplémentaire : corrélation locale ≥ 0,65 sur un voisinage 9 × 9, au moins 25 échantillons matériels et exclusion des nouveaux replis ou de leurs nouvelles contributions voisines. Les autres restent magenta. Les seuils du solveur local ne sont pas changés.</p>
</section><section><h2>Limites</h2><p>Test sur une seule paire au niveau 191. Aucun déplacement vrai n'est disponible. Le contrôle photométrique additionnel n'est pas une validation mécanique indépendante. Le champ assemblé reste un diagnostic : les 125 déterminants non positifs préexistants sont toujours présents ; aucun pixel déjà couvert n'a été corrigé par cette reprise.</p>
<p>Les résolutions locales utilisent WebGPU. L'orchestration et le remapping utilisent le processeur central. Les affines locales et leurs tailles de support sont exportées séparément : le catalogue contient plusieurs candidats et ne doit pas être interprété comme une grille unique de paramètres déjà fusionnés.</p>
<p><a href="local_initializers.npz">Initialisations locales et supports</a> · <a href="summary.json">Chiffres bruts</a> · <a href="before.png">Avant</a> · <a href="after.png">Après</a> · <a href="../index.html">Propagations précédentes</a></p></section></html>'''
    (gallery/'index.html').write_text(page)
    section='<section id="bords-exterieurs"><h2>Nouvelle reprise : bords extérieurs</h2><p>Moins de zones perdues, sans modifier le champ près des trous. Les ajouts douteux restent en magenta.</p><a href="near_hole_propagation/outer_edge/index.html">Résultats et contrôles détaillés</a><img src="near_hole_propagation/outer_edge/before_after.gif" alt="Reprise des bords extérieurs"></section>'
    page=Path('results/hudge_dense_remap_20261115/index.html')
    text=page.read_text()
    if 'id="bords-exterieurs"' not in text:text=text.replace('</h1>','</h1>'+section,1)
    page.write_text(text)
    parent=gallery.parent/'index.html';text=parent.read_text()
    if 'outer_edge/index.html' not in text:text=text.replace('</h1>','</h1><p><a href="outer_edge/index.html">Nouveau : reprise des bords extérieurs, avant/après</a></p>',1)
    parent.write_text(text)


if __name__=='__main__':main()
