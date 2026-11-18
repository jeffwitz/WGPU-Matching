# Diagnostic visuel des reprises affines — 15 novembre 2026

Document principal : [PDF explicatif](../../paper/hudge_remap_diagnostic.pdf),
[source LaTeX](../../paper/hudge_remap_diagnostic.tex).

Les figures montrent les **mouchetis réellement remappés**, y compris les propositions
non retenues. Les cartes de couverture ne remplacent pas les images.

## Résultat

| Niveau cible | Avant | Après | Restantes avec au moins un essai natif accepté | Dernier rejet par gradient |
|---|---:|---:|---:|---:|
| 50 | 26/36 | 34/36 | 2/2 | 2 |
| 150 | 21/36 | 27/36 | 8/9 | 7 |
| 191 | 18/36 | 18/36 | 13/18 | 7 |

Un statut accepté par le solveur ne signifie pas une acceptation par la propagation.
La fenêtre 102 au niveau 191 illustre ce cas : corrélation finale 0,889,
correction du déplacement 4,96 pixels, correction du gradient 0,403, rejet au seuil
de cohérence du gradient 0,25. D'autres fenêtres ne sont jamais essayées lors de
la reprise. Ne pas les interpréter comme des échecs photométriques démontrés.

Ces corrélations sont recalculées sur les images originales après résolution.
Elles ne prouvent pas la précision des déplacements. Le fichier de déplacements
de l'éprouvette en croix n'est pas utilisé : il ne correspond pas à cette plaque.

## Fichiers

- `figures/sequence_*.pdf` et `.png` : référence, proposition silhouette, remappages
  globaux de chaque tour, résidus absolus à échelle fixe.
- `figures/patch_*.pdf` et `.png` : agrandissements, proposition locale, sortie et
  éventuel état finalement retenu. Une sortie non retenue reste visible.
- `figures/map_*.pdf` : position et identifiant des fenêtres sur le mouchetis.
- `images/NNN_00.png` : référence complète 1024 × 1024, puis `01` : silhouette,
  `02` : affine du tour 1, `03` si présent : affine du tour 2.
  Chaque rectification repart de la cible originale.
- `windows_*.json` : historique des propositions natives, sorties et décisions
  pour les 36 fenêtres éligibles de chaque paire.
- `diagnostic.json` : synthèse et données des exemples illustrés.
- `recorded/rectified_native/trace_*/` : appels du solveur, statuts, itérations,
  états avant/après chaque tour et images rectifiées effectivement utilisées.
- `manifest.json` : empreintes des sources, entrées, traces et livrables.

Le dossier `recorded/` contient le traçage complet, incluant la composition des
paramètres. Les premières traces situées à la racine sont conservées mais ne sont
pas celles utilisées pour produire ce document.

## Reproduction

Depuis la racine du dépôt, choisir **un nouveau dossier** de sortie pour une relance :

```bash
.venv/bin/python python/benchmarks/qualify_hudge_affine_restart.py \
  --levels 50 150 191 --trace --methods rectified_native \
  --output results/hudge_remap_diagnostic_20261115/recorded
python3 python/benchmarks/diagnose_hudge_remaps.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error hudge_remap_diagnostic.tex
```

La première commande ci-dessus est la commande archivée, pas une invitation à
écraser ses résultats. Le générateur de figures utilise les chemins archivés.
Il vérifie par assertions l'égalité exacte des paramètres et masques avec les
résultats antérieurs de `hudge_affine_restart_20261115/all_levels`.

Contrôles : trois relances identiques aux archives, sept tests des helpers et du
benchmark réussis, contrôle Ruff des deux scripts modifiés réussi. Le solveur
externe et les seuils n'ont pas été modifiés. Le temps des relances instrumentées
inclut les écritures de traces et n'est pas un benchmark de performance.

Les traces affichent le dernier motif de rejet, pas une cause exhaustive de tous
les essais. Code 0 : aucun motif enregistré à cette étape ; 1 : solveur non accepté
ou résultat non fini ; 2 : correction du déplacement ; 3 : correction du gradient.
Les essais ne sont pas des observations indépendantes : plusieurs concernent la
même fenêtre. Les totaux du tableau comptent des fenêtres uniques.
