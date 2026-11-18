# Reprise ciblée du bord extérieur — niveau 191

Le champ de départ est `partial12`. Son masque couvert et tous ses déplacements existants sont préservés exactement. La couronne de 16 pixels autour des trous est également préservée, qu'elle soit couverte ou non. Les tests d'égalité portent sur les tableaux natifs, pas seulement sur des images arrondies.

## Méthode

Une reprise identique sans marge est comparée à une reprise avec marge technique constante de 32 pixels, exclue du masque matériel. Cette marge seule n'apporte que peu de pixels. Les supports tronqués sont ensuite agrandis à 16, 24 et 32 pixels, avec pas 4 et centres dans les 20 pixels du bord extérieur. Une dernière grille est décalée de 2 pixels (marge 34). Toutes les reprises sont initialisées depuis les mêmes observations `partial12`, puis propagées par les liens du masque.

Les contributions sont utilisées dans cet ordre uniquement sur les pixels encore non couverts de la bande extérieure, hors couronne des trous. Il n'y a pas d'extrapolation. Le contrôle final impose que tous les échantillons matériels restent dans l'image cible originale, même lorsque les buffers sont élargis. Les coordonnées exportées sont ramenées dans le repère natif.

## Résultats

Avant filtrage supplémentaire, les reprises proposent 4 732 pixels additionnels dans la bande de 20 pixels. Le contrôle photométrique local et des nouveaux replis en rejette 1 610, en conserve 3 122.

Dans la bande extérieure de 12 pixels, les pixels manquants passent de 5 141 à 2 236 (−56,5 %). Dans les quatre premiers pixels, la couverture passe de 1 519/5 008 à 3 862/5 008. La corrélation des intensités sur les seuls ajouts de cette bande est 0,866.

La sélection additionnelle examine un voisinage matériel de 9 × 9 pixels, avec au moins 25 échantillons et une corrélation ≥ 0,65. Elle rejette aussi les nouveaux pixels produisant un déterminant non positif ou contribuant à un tel voisinage ; elle ne modifie aucun pixel précédemment couvert. Ces critères additionnels ne changent pas les seuils du solveur. Ils sont évalués sur les images ayant servi à construire le champ : ce n'est pas une validation indépendante de précision mécanique.

Les 125 déterminants non positifs préexistants du champ assemblé subsistent ; le diagnostic final n'en compte pas davantage. Ce champ reste un remapping d'illustration non qualifié comme transformation mécanique globale. Le catalogue `local_initializers.npz` conserve séparément les affines locales, centres, tailles de support et sources ; il contient plusieurs candidats et n'est pas une grille fusionnée prête à consommer sans sélection.

## Reproduction

Depuis la racine, utiliser de nouveaux dossiers de sortie :

```bash
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 12 --stride 4 --seed-archive results/hudge_boundary_frontier_20261115/partial12/seeds.npz --output /tmp/outer_control
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 12 --stride 4 --padding 32 --seed-archive results/hudge_boundary_frontier_20261115/partial12/seeds.npz --output /tmp/outer_padding32
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 16 --stride 4 --padding 32 --outer-band 20 --seed-archive results/hudge_boundary_frontier_20261115/partial12/seeds.npz --output /tmp/outer16
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 24 --stride 4 --padding 32 --outer-band 20 --seed-archive results/hudge_boundary_frontier_20261115/partial12/seeds.npz --output /tmp/outer24
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 32 --stride 4 --padding 32 --outer-band 20 --seed-archive results/hudge_boundary_frontier_20261115/partial12/seeds.npz --output /tmp/outer32
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 32 --stride 4 --padding 34 --outer-band 20 --seed-archive results/hudge_boundary_frontier_20261115/partial12/seeds.npz --output /tmp/outer32_phase2
```

`complete_hudge_outer_edge.py` combine les dossiers archivés sous `results/hudge_boundary_frontier_20261115/`, puis `publish_hudge_outer_edge.py` actualise la galerie. Les étapes de reprise, parents et statuts sont conservés dans chaque dossier. La section `steps` de `summary.json` décrit la couverture brute avant filtrage ; `bands` donne le résultat final filtré. Le champ `selected` des essais de reprise compte les fenêtres de la région d'intérêt préparée ; les fenêtres effectivement autorisées pour la reprise extérieure sont données par `seeds.npz['selected']` et les essais effectifs par les archives de propositions.

Six tests ciblés passent, notamment conservation des observations, exclusion des ajouts hors région et invariance du transport sous décalage commun des coordonnées. Ruff passe. Aucun fichier du solveur externe n'a été modifié. Les résolutions locales sont sur WebGPU ; orchestration, filtrage supplémentaire et remapping restent sur processeur central.
