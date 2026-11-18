# Plaque trouée : subdivision, courbure locale et continuation

[Rapport illustré](../../paper/hudge_adaptive_support.pdf) ·
[Paramètres](all_levels/protocol.json) · [Comparaison](comparison.json)

Deux campagnes de 764 configurations (191 cibles × quatre méthodes) ont été exécutées.
Les images sont toujours comparées directement à la même référence. Aucun déplacement
analytique ni image intermédiaire ne participe aux calculs.

## Résultats principaux

Point de départ : `hudge_rbf_restart_20261115/all_levels/rbf_gradient_g0.50`,
6 549 centres retenus sur 6 876. Les expériences portent sur les 327 anciens supports
32×32 restants. RBF désigne les fonctions de base radiale ; IC-GN désigne le solveur
Gauss–Newton à composition inverse utilisé pour la corrélation d'images numériques.

| Modèle | Anciens centres récupérés, 1 appel/proposition | Avec continuation | Points retenus avec continuation |
|---|---:|---:|---:|
| Affine 32, pas 32 | 295/327 | 327/327 | 327/327 |
| Quadratique 32, courbure initiale nulle | 181/327 | 327/327 | 327/327 |
| Quadratique 32, courbure initiale RBF | 161/327 | 327/327 | 327/327 |
| Affine 16, pas 8 | 294/327 | 327/327 | 2 843/2 943 |

La subdivision place neuf supports fins par ancienne fenêtre, aux décalages
`{-8,0,8}²`. Chaque support fin reste dans l'ancienne fenêtre ; l'ancien centre est
conservé exactement. Il y a donc 2 516 **nouveaux centres distincts** retenus, en plus
des 327 anciens centres. La surface de la zone en échec couverte par l'union des
supports fins retenus atteint 98,38 % ; elle n'est pas assimilée à 100 %.

À support identique, le quadratique augmente la corrélation médiane de 0,8463 à
0,9307 sur les 327 anciennes fenêtres. Les déplacements affine/quadratique diffèrent
de 0,703 pixel en médiane et de 2,583 pixels au maximum. Sans vérité mécanique associée,
ces écarts ne permettent pas de désigner le modèle exact.

## Ce qui change et ce qui ne change pas

- Même initialisation par champ RBF ajusté aux états acceptés de la phase précédente.
  Le quadratique peut recevoir les dérivées secondes analytiques de ce champ ; une
  variante à courbure initiale nulle isole leur effet.
- Plus de veto sur la correction du gradient. Les nouvelles sorties sont retenues
  si statut solveur 1, paramètres finis, corrélation finale ≥ 0,65, pixels dans l'image
  et déterminant de jacobienne positif sur le support échantillonné.
- Il n'y a pas la confirmation topologique indépendante de l'ancienne chaîne.
  Les résultats sont des **candidats convergés**, pas une qualification de faux positifs.
- Au plus trois tours de proposition/résolution/réajustement. Les points retenus
  alimentent l'ajustement suivant ; les anciens états restent inchangés.
- Le solveur existant limite chaque appel à 30 itérations. La campagne de continuation
  effectue jusqu'à quatre appels par proposition, en réinitialisant seulement les
  statuts 0 finis avec leur dernière sortie. Il peut donc y avoir plus de 120 itérations
  cumulées sur une fenêtre si plusieurs tours de proposition ont lieu.
- Tolérance 1e-3, interpolation bilinéaire et seuil photométrique 0,65 inchangés.
  Le solveur natif WebGPU est utilisé sans modification de Live-DIC.

L'API quadratique refuse un masque de région d'intérêt. Nous utilisons son masque
d'activation pour sélectionner exactement les mêmes supports pleins que le solveur
affine 32. Aucun support partiel n'est demandé au quadratique. Les appels ne sont
pas effectués pendant un flux actif.

## Export de graines mixtes sans continuation

`mixed_centres/level_*.npz` assemble les sorties de la campagne à un appel par
proposition : garder la phase précédente, sinon prendre la sortie affine 32 acceptée,
sinon la sortie affine 16 acceptée au **même centre**. Cette règle ne consulte aucune
vérité analytique. Elle récupère les 327 centres : 295 en support 32 et 32 en support 16.

Les fichiers enregistrent les six paramètres, les centres, `support_size` et `source` :
0 = phase précédente ; 1 = nouvel affine 32 ; 2 = affine 16 ; -1 = hors sélection.
Ce sont des sorties réellement résolues, sans interpolation de centres manquants.
L'assemblage n'a pas fait l'objet d'une nouvelle résolution. Un état obtenu sur 16 pixels
ne doit pas être présenté comme une mesure validée sur un support de 32 pixels.

## Archives et figures

- `all_levels/` : 191 × quatre modèles, un appel par proposition.
- `all_levels_continuation4/` : même série, jusqu'à quatre appels par proposition.
- `pilot_v2/`, `continuation4/` : pilotes relancés dans les campagnes complètes.
- `pilot/` : préparation interrompue par le refus du masque quadratique, aucun résultat.
- `iterations100/` : préparation refusée par la limite de 30 itérations, aucun résultat.
- `figures/` : comparaison des couvertures, carte des centres et remappages locaux 65/164.
- Chaque `level_*.npz` conserve centres, sorties, statuts, corrélations, tailles et
  régions physiques. Les fichiers `_round_*` conservent les propositions et sorties
  même refusées. Les `_block_*` enregistrent les continuations supplémentaires.

## Reproduction

Depuis la racine du dépôt, avec les données et archives de départ disponibles :

```bash
.venv/bin/python python/benchmarks/qualify_hudge_adaptive_support.py \
  --levels $(seq 1 191) --output results/hudge_adaptive_support_20261115/all_levels
.venv/bin/python python/benchmarks/qualify_hudge_adaptive_support.py \
  --levels $(seq 1 191) --continuations 4 \
  --output results/hudge_adaptive_support_20261115/all_levels_continuation4
python3 python/benchmarks/report_hudge_adaptive_support.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error hudge_adaptive_support.tex
```

Ces commandes sont celles archivées. Choisir de nouveaux dossiers pour les relances :
le programme refuse les dossiers existants. Le générateur de rapport pointe explicitement
vers les archives ci-dessus. Les temps enregistrés incluent l'orchestration et les écritures
et ne constituent pas une comparaison de performance GPU isolée.

## Vérification et portée

Treize tests ciblés passent : dérivées RBF et Hessiennes par différences finies,
reproduction affine, exclusion des états non retenus, ordre des paramètres quadratiques,
remappage analytique, maintien des centres et inclusion des supports fins.
Les 28 configurations pilotes donnent exactement les mêmes masques et paramètres
dans les campagnes complètes. Contrôle Ruff réussi. Les pages du PDF sont rendues
et inspectées visuellement.

Les résultats concernent une seule plaque. Le fichier de déplacements de l'éprouvette
en croix n'est pas utilisé et aucun taux de précision mécanique n'est revendiqué.
Les sources et empreintes sont conservées dans `sources/` et `manifest.json`.
