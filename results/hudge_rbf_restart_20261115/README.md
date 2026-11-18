# Reprise affine puis RBF avec contraintes de gradient

[Document illustré, sept pages](../../paper/hudge_rbf_restart.pdf) ·
[Résultats agrégés](RESULTS.md) · [Paramètres figés](all_levels/protocol.json)

RBF signifie fonctions de base radiale. IC-GN signifie Gauss–Newton à composition
inverse, le solveur local de corrélation d'images numériques.

## Ce qui a réellement été exécuté

191 cibles de la plaque trouée, comparées directement à `Img_F.tif`, sans séquence
temporelle. Trois représentations et trois seuils : **1 719 configurations**.
Les 36 fenêtres par paire restent identiques. Aucun fichier de déplacement de
l'éprouvette en croix n'est employé : aucune vérité mécanique associée n'est disponible.

Chaque variante part des mêmes états archivés de
`hudge_affine_restart_20261115/all_levels/rectified_native` : 5 804 fenêtres retenues.
La nouvelle étape propose un champ à tous les centres (pas de sous-grille), résout,
confirme et propage, puis réajuste aux nouvelles observations. L'affine de contrôle
utilise la même densité d'essais. Les états déjà acceptés sont immuables.

| Proposition | Seuil gradient 0,25 | 0,50 | 1,00 |
|---|---:|---:|---:|
| Affine | 6 035 | 6 549 | 6 762 |
| RBF déplacements | 6 056 | 6 549 | 6 772 |
| RBF déplacements + gradients | 6 056 | 6 549 | 6 772 |

Les deux RBF donnent exactement les mêmes masques d'acceptation, pas seulement les
mêmes totaux. Les états résolus diffèrent légèrement (écart absolu maximal sur un
coefficient : 0,00232). Comparées à la nouvelle affine, les RBF augmentent le nombre
retenu sur 16 paires au seuil 0,25 et 10 paires au seuil 1,00, sans baisse sur une paire.
Ne pas attribuer à la RBF le gain provenant de la densification des essais ou du seuil.

## Formulation et choix fixés

Le déplacement est une affine robuste plus un résidu de gaussiennes centrées sur
les fenêtres acceptées, longueur 64 pixels (deux pas). Sa dérivée analytique fournit
les quatre composantes du gradient de déplacement : elles ne sont pas interpolées
séparément. La variante à gradients minimise simultanément les erreurs de position
et de gradient, avec une longueur de pondération de 16 pixels et une pénalisation
quadratique des coefficients de 1e-4. Ces valeurs n'ont pas été optimisées sur les
résultats. Les contraintes de gradient sont souples, pas des égalités exactes.

Les prédictions dont la jacobienne a un déterminant non positif ou un conditionnement
supérieur à 100 sont exclues des essais initiaux. L'algorithme ne déclare jamais une
fenêtre valide parce qu'elle appartient au champ interpolé : elle passe par le
solveur et la confirmation. Les règles de propagation et de confirmation existantes
sont conservées, seul leur paramètre de seuil de gradient varie.

L'ajustement utilise le processeur central, sur au plus 36 sites. IC-GN reste sur
WebGPU. Le modèle RBF initialise directement les six paramètres sur les pixels
originaux. Les remappages sont des diagnostics visuels, pas les entrées du solveur.
Il ne s'agit ni d'une nouvelle API publique ni d'un pipeline entièrement résident GPU.

## Contrôle hors ajustement

Pour chaque observation de la phase affine, retirer ensemble son déplacement et
son gradient, réajuster sur les autres observations puis prédire le site retiré.
5 804 sites contrôlés pour chaque méthode ; aucune sélection d'hyperparamètres.

| Modèle | Erreur quadratique déplacement (px) | Erreur quadratique gradient |
|---|---:|---:|
| Affine | 4,774 | 0,2410 |
| RBF déplacements | 2,092 | 0,1465 |
| RBF déplacements et gradients | 1,755 | 0,1393 |

Ce contrôle montre un intérêt des gradients pour prédire les **observations du
solveur**. Ce n'est pas une validation par vérité mécanique. Les médianes et tous
les résidus individuels sont dans `holdout_summary.json` et `holdout.json`.

## Limites et décision

- Aucune acceptation hors image, sous 0,65 de corrélation recalculée, à jacobienne
  non positive ou avec corrélation non finie. Ces contrôles n'excluent pas tous les
  faux appariements ; aucune couverture correcte n'est revendiquée.
- Niveau 191 : 18 fenêtres initiales, puis 27/31/34 avec les RBF aux trois seuils.
  Les fenêtres 65 et 164 restent non convergées au seuil 1,00. Leurs dernières
  corrélations sont 0,765 et 0,338, mais leur statut solveur reste 0.
- Le seuil 0,50 est une variante de compromis à qualifier sur vérité connue ;
  le seuil 1,00 est plus permissif. Les paramètres par défaut restent inchangés.
- Une seule plaque et une seule série : pas d'intervalles statistiques prétendant
  traiter les 6 876 fenêtres comme indépendantes.
- Les distances gaussiennes ne suivent pas la matière autour des trous ; ne pas
  extrapoler ce résultat aux fissures ou aux domaines matériellement séparés.
- Les temps sont ceux de la seule nouvelle étape, dans un ordre fixe, avec traces
  écrites sur disque. Ne pas conclure à une accélération générale à partir de ces médianes.

## Reproduction

Depuis la racine du dépôt, avec les archives et données externes disponibles :

```bash
.venv/bin/python python/benchmarks/qualify_hudge_rbf_restart.py \
  --levels $(seq 1 191) --output results/hudge_rbf_restart_20261115/all_levels
OPENBLAS_NUM_THREADS=1 .venv/bin/python python/benchmarks/check_hudge_rbf_holdout.py
python3 python/benchmarks/report_hudge_rbf_restart.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error hudge_rbf_restart.tex
```

La première commande est la commande archivée. Choisir un dossier neuf pour toute
nouvelle mesure : le programme refuse de réutiliser un dossier existant.
Les deux scripts d'analyse ont des chemins d'archive explicites.

`all_levels/*/level_*.npz` : sorties par fenêtre ; `_round_*.npz` : coefficients,
propositions et décisions par tour ; `_calls.npz` : chaque proposition et chaque
sortie du solveur, même refusée. `images/191_00.png` est la référence recadrée,
`01` l'affine, `02` la RBF du premier tour et `03` celle du second tour.

## Vérifications

Les 36 configurations pilotes sont exactement reproduites pendant la campagne
complète (masques et paramètres bit à bit). Dix tests ciblés passent, dont les
dérivées par différences finies, la reproduction affine, l'exclusion des observations
non retenues et un champ gaussien analytique. Ruff passe sur les cinq nouveaux scripts
et tests. Le PDF est rendu puis inspecté visuellement ; aucune alerte LaTeX.
Les sources et empreintes sont archivées avec les résultats. Aucun code du solveur
externe, seuil par défaut, résultat historique ou conclusion du manuscrit n'est modifié.
