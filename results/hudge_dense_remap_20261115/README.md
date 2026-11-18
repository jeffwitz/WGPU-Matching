# Plaque trouée : initialisation dense et remapping natif

> **Rectification après examen des textures : la recommandation du champ corrigé comme initialisation près des trous est retirée.** La correction des contours dégrade le mouchetis entre 4 et 16 pixels des trous. Aucun des 825 centres de contrôle n'est situé à moins de 8 pixels d'un trou. Voir [l'audit spatial et les résidus](near_hole_audit/README.md). Les images ci-dessous restent des diagnostics, pas une initialisation qualifiée de cette couronne.

15 novembre 2026. Galerie : [index.html](index.html). Rapport : [PDF](../../paper/hudge_dense_remap.pdf), [source LaTeX](../../paper/hudge_dense_remap.tex).

## Résultat et limites

Les images remappées sont de véritables échantillonnages de la cible originale, sans peinture de masque ni cercle imposé. Les exports de diagnostic se trouvent dans `refined/level_050`, `refined/level_150` et `refined/level_191`. Chaque paire utilise directement la même référence ; aucune image intermédiaire de la séquence n'est utilisée.

Le champ dense provient de fenêtres de corrélation d'images numériques (DIC), résolues par Gauss–Newton inverse compositionnel (IC-GN) sur WebGPU. Une correction par fonctions de base radiale (RBF) exploite les contours **observés** de la matière dans les deux images, avec maintien souple des déplacements intérieurs. Cette segmentation par fond uniforme est spécifique aux rendus fournis.

| Niveau | Ancrages fins retenus / 979 | Recouvrement matière avant / après contours | Corrélation globale avant / après |
|---|---:|---:|---:|
| 50 | 974 | 97,74 % / 99,65 % | 0,928 / 0,933 |
| 150 | 826 | 92,21 % / 96,48 % | 0,808 / 0,841 |
| 191 | 790 | 92,07 % / 98,37 % | 0,806 / 0,828 |

Le recouvrement est l'intersection des masques divisée par leur union. Comme les contours interviennent dans l'ajustement, ce résultat n'est **pas** une validation mécanique indépendante. Aucun pixel de matière ne sort de la cible ni ne présente de déterminant non positif dans les trois champs exportés. Le contrôle porte sur la grille pixel, pas sur tout le domaine continu. Les gradients analytiques du champ incluent les dérivées des poids de mélange.

Au niveau 191, les cinq plus grands trous ont des rapports d'axes entre 1,014 et 1,077 après remapping. Le plus petit reste mal restitué : rapport 1,479, recouvrement 66,3 %, décalage du centre 2,50 pixels. Au niveau 150, la recherche de pas refuse plusieurs corrections qui créeraient un repli. Les échecs ne sont pas masqués.

Les trois images font 1024 × 1024 pixels ; la matière de référence n'occupe qu'une partie du canevas. Les agrandissements de la galerie sont des recadrages identiques, sans changement de coordonnées dans les calculs.

## Vérification de l'initialisation

Le solveur est réinitialisé sur **825 centres de fenêtres 24 × 24**, espacés de 8 pixels, décalés de (4,4) pixels par rapport aux ancrages 16 × 16. Ils ne sont pas les points d'ajustement ; leurs supports recouvrent néanmoins les observations. Ce n'est pas un test sur de nouvelles images.

| Champ initial | Statuts convergés après le premier appel de 30 itérations | Retenues après au plus 4 appels |
|---|---:|---:|
| Champ grossier de 36 ancrages | 699 | 764 / 825 |
| Champ dense des fenêtres 16 | 734 | 780 / 825 |
| Dense + contours, rayon 16, 196 bases | 727 | 784 / 825 |
| Dense + contours, rayon 12, 384 bases, images recommandées | 729 | 782 / 825 |

Le champ de rayon 12 améliore la restitution des images mais retient deux fenêtres de moins que la variante de rayon 16. Il n'est donc pas présenté comme optimal pour la convergence. Les sorties retenues ont statut convergé, paramètres finis, corrélation finale ≥ 0,65, support dans l'image et déterminant positif. **Retenue ne signifie pas déplacement mécaniquement exact.** Il n'existe pas de vérité de déplacement associée à cette plaque ; le maillage en croix du dossier voisin n'est pas utilisé.

Le niveau 191 passe de 784 à 790 ancrages fins après réinjection de la première correction de contour et reprise locale. Cette correction est ensuite recalculée à partir des 790 ancrages. Les paramètres locaux mesurés et les paramètres prédits par le champ sont exportés séparément.

Les temps dans les contrôles ne constituent pas un benchmark : les coûts de première compilation sont différents et des reconstructions sur processeur central étaient exécutées simultanément. Aucun gain temporel n'est revendiqué.

## Fichiers utilisables

Dans chaque `refined/level_XXX/` :

- `image1_ref.png` : référence native ; `image2_def_cor.png` : cible remappée sur le même canevas ; `image0_def.png` : cible originale.
- `reference_corrected.gif` et `reference_corrected_zoom.gif` : alternances sans annotation peinte dans les pixels.
- `field.npz` : déplacement `(H,W,2)`, gradient du déplacement `(H,W,2,2)`, déterminant et diagnostics.
- `initialization.npz` : centres natifs, six paramètres prédits `[u, du/dx, du/dy, v, dv/dx, dv/dy]`, indicateur `observed_anchor`, paramètres et statuts réellement mesurés.
- `contour_field.npz` : coefficients, centres, rayon de la correction et données du champ de base ; permet de réévaluer le champ et ses dérivées.
- `summary.json` : empreintes SHA-256 des entrées, configuration, étapes de la recherche de pas, scores et six diagnostics de trous.

Le champ va **de la référence vers la cible originale** : `I_corr(x) = I_target(x + u(x))`. Les coordonnées sont des centres de pixels natifs, avec x à droite et y vers le bas. L'interpolation des pixels est bilinéaire. Les valeurs du fond observé remplissent seulement les échantillonnages hors image ; aucune matière de référence n'est dans ce cas dans les exports retenus.

## Reproduction

À la racine du dépôt. Les archives antérieures des ancrages grossiers sont nécessaires. L'environnement `.venv` contient le solveur WebGPU ; le Python système contient les outils de tracé et de reconstruction. Le solveur externe est importé depuis `/home/jeff/Code/tests_gpuio/DIC/LucasKanadeICGN/python/src` et n'est pas modifié.

```bash
# Calcul dense sur chaque paire, sans données temporelles
.venv/bin/python python/benchmarks/qualify_hudge_dense_remap.py \
  --levels 50 150 191 --models affine16 --interpolation bspline \
  --output /tmp/hudge_dense_new

# Première correction et sa réinjection au niveau 191
OPENBLAS_NUM_THREADS=1 python3 python/benchmarks/export_hudge_contour_remap.py \
  --level 191 --seeds /tmp/hudge_dense_new/level_191/affine16/seeds.npz \
  --output /tmp/hudge_contour_first
.venv/bin/python python/benchmarks/qualify_hudge_dense_remap.py \
  --levels 191 --models affine16 --interpolation bspline \
  --dense-seed-root /tmp/hudge_dense_new \
  --initial-contour /tmp/hudge_contour_first/contour_field.npz \
  --output /tmp/hudge_retry_new

# Champ livré, remapping natif et initialisations
OPENBLAS_NUM_THREADS=1 python3 python/benchmarks/export_hudge_contour_remap.py \
  --level 191 --seeds /tmp/hudge_retry_new/level_191/affine16/seeds.npz \
  --length 12 --max-sites 384 --output /tmp/hudge_remap_new
.venv/bin/python python/benchmarks/validate_hudge_remap_init.py \
  --field /tmp/hudge_remap_new/contour_field.npz \
  --output /tmp/hudge_offset_new

# Inspection des deux images en alternant dans eog
eog results/hudge_dense_remap_20261115/refined/level_191/zoom_1.png \
    results/hudge_dense_remap_20261115/refined/level_191/zoom_2.png
```

Pour les niveaux 50 et 150, l'export final utilise directement leurs ancrages de `/tmp/hudge_dense_new`, sans la réinjection réservée au niveau 191. Les dossiers de sortie doivent être nouveaux : les programmes refusent d'écraser les archives.

## Expériences non retenues et périmètre

- `pilot/` et `bspline/` : fenêtres affines 16 et quadratiques 16/32 sur toute la matière, pas seulement les anciennes régions d'échec.
- `fine8/` : 2 000 / 4 624 fenêtres retenues, mais 468 pixels de matière avec repli dans le champ combiné. Non retenu.
- `rbf_pilot/` : RBF globale réduite, longueurs 16 à 64, poids de gradient 0, 4, 8. Le meilleur recouvrement observé reste inférieur au champ local de départ.
- `local_pilot/` : raccourcir excessivement le mélange local améliore certaines corrélations mais crée des replis.
- `photometric_pilot/` : essai exploratoire de flot optique Farnebäck sur processeur central, composition avec le champ existant. Les itérations finissent par produire des replis ; ce chemin n'est pas utilisé dans les exports ni dans la qualification.
- `contour_pilot/`, `contour_fine_pilot/` : essais de rayons ; le rayon 8 s'arrête sur la contrainte de non-repli, le rayon 12 fournit le compromis retenu pour les images. Les paramètres ont été explorés sur cette plaque ; il ne s'agit pas d'une qualification indépendante.
- `final/` et `offset_validation/` : première correction de rayon 16, conservée. `refined/` et `offset_validation_refined/` : rayon 12, 384 bases.

Les programmes permanents permettent de reproduire les calculs retenus. Les petits essais exploratoires conservent leurs images et/ou coefficients, mais ne sont pas présentés comme une campagne de performance reproductible.

**Portabilité :** les calculs locaux IC-GN utilisent WebGPU. La combinaison des champs, l'ajustement des contours et le remapping utilisent NumPy/SciPy sur le processeur central. Il ne s'agit pas d'un pipeline entièrement sur processeur graphique ni d'une intégration de production dans Live-DIC.

**Contrôles logiciels :** 14 tests ciblés passent (dérivées, conventions de centres, champs locaux, RBF et contours). Ruff passe sur les nouveaux programmes. Le PDF est compilé puis inspecté par rendu de ses pages. Le manuscrit principal et le solveur externe ne sont pas modifiés.
