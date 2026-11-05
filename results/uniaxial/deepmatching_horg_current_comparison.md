# DeepMatching WebGPU et HORG-DIC WebGPU — comparaison actuelle

Comparaison des deux initialiseurs sur l’essai uniaxial Live-DIC à vérité analytique connue. Ce bilan remplace les anciennes lignes DeepMatching de `summary.csv`, calculées avec le matcher direct historique.

## Protocole

- Images UHD 3840 × 2160 : `reference.png` et les frames 040, 100 et 199 du jeu `uhd_tensile`.
- Même grille logique de 32 POI, 8 × 4, de pas 512 px, centrée en `(63,5 + 512 i, 63,5 + 512 j)`.
- DeepMatching : fenêtre 32 px, réduction ×4, rayon 256 px, racines de 512 px ; toutes les 40 racines sont calculées, puis le champ global est testé en réciprocité. Les centres sont projetés sur la grille atomique de 16 px.
- HORG-DIC : fenêtre locale de 128 px, réduction ×4, pas 512 px, plafond de 1024 descripteurs par image, ratio Lowe et vérification affine locale RANSAC. Le centre de sa fenêtre coïncide avec le centre logique demandé. Les images contiennent de 887 à 979 points-clés : le plafond de 1024 n’a pas tronqué cette paire.
- Les temps sont la médiane de trois appels par frame et méthode, images décodées avant le chronométrage. Ils incluent l’envoi des niveaux de gris au GPU, le calcul des descripteurs, la recherche et la sélection des matchs ; le RANSAC JavaScript sur CPU est inclus pour HORG. Mesure du 5 novembre 2026 dans Chromium headless 156, WebGPU/Vulkan, adaptateur Intel `xe-2lpg`.
- L’erreur est `sqrt((dx − du)² + (dy − dv)²)` en pixels, évaluée au centre source effectivement utilisé. La RMSE ne porte que sur les initialisations retenues. « ≤2 px » donne le nombre correct parmi les retenues et sa précision correspondante.

## Résultats d’erreur

| Frame | DeepMatching retenues / 32 | DeepMatching ≤2 px parmi retenues | RMSE DeepMatching | HORG retenues / 32 | HORG ≤2 px parmi retenues | RMSE HORG |
|---:|---:|---:|---:|---:|---:|---:|
| 040 | 32/32 | 30/32 (93,8 %) | 1,314 px | 31/32 | 31/31 (100 %) | 0,872 px |
| 100 | 28/32 | 22/28 (78,6 %) | 1,625 px | 26/32 | 26/26 (100 %) | 0,725 px |
| 199 | 28/32 | 27/28 (96,4 %) | 1,165 px | 5/32 | 4/5 (80 %) | 1,947 px |
| **Total** | **88/96** | **79/88 (89,8 %)** | **1,378 px** | **62/96** | **61/62 (98,4 %)** | **0,952 px** |

Les erreurs RMSE sont calculées sur des ensembles différents : HORG rejette davantage de POI, notamment à 10 % d’allongement. Sur l’ensemble des 96 graines demandées, DeepMatching fournit 79 initialisations à ≤2 px contre 61 pour HORG. HORG est plus sélectif et ses matchs retenus sont plus précis en proportion ; à la frame 199, il n’en conserve que 5, contre 28 pour DeepMatching.

## Temps de calcul

| Frame | DeepMatching médian | HORG médian | HORG plus rapide |
|---:|---:|---:|---:|
| 040 | 1,335 s | 0,173 s | ×7,7 |
| 100 | 1,333 s | 0,152 s | ×8,8 |
| 199 | 1,290 s | 0,144 s | ×9,0 |
| **Moyenne des médianes** | **1,319 s** | **0,156 s** | **×8,4** |

DeepMatching dépense ce temps pour calculer la hiérarchie dense complète sur 40 racines et appliquer la réciprocité globale. HORG extrait des points-clés, fait la recherche de descripteurs, puis ajuste une affine locale par graine. Il est plus rapide dans cette configuration, tout en faisant intervenir le RANSAC CPU.

Les critères de rejet ne sont pas équivalents : DeepMatching exige une correspondance réciproque globale, HORG exige un consensus affine local. La couverture, la précision des matchs retenus et l’erreur RMSE doivent donc être lues ensemble. Les résultats par graine restent disponibles dans les CSV INRIA/WebGPU et HORG du même dossier ; ce calcul frais concorde avec les vecteurs qu’ils contiennent.

## Mesures brutes

`deepmatching_horg_current_comparison.csv` conserve, pour chaque méthode et chaque frame, la couverture, les erreurs analytiques et les trois durées mesurées.
