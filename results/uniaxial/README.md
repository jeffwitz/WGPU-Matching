# Comparaison des initialisations sur l’essai uniaxial UHD

> **Résultats historiques :** ces mesures proviennent de versions antérieures du matcher. La comparaison actuelle des déplacements avec les sorties INRIA exactes est dans [le rapport hiérarchique](inria_webgpu_hierarchy_comparison.md) et [son CSV seed par seed](inria_webgpu_hierarchy_comparison.csv).

L’ancienne corrélation directe du patch complet est documentée dans [son rapport archivé](inria_webgpu_patch_comparison.md) et [son CSV associé](inria_webgpu_patch_comparison.csv). Les fichiers historiques frame par frame ci-dessous restent conservés pour provenance.

## Données et vérité analytique

Les images proviennent de `tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile` : une référence 3840 × 2160 à la frame 0 et les cibles des frames 40, 100 et 199. Il s’agit du jeu virtuel archivé pour Live-DIC, généré avant encodage vidéo. Le dépôt source a été lu sans modification.

Le générateur déforme la référence autour de `(1920,1080)`. Avec les étirements `sx`, `sy` du `machine_truth.csv`, le champ de référence est :

```text
du = (sx - 1) * (x - 1920)
dv = (sy - 1) * (y - 1080)
```

Les coordonnées sont des centres de pixels indexés à partir de zéro, selon la convention du jeu de données. Les images cible sont rééchantillonnées par interpolation bilinéaire puis quantifiées en 8 bits ; le champ cinématique est analytique, tandis que les pixels restent des échantillons discrets.

## Résultats

Chaque méthode fournit 32 POI sur une grille 8 × 4. « Précis ≤2 px » compte les POI retenus par la méthode dont l’erreur vectorielle par rapport au champ analytique est au plus 2 px, rapportés aux 32 POI. La RMSE est calculée seulement sur les initialisations retenues. Les temps sont les médianes de trois exécutions de la fonction d’initialisation, images déjà chargées.

| Frame | Allongement axial | Méthode | Initialisations retenues | Précises ≤2 px | RMSE retenues | Temps médian |
|---:|---:|---|---:|---:|---:|---:|
| 40 | 0,112 % | DeepMatching | 21/32 | 21/32 (65,6 %) | 1,175 px | 0,14 s |
| 40 | 0,112 % | HORG | 31/32 | 31/32 (96,9 %) | 0,872 px | 0,20 s |
| 100 | 3,843 % | DeepMatching | 21/32 | 1/32 (3,1 %) | 17,811 px | 0,14 s |
| 100 | 3,843 % | HORG | 26/32 | 26/32 (81,3 %) | 0,725 px | 0,18 s |
| 199 | 10,000 % | DeepMatching | 27/32 | 1/32 (3,1 %) | 881,356 px | 0,14 s |
| 199 | 10,000 % | HORG | 5/32 | 4/32 (12,5 %) | 1,947 px | 0,17 s |

À 10 % de déformation, les seuils score/unicité de DeepMatching laissent passer trois graines dont l’erreur dépasse 2500 px. HORG est beaucoup plus sélectif : il perd la couverture, mais ses cinq affines retenues restent proches du champ exact (p95 = 4,05 px). Il faut donc distinguer couverture et précision des graines retenues.

HORG trouve respectivement 265, 116 et 49 paires au ratio Lowe pour les frames 40, 100 et 199. Il détecte 887 points-clés dans la référence ; les trois cibles en ont 887, 935 et 979. Le nombre de paires cohérentes baisse avec l’allongement, ce qui explique la chute de couverture à 10 %.

La figure [analytical_comparison.png](analytical_comparison.png) trace la couverture à 2 px, la RMSE des graines retenues et les temps. Les CSV source, ainsi que leurs colonnes d’erreur analytique, sont conservés frame par frame dans ce dossier.

## Réglages et limites de la comparaison

- Fenêtre DIC 128 px, pas 512 px, même grille et mêmes images pour les deux méthodes.
- DeepMatching : réduction ×8, seuil score 0,12 et marge d’unicité minimale 0,01.
- HORG : réduction ×4, plafond de 2048 descripteurs/image, ratio Lowe 0,75 ; affine locale RANSAC à partir de 4 inliers, au moins 8 candidats et 20 % d’inliers. Le seuil de 1,5 px dans l’image réduite est converti en 6 px dans les coordonnées originales.
- Le temps HORG comprend le RANSAC local en JavaScript sur CPU. Les deux pipelines GPU utilisent WebGPU dans Chromium headless ; l’adaptateur indique Intel `xe-2lpg`.
- Les critères d’acceptation des méthodes diffèrent et les images ont été synthétisées avec rééchantillonnage ; les taux mesurent cette initialisation sur ce jeu connu, pas une généralisation à des essais physiques.

## Fichiers de résultats

- `frame_040|100|199_deepmatching.csv` et `frame_040|100|199_horg.csv` : sorties brutes de l’application.
- `frame_040|100|199_deepmatching_analytic_error.csv` et `frame_040|100|199_horg_analytic_error.csv` : déplacement analytique et erreur par POI.
- `summary.csv` : couverture, RMSE et temps médian par méthode et frame.
