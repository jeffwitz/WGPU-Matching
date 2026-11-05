# WGPU Matching — graines DeepMatching pour la DIC

Prototype navigateur de correspondances pour initialiser une corrélation d’images numériques (DIC). Le calcul des descripteurs et la recherche sont effectués sur WebGPU.

Ce dépôt regroupe les matchers que nous avons implémentés en JavaScript/WGSL : DeepMatching et HORG-DIC, leur interface navigateur et leurs résultats de comparaison. Il ne contient ni les sources INRIA ni les PDF des articles.

## Lancer

Depuis ce dossier, lancez un serveur HTTP local :

~~~sh
python3 -m http.server 8000
~~~

Ouvrez http://localhost:8000/ dans un navigateur compatible WebGPU. Chargez deux images de mêmes dimensions, choisissez la fenêtre DIC, l’échelle racine, le rayon, la réduction et les seeds, puis lancez le calcul. Le CSV donne le déplacement, le score hiérarchique et le résultat du test de réciprocité.

## Méthode

Le descripteur reprend la configuration CPU par défaut de DeepMatching :

1. descripteur HOG pixel par pixel à pleine résolution, avec le pré-lissage gaussien, le gradient fini, les huit projections positives, les lissages HOG, la sigmoïde et la neuvième composante constante de la configuration CPU par défaut ;
2. réduction par average-pooling des descripteurs après leur extraction, comme dans la référence ;
3. échantillonnage des patchs atomiques source, normalisation L2 des descripteurs source et cible et calcul des réponses de corrélation dans le rayon demandé ;
4. agrégation ascendante par max-pooling 3×3 / sous-échantillonnage 2 et convolution sparse pondérée des quatre enfants, avec offsets, correction des bords de l’agrégation et puissance non linéaire 1,4 ;
5. parcours descendant des réponses par argmax local 3×3, accumulation des scores, puis intersection réciproque des correspondances.

Les descripteurs sont calculés en tuiles avec des tampons temporaires float32. Le pré-lissage conserve une quantification 8 bits ; les descripteurs réduits sont stockés en float16. À l’échelle choisie, le calcul évalue toutes les racines de la grille INRIA sur l’image, puis fusionne les correspondances source et cible avant l’intersection réciproque globale. La grille par défaut ne chevauche pas les racines et son pas vaut leur taille, comme dans les paramètres CPU par défaut d’INRIA ; sur les images Live-DIC UHD, l’échelle racine par défaut est 512 px. Les seeds définissent les bins exportés, tandis que le matcher calcule la grille entière. Les variantes d’échelle/rotation et EpicFlow ne sont pas incluses. Les patchs cibles incomplets au bord de l’image sont écartés plutôt que rectifiés comme par `rectify_conv` ; les seeds près des bords peuvent diverger. Un calcul est limité à 8192 racines au total. La comparaison validée sur Live-DIC est documentée dans [le rapport hiérarchique INRIA/WebGPU](results/uniaxial/inria_webgpu_hierarchy_comparison.md). Les sorties de l’ancienne corrélation directe restent dans [le rapport historique](results/uniaxial/inria_webgpu_patch_comparison.md).

Le rayon borne les translations évaluées autour de chaque cellule atomique source ; `0` lance une recherche complète, qui peut dépasser la mémoire WebGPU. Saisissez les centres source manuels `x ; y` en pixels, un par ligne, ou utilisez la grille. Les centres sont projetés sur la grille atomique INRIA ; le CSV reporte ensuite la feuille effectivement appariée et la racine gagnante. La taille racine réduite doit contenir 1, 2, 4, 8… cellules atomiques de 4 pixels par côté. La fenêtre DIC reste indépendante de cette échelle racine.

## HORG-DIC WebGPU

Le sélecteur de méthode lance aussi une adaptation WebGPU de l’initialisation HORG-DIC : pyramide gaussienne/DoG, détection SIFT, descripteur HORG signé de 64 composantes, recherche exacte des deux plus proches descripteurs et ratio test de Lowe à 0,75. La sélection plafonne le nombre de points-clés comparés par image. Les affines locales sont ajustées par RANSAC en JavaScript sur le CPU ; le noyau WebGPU couvre la détection, les descripteurs et la recherche. Le raffinement sous-pixel SIFT et l’étape IC-GN ne sont pas inclus.

Les résultats HORG sont marqués utilisables seulement avec au moins quatre inliers, huit candidats locaux et 20 % d’inliers. Le seuil RANSAC de 1,5 px est ramené aux coordonnées de l’image d’origine selon le facteur de réduction. Le CSV conserve également les ajustements rejetés, identifiés par `usable=0`.

Le compte rendu de performance et le diagnostic sur l’ROI masquée sont dans [le rapport de comparaison](results/horg_vs_deepmatching_benchmark.md), [le CSV HORG](results/horg_000000_to_000075_masked.csv) et [son PNG](results/horg_000000_to_000075_masked.png). Pour ce couple d’images, HORG n’a pas produit d’affine locale assez soutenue pour initialiser la DIC ; ce résultat est visible dans le PNG.

Une comparaison avec les sorties denses INRIA est disponible pour les images de traction uniaxiale UHD de Live-DIC : [rapport de la hiérarchie WebGPU](results/uniaxial/inria_webgpu_hierarchy_comparison.md) et [CSV seed par seed](results/uniaxial/inria_webgpu_hierarchy_comparison.csv). L’ancien essai de corrélation directe reste archivé dans [son rapport](results/uniaxial/inria_webgpu_patch_comparison.md). Les images source dans `tests_gpuio` sont lues sans modification.

Le [bilan actuel DeepMatching–HORG](results/uniaxial/deepmatching_horg_current_comparison.md) compare l’erreur analytique et le temps de calcul sur les mêmes trois frames Live-DIC ; les durées brutes sont dans son CSV associé.

## Références et provenance

- Article : [DeepMatching: Hierarchical Deformable Dense Matching](https://arxiv.org/abs/1506.07656), Revaud, Weinzaepfel, Harchaoui et Schmid.
- [Page officielle DeepMatching](https://thoth.inrialpes.fr/src/deepmatching.html) et [archives de code](https://lear.inrialpes.fr/src/deepmatching/code/).
- Article HORG-DIC : « Chirality-aware initialization for reliable digital image correlation under large rotation and heterogeneous deformation », *Measurement* 292 (2026), 123133, [DOI 10.1016/j.measurement.2026.123133](https://doi.org/10.1016/j.measurement.2026.123133) ; [code compagnon HORG-DIC](https://github.com/123456shiro/HORG-DIC).
- Les noyaux WGSL de ce dossier sont une réimplémentation à partir de la description publiée ; ils n’intègrent pas de code du projet officiel. La version GPU officielle 1.0 documentée par les auteurs dépend de Caffe/CUDA et annonce une licence GPL version 3 ou ultérieure ainsi qu’un usage scientifique ou personnel.
