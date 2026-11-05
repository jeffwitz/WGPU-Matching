# DeepMatching INRIA et WebGPU — grille dense globale

Comparaison des sorties INRIA CPU v1.2.2 et WebGPU sur les images de traction uniaxiale Live-DIC. Les images de fixture, les sorties INRIA et la vérité analytique ont été lues depuis `tests_gpuio`; ce dépôt source n’a pas été modifié.

## Protocole

- Paires `reference.png` → `frame_040.png`, `frame_100.png` et `frame_199.png`, 3840 × 2160.
- 32 bins demandés par frame : `x = 63.5 + 512 i`, `y = 63.5 + 512 j`, `i=0…7`, `j=0…3`. WebGPU projette chaque centre sur la grille atomique source de pas 16 px ; l’INRIA est comparé au même bin.
- WebGPU : HOG pixelwise, average-pooling ×4, patch atomique 16 px, fenêtre DIC 32 px, rayon 256 px, hiérarchie racine 512 px sur 6 niveaux. Il calcule la grille complète de 40 racines INRIA (8 × 5) puis l’intersection réciproque globale ; seules les 32 positions demandées sont exportées.
- INRIA : exécutable CPU v1.2.2 avec `-downscale 2 -ngh_rad 256`, options par défaut pour le chevauchement, les couches et le scoring. Son `max_psize=999` construit ici le niveau racine 512 px ; les racines sont espacées de 512 px. Les cartes denses sont comparées au même centre atomique source, sans recherche du voisin le plus proche.
- Vérité analytique : `du=(stretch_x−1)(x−1920)`, `dv=(stretch_y−1)(y−1080)`, évaluée au centre atomique source.

## Résultats

| Frame | Matchs INRIA / WebGPU sur 32 | Erreur analytique INRIA = WebGPU (RMSE / médiane / P95 / max, px) | Matchs ≤2 px INRIA / WebGPU | Temps INRIA, 1 thread | Temps INRIA, 8 threads | Temps WebGPU |
|---:|---:|---:|---:|---:|---:|---:|
| 040 | 32/32 · 32/32 | 1,314 / 1,177 / 2,002 / 2,102 | 30/32 (93,8 %) · 30/32 (93,8 %) | 33,93 s | 8,87 s | 1,45 s |
| 100 | 28/32 · 28/32 | 1,625 / 1,530 / 2,332 / 2,466 | 22/28 (78,6 %) · 22/28 (78,6 %) | 33,32 s | 9,26 s | 1,32 s |
| 199 | 28/32 · 28/32 | 1,165 / 0,893 / 1,787 / 2,433 | 27/28 (96,4 %) · 27/28 (96,4 %) | 33,37 s | 11,07 s | 1,30 s |
| **Moyenne / total** | **88/96 · 88/96** | **1,378 / 1,356 / 2,094 / 2,466** | **79/88 (89,8 %) · 79/88 (89,8 %)** | **33,54 s** | **9,73 s** | **1,36 s** |

L’erreur vectorielle par rapport à l’analytique vaut `sqrt((dx − du)² + (dy − dv)²)` en pixels ; le tableau donne sa RMSE, sa médiane, son P95 et son maximum parmi les matchs trouvés. La précision indique la part des correspondances présentes/réciproques dont cette erreur est ≤2 px. Sur les 88 bins rendus par les deux méthodes, les 88 vecteurs sont identiques au pixel près (RMSE WebGPU–INRIA = 0 px) ; les deux implémentations ont donc les mêmes statistiques d’erreur sur ce jeu. Elles trouvent une correspondance sur 88 des 96 positions demandées, et 79 de ces 88 vecteurs sont à ≤2 px de la vérité analytique. Les correspondances qui dépassent ce seuil restent dans le décompte et ne sont pas masquées.

Le calcul couvre toutes les racines non chevauchantes du niveau supérieur et met en commun les meilleurs candidats vers la source et vers la cible avant le test réciproque. Il reproduit donc le champ dense de correspondances au niveau racine configuré, pour la configuration sans rotation/variation d’échelle utilisée ici. La sortie de l’application est ensuite échantillonnée aux seeds demandées. Le redressement CPU des patchs incomplets au bord (`rectify_conv`) et les variantes d’invariance rotation/échelle ne sont pas encore portés.

Les temps INRIA ont été mesurés sur le binaire CPU v1.2.2 avec `-downscale 2 -ngh_rad 256`, une exécution par frame : 1 thread par défaut, puis `-nt 0` pour les 8 cœurs disponibles. Le chronométrage CLI inclut le chargement PNG et l’écriture des sorties vers `/dev/null`. Le temps WebGPU inclut la préparation HOG, la création des pipelines et les 40 racines de la grille ; il exclut le décodage PNG et la conversion gris. Mesure du 5 novembre 2026 avec Chromium headless 156, WebGPU/Vulkan et adaptateur Intel `xe-2lpg`. En moyenne, WebGPU est environ ×24,7 plus rapide que le CPU INRIA à un thread et ×7,2 plus rapide que la version INRIA à huit threads. Les limites de chronométrage ne sont pas strictement identiques : ces ratios sont indicatifs.

La descente récursive des argmax est exécutée sur le GPU ; seules les coordonnées et les scores des candidats sont rapatriés, au lieu des cartes denses complètes. Deux racines sont encodées dans chaque soumission. Les temps ne comparent pas le champ CPU précalculé d’INRIA séparément.

## Données

- `inria_webgpu_hierarchy_comparison.csv` : centres atomiques, racines gagnantes, vecteurs, réciprocité, écarts INRIA et erreurs analytiques par graine.
- `inria_webgpu_patch_comparison.md` et son CSV : résultats historiques de l’ancienne corrélation directe.
