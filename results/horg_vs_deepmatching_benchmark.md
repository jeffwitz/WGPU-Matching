# HORG-DIC contre DeepMatching sur l’ROI masquée

> **Résultats historiques :** les mesures DeepMatching ci-dessous proviennent de l’ancien descripteur simplifié, calculé directement sur l’image réduite. Elles ne décrivent pas le pipeline de descripteur HOG pixelwise ajouté depuis ; refaire le benchmark avant d’interpréter les temps ou le nombre de graines.

Mesure du 5 novembre 2026 sur `img_000000_0.09422.tiff` et `img_000075_1608.88508.tiff`. Le masque `Mask.tiff` définit l’ROI `x=[384,7576)`, `y=[888,5088)`, soit `7192 × 4200` pixels. Les deux méthodes ont tourné dans la même session Chromium 148 headless, avec WebGPU/Vulkan ; l’adaptateur déclaré par le navigateur est Intel, architecture `xe-2lpg`.

## Réglages

- Grille DIC : fenêtre 128 px, pas 1024 px, soit 28 POI.
- DeepMatching : réduction ×8, seuil score 0,12 et marge d’unicité 0,01.
- HORG : réduction ×4, maximum de 2048 descripteurs par image, ratio Lowe 0,75. Le seuil RANSAC de 1,5 px dans les coordonnées réduites équivaut à 6 px dans l’image d’origine. Une affine est retenue à partir de quatre inliers, huit candidats locaux et 20 % d’inliers.

## Temps observés

Chaque méthode a été lancée trois fois après chargement des images. Le temps comprend toute l’extraction et l’initialisation ; pour HORG, il comprend aussi le RANSAC local exécuté en JavaScript sur le CPU.

| Méthode | Durées des 3 exécutions | Médiane | Écart à DeepMatching |
|---|---:|---:|---:|
| DeepMatching WebGPU | 0,41 · 0,43 · 0,42 s | 0,42 s | — |
| HORG-DIC WebGPU | 0,55 · 0,52 · 0,51 s | 0,52 s | +0,10 s, soit +23,8 % |

## Résultat sur cette paire

| Méthode | Correspondances / graines | POI avec initialisation retenue |
|---|---:|---:|
| DeepMatching | 25 graines sur 28 passent les seuils de score et d’unicité | 25/28 |
| HORG-DIC | 131 paires passent le ratio Lowe | 0/28 affines locales passent le consensus RANSAC |

Le passage des seuils DeepMatching n’est pas une validation par vérité terrain. HORG détecte 8069 et 8315 points-clés dans les deux images, mais les paires retenues ne donnent pas ici un consensus local assez soutenu. Le PNG affiche les 28 POI rejetés sans dessiner les ajustements aberrants. Il n’y a donc pas de champ HORG exploitable pour cette paire, malgré un temps de calcul proche de DeepMatching.

Ces durées comparent deux réglages utiles sur la même grille et le même ROI, pas deux quantités de calcul strictement équivalentes : les réductions diffèrent et HORG compare 2048 descripteurs par image. L’implémentation s’arrête à l’initialisation et ne lance pas l’étape IC-GN.
