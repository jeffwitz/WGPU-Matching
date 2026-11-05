# Comparaison DeepMatching INRIA / WebGPU, essai uniaxial

> Archive de l’ancienne corrélation directe de patch. Elle ne décrit pas l’implémentation hiérarchique actuelle ; les mesures comparables à l’INRIA sont maintenant dans [le rapport hiérarchique](inria_webgpu_hierarchy_comparison.md).

Cette mesure historique compare les déplacements sur trois frames Live-DIC avec vérité analytique connue. Les CSV sont conservés pour tracer l’évolution.

## Protocole

- Images `reference.png` vers `frame_040.png`, `frame_100.png` et `frame_199.png` du jeu `uhd_tensile` (3840 × 2160), lu depuis `tests_gpuio/DIC/LucasKanadeICGN/assets/uhd_tensile` sans le modifier.
- Même grille de 32 centres pour WebGPU et INRIA : `x = 63.5 + 512 i`, `y = 63.5 + 512 j`, avec `i = 0…7` et `j = 0…3`.
- WebGPU : fenêtre 128 px, réduction ×8, rayon 256 px, score minimal 0.12, marge minimale 0.04. Corrélation directe du patch complet, pic interpolé paraboliquement ; les maxima au bord du rayon et les scores ambigus sont rejetés.
- INRIA : DeepMatching CPU v1.2.2, réduction `-downscale 2`, voisinage `-ngh_rad 256`, sorties denses. Pour chaque seed, comparaison avec la correspondance officielle la plus proche à moins de 16 px.
- Champ analytique : `du = (sx − 1)(x − 1920)`, `dv = (sy − 1)(y − 1080)`, avec les étirements du `machine_truth.csv`.

La RMSE analytique et l’écart WGPU/INRIA sont calculés sur les seeds WebGPU retenues. Les critères de sélection ne sont pas identiques : les sorties INRIA sont brutes, et le rapprochement spatial utilise la correspondance officielle la plus proche. La recherche WebGPU directe n’implémente pas l’agrégation hiérarchique complète ni le parcours top-down d’INRIA.

## Résultats

| Frame | Seeds WebGPU retenues | Erreur analytique WebGPU (RMSE; ≤2 px) | INRIA sur mêmes seeds (RMSE; ≤2 px) | Écart vecteur WebGPU–INRIA (RMSE; ≤2 px) | Temps WebGPU |
|---:|---:|---:|---:|---:|---:|
| 40 | 21/32 | 0.71 px; 21/21 | 1.19 px; 20/21 | 0.78 px; 20/21 | 0.24 s |
| 100 | 27/32 | 1.14 px; 26/27 | 1.55 px; 24/27 | 1.85 px; 16/27 | 0.18 s |
| 199 | 23/32 | 0.77 px; 23/23 | 1.69 px; 18/23 | 1.68 px; 18/23 | 0.18 s |

Le champ WebGPU retenu est donc proche du champ INRIA sur cet essai, avec un écart vectoriel RMSE de 0.78 à 1.85 px selon la frame. À 10 % d’allongement, 23 seeds sur 32 sont retenues et toutes sont à moins de 2 px du champ analytique. Les seeds de bord non fiables sont écartées par la marge et le contrôle du rayon.

Les temps WebGPU correspondent à une exécution headless Chromium/WebGPU sur l’adaptateur Intel intégré, images déjà chargées ; ils comprennent la génération des descripteurs et la recherche. Ils ne constituent pas un ratio de vitesse face à INRIA : INRIA renvoie un champ dense à réduction ×2, tandis que cette adaptation ne calcule que 32 graines à réduction ×8.

## Fichiers

- `frame_040|100|199_webgpu_direct_patch.csv` : sorties WebGPU complètes, scores, marges et indicateur `usable`.
- `inria_webgpu_patch_comparison.csv` : vérité analytique, correspondance INRIA la plus proche et erreurs, seed par seed.
