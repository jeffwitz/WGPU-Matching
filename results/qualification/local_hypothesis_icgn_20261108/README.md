# Hypothèses de rotation, modèles affines locaux et convergence IC-GN

## Résumé

Cette expérience raccorde la recherche coarse-to-fine des supports affines aux
modèles affines locaux WGSL et au solveur IC-GN WebGPU. Elle utilise le CNN DIC
compact existant, le détecteur DIC et le RANSAC affine local déjà présents dans
WGPU-Matching. Aucun apprentissage supplémentaire n'a été effectué.

La vérité analytique sert uniquement à noter les correspondances, à calculer
les métriques finales et à fournir le contrôle oracle du solveur. Elle ne sert
pas à choisir l'angle, les correspondances, les modèles locaux ni les fenêtres
actives. Le choix d'un modèle local repose sur les correspondances observées,
les seuils de consensus fixés et la position cible prédite dans l'image.

## Protocole

- Adaptateur : Vulkan, Intel Graphics (LNL).
- Solveur : IC-GN affine WebGPU, fenêtres 64 × 64 pixels, pas de 64 pixels,
  30 itérations au plus.
- Détection : détecteur DIC WGSL avec sélection spatiale.
- Matching : descripteur CNN DIC compact, matching réciproque WGSL.
- Recherche : 18 orientations grossières, raffinement local autour des deux
  meilleures, puis vérification complète des deux hypothèses retenues.
- Modèles : ajustement RANSAC affine WGSL par centre DIC, rayon 192 px,
  résidu maximal 4 px, au moins 6 inliers et 8 candidats.
- Contrôle apparié : mêmes points détectés et mêmes candidats, descripteurs
  sans compensation affine, puis même RANSAC local et même IC-GN.
- Critère principal : déplacement final correct si l'erreur de translation est
  inférieure à 2 px; la couverture est rapportée sur les centres visibles.

Le modèle local initialise directement la fenêtre au centre demandé. Le champ
rapporté ici n'est pas interpolé entre centres. Les correspondances utilisées
par chaque modèle proviennent néanmoins d'un voisinage de 192 px; la limite
spatiale n'est donc pas une frontière physique et ces résultats ne qualifient
pas le comportement au voisinage d'une fissure.

## Résultats

| Cas | Résolution | Hypothèse choisie | Correspondances directes <5 px | Modèles locaux actifs | IC-GN correct | Oracle IC-GN |
|---|---:|---:|---:|---:|---:|---:|
| 8102/9102 | 1024 × 576 | 140° | 0/349 | 0 | 0/65 | 45/65 |
| 8103/9103 | 1024 × 576 | 0° | 90/277 | 56 | 41/77 | 68/77 |
| 8103/9103 | 3840 × 2160 | 0° | 118/949 | 132 | 131/1 096 | 1 058/1 096 |

Les résultats affichés pour les deux derniers cas sont identiques au contrôle
sans compensation affine : mêmes correspondances, mêmes modèles locaux et même
nombre de convergences. La recherche d'orientation n'apporte donc aucun gain
mesuré sur ces champs. Sur 8102/9102, elle sélectionne une mauvaise hypothèse
et échoue entièrement. La recherche coarse-to-fine n'est pas encore une
solution démontrée pour les champs affines fortement hétérogènes.

Pour 8103/9103 à 1024 × 576, la RMSE des déplacements finaux acceptés est
0,279 px et la RMSE des gradients affines est 0,00407. Le contrôle oracle
converge sur 68/77 fenêtres. À 4K, ces valeurs sont respectivement 0,0872 px
et 0,000851; la couverture correcte demeure cependant 12,0 % des fenêtres
visibles. Ces erreurs conditionnelles ne doivent pas être confondues avec la
couverture.

Temps de bout en bout mesurés : 0,915 s pour 1024 × 576 et 2,803 s pour 4K
sur le cas 8103/9103. Le temps 4K dépasse la cible exploratoire de 1 s. Il
comprend détection, recherche de supports, préparation de référence, RANSAC
local et solveur IC-GN. Le décodage et la génération des images synthétiques
sont exclus. La référence a été préparée à chaque cas; ces temps ne mesurent
pas le mode référence persistante en séquence.

## Interprétation

Le test montre que les ajustements affines locaux peuvent fournir des seeds
IC-GN utiles lorsque les correspondances de départ sont suffisamment bonnes.
Le résultat positif sur 8103/9103 est toutefois déjà obtenu par le contrôle
sans compensation; il valide surtout l'association matching réciproque,
consensus affine local et IC-GN sur ce cas. Il ne prouve pas que la banque
d'hypothèses améliore la récupération.

Le cas 8102/9102 demeure un échec instructif : l'oracle initialise 45 fenêtres,
mais ni les correspondances sélectionnées ni leurs modèles locaux n'en
initialisent une. Le goulot d'étranglement est en amont du solveur et nécessite
une génération/sélection d'hypothèses affine plus discriminante que la seule
banque de rotations.

Le résultat 4K est exploratoire : un seul champ hétérogène est évalué. La
couverture de 131/1 096 et le temps de 2,803 s ne satisfont pas l'objectif de
forte couverture sous une seconde. Aucune conclusion statistique générale ne
peut être tirée de ces trois évaluations.

## Reproduction

```bash
PYTHONPATH=python/src:python/benchmarks .venv/bin/python \
  python/benchmarks/qualify_local_hypothesis_icgn.py \
  --cnn-weights results/descriptors/dic_descriptor_hardneg_training_20261108/dic_patch_cnn_descriptor.npz \
  --output results/qualification/local_hypothesis_icgn_20261108/reproduced \
  --width 3840 --height 2160 \
  --candidate-budget 8192 --query-budget 2048 \
  --coarse-queries 128 --coarse-candidates 2048 \
  --textures 1 --seed 8103 --warp-offset 1000
```

Les sorties JSON conservent les statistiques par étape; les fichiers NPZ
conservent les paires directement observées, les modèles locaux, les seeds et
les résultats du solveur. Les autres cas sont exécutables en remplaçant
`--seed` et `--warp-offset`.
