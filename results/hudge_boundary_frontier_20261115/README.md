# Propagation vers les trous sur supports partiels

15 novembre 2026. Pilote niveau 191 uniquement, référence `Img_F.tif`, cible originale `Img_wa_g191.tif`. Aucune correction de contour, aucun déplacement analytique. Les images ne sont pas prises dans une séquence temporelle pour le calcul.

## Protocole

Le solveur Gauss–Newton inverse compositionnel (IC-GN) de Live-DIC est utilisé sans modification, sur WebGPU. La région d'intérêt est définie par le masque de matière observé dans le rendu. `roi_policy="partial"`, `roi_truncation="visibility"` : seule la composante matérielle visible depuis le centre contribue au support. Le conditionnement est contrôlé par la préparation native.

Départ : 784 observations affines 16 × 16 retenues, avant toute correction de contour. Chaque nouvelle fenêtre reçoit l'affine d'un voisin retenu, transportée à son centre exact. Les permissions `neighbour_bits` préparées à partir du masque interdisent les liens traversant un trou. Les voisins sont proposés par corrélation décroissante, au plus trois voisins distincts par fenêtre. Seul le front actif est résolu. Jusqu'à quatre appels de 30 itérations continuent les états non convergés finis ; tolérance 0,001 et seuil 0,65 inchangés. Aucune pénalisation de l'amplitude du gradient.

Retenue = statut convergé, paramètres finis, corrélation recalculée sur le support tronqué ≥ 0,65, échantillons du support dans la cible et déterminant affine positif. Aucun déplacement vrai n'est disponible : ce ne sont pas des mesures mécaniquement certifiées.

Les subdivisions 12 × 12 puis 8 × 8, pas 4, sont initialisées depuis les observations retenues à l'étape précédente. Un test de ligne matérielle relie les centres ; chaque nouvelle graine doit être réellement résolue au nouveau support avant de propager. Ce transfert initial n'est pas compté comme une fenêtre validée sans résolution.

## Résultats

- Support 16, pas 8 : 784 → 909 fenêtres retenues. 25 centres retenus à moins de 4 pixels d'un trou ; 42 entre 4 et 8 pixels.
- Support 12, pas 4 : 3 061 fenêtres après transfert résolu, puis 3 163 après propagation. 85 centres retenus à moins de 4 pixels ; 174 entre 4 et 8 pixels.
- Support 8, pas 4 : 2 050 après transfert résolu, puis 2 241 après propagation. La diminution du support perd de la couverture ; elle n'est pas retenue comme remplacement général.

Sur la bande 0–4 pixels, le support 12 couvre 2 793 / 2 844 pixels, mais la corrélation des intensités sur cette matière couverte reste à 0,640. Sur la bande 4–8 pixels, il couvre 3 635 / 3 637 pixels, corrélation 0,883. Le support 8 atteint 0,703 sur seulement 1 878 / 2 844 pixels de la première bande : cette augmentation conditionnelle ne démontre pas une amélioration sur tous les pixels.

**Le bord est désormais atteint par des calculs, mais le bord immédiat n'est pas correctement recalé partout.** La couverture d'un support ne prouve pas la précision de son déplacement.

## Remapping et assemblage : limitation ouverte

Les pixels sont échantillonnés dans la cible originale après combinaison des déplacements des seules fenêtres retenues qui les contiennent. Il n'y a pas d'extrapolation dans les zones non couvertes. Les images de la galerie montrent la matière non couverte en magenta et les trous/extérieurs par des hachures : la forme des trous hachurés n'est pas un résultat de recalage.

Les affines individuelles ont un déterminant positif, mais leur combinaison peut créer des replis. Le diagnostic par différences finies, sur les pixels couverts dont le voisinage direct est couvert, détecte 29 déterminants non positifs avec le support 16 et 125 avec le support 12. **Le champ assemblé n'est donc pas qualifié comme transformation matérielle globale.** Les paramètres locaux et les champs d'illustration restent séparés dans les exports. La corrélation globale ne doit pas servir à masquer ce défaut.

## Reproduction

Depuis la racine WGPU-Matching, avec l'environnement `.venv` et l'installation locale du solveur :

```bash
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --output /tmp/boundary16
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 12 --stride 4 --seed-archive /tmp/boundary16/seeds.npz --output /tmp/boundary12
.venv/bin/python python/benchmarks/qualify_hudge_boundary_frontier.py --size 8 --stride 4 --seed-archive /tmp/boundary12/seeds.npz --output /tmp/boundary8
```

Les sorties doivent être nouvelles. Les dossiers `partial16`, `partial12`, `partial8` conservent tous les parents, propositions, statuts, contrôles et paramètres par anneau. `comparison.json` compare l'ancien champ corrigé et les nouveaux champs sur les mêmes pixels couverts pour chaque configuration.

La résolution est sur WebGPU. L'orchestration du front, les contrôles et le remapping de ce prototype sont sur processeur central. Le solveur externe n'a pas été modifié. Le test de transport affine et les trois tests de supports adaptatifs passent ; Ruff passe sur les nouveaux scripts.
